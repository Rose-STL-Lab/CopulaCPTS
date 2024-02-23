
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pickle
from src.models import rnn, lstm, bjrnn, cfrnn, copulaCPTS, dplstm, vanila_copula


datapath = '../../robotics_project/PythonRobotics/AerialNavigation/drone_3d_trajectory_following/data/dataset_smallnoise.npy'

data = np.load(datapath, allow_pickle=True)
x_downsample = data[:,::3,:].astype(np.float32)

train_len=60
pred_len=10

x_train = torch.tensor(x_downsample[:600,:train_len,:], dtype=torch.float)
y_train = torch.tensor(x_downsample[:600,train_len:train_len+pred_len,:3], dtype=torch.float)

x_cali = torch.tensor(x_downsample[600:800,:train_len,:], dtype=torch.float)
y_cali = torch.tensor(x_downsample[600:800,train_len:train_len+pred_len,:3], dtype=torch.float)

x_test = torch.tensor(x_downsample[800:,:train_len,:], dtype=torch.float)
y_test = torch.tensor(x_downsample[800:, train_len:train_len+pred_len,:3], dtype=torch.float)

def experiment(x_downsample, train_len=60, pred_len = 10, name='exp', 
            x_train=x_train,
            y_train=y_train,
            x_cali=x_cali,
            y_cali=y_cali,
            x_test=x_test,
            y_test=y_test):

    rnn_model = rnn.rnn(embedding_size=128, input_size=6, output_size=3, horizon=pred_len)
    encdec_model = lstm.lstm_seq2seq(input_size=6, output_size=3, embedding_size=128, target_len = pred_len)
    models = [rnn_model,encdec_model]
    
    models[0].train_model(x_train,y_train, n_epochs=500, batch_size=150)
    #models[1].train_model(x_train,y_train, n_epochs=500, batch_size=150)

    with open('./trained/models_%s.pkl'%name, 'wb') as f:
        pickle.dump(models , f)
    
        
    UQ = {}

    
    bj_class = bjrnn.bj_rnn(models[0], recursion_depth=10)
    UQ['bjrnn'] = bj_class   
    
    dprnn = dplstm.DPRNN(epochs=150,  input_size=6, output_size=3, n_steps=pred_len, batch_size=64, dropout_prob=0.1)
    dprnn.fit(x_train, y_train)
    UQ['dprnn'] = dprnn
    
    cf = cfrnn.CFRNN(models[0], x_cali, y_cali)
    cf.calibrate()
    UQ['cfrnn'] = cf
    
    cf = cfrnn.CFRNN(models[1], x_cali, y_cali)
    cf.calibrate()
    UQ['cf-EncDec'] = cf

    copula = copulaCPTS.copulaCPTS(models[0], x_cali, y_cali)
    copula.calibrate()
    UQ['copula-rnn'] = copula
    
    copula = copulaCPTS.copulaCPTS(models[1], x_cali, y_cali)
    copula.calibrate()
    UQ['copula-EncDec'] = copula
    
    
    
    areas = {}
    coverages = {}

    epsilon_ls = np.linspace(0.05, 0.50, 10)

    for k, uqmethod in UQ.items():
        print(k)
       
        area = []
        coverage = []
        for eps in epsilon_ls:
            pred, box = uqmethod.predict(x_test, epsilon=eps)
            area.append(uqmethod.calc_area_3d(box))
            pred = torch.tensor(pred)
            coverage.append(uqmethod.calc_coverage_3d(box, pred, y_test))
        areas[k] = area
        coverages[k] = coverage
    
    with open('./trained/uq_%s.pkl'%name, 'wb') as f:
        pickle.dump(UQ , f)
    with open('./trained/results_%s.pkl'%name, 'wb') as f:
        pickle.dump((areas, coverages) , f)
        
    return areas, coverages, (models, UQ)

def main():
    lens = [1, 5, 10, 15]
    for pred_len in lens:
        x_train = torch.tensor(x_downsample[:600,:train_len,:], dtype=torch.float)
        y_train = torch.tensor(x_downsample[:600,train_len:train_len+pred_len,:3], dtype=torch.float)

        x_cali = torch.tensor(x_downsample[600:800,:train_len,:], dtype=torch.float)
        y_cali = torch.tensor(x_downsample[600:800,train_len:train_len+pred_len,:3], dtype=torch.float)

        x_test = torch.tensor(x_downsample[800:,:train_len,:], dtype=torch.float)
        y_test = torch.tensor(x_downsample[800:, train_len:train_len+pred_len,:3], dtype=torch.float)

        for i in range(3):
            res = experiment(x_downsample, pred_len= pred_len, 
                name='drone_split_len_'+str(pred_len)+'_run_'+str(i),
                x_train=x_train,
                y_train=y_train,
                x_cali=x_cali,
                y_cali=y_cali,
                x_test=x_test,
                y_test=y_test
                )
            print('finished run ' + str(pred_len))
            del res 

    
    for i in range(3):
        res = experiment(x_downsample, name='drone_vanila_'+str(i))
        print('finished run ' + str(i))
        del res 


if __name__ == '__main__':

    main()
