import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import pickle
import os 


from models import rnn, lstm, bjrnn, cfrnn, copulaCPTS, dplstm

def experiment(train, valid, test, name='exp'):
    rnn_model = rnn.rnn(embedding_size=24, input_size=2, output_size=2, horizon=24)
    encdec_model = lstm.lstm_seq2seq( input_size=2, embedding_size=24, target_len = 24)
    models = [rnn_model, encdec_model]
    
    a = np.load(train)
    x_train = torch.tensor(a[:,:35,0,:], dtype=torch.float)
    y_train = torch.tensor(a[:,35:,0,:], dtype=torch.float)
    
    for m in models:
        m.train_model(x_train,y_train, n_epochs=150, batch_size=150)
    
    with open('./trained/models_%s.pkl'%name, 'wb') as f:
        pickle.dump(models , f)
        
    b = np.load(valid)
    x_cali = torch.tensor(b[:,:35,0,:], dtype=torch.float)
    y_cali = torch.tensor(b[:,35:,0,:], dtype=torch.float)
    
    UQ = {}
    
    print('dprnn')
    dprnn = dplstm.DPRNN(epochs=90,  input_size=2, output_size=2, n_steps=24, dropout_prob=0.1)
    dprnn.fit(x_train, y_train)
    UQ['dprnn'] = dprnn

    bj_class = bjrnn.bj_rnn(models[0],recursion_depth=20)
    UQ['bjrnn'] = bj_class   

    cf = cfrnn.CFRNN(models[0], x_cali, y_cali)
    cf.calibrate()
    UQ['cfrnn'] = cf
    
    copula = copulaCPTS.copulaCPTS(models[0], x_cali, y_cali)
    copula.calibrate()
    UQ['copula-rnn'] = copula

    cf = cfrnn.CFRNN(models[1], x_cali, y_cali)
    cf.calibrate()
    UQ['cf-EncDec'] = cf

    copula = copulaCPTS.copulaCPTS(models[1], x_cali, y_cali)
    copula.calibrate()
    UQ['copula-EncDec'] = copula

    

    c = np.load(test)
    x_test = torch.tensor(c[:,:35,0,:], dtype=torch.float)
    y_test = torch.tensor(c[:,35:,0,:], dtype=torch.float)

    areas = {}
    coverages = {}

    epsilon_ls = np.linspace(0.05, 0.50, 10)

    for k, uqmethod in UQ.items():
        print(k)
        area = []
        coverage = []
        for eps in epsilon_ls:
            pred, box = uqmethod.predict(x_test, epsilon=eps)
            area.append(uqmethod.calc_area(box))
            pred = torch.tensor(pred)
            coverage.append(uqmethod.calc_coverage(box, pred, y_test))
        areas[k] = area
        coverages[k] = coverage
    
    with open('./trained/uq_%s.pkl'%name, 'wb') as f:
        pickle.dump(UQ , f)
    with open('./trained/results_%s.pkl'%name, 'wb') as f:
        pickle.dump((areas, coverages) , f)
        
    return areas, coverages, (models, UQ)


def main():

    os.makedirs("trained", exist_ok = True)

    train = 'nridata/loc_train_springs2_noise_0.05.npy'
    valid = 'nridata/loc_valid_springs2_noise_0.05.npy'
    test = 'nridata/loc_test_springs2_noise_0.05.npy'

    for i in range(3):
        res = experiment(train, valid, test, name='particle5_run'+str(i))
        print('run ' +str(i)+ 'done')
        del res

    train = 'nridata/loc_train_springs2_noise_0.01.npy'
    valid = 'nridata/loc_valid_springs2_noise_0.01.npy'
    test = 'nridata/loc_test_springs2_noise_0.01.npy'

    for i in range(3):
        res = experiment(train, valid, test, name='particle1_run'+str(i))
        print('run ' +str(i)+ 'done')
        del res



if __name__ == '__main__':
    main()
    