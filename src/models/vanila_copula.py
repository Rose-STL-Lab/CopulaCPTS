import numpy as np
import pandas as pd
import torch
from copulae import GumbelCopula
from copulae.core import pseudo_obs
from .utils import search_alpha

def gumbel_copula_loss(x, cop, data, epsilon):
    return np.fabs(cop.cdf([x] * data.shape[1]) - 1 + epsilon)


def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(np.mean(np.all(np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1)
                           ) - 1 + epsilon)

class vanila_copula:
    '''
    copula as implemented by Messoudi et al.
    '''
    def __init__(self, model, cali_x, cali_y):
        
        self.model = model

        self.cali_x = cali_x
        self.cali_y = cali_y

        self.nonconformity = None
        self.results_dict = {}

    def calibrate(self):

        pred_y = self.model.predict(self.cali_x)
        nonconformity = torch.norm((pred_y-self.cali_y), p=2, dim = -1).detach().numpy()
        self.nonconformity = nonconformity


    def predict(self, X_test, epsilon=0.1):

        #alphas = self.nonconformity
        pred_y = self.model.predict(self.cali_x)
        scores = torch.norm((pred_y-self.cali_y), p=2, dim = -1).detach().numpy()
        alphas = []
        for i in range(scores.shape[0]):
            a = (scores[i]>self.nonconformity).mean(axis=0)
            alphas.append(a)
        alphas = np.array(alphas)

        x_candidates = np.linspace(0.0001, 0.999, num=300)
        x_fun = [empirical_copula_loss(x, alphas, epsilon) for x in x_candidates]
        x_sorted = sorted(list(zip(x_fun, x_candidates)))

        mapping_shape = self.nonconformity.shape[0]
        mapping = {i: sorted(self.nonconformity[:, i].tolist()) for i in range(alphas.shape[1])}
        quantile = int(x_sorted[0][1]*alphas.shape[0])
        
        radius = np.array([mapping[i][quantile] for i in range(alphas.shape[1])])

        y_pred = self.model.predict(X_test)
        
        self.results_dict[epsilon] = {'y_pred': y_pred, 'radius': radius}

        return y_pred, radius


    def calc_area(self, radius):
        
        area = sum([np.pi * r**2 for r in radius])

        return area


    def calc_area_3d(self, radius):
        
        area = sum([4/3.0 * np.pi * r**3 for r in radius])

        return area

    def calc_area_1d(self, radius):
        
        area = sum(radius)

        return area

    def calc_coverage(self, radius, y_pred, y_test):

        testnonconformity = torch.norm((y_pred-y_test), p=2, dim = -1).detach().numpy()

        circle_covs = []
        for j in range(y_test.shape[-2]):
            circle_covs.append(testnonconformity[:,j]<radius[j])
        

        circle_covs = np.array(circle_covs)
        coverage = np.mean(np.all(circle_covs, axis=0))
        return coverage


    def calc_coverage_3d(self, radius, y_pred, y_test):
        
        return self.calc_coverage(radius, y_pred, y_test)


    def calc_coverage_1d(self, radius, y_pred, y_test):
        
        return self.calc_coverage(radius, y_pred, y_test)



