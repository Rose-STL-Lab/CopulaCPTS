

# adapted from Kamilė Stankevičiūtė
# https://github.com/kamilest/conformal-rnn/tree/5f6dc9e3118bcea631745391f4efb246733a11c7

""" CFRNN model. """


import numpy as np
import torch

class CFRNN:
    """
    The auxiliary RNN issuing point predictions.
    Point predictions are used as baseline to which the (normalised)
    uncertainty intervals are added in the main CFRNN network.
    """

    def __init__(self, model, cali_x, cali_y):
        """
        Initialises the auxiliary forecaster.
        Args:
            embedding_size: hyperparameter indicating the size of the latent
                RNN embeddings.
            input_size: dimensionality of the input time-series
            output_size: dimensionality of the forecast
            horizon: forecasting horizon
            rnn_mode: type of the underlying RNN network
            path: optional path where to save the auxiliary model to be used
                in the main CFRNN network
        """
        self.model = model

        self.cali_x = cali_x
        self.cali_y = cali_y
        self.nonconformity = None
        self.results_dict = {}



    def calibrate(self):
        dim = self.cali_y.shape[-1]
        pred_y = self.model.predict(self.cali_x)
        nonconformity = torch.norm((pred_y[...,:dim]-self.cali_y), p=2, dim = -1).detach().numpy()
        self.nonconformity = nonconformity

    def calibrate_l1(self):
        dim = self.cali_y.shape[-1]
        pred_y = self.model.predict(self.cali_x)
        nonconformity = torch.norm((pred_y[...,:dim]-self.cali_y), p=1, dim = -1).detach().numpy()
        self.nonconformity = nonconformity



    def predict(self, X_test, epsilon=0.1):

        nonconformity = self.nonconformity
        n_calibration = nonconformity.shape[0] 
        new_quantile = min((n_calibration + 1.0) * (1 - (epsilon / self.cali_y.shape[-2])) / n_calibration, 1)     
        
        radius = [np.quantile(nonconformity[:,r], new_quantile) for r in range(nonconformity.shape[1])]
        y_pred = self.model.predict(X_test)
        
        self.results_dict[epsilon] = {'y_pred': y_pred, 'radius': radius}

        return y_pred, radius


    def calc_area(self, radius):
        
        area = sum([np.pi * r**2 for r in radius])

        return area

    def calc_area_l1(self, radius):
        
        area = sum([2 * r**2 for r in radius])

        return area


    def calc_area_3d(self, radius):
        
        area = sum([4/3.0 * np.pi * r**3 for r in radius])

        return area

    def calc_area_1d(self, radius):
        
        area = sum(radius)

        return area


    def calc_coverage(self, radius, y_pred, y_test):
        dim = y_test.shape[-1]
        testnonconformity = torch.norm((y_pred[...,:dim]-y_test), p=2, dim = -1).detach().numpy() 

        circle_covs = []
        for j in range(y_test.shape[-2]):
            circle_covs.append(testnonconformity[:,j]<radius[j])

        circle_covs = np.array(circle_covs)
        coverage = np.mean(np.all(circle_covs, axis=0))
        return coverage

    def calc_coverage_l1(self, radius, y_pred, y_test):
        dim = y_test.shape[-1]
        testnonconformity = torch.norm((y_pred[...,:dim]-y_test), p=1, dim = -1).detach().numpy() 

        circle_covs = []
        for j in range(y_test.shape[-2]):
            circle_covs.append(testnonconformity[:,j]<radius[j])

        circle_covs = np.array(circle_covs)
        coverage = np.mean(np.all(circle_covs, axis=0))
        return coverage




    def calc_coverage_3d(self, radius, y_pred, y_test):
        
        return self.calc_coverage(radius, y_pred[:3], y_test[:3])

    def calc_coverage_1d(self, radius, y_pred, y_test):
        
        return self.calc_coverage(radius, y_pred, y_test)

