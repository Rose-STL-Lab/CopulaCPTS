# Copyright (c) 2021, Kamilė Stankevičiūtė
# Adapted from Ahmed M. Alaa github.com/ahmedmalaa/rnn-blockwise-jackknife
# Licensed under the BSD 3-clause license

import numpy as np
import torch

from .influence import *

# influence_function, perturb_model_


class bj_rnn:
    def __init__(self, model, mode="stochastic", damp=1e-4, rnn_mode='RNN', recursion_depth=20):

        self.model = model
        self.rnn_mode = rnn_mode
        self.IF = influence_function(model, train_index=list(range(model.X.shape[0])[:200]), mode=mode, damp=damp, recursion_depth=recursion_depth)

        #X_ = [model.X[k][: int(torch.sum(model.masks[k, :]).detach().numpy())] for k in range(model.X.shape[0])]
        X_ = model.X
        self.LOBO_residuals = []
        self.dim = model.output_size
        self.variable_preds = None

        for k in range(len(self.IF)):
            perturbed_models = perturb_model_(self.model, self.IF[k])

            self.LOBO_residuals.append(
                np.abs(
                    np.array(self.model.y[k].unsqueeze(0)) - np.array(perturbed_models.predict(X_[k].unsqueeze(0)).detach())
                )
            )

            del perturbed_models

        self.LOBO_residuals = np.squeeze(np.array(self.LOBO_residuals))

        self.results_dict = {}


    def variable_predict(self, X_test):
        variable_preds = []

        for k in range(len(self.IF)):
            perturbed_models = perturb_model_(self.model, self.IF[k])
            variable_preds.append(perturbed_models.predict(X_test).detach().numpy())

            del perturbed_models

        variable_preds = np.array(variable_preds)
        self.variable_preds = variable_preds



    def predict(self, X_test, epsilon=0.1):
        
        if self.variable_preds is None:
                self.variable_predict(X_test)

        variable_preds = self.variable_preds

        num_sequences = X_test.shape[0]
        y_l_approx = np.zeros(variable_preds.shape[1:])
        y_u_approx = np.zeros(variable_preds.shape[1:])

        if len(self.LOBO_residuals.shape) == 2:
            self.LOBO_residuals = np.expand_dims(self.LOBO_residuals, axis=1)

        for i in range(self.dim):
            y_u_approx[:,:,i] = np.quantile(
                variable_preds[...,i] + np.repeat(np.expand_dims(self.LOBO_residuals[...,i], axis=1), num_sequences, axis=1)*5,
                1 - epsilon / 2,
                axis=0,
                keepdims=False,
            )
            y_l_approx[:,:,i] = np.quantile(
                variable_preds[...,i] - np.repeat(np.expand_dims(self.LOBO_residuals[...,i], axis=1), num_sequences, axis=1)*5,
                1 - epsilon / 2,
                axis=0,
                keepdims=False,
            )
        y_pred = self.model.predict(X_test)

        self.results_dict[epsilon] = {'y_pred': y_pred, 'y_l_approx': y_l_approx, 'y_u_approx': y_u_approx}

        return y_pred, (y_l_approx, y_u_approx)


    def calc_area(self, box):

        y_l_approx, y_u_approx = box
        delta = y_u_approx - y_l_approx
        all_area = np.multiply(delta[:,:,0],delta[:,:,1]) # (num_samples, len)
        area = all_area.sum(axis=1).mean()
        return area

    def calc_area_1d(self, box):

        y_l_approx, y_u_approx = box
        delta = y_u_approx - y_l_approx
        area = delta.sum(axis=1).mean()
        return area

    def calc_area_3d(self, box):

        y_l_approx, y_u_approx = box
        delta = y_u_approx - y_l_approx
        all_area = np.multiply(delta[:,:,0],delta[:,:,1], delta[:,:,2]) # (num_samples, len)
        area = all_area.sum(axis=1).mean()
        return area


    def calc_coverage(self, box, y_pred, y_test):

        y_l_approx, y_u_approx = box
        lower = y_test.detach().numpy()>y_l_approx
        upper = y_test.detach().numpy()<y_u_approx
        an = np.logical_and(lower, upper)
        cov = np.all(np.all(an, axis = 2), axis=1).mean()
        return cov

    def calc_coverage_3d(self, box, y_pred, y_test):

        return self.calc_coverage(box, y_pred, y_test)


    def calc_coverage_1d(self, box, y_pred, y_test):

        return self.calc_coverage(box, y_pred, y_test)

        