import numpy as np
import pandas as pd
import torch
from copulae.core import pseudo_obs
from .utils import search_alpha


def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(
        np.mean(
            np.all(
                np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1
            )
        )
        - 1
        + epsilon
    )


class copulaCPTS:
    def __init__(self, model, cali_x, cali_y):
        """
        Copula conformal prediction with two-step calibration.
        """
        self.model = model

        self.cali_x = None
        self.cali_y = None
        self.copula_x = None
        self.copula_y = None
        self.split_cali(cali_x, cali_y)

        self.nonconformity = None
        self.results_dict = {}

    def split_cali(self, cali_x, cali_y, split=0.6):
        if self.copula_x:
            print("already split")
            return
        size = cali_x.shape[0]
        halfsize = int(split * size)

        idx = np.random.choice(range(size), halfsize, replace=False)

        self.cali_x = cali_x[idx]
        self.copula_x = cali_x[list(set(range(size)) - set(idx))]
        self.cali_y = cali_y[idx]
        self.copula_y = cali_y[list(set(range(size)) - set(idx))]

    def calibrate(self):

        pred_y = self.model.predict(self.cali_x)
        nonconformity = torch.norm((pred_y - self.cali_y), p=2, dim=-1).detach().numpy()
        self.nonconformity = nonconformity

    def calibrate_l1(self):

        pred_y = self.model.predict(self.cali_x)
        nonconformity = torch.norm((pred_y - self.cali_y), p=1, dim=-1).detach().numpy()
        self.nonconformity = nonconformity

    def predict(self, X_test, epsilon=0.1):

        # alphas = self.nonconformity
        pred_y = self.model.predict(self.copula_x)
        scores = torch.norm((pred_y - self.copula_y), p=2, dim=-1).detach().numpy()
        alphas = []
        for i in range(scores.shape[0]):
            a = (scores[i] > self.nonconformity).mean(axis=0)
            alphas.append(a)
        alphas = np.array(alphas)

        threshold = search_alpha(alphas, epsilon, epochs=800)

        mapping_shape = self.nonconformity.shape[0]
        mapping = {
            i: sorted(self.nonconformity[:, i].tolist()) for i in range(alphas.shape[1])
        }

        quantile = []
        mapping_shape = self.nonconformity.shape[0]

        for i in range(alphas.shape[1]):
            idx = int(threshold[i] * mapping_shape) + 1
            if idx >= mapping_shape:
                idx = mapping_shape - 1
            quantile.append(mapping[i][idx])

        radius = np.array(quantile)

        y_pred = self.model.predict(X_test)

        self.results_dict[epsilon] = {"y_pred": y_pred, "radius": radius}

        return y_pred, radius

    def calc_area(self, radius):

        area = sum([np.pi * r**2 for r in radius])

        return area

    def calc_area_l1(self, radius):

        area = sum([2 * r**2 for r in radius])

        return area

    def calc_area_3d(self, radius):

        area = sum([4 / 3.0 * np.pi * r**3 for r in radius])

        return area

    def calc_area_1d(self, radius):

        area = sum(radius)

        return area

    def calc_coverage(self, radius, y_pred, y_test):

        testnonconformity = torch.norm((y_pred - y_test), p=2, dim=-1).detach().numpy()

        circle_covs = []
        for j in range(y_test.shape[-2]):
            circle_covs.append(testnonconformity[:, j] < radius[j])

        circle_covs = np.array(circle_covs)
        coverage = np.mean(np.all(circle_covs, axis=0))
        return coverage

    def calc_coverage_l1(self, radius, y_pred, y_test):
        testnonconformity = (
            torch.norm((y_pred - y_test), p=1, dim=-1).detach().numpy()
        )  # change back to p=2
        circle_covs = []
        for j in range(y_test.shape[-2]):
            circle_covs.append(testnonconformity[:, j] < radius[j])

        circle_covs = np.array(circle_covs)
        coverage = np.mean(np.all(circle_covs, axis=0))
        return coverage

    def calc_coverage_3d(self, radius, y_pred, y_test):

        return self.calc_coverage(radius, y_pred, y_test)

    def calc_coverage_1d(self, radius, y_pred, y_test):

        return self.calc_coverage(radius, y_pred, y_test)
