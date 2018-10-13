'''
Created on Oct 10th, 2018
@author: Rio Li
'''

import numpy as np
from numpy.dual import inv
import matplotlib.pyplot as plt
from os import path
import logging


class Model:
    @property
    def theta(self):
        return self._theta

    @property
    def plot(self):
        return self._plot
    
    def theta(self, x, y):
        logging.INFO("Computing theta...")

    def predictor(self, x):
        logging.INFO("Predicting...")
        return x.T.dot(self.theta)

# least-square
class LS(Model):
    # parameter estimate theta for least square
    def fit(self, x, y):
        self.theta = inv(x.dot(x.T)).dot(x).dot(y)



# L1-regularized LS
class LASSO(Model):
    def fit(self, x, y, l):
        pass

# regularized least-square
class RLS:
    # parameter estimate theta for regularized least square
    def fit(self, x, y, l=None):
        if l is None:
            self.theta = inv(x.dot(x.T) + 1).dot(x).dot(y)
        else:   
            self.theta = inv(x.dot(x.T) + l).dot(x).dot(y)