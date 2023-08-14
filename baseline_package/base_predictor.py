from __future__ import print_function, absolute_import

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator





class BasePredictor:
    def __init__(self, model:BaseEstimator, param_grid = None, standardize = True, n_splits = 5, split_size = 0.2):
        self.model              = model
        self.params             = param_grid
        self.standardize        = standardize

        self.n                  = n_splits
        self.skf                = StratifiedKFold(n_splits=self.n, random_state=0, shuffle=False)

        self.X_train            = []
        self.X_test             = []
        self.Y_train            = []
        self.Y_test             = []

        self.split              = split_size

    
    def prepare_data(self,data_dict):
        self.X_train            = data_dict['X']
        self.Y_train            = data_dict['Y']