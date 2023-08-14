from __future__ import print_function, absolute_import

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator





class BasePredictor:
    def __init__(self, d_path, s_path, model:BaseEstimator, param_grid = None, n_splits = 5):
        self.d_path             = d_path
        self.s_path             = s_path

        self.t_data             = pd.read_csv(self.d_path)
        self.select             = pd.read_csv(self.s_path)

        self.model              = model
        self.params             = param_grid

        self.n                  = n_splits
        self.skf                = StratifiedKFold(n_splits=self.n, random_state=0, shuffle=False)

        self.data               = {}


    def read_data(self):
        self.data['X']          = self.t_data.iloc[:,:-1]
        self.data['Y']          = self.t_data.iloc[:,-1]


    def select_features(self):
        self.select             = self.select[self.select['Significant'] == 'S']['Feature']

        self.data['X']          = self.data['X'][list(self.select['Feature'])]
    

    def prepare_data(self):
        self.read_data()
        self.select_features()

        self.X_train            = self.data['X']
        self.Y_train            = self.data['Y']


    def train(self, data_path, selection_path):
        