import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score




class BasePredictor:
    def __init__(self, model:BaseEstimator, d_path, param_grid = None, n_splits = 5):
        self.d_path             = d_path
        self.t_data             = pd.read_csv(self.d_path)

        self.model              = model
        self.models             = []
        
        self.params             = param_grid

        self.n                  = n_splits
        self.skf                = StratifiedKFold(n_splits=self.n, shuffle=False)

        self.data               = {}
        self.data['X']          = self.t_data.iloc[:,:-1]
        self.data['Y']          = self.t_data.iloc[:,-1]

        self.preds              = []

        self.rmses              = []
        self.confs              = []
        self.accs               = []
        self.f1s                = []
        self.sens               = []
        self.specs              = []


    def optimise_hyperparams(self):
        ## Get best params for current split
        self.gridsearch         = GridSearchCV(estimator = self.model, param_grid = self.params, cv = self.skf)

        self.gridsearch.fit(self.data['X'], self.data['Y'])
        self.best_params        = self.gridsearch.best_params_


    def train(self):
        ## Train-Test split
        self.Xtr            = np.array(self.data['X'].iloc[self.tr])
        self.Xts            = np.array(self.data['X'].iloc[self.ts])

        self.Ytr            = np.array(self.data['Y'].iloc[self.tr])
        self.Yts            = np.array(self.data['Y'].iloc[self.ts])
        
        self.models.append(self.model)

        ## Fit the model on the entire training set
        self.model.fit(self.Xtr,self.Ytr)


    def predict(self):
        self.Ypr            = self.model.predict(self.Xts)
        self.preds.append(self.Ypr)


    def run(self, features = None, select_path = "feature_select/feature_tests.xlsx"):
        ## Select features to be considered, if not given use the most optimal
        if(features):
            self.data['X']      = self.data['X'][features]
        elif(select_path):
            self.select         = pd.read_excel(select_path)
            self.params         = list(self.select[self.select['Significant'] == 'S']['Feature'])

            self.data['X']      = self.data['X'][self.params]
        else:
            self.params         = None


        ## Get best hyperparams for current iteration
        if(self.params):
            self.optimise_hyperparams()

            for param,val in self.best_params.items():
                setattr(self.model,param,val)


        ## Train-test split (stratified to account for class imbalance)
        for train_ind, test_ind in self.skf.split(self.data['X'],self.data['Y']):
            
            ## Split into training set and testing set
            self.tr             = train_ind
            self.ts             = test_ind

            ## Train the model on the current split
            self.train()

            ## Get predictions for the current split
            self.predict()

            ## Get RMSE error and Confusion matrix for current split
            self.get_errors()

            ## Get performance metrics for current split
            self.get_metrics()

    
    def get_metrics(self):
        ## Accuracy
        self.acc                = accuracy_score(self.Yts,self.Ypr)
        self.accs.append(self.acc)

        ## Sensitivity and Specificity
        TN, FP, FN, TP          = self.conf.ravel()
        self.sen                = (TP / (TP+FN)) * 100
        self.spec               = (TN / (TN+FP)) * 100

        self.sens.append(self.sen)
        self.specs.append(self.spec)

        ## F1-scores
        self.f1                 = f1_score(self.Yts,self.Ypr,average='weighted')
        self.f1s.append(self.f1)


    def get_errors(self):
        ## RMSE
        self.rmse               = mean_squared_error(y_true = self.Yts, y_pred = self.Ypr, squared = False)
        self.rmses.append(self.rmse)

        ## Binary Confusion Matrix
        self.conf               = confusion_matrix(y_true = self.Yts, y_pred = self.Ypr)
        self.confs.append(self.conf)