import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, f1_score, make_scorer\




class BasePredictor:
    def __init__(self, model:BaseEstimator, d_path, target, param_grid = None, n_splits = 5):
        self.d_path             = d_path
        self.t_data             = pd.read_csv(self.d_path)

        self.target             = target

        self.model              = model
        
        self.params             = param_grid

        self.best_params        = None

        self.n                  = n_splits
        self.skf                = StratifiedKFold(n_splits=self.n, shuffle=False)

        self.data               = {}
        self.data['X']          = self.t_data.drop(columns=[target],axis=1)
        self.data['Y']          = self.t_data[target]

        self.data_re            = {}

        self.preds              = []
        self.preds_re           = []
        self.preds_un           = []

        self.splits             = []
        self.splits_re          = []

        self.rmses_re           = []
        self.confs_re           = []
        self.accs_re            = []
        self.f1s_re             = []
        self.sens_re            = []
        self.specs_re           = []

        self.rmses_un           = []
        self.confs_un           = []
        self.accs_un            = []
        self.f1s_un             = []
        self.sens_un            = []
        self.specs_un           = []


    def optimise_hyperparams(self, data):
        ## Custom scoring metric for GridSearch
        def custom_metric(Y_true, Y_pred):
            tn, fp, fn, tp      = confusion_matrix(y_true = Y_true, y_pred = Y_pred).ravel()

            sensitivity         = (float(tp)/(tp+fn))
            specificity         = (float(tn)/(tn+fp))

            return (sensitivity * specificity)
        
        scorer = make_scorer(custom_metric, greater_is_better = True)


        ## Get best params for current split
        self.gridsearch         = GridSearchCV(estimator = self.model, param_grid = self.params, cv = self.skf, scoring = scorer)

        self.gridsearch.fit(data['X'], data['Y'])
        
        self.best_params        = self.gridsearch.best_params_
        
        self.model              = self.gridsearch.best_estimator_


    def train(self, data):
        # ## Train-Test split
        # self.Xtr            = np.array(data['X'].iloc[self.tr])
        # self.Xts            = np.array(data['X'].iloc[self.ts])

        # self.Ytr            = np.array(data['Y'].iloc[self.tr])
        # self.Yts            = np.array(data['Y'].iloc[self.ts])

        ## Fit the model on the entire training set
        self.model.fit(self.Xtr,self.Ytr)


    def predict(self,lis):
        self.Ypr            = self.model.predict(self.Xts)
        lis.append(self.Ypr)


    def run(self, features = None, select_path = None):
        ## Select features to be considered, if not given use the most optimal
        if(features):
            self.data['X']      = self.data['X'][features]
        elif(select_path):
            self.select         = pd.read_excel(select_path)
            self.features       = list(self.select[self.select['Significant'] == 'S']['Feature'])

            self.data['X']      = self.data['X'][self.params]

        
        ## Resampling
        ## Development data, kept separate from unseen data
        X_dev, X_un, Y_dev, Y_un    = train_test_split(self.data['X'], self.data['Y'], test_size=0.2, random_state=0)
        self.dev            = pd.concat([X_dev,Y_dev],axis=1)
        
        minority            = self.dev[self.dev[self.target] == 1]
        majority            = self.dev[self.dev[self.target] == 0]

        ## Resample to 1000 sampled for each class
        min_resampled       = resample(minority, n_samples=1200, replace=True, random_state=8)
        maj_resampled       = resample(majority, n_samples=1200, replace=True, random_state=8)
        
        ## Revised dataset
        self.data_re['X']   = pd.concat([maj_resampled, min_resampled],axis=0)
        self.data_re['Y']   = self.data_re['X'][self.target]
        self.data_re['X']   = self.data_re['X'].drop(labels = [self.target], axis=1)

        
        ## Get best hyperparams for current iteration
        if(self.params):
            self.optimise_hyperparams(self.data)

        
        # ## Stratified splitting on the dev data
        # for tr_ind, ts_ind in self.skf.split(self.data_re['X'], self.data_re['Y']):

            # ## Split into training and testing set
            # self.tr         = tr_ind
            # self.ts         = ts_ind

        self.Xtr, self.Xts, self.Ytr, self.Yts = train_test_split(self.data['X'], self.data['Y'], test_size=0.3, stratify=self.data['Y'], random_state=8)

        ## Train the model on the current split
        self.train(self.data_re)

        ## Get predictions for the current split
        self.predict(self.preds_re)

        ## Get RMSE score and Confusion matrix for current split
        self.get_errors(self.rmses_re, self.confs_re)

        ## Get performance metrics for current split
        self.get_metrics(self.accs_re, self.sens_re, self.specs_re, self.f1s_re)

        
        ## Results on unseen data
        self.Xts        = X_un
        self.Yts        = Y_un

        ## Predict for unseen data
        self.predict(self.preds_un)

        ## Get errors for unseen data
        self.get_errors(self.rmses_un, self.confs_un)

        ## Get performance metrics for unseen data
        self.get_metrics(self.accs_un, self.sens_un, self.specs_un, self.f1s_un)

    
    def get_metrics(self,l_acc,l_sens,l_specs,l_f1s):
        ## Accuracy
        self.acc                = accuracy_score(self.Yts,self.Ypr)
        l_acc.append(self.acc)

        ## Sensitivity and Specificity
        TN, FP, FN, TP          = self.conf.ravel()
        self.sen                = (TP / (TP+FN)) * 100
        self.spec               = (TN / (TN+FP)) * 100

        l_sens.append(self.sen)
        l_specs.append(self.spec)

        ## F1-scores
        self.f1                 = f1_score(self.Yts,self.Ypr,average='weighted')
        l_f1s.append(self.f1)


    def get_errors(self,l_rmse,l_conf):
        ## RMSE
        self.rmse               = mean_squared_error(y_true = self.Yts, y_pred = self.Ypr, squared = False)
        l_rmse.append(self.rmse)

        ## Binary Confusion Matrix
        self.conf               = confusion_matrix(y_true = self.Yts, y_pred = self.Ypr)
        l_conf.append(self.conf)