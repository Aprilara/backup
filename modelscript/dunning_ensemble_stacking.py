# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:56:49 2017

@author: Administrator
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.externals import joblib

class Ensemble(object):
    
    def __init__(self, n_folds, save_path):
        self.n_folds = n_folds
#       self.stacker = stacker
#        self.base_models = base_models
        self.save_path = save_path
        
    def fit_predict(self,stacker, base_models, X, y, T):
        self.stacker = stacker
        self.base_models = base_models
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], folds.n_splits))
            for j, (train_idx, test_idx) in enumerate(folds.split(X)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(1)
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred
    
    def fit(self,stacker, base_models, X, y):
        self.stacker = stacker
        self.base_models = base_models
        X_mean = X.mean(axis = 0)
        X_std = X.std(axis = 0)
        X = (X - X_mean)/X_std
        X = np.array(X)
        y = np.array(y)
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        model_collect = []
        for i, clf in enumerate(self.base_models):
            model_tmp = []
            for j, (train_idx, test_idx) in enumerate(folds.split(X)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                model_tmp.append(clf)
            model_collect.append(model_tmp)
        self.stacker.fit(S_train, y)
        
        if self.save_path != '':
            joblib.dump((self.stacker, model_collect, X_mean, X_std), self.save_path)
            
    def predict(self, T):
        try:
            stackers, model_collect, X_mean, X_std = joblib.load(self.save_path)
        except Exception:
            print 'no models in the save_path offered!'
        else:
            T = (T - X_mean)/X_std
            S_test = np.zeros((T.shape[0], len(model_collect)))
            for i, clfs in enumerate(model_collect):
                S_test_i = np.zeros((T.shape[0], len(clfs)))
                for j in range(len(clfs)):
                    S_test_i[:, j] = clfs[j].predict(T)[:]
                S_test[:, i] = S_test_i.mean(1)
            y_pred = stackers.predict(S_test)[:]
            return y_pred
        
    