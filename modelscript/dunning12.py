# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 10:09:28 2018

@author: Administrator
"""

import os
os.chdir('F:/company/dunning/model')
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from datetime import datetime


engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')
df = pd.read_sql_query("select * from test.xxydunningfeature12", engine)
df_train = df.loc[df.flag>-1,:][:]
start_time = datetime.strptime('2017-11-20', '%Y-%m-%d').date()
df_train = df_train.loc[df_train.startdunningtime>start_time,:][:]
df_train = df_train.fillna(0)
df.to_pickle('./df_Dec.pkl')
df_train.to_pickle('./df_train_Dec.pkl')

y = df_train['flag']
X = df_train[np.setdiff1d(df.columns, ['loanborrowid', 'custid', 'startdunningtime', 'flag'])]
X = X.astype('float32')
y = y.astype('float32')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_mean = X_train.mean()
X_std = X_train.std()
X_train = (X_train - X_mean)/X_std
X_test = (X_test - X_mean)/X_std

######################################ml######################################
def rfrcv(n_estimators, min_samples_split, max_features):
    val = cross_val_score(
        RFR(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            random_state=42
        ),
        X_train, y_train, cv=2
    ).mean()
    return val

rfrBO = BayesianOptimization(
        rfrcv,
        {'n_estimators': (100, 400),
        'min_samples_split': (20, 100),
        'max_features': (0.1, 0.999)}
    )
gp_params = {"alpha": 1e-5}
rfrBO.maximize(n_iter=10, **gp_params)

rf = RFR(n_estimators=268, min_samples_split=20, max_features=9, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

def ridgecv(alpha):
    val = cross_val_score(
            Ridge(alpha=alpha, random_state=42),
            X_train, y_train, cv=2
            ).mean()
    return val

ridgeBO = BayesianOptimization(
        ridgecv,
        {'alpha':(0.01,11)})
ridgeBO.maximize(n_iter=20, **gp_params)

rg = Ridge(alpha=1.0, random_state=42)
rg.fit(X_train, y_train)
pred = rg.predict(X_test)

f = np.percentile(pred, 40)
sum(y_test[pred>f])*1.0/len(y_test[pred>f])

def gbdtcv(n_estimators, max_depth, min_sample_split):
    val = cross_val_score(
            GBR(n_estimators=int(n_estimators), 
                min_samples_split=int(min_sample_split),
                max_depth=int(max_depth)),
                X_train, y_train).mean()
    return val
gbdtBO = BayesianOptimization(
        gbdtcv,
        {'n_estimators':(50, 300),
         'min_sample_split':(20, 100),
         'max_depth':(2, 8)})
gbdtBO.maximize(n_iter=20, **gp_params)

gbdt = GBR(n_estimators=300, max_depth=8, min_samples_split=20, random_state=42)
gbdt.fit(X_train, y_train)
pred = gbdt.predict(X_test)
            
f = np.percentile(pred, 40)
sum(y_test[pred>f])*1.0/len(y_test[pred>f])

param_grid = {'alpha':[0.0001,0.00005,0.00001], 'hidden_layer_sizes':[(10,5),(10,2)]}
mlprt = MLPRegressor(activation='logistic',solver='adam', random_state=42)
model = GridSearchCV(estimator=mlprt, param_grid=param_grid, n_jobs=1, cv=5, verbose=20)
model.fit(X_train, y_train)
model.best_params_
clf = MLPRegressor(hidden_layer_sizes=(10, 2), activation='logistic',solver='adam', alpha=1e-05,random_state=42)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

clf = MLPRegressor(hidden_layer_sizes=(10, 2), activation='relu',solver='adam', alpha=1e-05,random_state=42)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

clf = MLPRegressor(hidden_layer_sizes=(10, 5), activation='tanh',solver='adam', alpha=1e-05,random_state=42)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

###############################################################################
params_lrr = {'alpha':1.0, 'random_state':42}
params_rfr = {'max_features':9, 'n_estimators':268, 'min_samples_split':20, 'random_state':42}
params_gbdt = {'max_features':8, 'n_estimators':300, 'min_samples_split':20, 'random_state':42}
params_nntanh = {'hidden_layer_sizes':(10, 5), 'alpha':1e-05, 'activation':'tanh', 'random_state':42}
params_nnlg = {'hidden_layer_sizes':(10, 2), 'alpha':1e-05, 'activation':'logistic', 'random_state':42}
params_nnrelu = {'hidden_layer_sizes':(10,2), 'alpha':1e-05, 'activation':'relu', 'random_state':42}
lrr = Ridge(**params_lrr)
rft = RFR(**params_rfr)
gbdt = GBR(**params_gbdt)
nntanh = MLPRegressor(**params_nntanh)
nnlg = MLPRegressor(**params_nnlg)
nnrelu = MLPRegressor(**params_nnrelu)
base_models = [lrr, rft, nntanh, nnlg, nnrelu,gbdt]
from ensemble_stacking import Ensemble
gbdt1 = GBR(random_state=42)
en_model1 = Ensemble(5,'model_dunning12.pkl')
en_model1.fit(gbdt1,base_models,X_train, y_train)

pred = en_model1.predict(X_test)
f = np.percentile(pred, 40)
sum(y_test[pred>f])*1.0/len(y_test[pred>f])