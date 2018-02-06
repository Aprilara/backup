# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:16:03 2017

@author: Administrator
"""


import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
import os
import xgboost as xgb
os.chdir('F:/company')

engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')
df = pd.read_sql_query('select * from test.tongdunselect', con=engine)
colnames = df.columns.values

for col in colnames:
    df[col] = df[col].fillna(0)
X = df[np.setdiff1d(df.columns, ['custid', 'flag','reportid','bankconsumerfinance24_60m','bigconsumerfinance18_24m','bigconsumerfinance24_60m','creditcardcenter18_24m','financialinstitutions18_24m','financialinstitutions24_60m','propertyinsurance24_60m','thirdpartfacilitator12_18m','thirdpartfacilitator18_24m','thirdpartfacilitator24_60m'])]
X_test = df[np.setdiff1d(df.columns, ['custid', 'flag','reportid','custwholeinfoid','bankconsumerfinance24_60m','bigconsumerfinance18_24m','bigconsumerfinance24_60m','creditcardcenter18_24m','financialinstitutions18_24m','financialinstitutions24_60m','propertyinsurance24_60m','thirdpartfacilitator12_18m','thirdpartfacilitator18_24m','thirdpartfacilitator24_60m'])]
X = df[np.setdiff1d(df.columns, ['custid', 'flag','reportid','bankconsumerfinance60m','bigconsumerfinance24m','bigconsumerfinance60m','creditcardcenter24m','financialinstitutions24m','financialinstitutions60m','propertyinsurance60m','thirdpartfacilitator18m','thirdpartfacilitator24m','thirdpartfacilitator60m'])]
X = df[np.setdiff1d(df.columns, ['custid', 'flag','reportid','bankconsumerfinance60m','bigconsumerfinance24m','bigconsumerfinance60m','creditcardcenter24m','financialinstitutions24m','financialinstitutions60m','propertyinsurance60m','thirdpartfacilitator18m','thirdpartfacilitator24m','thirdpartfacilitator60m','bankconsumerfinance24_60m','bigconsumerfinance18_24m','bigconsumerfinance24_60m','creditcardcenter18_24m','financialinstitutions18_24m','financialinstitutions24_60m','propertyinsurance24_60m','thirdpartfacilitator12_18m','thirdpartfacilitator18_24m','thirdpartfacilitator24_60m'])]
y = df['flag']
X_train, X_test_validate, y_train, y_test_validate = train_test_split(\
        X, y, test_size=0.3, random_state=42)
X_test, X_validate, y_test, y_validate = train_test_split(X_test_validate, y_test_validate, test_size=0.5,  random_state=42)

from sklearn import preprocessing
X_mean = X_train.mean(axis = 0)
X_std = X_train.std(axis=0)
X_train_scale = (X_train - X_mean)/X_std
X_test_scale = (X_test - X_mean)/X_std
X_validate_scale = (X_validate - X_mean)/X_std

#-----ridge regression--------------------------------------------------------
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha':[0.8,1.0,1.2]}
lrr = linear_model.Ridge(random_state=42)
model = GridSearchCV(estimator=lrr, param_grid=param_grid, n_jobs=1, cv=10, verbose=20)
model.fit(X_train_scale, y_train)
model.best_params_

clf = linear_model.Ridge(alpha=1.2, random_state = 42)
clf.fit(X_train_scale, y_train)
pred1 = clf.predict(X_validate_scale)
flag = np.percentile(pred1,80,axis=0)
sum(y_validate[pred1<flag])*1.0/len(y_validate[pred1<flag]) 


#-----random forests regression -----------------------------------------------
from sklearn.ensemble import RandomForestRegressor
param_grid = {'n_estimators':[50,80], 'max_features':[9,11,13]}
rfr = RandomForestRegressor(random_state=42)
model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, verbose=20)
model.fit(X_train_scale, y_train)
model.best_params_
#clf = RandomForestRegressor(max_features=9, n_estimators=80, random_state=42)#0.16874537575309165
clf = RandomForestRegressor(max_features=9, n_estimators=80, random_state=42)
clf.fit(X_train_scale, y_train)
pred2 = clf.predict(X_validate_scale)
flag = np.percentile(pred2,80,axis=0)
sum(y_validate[pred2<flag])*1.0/len(y_validate[pred2<flag]) 

#------extra tree regressor----------------------------------------------------
from sklearn.ensemble import ExtraTreesRegressor
param_grid = {'n_estimators':[50,80], 'max_features':[9,11,13]}
etr = ExtraTreesRegressor(random_state=42)
model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=1, cv=10, verbose=20)
model.fit(X_train_scale, y_train)
model.best_params_
clf = ExtraTreesRegressor(max_features=9, n_estimators=80, random_state=42 )
clf.fit(X_train_scale, y_train)
pred3 = clf.predict(X_validate_scale)
flag = np.percentile(pred3,80,axis=0)
sum(y_validate[pred3<flag])*1.0/len(y_validate[pred3<flag])

#---------gradient boosting decision tree--------------------------------------
from sklearn.ensemble import GradientBoostingRegressor
param_grid = {'n_estimators':[80,100], 'max_features':[9,11,13]}
gbdt = GradientBoostingRegressor(random_state=42)
model = GridSearchCV(estimator=gbdt, param_grid=param_grid, n_jobs=1, cv=10, verbose=20)
model.fit(X_train_scale, y_train)
model.best_params_
clf = GradientBoostingRegressor(max_features=13, n_estimators=180, random_state=42 )
clf.fit(X_train_scale, y_train)
pred7 = clf.predict(X_validate_scale)
flag = np.percentile(pred7,80,axis=0)
sum(y_validate[pred7<flag])*1.0/len(y_validate[pred7<flag])

#------nn tanh ----------------------------------------------------------------
from sklearn.neural_network import MLPRegressor
param_grid = {'alpha':[0.0001,0.00005,0.00001], 'hidden_layer_sizes':[(10,10),(10,2)]}
mlprt = MLPRegressor(activation='relu',solver='adam', random_state=42)
model = GridSearchCV(estimator=mlprt, param_grid=param_grid, n_jobs=1, cv=5, verbose=20)
model.fit(X_train_scale, y_train)
model.best_params_

y_tr = y_train[:]
y_tr[y_tr == 0] = -1
model.fit(X_train_scale, y_tr)

clf = MLPRegressor(hidden_layer_sizes=(10, 2), activation='tanh',solver='adam', alpha=0.0001,random_state=42)
clf.fit(X_train_scale,y_train)
pred4 = clf.predict(X_validate_scale)
flag = np.percentile(pred4,80,axis=0)
sum(y_validate[pred4<flag])*1.0/len(y_validate[pred4<flag]) 



clf = MLPRegressor(hidden_layer_sizes=(10, 2), activation='logistic',solver='adam', alpha=0.0001,random_state=42)
clf.fit(X_train_scale,y_train)
pred5 = clf.predict(X_validate_scale)
flag = np.percentile(pred5,80,axis=0)
sum(y_validate[pred5<flag])*1.0/len(y_validate[pred5<flag]) 



clf = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu',solver='adam', alpha=1e-05,random_state=42)
clf.fit(X_train_scale,y_train)
pred6 = clf.predict(X_validate_scale)
flag = np.percentile(pred6,80,axis=0)
sum(y_validate[pred6<flag])*1.0/len(y_validate[pred6<flag]) 


