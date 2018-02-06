# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:16:03 2017

@author: Administrator
"""

# change pandas dataframe to svmlib format
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
#from sklearn.datasets import dump_svmlight_file
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
#df = pd.read_pickle('tongdun/tongdundata/tongdunselect.pkl')
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
#0.16423244251087632
#0.16883523362396602
0.16887667741807033

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
#0.174
#0.17219315895372234
0.17057165506914471
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
#0.14889080433697774
#0.17381965552178319
#----------------svm---------------------------
#from sklearn import svm
#clf = svm.SVR()
#clf.fit(X_train_scale, y_train)
#pred8 = clf.predict(X_validate_scale)
#flag = np.percentile(pred8,80,axis=0)
#sum(y_validate[pred8<flag])*1.0/len(y_validate[pred8<flag])

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
#0.16585035638912118
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
#0.16583799461363166
#0.17200983679856918
#0.16895631744514794

#clf = MLPRegressor(hidden_layer_sizes=(10,2), activation='logistic',solver='adam', alpha=0.00001,random_state=42)
clf = MLPRegressor(hidden_layer_sizes=(10, 2), activation='logistic',solver='adam', alpha=0.0001,random_state=42)
clf.fit(X_train_scale,y_train)
pred5 = clf.predict(X_validate_scale)
flag = np.percentile(pred5,80,axis=0)
sum(y_validate[pred5<flag])*1.0/len(y_validate[pred5<flag]) 
#0.16376631448104412
#0.16722557567627991
#0.16684585672759128

#clf = MLPRegressor(hidden_layer_sizes=(50,3), activation='relu',solver='adam', alpha=0.0001,random_state=42)
clf = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu',solver='adam', alpha=1e-05,random_state=42)
clf.fit(X_train_scale,y_train)
pred6 = clf.predict(X_validate_scale)
flag = np.percentile(pred6,80,axis=0)
sum(y_validate[pred6<flag])*1.0/len(y_validate[pred6<flag]) 
# 0.16526828257717008
#0.16732369689005694

#-----------stacking the 6 results --------------------------------------------
#params_lrr = {'alpha':1.2, 'random_state':42}
#params_rfr = {'max_features':9, 'n_estimators':80, 'random_state':42}
#params_etr = {'max_features':9, 'n_estimators':80, 'random_state':42}
#params_nntanh = {'hidden_layer_sizes':(10,2), 'alpha':0.0001, 'activation':'tanh', 'random_state':42}
#params_nnlg = {'hidden_layer_sizes':(10,2), 'alpha':0.00001, 'activation':'logistic', 'random_state':42}
#params_nnrelu = {'hidden_layer_sizes':(50,3), 'alpha':0.0001, 'activation':'relu', 'random_state':42}
params_lrr = {'alpha':1.2, 'random_state':42}
params_rfr = {'max_features':9, 'n_estimators':80, 'random_state':42}
params_etr = {'max_features':9, 'n_estimators':80, 'random_state':42}
params_gbdt = {'max_features':13, 'n_estimators':200, 'random_state':42}
params_nntanh = {'hidden_layer_sizes':(10, 2), 'alpha':0.0001, 'activation':'tanh', 'random_state':42}
params_nnlg = {'hidden_layer_sizes':(10, 2), 'alpha':0.0001, 'activation':'logistic', 'random_state':42}
params_nnrelu = {'hidden_layer_sizes':(10,10), 'alpha':1e-05, 'activation':'relu', 'random_state':42}
lrr = linear_model.Ridge(**params_lrr)
rft = RandomForestRegressor(**params_rfr)
etr = ExtraTreesRegressor(**params_etr)
gbdt = GradientBoostingRegressor(**params_gbdt)
nntanh = MLPRegressor(**params_nntanh)
nnlg = MLPRegressor(**params_nnlg)
nnrelu = MLPRegressor(**params_nnrelu)
base_models = [lrr, rft, etr, nntanh, nnlg, nnrelu,gbdt]

os.chdir('tongdun/model_utils')
from ensemble_stacking import Ensemble
gbdt1 = GradientBoostingRegressor(random_state=42)
#rfr1 = RandomForestRegressor(random_state=42)
en_model1 = Ensemble(5,'model1_all_gbdt.pkl')
en_model1.fit(gbdt1,base_models,X_train_scale, y_train)
en_model1.fit(gbdt1,base_models,X_train, y_train)
pred = en_model1.predict(X_test_scale)
pred = en_model1.predict(X_test)
pred = en_model1.fit_predict(X_train_scale, y_train, X_test_scale)
flag = np.percentile(pred,40,axis=0)
from sklearn.externals import joblib
joblib.dump(flag, 'flag.pkl')


sum(y_test[pred<flag])*1.0/len(y_test[pred<flag]) 
# 0.1621089703749741
#0.14196288844176169
#0.14200760116253075
# 0.16740244583143046
#0.16730115222577535gbdt
#0.16814018319394664