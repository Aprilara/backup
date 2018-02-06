# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:45:34 2017

@author: Administrator
"""

import os
os.chdir('F:/company/model_taobao/ml')
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
from ensemble_stacking import Ensemble
from sklearn.svm import SVR

#从数据库读取数据，预处理
engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')
df = pd.read_sql_query('select * from test.xxytaobaofeature', con=engine)
for col in df.columns.values:
    if df[col].dtypes == 'O':
        df[col] = df[col].astype('float')
df.to_pickle('data/df.pkl')
df_tr, df_te = train_test_split(df, test_size=0.3, random_state=42)
X_tr = df_tr[np.setdiff1d(df.columns, ['custid', 'flag'])]
y_tr = df_tr['flag']
X_te = df_te[np.setdiff1d(df.columns, ['custid', 'flag'])]
y_te = df_te['flag']
X_val, X_te, y_val, y_te = train_test_split(X_te, y_te, test_size=0.5, random_state=42)
X_tr = fillnas(X_tr, fillin='med')
X_te = fillnas(X_te, fillin='med')
X_val = fillnas(X_val, fillin='med')

X_mean = X_tr.mean()
X_std = X_tr.std()
X_tr = (X_tr - X_mean)/X_std
X_te = (X_te - X_mean)/X_std
X_val = (X_val - X_mean)/X_std

joblib.dump([X_tr, y_tr], '../data/tr_new.pkl')
joblib.dump([X_val, y_val], '../data/val_new.pkl')
joblib.dump([X_te, y_te], '../data/ts_new.pkl')

##########################数据处理结束##########################################
gbdt = GradientBoostingRegressor(random_state=42)
param = {'max_features':[8,9,10], 'max_depth':[4,5,6], 'n_estimators':[100,200]}
model = GridSearchCV(estimator=gbdt, param_grid=param, cv=5, n_jobs=1, verbose=20)
model.fit(X_tr, y_tr)
model.best_params_
clf = GradientBoostingRegressor(n_estimators=200, max_depth=4, max_features=9, random_state=42)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_val)
overduerate(pred, y_val, 50)

rfr = RandomForestRegressor(random_state=42)
param = {'max_features':[8,9,10], 'n_estimators':[100,200]}
model = GridSearchCV(estimator=gbdt, param_grid=param, cv=5, n_jobs=1, verbose=20)
model.fit(X_tr, y_tr)
model.best_params_
clf = RandomForestRegressor(n_estimators=200, max_features=8)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_val)
overduerate(pred, y_val, 50)

lr = Ridge(random_state=42)
param = {'alpha':[1, 10, 5, 50]}
model = GridSearchCV(lr, param, cv=5, verbose=20, n_jobs=1)
model.fit(X_tr, y_tr)
model.best_params_
clf = Ridge(alpha=50, random_state=42)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_val)
overduerate(pred, y_val, 50)

mlp = MLPRegressor(random_state=42, activation='relu', solver='adam')
param = {'hidden_layer_sizes':[(20,10),(10,10)], 'alpha':[0.0001,0.0005,0.001]}
mlp = MLPRegressor(random_state=42, activation='relu', solver='adam')
model = GridSearchCV(mlp, param_grid=param, cv=5, n_jobs=1, verbose=20)
model.fit(X_tr, y_tr)
model.best_params_
clf = MLPRegressor(hidden_layer_sizes=(20,10), activation='relu', solver='adam', alpha=0.0001, random_state=42)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_val)
overduerate(pred, y_val, 50)

mlp = MLPRegressor(random_state=42, activation='tanh', solver='adam')
param = {'hidden_layer_sizes':[(20,10),(10,10)], 'alpha':[0.0001,0.0005,0.001]}
mlp = MLPRegressor(random_state=42, activation='tanh', solver='adam')
model = GridSearchCV(mlp, param_grid=param, cv=5, n_jobs=1, verbose=20)
model.fit(X_tr, y_tr)
model.best_params_
clf = MLPRegressor(hidden_layer_sizes=(20,10), activation='tanh', solver='adam', alpha=0.001, random_state=42)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_val)
overduerate(pred, y_val)

mlp = MLPRegressor(random_state=42, activation='logistic', solver='adam')
param = {'hidden_layer_sizes':[(20,10),(10,10)], 'alpha':[0.0001,0.0005,0.001]}
mlp = MLPRegressor(random_state=42, activation='logistic', solver='adam')
model = GridSearchCV(mlp, param_grid=param, cv=5, n_jobs=1, verbose=20)
model.fit(X_tr, y_tr)
model.best_params_
clf = MLPRegressor(hidden_layer_sizes=(20,10), activation='logistic', solver='adam', alpha=0.0001, random_state=42)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_val)
overduerate(pred, y_val, 50)

clf = SVR(degree=3, gamma=1.0, C=1.0)


param_lr = {'alpha':50, 'random_state':42}
param_gbdt = {'max_features':9, 'max_depth':4, 'n_estimators':200, 'random_state':42}
param_rf = {'max_features':8, 'n_estimators':200, 'random_state':42}
param_relu = {'hidden_layer_sizes':(20, 10), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'random_state':42}
param_tanh = {'hidden_layer_sizes':(20, 10), 'activation':'tanh', 'solver':'adam', 'alpha':0.001, 'random_state':42}
param_lg = {'hidden_layer_sizes':(20, 10), 'activation':'logistic', 'solver':'adam', 'alpha':0.0001, 'random_state':42}
lrr = Ridge(**param_lr)
gbdt = GradientBoostingRegressor(**param_gbdt)
rf = RandomForestRegressor(**param_rf)
relu = MLPRegressor(**param_relu)
tanh= MLPRegressor(**param_tanh)
lg = MLPRegressor(**param_lg)
#stacking model
gbdt1 = GradientBoostingRegressor(random_state=42, max_depth=6)
model = Ensemble(5, 'model1.pkl')
model.fit(gbdt1, [lrr, gbdt, rf, relu, tanh, lg], X_tr, y_tr)
pred = model.predict(X_val)
overduerate(pred, y_val)

#vote model
base_model = [lrr, gbdt, rf, relu, tanh, lg]
y_models = np.zeros((X_val.shape[0], len(base_model)))
for i in range(len(base_model)):
    base_model[i].fit(X_tr, y_tr)
    pred = base_model[i].predict(X_val)
    flag = np.percentile(pred, 50)
    y_models[pred<flag,i] = 1

------------------------------------------------------------------------
画feature importance
------------------------------------------------------------------------
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
# Scatter plot 
trace = go.Scatter(
    y = fea,
    x = X_tr.columns.values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = fea,
        colorscale='Portland',
        showscale=True
    ),
    text = X_tr.columns.values
)
    
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'GBDT Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
#命令行用
py.plot(fig,filename='gbdt')
#notebook用
py.iplot(fig,filename='scatter2010')