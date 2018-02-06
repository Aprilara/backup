# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:42:47 2017

@author: Administrator
"""

from sqlalchemy import create_engine
import pandas as pd
import os
import numpy as np
os.chdir('F:/company/android/openacc/')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')
df = pd.read_sql_query('select * from test.xxyandroiduser', con = engine)
df_tr, df_te = train_test_split(df, test_size=0.3, random_state=42)
df.to_pickle('./data/df.pkl')

#1.剔除异常值(短信银行卡余额等有异常值,呼入呼出时间有负数)

#数据预处理：先不处理异常值，直接用中位数填充控制；之后再做一个处理异常值的版本，比较两者的不同
for col in df_tr.columns.values:
    tmp = np.nanpercentile(df_tr[col], 50)
    df_tr.loc[:,col] = df_tr[col].fillna(tmp)
    df_te.loc[:,col] = df_te[col].fillna(tmp)
#把时长由秒数改为分钟
call_col = [x for x in df_tr.columns.values if 'time' in x]
for col in call_col:
    df_tr[col] = df_tr[col]*1.0/60
    df_te[col] = df_te[col]*1.0/60
#数据标准化
X_tr = df_tr[np.setdiff1d(df_tr.columns,['custid','flag'])]
y_tr = df_tr['flag']
X_te = df_te[np.setdiff1d(df_te.columns,['custid','flag'])]
y_te = df_te['flag']

means = X_tr.mean(axis=0)
std = X_tr.std(axis=0)
X_tr_scale = (X_tr - means)/std
X_te_scale = (X_te - means)/std

X_val_scale, X_te_scale, y_val, y_te = train_test_split(X_te_scale, y_te, test_size=0.5, random_state=42)

#用pearson检验挑选出线性相关的变量(列数增加，树方法较慢)
def pvalidcol(df, y, pvalid=0.05):
    validcol = []
    for col in df.columns:
        if pearsonr(df[col], y)[1]<pvalid:
            validcol.append(col)
    return validcol
validcol = pvalidcol(X_tr, y_tr)
X_tr_scale = X_tr_scale[validcol]
X_val_scale = X_val_scale[validcol]
X_te_scale = X_te_scale[validcol]
#随机森林
param_grid = {'n_estimators':[80,100], 'max_features':[15,24,19]}
min_rate = 1
best_param = {'n_estimators':1, 'max_features':1}
for n in param_grid['n_estimators']:
    for max_fea in param_grid['max_features']:
        clf = RandomForestRegressor(max_features=max_fea, n_estimators=n, random_state=42)
        clf.fit(X_tr_scale, y_tr)
        pred = clf.predict(X_val_scale)
        rate = overduerate(pred, y_val,50)
        if rate<min_rate:
            min_rate = rate
            best_param['n_estimators'] = n
            best_param['max_features'] = max_fea
            
clf = RandomForestRegressor(n_estimators=500, max_features=16, random_state=42)
clf.fit(X_tr_scale, y_tr)
pred = clf.predict(X_val_scale)
overduerate(pred, y_val, 50)    
0.14110156944033164 #减少变量之后
overduerate(pred, y_val, 80) 
0.17277777777777778
    
#gbdt
param_grid = {'n_estimators':[500,300], 'max_features':[4,8,14]}
min_rate = 1
best_param = {'n_estimators':1, 'max_features':1}
for n in param_grid['n_estimators']:
    for max_fea in param_grid['max_features']:
        clf = GradientBoostingRegressor(max_depth=max_fea, n_estimators=n, random_state=42)
        clf.fit(X_tr_scale, y_tr)
        pred = clf.predict(X_val_scale)
        rate = overduerate(pred, y_val,50)
        print 'once'
        if rate<min_rate:
            min_rate = rate
            best_param['n_estimators'] = n
            best_param['max_features'] = max_fea  
    
clf = GradientBoostingRegressor(n_estimators=80, max_depth=9, random_state=42)
clf.fit(X_tr_scale, y_tr)
pred = clf.predict(X_val_scale)
overduerate(pred, y_val, 50) 
0.14232433238848277  

#l1
param_grid = {'C':[1,0.5,2]}
min_rate = 1
best_param = {'C':1}
for c in param_grid['C']:
    clf = LogisticRegression(C=c, penalty='l1', random_state=42)
    clf.fit(X_tr_scale, y_tr)
    pred = clf.predict(X_val_scale)
    rate = overduerate(pred, y_val,50)
    print 'once'
    if rate<min_rate:
        min_rate = rate
        best_param['C'] = c
    

#l2
param_grid = {'alpha':[1,5,10]}
min_rate = 1
best_param = {'alpha':1}
for a in param_grid['alpha']:
    clf = Ridge(alpha=a, random_state=42)
    clf.fit(X_tr_scale, y_tr)
    pred = clf.predict(X_val_scale)
    rate = overduerate(pred, y_val,50)
    print 'once'
    if rate<min_rate:
        min_rate = rate
        best_param['alpha'] = a
        

#1.写一个函数，输入值是分位数，flag是否剔除，输出值是 空值比例多于分位数的列名list
def isnullcol(df,percent=60):
    for col in df.columns.values:
        
flag = np.percentile(pred,80)
sum(y_val[pred<flag])*1.0/len(y_val[pred<flag])
        
#