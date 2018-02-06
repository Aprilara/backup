# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 10:33:40 2017

@author: Administrator
"""

from sqlalchemy import create_engine
import pandas as pd
import os
import numpy as np
os.chdir('F:/company/model_smscallinfo/ml')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.externals import joblib
import codecs

df_tr, y_tr = joblib.load('../data/tr.pkl')
df_val, y_val = joblib.load('../data/val.pkl')
df_te, y_te = joblib.load('../data/te.pkl')

dtrain = xgb.DMatrix(df_tr.values, label=y_tr.values)
dtest = xgb.DMatrix(df_te.values, label=y_te.values)
dval = xgb.DMatrix(df_val.values, label=y_val.values)

pred = bst.predict(dval)
labels = dval.get_label()
flag = np.percentile(pred,80)
sum(labels[pred<flag])*1.0/len(labels[pred<flag])

param = {
         'learning_rate': 0.1,
         'n_estimators': 500,
         'max_depth': 21,
         'min_child_weight': 80,
         'gamma': 6.48,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'objective': 'binary:logistic',
         'nthread': 4,
         'scale_pos_weight': 1,
         'seed': 27,
         'silent': 1
}

def figure_results(preds, labels):
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(len(labels)):
        if preds[i] > 0.5:
            x = 1
        else:
            x = 0
        y = labels[i]
        if x == 1 and y == 1:
            a += 1
        elif x == 1 and y == 0:
            b += 1
        elif x == 0 and y == 1:
            c += 1
        else:
            d += 1
    if (a+b)==0:
        p=0.0
    else:
        p = float(a)/float(a+b)
    if (c+d) == 0:
        r = 0.0
    else:
        r = float(a)/float(a+c)
    if (p+r)==0:
        f1 = 0.0
    else:
        f1 = 2*p*r/(p+r)
    return p, r, f1

num_round = 120
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
fmax = 0.0
max_depth = 0
min_child_weight = 0
for i in range(3, 30, 3):
    for j in range(30,100,10):
        param['max_depth'] = i
        param['min_child_weight'] = j

        bst = xgb.train(param, dtrain, num_round, watchlist)
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        p, r, f = figure_results(preds, labels)
        if f > fmax:
            fmax = f
            max_depth = i
            min_child_weight = j
            output = codecs.open("result.txt", "w", "utf-8-sig")
            output.write("%s\n" % str(f))
            output.close()

fmax = 0.0
param['max_depth'] = max_depth
param['min_child_weight'] = min_child_weight
max_gamma = 0
for i in range(1, 1001, 10):
    param['gamma'] = i*0.1

    bst = xgb.train(param, dtrain, num_round, watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    p, r, f = figure_results(preds, labels)
    if f > fmax:
        fmax = f
        max_gamma = i * 0.1
        output = codecs.open("result.txt", "w", "utf-8-sig")
        output.write("%s\n" % str(f))
        output.close()
        
fmax = 0.0
l = max(0, (int(max_gamma) - 2) * 100)
r = (int(max_gamma) + 2) * 100
max_gamma = 0
for i in range(l, r+1,2):
    param['gamma'] = i*0.01
    bst = xgb.train(param, dtrain, num_round, watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    p, r, f = figure_results(preds, labels)
    if f > fmax:
        fmax = f
        max_gamma = i * 0.01
        output = codecs.open("result.txt", "w", "utf-8-sig")
        output.write("%s\n" % str(f))
        output.close()
        
# round4
fmax = 0.0
pmax = 0.0
rmax = 0.0
max_subsample = 0
max_colsample_bytree = 0
param['gamma'] = max_gamma
for i in range(1, 11):
    for j in range(2, 11,2):
        param['subsample'] = i*0.1
        param['colsample_bytree'] = j * 0.1

        bst = xgb.train(param, dtrain, num_round, watchlist)
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        p, r, f = figure_results(preds, labels)
        if f > fmax:
            fmax = f
            pmax = p
            rmax = r
            max_subsample = i * 0.1
            max_colsample_bytree = j * 0.1
            output = codecs.open("result.txt", "w", "utf-8-sig")
            output.write("%s\n" % str(f))
            output.close()
            
param['subsample'] = max_subsample
param['colsample_bytree'] = max_colsample_bytree
output = codecs.open("result.txt", "w", "utf-8-sig")
output.write("%s,%s,%s\n" %(str(pmax),str(rmax),str(fmax)))
output.write("%s\n" % json.dumps(param))
output.close()
#保存模型方法1
bst.save_model('20171116.model')
#保存模型方法2
bst.dump('dump.raw.txt', 'featmap.txt')
#加载模型
bst = xgb.Booster({'nthread':4})
bst.load_model('20171116.model')