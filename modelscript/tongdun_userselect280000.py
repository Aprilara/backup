# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:28:20 2017

@author: Xiaoyan Xu
"""
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
import os
from sklearn.externals import joblib

os.chdir('F:/company/tongdun/tongdun_score/20170711')
engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')
df = pd.read_sql_query('select * from test.tongdunselect1', con=engine)
colnames = df.columns.values

for col in colnames:
    df[col] = df[col].fillna(0)
X_train = df[np.setdiff1d(df.columns, ['custid', 'flag','custwholeinfoid','reportid','bankconsumerfinance24_60m','bigconsumerfinance18_24m','bigconsumerfinance24_60m','creditcardcenter18_24m','financialinstitutions18_24m','financialinstitutions24_60m','propertyinsurance24_60m','thirdpartfacilitator12_18m','thirdpartfacilitator18_24m','thirdpartfacilitator24_60m'])]
y_train = df['flag']
X_mean = X_train.mean(axis = 0)
X_std = X_train.std(axis=0)
X_train_scale = (X_train - X_mean)/X_std

df_t = pd.read_sql_query('select * from test.tongdunselect280000_1', con=engine)
for col in df_t.columns:
    df_t[col] = df_t[col].fillna(0)
X_test = df_t[np.setdiff1d(df_t.columns, ['custid', 'flag','custwholeinfoid','reportid','bankconsumerfinance24_60m','bigconsumerfinance18_24m','bigconsumerfinance24_60m','creditcardcenter18_24m','financialinstitutions18_24m','financialinstitutions24_60m','propertyinsurance24_60m','thirdpartfacilitator12_18m','thirdpartfacilitator18_24m','thirdpartfacilitator24_60m'])]
X_test_scale = (X_test - X_mean)/X_std

params_lrr = {'alpha':1.2, 'random_state':42}
params_rfr = {'max_features':9, 'n_estimators':80, 'random_state':42}
params_etr = {'max_features':9, 'n_estimators':80, 'random_state':42}
params_gbdt = {'max_features':13, 'n_estimators':200, 'random_state':42}
params_nntanh = {'hidden_layer_sizes':(10,2), 'alpha':0.0001, 'activation':'tanh', 'random_state':42}
params_nnlg = {'hidden_layer_sizes':(50,3), 'alpha':0.0001, 'activation':'logistic', 'random_state':42}
params_nnrelu = {'hidden_layer_sizes':(50,3), 'alpha':1e-05, 'activation':'relu', 'random_state':42}
lrr = linear_model.Ridge(**params_lrr)
rft = RandomForestRegressor(**params_rfr)
etr = ExtraTreesRegressor(**params_etr)
gbdt = GradientBoostingRegressor(**params_gbdt)
nntanh = MLPRegressor(**params_nntanh)
nnlg = MLPRegressor(**params_nnlg)
nnrelu = MLPRegressor(**params_nnrelu)
base_models = [lrr, rft, etr, nntanh, nnlg, nnrelu,gbdt]

from ensemble_stacking import Ensemble
gbdt1 = GradientBoostingRegressor(random_state=42)
rfr1 = RandomForestRegressor(random_state=42)
en_model1 = Ensemble(5,gbdt1,base_models)
en_model1 = Ensemble(5,'tongdunscore1.pkl')
en_model1.fit(gbdt1, base_models, X_train, y_train)
pred = en_model1.fit_predict(gbdt1, base_models,X_train_scale, y_train, X_test_scale)
flag = np.percentile(pred,40,axis=0)
df_t['pred'] = pred
userselected = df_t[pred<flag][['custid','custwholeinfoid','pred']]
#userselected.to_sql(name='test.userselected280000', con = engine, if_exists='append', index=False)

import psycopg2
conn = psycopg2.connect(dbname='credit', user='dev', password='123456', host='192.168.91.11', port=5433)
cursor = conn.cursor()
for i in userselected.index:
    custid = userselected.loc[i,'custid']
    custwholeinfoid = userselected.loc[i,'custwholeinfoid']
    pred = userselected.loc[i,'pred']
    query = "insert into test.userselected280000(custid, custwholeinfoid, pred) values (%s, %s, %s)"
    data = (int(custid), int(custwholeinfoid), float(pred))
    cursor.execute(query, data)
conn.commit()

st = time.time()
args_str = ','.join(cursor.mogrify("(%s,%s,%s)", x) for x in userselected280000tuple)
cursor.execute("insert into test.userselected280000 values" + args_str)
ed = time.time()