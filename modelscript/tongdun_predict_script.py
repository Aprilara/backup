# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:28:24 2017

@author: Administrator
"""

#同盾模型预测脚本
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.chdir('F:/company')
df = pd.read_pickle('tongdun/tongdundata/tongdunselect.pkl')
colnames = df.columns.values
for col in colnames:
    df[col] = df[col].fillna(0)    
X = df[np.setdiff1d(df.columns, ['custid', 'flag','reportid','custwholeinfoid','bankconsumerfinance60m','bigconsumerfinance24m','bigconsumerfinance60m','creditcardcenter24m','financialinstitutions24m','financialinstitutions60m','propertyinsurance60m','thirdpartfacilitator18m','thirdpartfacilitator24m','thirdpartfacilitator60m','bankconsumerfinance24_60m','bigconsumerfinance18_24m','bigconsumerfinance24_60m','creditcardcenter18_24m','financialinstitutions18_24m','financialinstitutions24_60m','propertyinsurance24_60m','thirdpartfacilitator12_18m','thirdpartfacilitator18_24m','thirdpartfacilitator24_60m'])]
y = df['flag']
cust = df['custid']
os.chdir('tongdun/model_utils')
from ensemble_stacking import Ensemble
from sklearn.externals import joblib

en_model1 = Ensemble(5,'model1_all_gbdt.pkl')
pred = en_model1.predict(X)
flag = joblib.load('flag.pkl')
flags = np.zeros(y.shape)
flags[pred>=flag] = 1
#输出文件或者别的什么

#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
os.chdir('F:/company/tongdun/tongdun_score/20170711_1')
from sqlalchemy import create_engine

engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')
#df = pd.read_sql_query('select * from test.xxytongdun100', con=engine)
df = pd.read_sql_query('select * from test.xxytongduntest', con=engine)
df = pd.read_csv('xxytongdun100.csv', sep = ',')
colnames = df.columns.values
for col in colnames:
    df[col] = df[col].fillna(0)  
X = df[np.setdiff1d(df.columns, [['custid', 'flag','custwholeinfoid','reportid','breportid','bankconsumerfinance24_60m','bigconsumerfinance18_24m','bigconsumerfinance24_60m','creditcardcenter18_24m','financialinstitutions18_24m','financialinstitutions24_60m','propertyinsurance24_60m','thirdpartfacilitator12_18m','thirdpartfacilitator18_24m','thirdpartfacilitator24_60m']])]
os.chdir('tongdun/model_utils')
from ensemble_stacking import Ensemble
from sklearn.externals import joblib
en_model1 = Ensemble(5,'tongdunscore.pkl')
pred = en_model1.predict(X)
flag = joblib.load('flag.pkl')

flagall = joblib.load('flagall.pkl')




engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')
#df_t = pd.read_sql_query('select * from test.xxytongdun100', con=engine)
df = pd.read_sql_query('select * from test.xxytongduntest', con=engine)
df = pd.read_sql_query('select b.* from (select custid, max(custwholeinfoid) as custwholeinfoid from test.xxytongdun09 group by 1) a left join test.xxytongdunall b on a.custwholeinfoid = b.custwholeinfoid', con=engine)
df_t = pd.read_csv('xxytongdun100.csv', sep = ',')
colnames_t = df_t.columns.values
for col in colnames_t:
    df_t[col] = df_t[col].fillna(0)  
X_t = df_t[np.setdiff1d(df_t.columns, [['custid','flag', 'flag','custwholeinfoid','reportid','bankconsumerfinance24_60m','bigconsumerfinance18_24m','bigconsumerfinance24_60m','creditcardcenter18_24m','financialinstitutions18_24m','financialinstitutions24_60m','propertyinsurance24_60m','thirdpartfacilitator12_18m','thirdpartfacilitator18_24m','thirdpartfacilitator24_60m']])]
y = df_t['flag']
os.chdir('tongdun/model_utils')
from ensemble_stacking import Ensemble
from sklearn.externals import joblib
en_model1 = Ensemble(5,'tongdunscore.pkl')
pred = en_model1.predict(X)
flag = joblib.load('flag.pkl')

