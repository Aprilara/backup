# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:03:55 2017

@author: Administrator
"""

from ensemble_stacking import Ensemble
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from preprocess import fillnas
import datetime

df = pd.read_csv('./input/feature.csv')
X = df[np.setdiff1d(df.columns, ['samephone', 'sameidcard', 'custid', 'wholeinfoid'])]
for col in X.columns.values:
    if X[col].dtypes == 'O':
        X[col] = X[col].astype('float')
X = fillnas(X, fillin='med')
model = Ensemble(5, 'model_taobao.pkl')
pred = model.predict(X)
flagall = joblib.load('flagall.pkl')
flagmax = max(flagall)
flagmin = min(flagall)

result = df[['custid', 'wholeinfoid']][:]
result['caldate'] = datetime.datetime.now().strftime('%Y-%m-%d')
result['modelcode'] = 'taobao1.0'
result['score'] = 600*(flagmax - pred)/(flagmax - flagmin)
result.loc[result['score']<1.1, 'score'] = 1
result['score'] = result['score'].astype('int')
result['otherinfo'] = ''
result.to_csv('./output/result.csv', )