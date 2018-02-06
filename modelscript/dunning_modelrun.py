# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 11:41:03 2017

@author: Xiaoyan Xu
"""
import pandas as pd
import numpy as np
import datetime
from ensemble_stacking import Ensemble



df = pd.read_csv('./input/data.csv')
X = df[np.setdiff1d(df.columns, ['custid','flag','loanborrowid','successdunningtimes','startdunningtime','duration'])]
en_model = Ensemble(5,'model_dunning1.pkl')
pred = en_model.predict(X)
result = df[['custid','loanborrowid']][:]
result['modelcode'] = 'dunningmodel1.1'
result['score'] = 600*(1-(pred - pred.min())/(pred.max()- pred.min()))
result['score'] = result['score'].astype(int)
result['caldate'] = datetime.datetime.now().strftime('%Y-%m-%d')
result.to_csv('./output/result.csv', header=False, index=False)