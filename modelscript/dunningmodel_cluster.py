# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 16:10:56 2018

@author: Administrator
"""
import os
os.chdir('F:/company/dunning/model')
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib


engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')
#df_all = pd.read_sql_query("select * from test.xxydunningfeaturem12 where successdunningtimes = 0 and date(startdunningtime)>'2017-12-01'", con=engine)
df_all = pd.read_sql_query("select * from test.dunningfeature ", con=engine)
X = df_all[np.setdiff1d(df_all.columns, ['custid','flag','loanborrowid','successdunningtimes','startdunningtime','duration'])]
y = df_all['flag']
df30 = df_all.loc[df_all.latedays>30,:][:]
X = df30[np.setdiff1d(df30.columns, ['custid','flag','loanborrowid','successdunningtimes','startdunningtime','duration'])]
y = df30['flag']



n_clusters = 3
km = KMeans(n_clusters=n_clusters, random_state=42)
km.fit(X)
for i in range(n_clusters):
    print('Cluster %d: %f, has sample %d' %(i,sum(y[km.labels_ == i])*1.0/len(y[km.labels_ == i]), len(y[km.labels_ == i])))
    
center0 = km.cluster_centers_[0]
dist = np.sqrt(np.sum(np.square(X_np - center0), axis=1))

[ 26.26635057,   1.58440581,   0.04560849,   0.77587115,
         2.39452096,   1.01016031,   1.51373523,   5.89079551,   1.53089486]
[u'age', u'cnt', u'cntlatedays', u'gender', u'latedays', u'maxlatedays',
       u'ostype', u'rank', u'sumlatedays']

##############################################################################
df = pd.read_csv('./input/feature.csv')
df = df.fillna(0)
X = df[np.setdiff1d(df.columns, ['custid','flag','loanborrowid','successdunningtimes','startdunningtime','duration','arrearageid'])]
X = np.array(X)
center = joblib.load('./center.pkl')
dist = np.sqrt(np.sum(np.square(X - center), axis=1))
result = df[['custid', 'loanborrowid','arrearageid']]
result['dist'] = 400+200.0*(212 - dist)/212
result.loc[result.dist<400,:] = 400
result['dist'] = result['dist'].astype(int)
result['caldate'] = datetime.datetime.now().strftime('%Y-%m-%d')
result.to_csv('./output/result.csv', header=False, index=False)

##############################################################################
df = pd.read_csv('./input/feature.csv')
df = df.fillna(0)
X = df[np.setdiff1d(df.columns, ['custid','flag','loanborrowid','successdunningtimes','startdunningtime','duration','arrearageid'])]
X = np.array(X)
center = joblib.load('./centernew.pkl')
dist = np.sqrt(np.sum(np.square(X - center), axis=1))
result = df[['custid', 'loanborrowid','arrearageid']]
result['score'] = dist
result_sort = result.sort_values(by='score', axis=0, ascending=False)

df_user = pd.read_csv('./input/dunner.csv')
df_user_sort = df_user.sort_values(by='score', axis=0, ascending=False)
inx = 0
for i in range(df_user_sort.shape[0]):
    cnt = df_user_sort.cnt[i]
    if ((inx+cnt)<=result.shape[0] and inx<result.shape[0]):
        result_sort.loc[inx:inx+cnt, 'dunningerid'] = df_user_sort.dunningerid[i]
    elif (inx<result.shape[0] and (inx+cnt)>result.shape[0]):
        result_sort.loc[inx:result.shape[0], 'dunningerid'] = df_user_sort.dunningerid[i]
    else:
        break
    inx += cnt
result_sort.to_csv('')

##################################################################

        