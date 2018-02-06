# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:04:46 2017

@author: Administrator
"""

import jieba
import jieba.posseg as pseg
import os
import pandas as pd
os.chdir('F:/company/nlp/sms/')
jieba.load_userdict('mydict.txt')

from sqlalchemy import create_engine
import time

engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')

df = pd.read_sql_query("select * from credit.mobilesms where smsbody is not null and custid is not null and senddate is not null and date(createtime) between '2017-09-06' and '2017-09-10'", con=engine)
df = pd.read_pickle('df.pkl')
df = pd.read_csv('./input/sms.csv', sep = ',')
pd.to_pickle(df, './input/sms09060910.pkl')
start = time.time()
for index, row in df.iterrows():
    acttype = ''
    apptype = ''
    appname = ''
    overdue = 0
    words = pseg.cut(row.smsbody, HMM=False)
    for word, flag in words:
        if flag == 'loan':
            appname = word
            apptype = 'loan'
        elif flag == 'bk' and appname == '':
            appname = word
            apptype = 'bk'
        elif flag == 'fp':
            overdue = 1
        elif flag == 'rp':
            acttype = 'repay'
        elif flag == 'fl' and acttype == 'repay':
            overdue = 1
    df.set_value(index, 'appname', appname)
    df.set_value(index, 'apptype', apptype)
    df.set_value(index, 'overdue', overdue)       
end = time.time()
result = df[['channelid', 'custid', 'createtime', 'senddate', 'appname', 'apptype', 'overdue']]
result.to_csv('./output/result09060910.csv', header = False, index = False)


