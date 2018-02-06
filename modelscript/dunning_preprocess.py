# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:05:12 2017

@author: Administrator
"""
import numpy as np
import os
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
"""
This is used to deal with the abnormal values in data.
The default set will print out the columns 
which contain values 1000 times larger than their 90 percentiles.
If flag is set to True, the abnormal values will be repaired by their medians.
"""

def abnormalcol(df, times=1000, flag=False, percentile=90):
    if os.path.isfile('./abnormalcol.pkl'):
        a = joblib.load('./abnormalcol.pkl')
    else:
        a = np.nanpercentile(df, 50, axis=0)
        joblib.dump(a, './abnormalcol.pkl')
    i=0
    for col in df.columns.values:
        tmp = np.nanpercentile(df[col], percentile)
        cnt = sum(abs(df[col])>times*abs(tmp))
        percent = cnt*1.0/len(df[col])
        if cnt>0:
            print 'abcolumn is %s, there are %d rows, which takes %.2f%%' %(col,cnt,percent*100)
        if flag == True:
#            tmp1 = np.nanpercentile(df[col], 50)
            df.loc[abs(df[col])>times*abs(tmp), col] = a[i]
            i += 1
    if flag == True:
        return df

"""
count overdue rate of the top percentile ones.
""" 
def overduerate(pred, y, percent=-1):
    if percent == -1:
        for i in range(1,10):
            flag = np.percentile(pred,i*10, axis=0)
            print str(sum(y[pred<flag])*1.0/len(y[pred<flag]))
    else:
        flag = np.percentile(pred,percent, axis=0)
        return sum(y[pred<flag])*1.0/len(y[pred<flag])
    
        

"""
fill the nas.
three choices: 
    'zero' -- filled with 0
    'med' -- filled with median
    'avg' -- filled with average
"""
def fillnas(df, fillin = 'zero'):
    i = 0
    if fillin == 'zero':
        df = df.fillna(0)
        return df
    elif fillin == 'med':
        if os.path.isfile('./colmedian.pkl'):
            a = joblib.load('./colmedian.pkl')
        else:
            a = df.median()
            joblib.dump(a, './colmedian.pkl')
        for col in df.columns.values:
            df[col] = df[col].fillna(a[i])
            i += 1
        return df
    elif fillin == 'avg':
        if os.path.isfile('./colmean.pkl'):
            a = joblib.load('./colmean.pkl')
        else:
            a = df.mean()
            joblib.dump(a, './colmean.pkl')        
        for col in df.columns.values:
            df[col] = df[col].fillna(a[i])
            i += 1
        return df
    else:
        raise Exception("not the customized fillins. Choose from ('zero', 'med', 'avg')")

"""
devide df into train set and test set.
target: flag column name
col_exclude:list
"""
def dfdevide(df, target='flag', col_exclude=['flag'], test_size=0.3):
    df_tr, df_te = train_test_split(df, test_size=test_size, random_state=42)
    X_tr = df_tr[np.setdiff1d(df.columns, col_exclude)]
    y_tr = df_tr[target]
    X_te = df_te[np.setdiff1d(df.columns, col_exclude)]
    y_te = df_te[target]
    return X_tr, y_tr, X_te, y_te

"""
use pearson to select the valid columns.
"""
def pvalidcol(df, y, pvalid=0.05):
    validcol = []
    pval = []
    for col in df.columns:
        if pearsonr(df[col], y)[1]<pvalid:
            validcol.append(col)
            pval.append(pearsonr(df[col], y)[1])
    return validcol, pval