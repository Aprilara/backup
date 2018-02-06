# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:45:09 2017

@author: Xu Xiaoyan
"""

# change pandas dataframe to svmlib format
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
import os
import xgboost as xgb
os.chdir('F:/company')

engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')
df = pd.read_sql_query('select * from test.tongdunselect1', con=engine)
colnames = df.columns.values

for col in colnames:
    df[col] = df[col].fillna(0)
X = df[np.setdiff1d(df.columns, ['custid', 'flag'])]
y = df['flag']
X_train, X_test, y_train, y_test = train_test_split(\
        X, y, test_size=0.3, random_state=42)
dump_svmlight_file(X_train, y_train, 'tongdunselect.txt.train')
dump_svmlight_file(X_test, y_test, 'tongdunselect.txt.test')

# deal with categorical type
# dummy = pd.get_dummies(df)
# mat = dummy.as_matrix()
# dump_svmlight_file(mat, y, 'svm-output.libsvm')

# --------------------------------------------------------------------------
# use xgboost to train models

dtrain = xgb.DMatrix('tongdunselect.txt.train')
dtest = xgb.DMatrix('tongdunselect.txt.test')
param = {'max_depth': 6, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'scale_pos_weight': 0.5}
num_round = 3
watchlist  = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train(param, dtrain, num_round, watchlist)
labels = dtest.get_label()
preds = bst.predict(dtest)
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
flag1 = np.percentile(preds, 80, axis=0)
sum(labels[preds<flag1])/sum(preds<flag1)
# max_depth 5 is better than 4 0.1707 0.174
# max_depth = 6 and num_round = 6
# ----------------------------------------------------------------------------
# a class to make train much easier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_

# feature selection: use random forest to see the importance of the features
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     # 'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}
from sklearn.ensemble import RandomForestClassifier
SEED = 42
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
rf_feature = rf.feature_importances(X_train,y_train)
import plotly.graph_objs as go
trace = go.Scatter(
    y = rf_feature,
    x = X_train.columns.values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = rf_feature,
        colorscale='Portland',
        showscale=True
    ),
    text = X_train.columns.values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
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
import plotly.offline as py
from plotly.offline import plot
plot(fig,filename='feature importance')

#-------------------model selection-----------------------
#------------------base  model ---------------------------
#------------------data preprocess -----------------------
from sklearn import preprocessing
X_mean = X_train.mean(axis = 0)
X_std = X_train.std(axis=0)
X_train_scale = (X_train - X_mean)/X_std
X_test_scale = (X_test - X_mean)/X_std
#------linear model ridge regression, tune alpha-----------
from sklearn import linear_model
clf = linear_model.Ridge(alpha=1.0, random_state = 42)
lrr = clf.fit(X_train_scale, y_train)
pred = lrr.predict(X_test_scale)
flag = np.percentile(pred,80,axis=0)
sum(y_test[pred<flag])*1.0/len(y_test[pred<flag]) #0.1664
#------svm, tune C, gamma --------------------------------
from sklearn.svm import SVC
clf = SVC(C=1.0, kernel='rbf',gamma='auto',probability=True,random_state=42,
          class_weight='balanced', max_iter=100)
svmm = clf.fit(X_train_scale,y_train)
pred = svmm.predict(X_test_scale)
flag = np.percentile(pred,80,axis=0)
sum(y_test[pred<flag])*1.0/len(y_test[pred<flag]) # 
#------random forests, tune max_depth, n_estimators,class_weight ------
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(max_features='auto', n_estimators=50, random_state=42)
clf.fit(X_train_scale,y_train)
pred = clf.predict(X_test_scale)
flag = np.percentile(pred,80,axis=0)
sum(y_test[pred<flag])*1.0/len(y_test[pred<flag]) #0.1741
#------extra tree regressor, tune n_estimator, max_features------------
from sklearn.ensemble import ExtraTreesRegressor
clf = ExtraTreesRegressor(n_estimators=50, max_features='auto',random_state=42)
