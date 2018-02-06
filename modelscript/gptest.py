# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:53:39 2017

@author: Administrator
"""
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR

X_tr, y_tr = joblib.load('./data3/tr.pkl')
X_tr = np.array(X_tr)
y_tr = np.array(y_tr)
def rfrcv(n_estimators, min_samples_split, max_features):
    val = cross_val_score(
        RFR(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            random_state=2
        ),
        X_tr, y_tr, cv=2
    ).mean()
    return val

rfrBO = BayesianOptimization(
        rfrcv,
        {'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999)}
    )
gp_params = {"alpha": 1e-5}
rfrBO.maximize(n_iter=10, **gp_params)

def gbdtcv(n_estimators, min_samples_split,  max_depth):
    val = cross_val_score(
        GBR(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_depth=int(max_depth),
            random_state=2
        ),
        X_tr, y_tr, cv=2
    ).mean()
    return val
gbdtBO = BayesianOptimization(
        gbdtcv,
        {'n_estimators': (10, 250),
        'min_samples_split': (20, 80),
        'max_depth': (2,8)}
    )
gbdtBO.maximize(n_iter=10, **gp_params)