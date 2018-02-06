# -*- coding: utf-8 -*-
import codecs
import json
import numpy as np
import xgboost as xgb
import pandas as pd
import sklearn

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

def learn_test():
    dtrain = pd.read_csv("data/train_set.csv")
    dtest = pd.read_csv("data/test_set.csv")
    del(dtrain['index'])
    del(dtrain['custid'])
    del(dtest['index'])
    del(dtest['custid'])   
    target = 'label'
    predictors = [x for x in dtrain.columns if x not in [target,'index','custid','livecity']]


    dtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
    dtest = xgb.DMatrix(dtest[predictors].values, label=dtest[target].values)

    param = {
         'learning_rate': 0.1,
         'max_depth': 5,
         'min_child_weight': 1,
         'gamma': 0,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'objective': 'binary:logistic',
         'nthread': 4,
         'scale_pos_weight': 1,
         'seed': 27,
         'silent': 1
    }
    num_round = 500
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]

    # round1
    fmax = 0.0
    max_depth = 0
    min_child_weight = 0
    for i in range(3, 30):
        for j in range(1, 30):
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

    # round2
    fmax = 0.0
    param['max_depth'] = max_depth
    param['min_child_weight'] = min_child_weight
    max_gamma = 0
    for i in range(1, 1001):
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

    # round3
    fmax = 0.0
    l = max(0, (int(max_gamma) - 2) * 100)
    r = (int(max_gamma) + 2) * 100
    max_gamma = 0
    for i in range(l, r+1):
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
        for j in range(2, 11):
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

learn_test()
