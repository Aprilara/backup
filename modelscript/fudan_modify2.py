# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 18:50:29 2017

@author: Administrator
"""

dtrain = pd.read_csv("data/train_set.csv")
dtest = pd.read_csv("data/test_set.csv")
target = 'flag'
predictors = [x for x in dtrain.columns if x not in [target,'index','custid','livecity']]
#predictors = [x for x in dtrain.columns if x not in [target,'index','custid','livecity','login_number','diff_ip','time_D','time_M','time_A','time_E','create2lastlogin']]


dtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
dtest = xgb.DMatrix(dtest[predictors].values, label=dtest[target].values)
dt = xgb.DMatrix(df[predictors].values, label=df[target].values)

param = {
         'learning_rate': 0.1,
         'n_estimators': 1000,
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
for i in range(3, 30,3):
    for j in range(1, 30,3):
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
#max_depth=5, min_child_weight=2
    # round2
fmax = 0.0
param['max_depth'] = max_depth
param['min_child_weight'] = min_child_weight
max_gamma = 0
for i in range(1, 1001,10):
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
#max_gamma = 2.0
    # round3
fmax = 0.0
l = max(0, (int(max_gamma) - 2) * 100)
r = (int(max_gamma) + 2) * 100
max_gamma = 0
for i in range(l, r+1,5):
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
#max_gamma=2.23
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