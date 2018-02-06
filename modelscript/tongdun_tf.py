# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
os.chdir('F:/company/tongdun/')
import utils
from sqlalchemy import create_engine

engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')
df = pd.read_sql_query('select * from test.tongdunselect', con=engine)
colnames = df.columns.values
for col in colnames:
    df[col] = df[col].fillna(0)
X = df[np.setdiff1d(df.columns, ['custid', 'flag','reportid'])]
y = df['flag']
#X_train, X_test, y_train, y_test = train_test_split(\
#        np.array(X), np.reshape(np.array(y),(-1,1)), test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(\
        np.array(X, dtype='float32'), np.array(y), test_size=0.3, random_state=42)
X_train_mean = X_train.mean(axis = 0)
X_train_std = X_train.std(axis=0)

X_train = (X_train - X_train_mean)/X_train_std
X_test = (X_test - X_train_mean)/X_train_std

train_dt = utils.Dataset(X_train, y_train)
batch_size = 1000

def add_layer(inputs, in_size, out_size, layer_name, activity_func=None):
    ''''''
    W = tf.Variable(tf.random_uniform([in_size, out_size], -1.0, 1.0), name='W')
    bias = tf.Variable(tf.constant(0.1, shape=[out_size]), name='bias')
    
    Wx_plus_b = tf.nn.xw_plus_b(inputs, W, bias)
    
    if activity_func is None:
        outputs = Wx_plus_b
    else:
        outputs = activity_func(Wx_plus_b)
    return outputs

hidden_layers = 2
hidden_units = [200,100]
n_inputs = 122
n_classes = 2
learning_rate = 0.01

n_epoch = 10
## define network
xs = tf.placeholder(tf.float32, [batch_size, n_inputs], name='input')
#ys1 = tf.placeholder(tf.int64, [None, 1], name='output')
ys1 = tf.placeholder(tf.int32, [batch_size,], name='output')
ys = tf.one_hot(ys1, depth=n_classes, on_value=1.0, off_value=0.0, axis=-1)
keep_prob = tf.placeholder(tf.float32)
l0 = add_layer(xs, n_inputs, hidden_units[0], 'hidden_layer_1', activity_func=tf.nn.softmax)

#for i in range(1,hidden_layers):
#    lo = add_layer(l0, hidden_units[i-1], hidden_units[i], 'hidden_layer_'+str(i), activity_func=tf.nn.softmax)
#    l0 = lo
pred = add_layer(l0,hidden_units[-1], n_classes, 'prediction_layer', activity_func=tf.nn.softmax)

#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ys))
loss = -tf.reduce_mean(12*tf.log(pred[:,0])*ys[:,0] + tf.log(pred[:,1])*ys[:,1])
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
tf.summary.scalar('loss', loss)

## accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

# para



with tf.Session() as sess:
    st = time.time()
    merged = tf.summary.merge_all()
    sess.run(init)
    write = tf.summary.FileWriter('./logs/', sess.graph)
    for epoch in range(n_epoch):
        n_batch = train_dt.num_examples / batch_size
        for i in range(int(n_batch)):
            batch_xs, batch_ys = train_dt.next_batch(batch_size)
            sess.run(train_op, feed_dict={xs: batch_xs, ys1: batch_ys, keep_prob:1.0})
            summry, _ = sess.run([merged, train_op], feed_dict={xs: batch_xs, ys1: batch_ys, keep_prob:1.0})
            write.add_summary(summry, epoch)
        print('epoch', epoch, 'accuracy:', sess.run(accuracy, feed_dict={xs: X_test[0:batch_size,:], ys1: y_test[0:batch_size], keep_prob:1.0}))
        print('pred:',sess.run(pred[1:10,:], feed_dict={xs: X_test[0:batch_size,:], ys1: y_test[0:batch_size], keep_prob:1.0}))
        print(tf.summary.scalar('loss', loss))
        
    
    ed = time.time()
    print('*'*30)
    print('time cost:', ed - st)
    