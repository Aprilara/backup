# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:34:35 2017

@author: Administrator
"""

import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#目录调整成自己需要的
os.chdir('F:/company/oldcustfeature/')
import preprocess
from sqlalchemy import create_engine
import time
from utils import Dataset
from sklearn.externals import joblib

engine = create_engine('postgresql://dev@192.168.91.11:5433/credit')
df = pd.read_sql_query('select * from test.xxyoldcustfeature', con=engine)
df = preprocess.fillnas(df)
y = df['flag']
X = df[np.setdiff1d(df.columns, ['custid','custwholeinfoid','reportid', 'finalscore', 'latedays','flag','updatetime'])]
X = preprocess.fillnas(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = 0.2)
colsum = X_tr.std(axis=0)
colzero = colsum[colsum == 0].index.values
X_tr = X_tr[np.setdiff1d(X_tr.columns, colzero)]
X_te = X_te[np.setdiff1d(X_tr.columns, colzero)]
X_val, X_te, y_val, y_te = train_test_split(X_te, y_te, test_size=0.5)
joblib.dump([X_tr, y_tr], './tf/data/tr.pkl')
joblib.dump([X_val, y_val], './tf/data/val.pkl')
joblib.dump([X_te, y_te], './tf/data/te.pkl')
X_mean = X_tr.mean(axis=0)
X_std = X_tr.std(axis=0)
X_tr = (X_tr - X_mean)/X_std
X_val = (X_val - X_mean)/X_std
X_te = (X_te - X_mean)/X_std


X_tr, y_tr = joblib.load('./tf/data/tr.pkl')
X_val, y_val = joblib.load('./tf/data/val.pkl')
X_te, y_te = joblib.load('./tf/data/te.pkl')
X_tr = np.array(X_tr, dtype='float32')
X_val = np.array(X_val, dtype='float32')
X_te = np.array(X_te, dtype='float32')
y_tr = np.array(pd.get_dummies(y_tr))
y_val = np.array(pd.get_dummies(y_val))
y_te = np.array(pd.get_dummies(y_te))



learning_rate = 0.001
train_dt = Dataset(X_tr, y_tr)
n_inputs = X_tr.shape[1]
hidden_units = [50,50,50,30]
n_classes = 2
xs = tf.placeholder(tf.float32, [None, n_inputs], name='input')
ys = tf.placeholder(tf.int32, [None, n_classes], name='output')

def variable_summuries(var):
    """attach a lot of summaries to a Tensor"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
def weight_variable(shape):
    """create a random varible and appropriatelly initialized"""
    initail = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initail)

def bias_variable(shape):
    """create a bias variable and appropriatelly initialized"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def nn_layer(input_tensor, input_size, output_size, layer_name, act = tf.nn.relu):
    
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_size, output_size])
            variable_summuries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_size])
            variable_summuries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32, name='kp')
    tf.summary.scalar('dropout_prob', keep_prob)
hidden1 = nn_layer(xs, n_inputs, hidden_units[0], 'layer1')
hidden1 = tf.nn.dropout(hidden1, keep_prob)
hidden2 = nn_layer(hidden1, hidden_units[0], hidden_units[1], 'layer2' )
hidden2 = tf.nn.dropout(hidden2, keep_prob)
hidden3 = nn_layer(hidden2, hidden_units[1], hidden_units[2], 'layer3')
hidden3 = tf.nn.dropout(hidden3, keep_prob)
hidden4 = nn_layer(hidden3, hidden_units[2], hidden_units[3], 'layer4')
hidden4 = tf.nn.dropout(hidden4, keep_prob)

pred = nn_layer(hidden4, hidden_units[3], n_classes, 'layer5', act=tf.identity)

with tf.name_scope('preds'):
    variable_summuries(pred[:,0])

with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=pred)
#    pred_s = tf.nn.softmax(pred)
#    diff = -(tf.multiply(tf.cast(ys[:,0], dtype='float32'),tf.log(pred_s[:,0]))*0.6 + tf.multiply(tf.cast(ys[:,1], dtype='float32'),tf.log(pred_s[:,1]))*1.5)
#    diff = -tf.reduce_sum(tf.cast(ys, dtype='float32')*tf.log(pred_s), [1])
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_sum(diff)
#        cross_entropy = tf.reduce_mean(tf.where(tf.greater(tf.arg_max(pred), tf.arg_max(ys)), 0.1, 0.93))
tf.summary.scalar('cross_entropy', cross_entropy)

vars = tf.trainable_variables()
lossl2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])*0.012
with tf.name_scope('regularization'):
    loss = tf.reduce_mean(lossl2+diff)
tf.summary.scalar('reg_loss', loss)
tf.summary.scalar('lossl2', lossl2)



with tf.name_scope('train'):
##梯度下降方法有很多，可以自己选几个测试一下效果 
#GradientDescentOptimizer,AdadeltaOptimizer,AdagradDAOptimizer,AdagradOptimizer等等
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.argmax(ys, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#        fg = np.percentile(pred[:,1],50)
#        accuracy = sum(ys[:,1][pred[:,1]<fg])*1.0/len(ys[:,1][pred[:,1]<fg])
tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver(max_to_keep=3)
sess = tf.InteractiveSession()
tf.add_to_collection('pred', pred)
#tf.add_to_collection('kp', keep_prob)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./boardlogs/train', sess.graph)
test_writer = tf.summary.FileWriter('./boardlogs/test')
tf.global_variables_initializer().run()


n_epoch = 100
batch_size = 512
steps = 0
max_acc = 0
delta = 0.00001
last_acc = 0


start_time = time.time()
for i in range(n_epoch):
    n_batch = train_dt.num_examples / batch_size
    for j in range(int(n_batch)):
        batch_xs, batch_ys = train_dt.next_batch(batch_size)
        sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.7})
        steps += 1    
    summary, acc = sess.run([merged, accuracy], feed_dict={xs:X_te, ys:y_te, keep_prob:1.0})
    train_writer.add_summary(summary, steps)
    test_writer.add_summary(summary, steps)
    saver.save(sess,'./output/model1',global_step=steps)
    print('Accuracy at step: %s: %s' % (steps, acc) )
    if acc>max_acc:
        pred1 = sess.run(pred, feed_dict={xs:X_te, keep_prob:1.0})
        max_acc = acc
    if (i+1)%200 == 0:
        learning_rate = learning_rate/2
    last_acc = acc

pf = sess.run(pred, feed_dict={xs:X_val, keep_prob:1.0})
end_time = time.time()
print('time cost: %s' % ((end_time - start_time)/60))

y_v = y_val[:,1]
pred_f = pf[:,1]
flag = np.percentile(pf[:,1],40)
sum(y_v[pred_f<flag])*1.0/len(y_v[pred_f<flag])

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('./output/model1-46000.meta')
  new_saver.restore(sess, './output/model1-46000')
#  model_file = tf.train.latest_checkpoint('./output/')
#  new_saver.restore(sess, model_file)
  # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
  pred_final = tf.get_collection('pred')[0]

  graph = tf.get_default_graph()

  # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
  input_x = graph.get_operation_by_name('input').outputs[0]
  keep_prob = graph.get_operation_by_name('dropout/kp').outputs[0]

  # 使用y进行预测  
  sess.run(pred_final, feed_dict={input_x:df_val,  keep_prob:1.0})