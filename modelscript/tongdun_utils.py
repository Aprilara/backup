# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:30:28 2017

@author: Xu Xiaoyan
"""

from tensorflow.python.framework import dtypes
import numpy as np

class Dataset(object):
    
    def __init__(self,
                 features,
                 labels,
                 one_hot=False,
                 dtype=dtypes.float32):
        
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid feature dtype %r, expected unit8 or float32' % dtype)
        
        assert features.shape[0] == labels.shape[0]
        self._num_examples = features.shape[0]
        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    @property
    def features(self):
        return self._features
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size, shuffle=True):
        """return the next 'batch_size' examples from this data set."""
        
        start = self._index_in_epoch
        #shuffle the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._labels = self.labels[perm0]
        #go to the next epoch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            #get the rest of the examples
            rest_num_examples = self._num_examples - start
            features_rest_aprt = self._features[start:self._num_examples]
            labels_rest_aprt = self._labels[start:self._num_examples]
            #shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._features = self.features[perm]
                self._labels = self.labels[perm]
            #start the next poch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((features_rest_aprt, features_new_part), axis = 0), np.concatenate((labels_rest_aprt, labels_new_part), axis = 0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._labels[start:end]
        
        
def balanced_subsample(X, y, ratio=[0.5,0.5], values=[0,1]):
    
    size = X.shape[0]*np.array(ratio)
    class_size = size.astype(int)
    xs = []
    ys = []
    for i in range(len(values)):
        elems = X[(y == values[i])]
        index_selected = np.random.choice(elems.shape[0],class_size[i])
        x_ = elems[index_selected]
        y_ = np.empty(len(index_selected))
        y_.fill(values[i])
        
        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    return xs, ys
    
    