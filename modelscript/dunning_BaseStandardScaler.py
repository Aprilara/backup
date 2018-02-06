# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:27:48 2017

@author: Administrator
"""


class BaseStandardScaler(object):
    def __init__(self, X_mean=0, X_std=1):
        self.X_mean = X_mean
        self.X_std = X_std

    def basescale(self, X_train):
        # calculate mean
        self.X_mean = X_train.mean(axis=0)
        # calculate std
        self.X_std = X_train.std(axis=0)
        # standardize X_train
        return (X_train - self.X_mean)/self.X_std

    def scale(self, X):
        return (X - self.X_mean) / self.X_std
