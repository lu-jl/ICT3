#!/usr/bin/python
# coding:utf8

"""
Created on 2017-06-28
Updated on 2017-06-28
NaiveBayes: 朴素贝叶斯
Author: 小瑶
GitHub: https://github.com/apachecn/AiLearning
"""
from __future__ import print_function


# GaussianNB_高斯朴素贝叶斯
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
print(clf_pf.predict([[-0.8, -1]]))

