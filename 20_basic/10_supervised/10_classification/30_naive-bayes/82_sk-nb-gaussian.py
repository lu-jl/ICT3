#!/usr/bin/python
# coding:utf8


import numpy as np
from sklearn.naive_bayes import GaussianNB

if __name__ == "__main__":
    dataSetArray = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    labelSetArray = np.array([1, 1, 1, 2, 2, 2])
    bayesClassifier = GaussianNB()
    bayesClassifier.fit(dataSetArray, labelSetArray)
    print('the predict result is: ', bayesClassifier.predict([[-0.8, -1]]))

