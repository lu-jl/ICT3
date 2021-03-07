#!/usr/bin/python
# coding:utf8

from matplotlib import pyplot
from numpy import arange, array, c_, meshgrid
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
    instanceSet = array([[-1.0, -1.1], [-1.0, -1.0], [0, 0], [1.0, 1.1],
                         [2.0, 2.0], [2.0, 2.1]])
    labelSet = array([0, 0, 0, 1, 1, 1])

    knnClassifier = KNeighborsClassifier(n_neighbors=3)
    knnClassifier.fit(instanceSet, labelSet)
    print("Predict: ", knnClassifier.predict([[-1.0, -1.0]]))
