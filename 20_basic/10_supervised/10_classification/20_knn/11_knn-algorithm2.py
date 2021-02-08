#!/usr/bin/env python
# coding: utf-8

from numpy import array, sum
from collections import Counter


def createDataSet():
    """
    创建数据集和标签
    """
    instanceSet = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labelSet = ['A', 'A', 'B', 'B']
    return instanceSet, labelSet


def classify(testInstance, instanceSet, labelSet, k):
    """
    testInstance: 用于分类的输入instance
    instanceSet: 训练样本集的，在KNN中也就是分类器的一部分
    labelSet: 训练样本集的标签，在KNN中也就是分类器的一部分
    k: KNN中的K，选择最近邻居的数目
    注意: 本程序使用欧式距离公式.
    """
    distanceMatrix = sum((testInstance - instanceSet)**2, axis=1)**0.5

    # k个最近的label
    kLabelSet = [labelSet[index] for index in distanceMatrix.argsort()[0: k]]

    """
    出现次数最多的标签即为最终类别
    使用collections.Counter可以统计各个标签的出现次数，most_common返回出现次数最多的标签tuple
    """
    maxLabelCount = Counter(kLabelSet).most_common(1)[0][0]
    return maxLabelCount


if __name__ == '__main__':
    instanceSet, labelSet = createDataSet()
    print(str(instanceSet))
    print(str(labelSet))
    print(classify([0.1, 0.1], instanceSet, labelSet, 3))
