#!/usr/bin/env python
# coding: utf-8
"""
Created on Sep 16, 2010
Update  on 2017-05-18
Author: Peter Harrington/羊三/小瑶
GitHub: https://github.com/apachecn/AiLearning
"""

from __future__ import print_function
from numpy import *
# 导入科学计算包numpy和运算符模块operator
import operator
from os import listdir
from collections import Counter


def createDataSet():
    """
    创建数据集和标签

    调用方式
    import kNN
    group, labels = kNN.createDataSet()
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    inX[1,2,3]
    DS=[[1,2,3],[1,2,0]]
    inX: 用于分类的输入向量
    dataSet: 训练样本集的 feature
    labels: 训练样本集的标签向量，labels元素数目和dataSet行数相同
    k: 选择最近邻居的数目
    注意: 程序使用欧式距离公式.

    预测数据所在分类可在输入下列命令
    kNN.classify0([0,0], group, labels, 3)
    """

    """
    1. 计算距离

    欧氏距离:  点到点之间的距离
       第一行:  同一个点 到 dataSet的第一个点的距离。
       第二行:  同一个点 到 dataSet的第二个点的距离。
       ...
       第N行:  同一个点 到 dataSet的第N个点的距离。

    [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    (A1-A2)^2+(B1-B2)^2+(c1-c2)^2

    inx - dataset 使用了numpy broadcasting，见 https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
    np.sum() 函数的使用见 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sum.html
    """
    dist = sum((inX - dataSet)**2, axis=1)**0.5

    """
    2. k个最近的标签

    对距离排序使用numpy中的argsort函数， 见 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sort.html#numpy.sort
    函数返回的是索引，因此取前k个索引使用[0 : k]
    将这k个标签存在列表k_labels中
    """
    k_labels = [labels[index] for index in dist.argsort()[0 : k]]
    """
    3. 出现次数最多的标签即为最终类别

    使用collections.Counter可以统计各个标签的出现次数，most_common返回出现次数最多的标签tuple，例如[('lable1', 2)]，因此[0][0]可以取出标签值
    """
    label = Counter(k_labels).most_common(1)[0][0]
    return label


def test():
    group, labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(classify0([0.1, 0.1], group, labels, 3))


if __name__ == '__main__':
    test()
