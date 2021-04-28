#!/usr/bin/env python
# coding: utf-8

from numpy import array, tile


def createDataSet():
    """
    创建数据集和标签
    """
    instanceSet = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  # instance是没有label的
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

    # 1. 距离计算
    instanceSetSize = instanceSet.shape[0]

    # tile生成和训练样本对应的矩阵，并与训练样本求差
    """
    tile: 复制给定内容，并生成指定行列的矩阵
    若 A = [0,1]，输入 B = tile(A, (4,1))，则
    B = array([[0,1],
        [0,1],
        [0,1],
        [0,1]])
    """
    diffMatrix = tile(testInstance, (instanceSetSize, 1)) - instanceSet
    # 取平方
    squareDiffMatrix = diffMatrix ** 2
    # 将矩阵的每一行相加
    squareDistanceMatrix = squareDiffMatrix.sum(axis=1)
    # 开方
    distanceMatrix = squareDistanceMatrix ** 0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    sortedDistanceIndexMatrix = distanceMatrix.argsort()

    # 2. 选择距离最小的k个点
    labelCount = {}
    for i in range(k):
        # 找到该样本的类型
        voteLabel = labelSet[sortedDistanceIndexMatrix[i]]
        # 在字典中将该类型加1
        labelCount[voteLabel] = labelCount.get(voteLabel, 0) + 1

    # 3.利用max函数直接返回字典中value最大的key
    maxLabelCount = max(labelCount, key=labelCount.get)
    return maxLabelCount


if __name__ == '__main__':
    instanceSet, labelSet = createDataSet()
    print(str(instanceSet))
    print(str(labelSet))
    print(classify([0.1, 0.1], instanceSet, labelSet, 3))
