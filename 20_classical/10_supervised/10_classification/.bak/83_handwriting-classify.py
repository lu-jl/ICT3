#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from numpy import *
# 导入科学计算包numpy和运算符模块operator
import operator
from os import listdir
from collections import Counter


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


def img2vector(fileName):
    """
    将图像数据转换为向量
    :param fileName: 图片文件 因为我们的输入数据的图片格式是 32 * 32的
    :return: 一维矩阵
    该函数将图像转换为向量: 该函数创建 1 * 1024 的NumPy数组，然后打开给定的文件，
    循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    returnVect = zeros((1, 1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


if __name__ == '__main__':
    # 1. 导入数据
    hwLabels = []
    trainingFileList = listdir('../10_knn/dataset/digits/training')
    trainingDataSize = len(trainingFileList)
    trainingMat = zeros((trainingDataSize, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(trainingDataSize):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('data/digits/training/%s' % fileNameStr)

    # 2. 导入测试数据
    testFileList = listdir('../10_knn/dataset/digits/test')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('data/digits/test/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


