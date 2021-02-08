#!/usr/bin/env python
# coding: utf-8

from numpy import shape, tile, zeros
from sklearn.neighbors import KNeighborsClassifier


def file2matrix(fileName):
    """
    导入训练数据
    :param fileName: 数据文件路径
    :return: 数据矩阵
    """
    fr = open(fileName)
    # 获得文件中的数据行的行数
    fileSize = len(fr.readlines())
    # 生成对应的空矩阵
    instanceMatrix = zeros((fileSize, 3))
    labelVector = []
    fr = open(fileName)
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) 返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
        # 每列的属性数据
        instanceMatrix[index, :] = listFromLine[0:3]
        # 每列的类别数据，就是 label 标签数据
        labelVector.append(int(listFromLine[-1]))
        index += 1
    return instanceMatrix, labelVector


def autoNorm(dataSet):
    """
    归一化特征值，消除属性之间量级不同导致的影响
    :param dataSet: 数据集
    :return: 归一化后的数据集normDataSet,ranges和minVals即最小值与范围，并没有用到

    归一化公式:
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。
        该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    dataSetSize = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - tile(minVals, (dataSetSize, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / tile(ranges, (dataSetSize, 1))
    return normDataSet, ranges, minVals


if __name__ == '__main__':
    # 从文件中加载数据
    datingInstanceMatrix, datingLabelVector = file2matrix('data/datingTestSet.txt')
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingInstanceMatrix)
    # 设置 trainingSet 与 testSet 的比例
    hoRatio = 0.1
    # 数据的行数
    normMatSize = normMat.shape[0]
    # 训练样本的数量
    testInstanceNb = int(normMatSize * hoRatio)
    errorCount = 0.0
    knnClassifier = KNeighborsClassifier(n_neighbors=3)
    knnClassifier.fit(datingInstanceMatrix, datingLabelVector)

    for i in range(testInstanceNb):
        result = knnClassifier.predict([datingInstanceMatrix[i]])
        print("the classifier came back with: %d, the real answer is: %d" % (result, datingLabelVector[i]))
        if result != datingLabelVector[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(testInstanceNb)))