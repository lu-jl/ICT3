#!/usr/bin/env python
# coding: utf-8

from numpy import shape, tile, zeros


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
    for i in range(testInstanceNb):
        classifierResult = classify(normMat[i, :], normMat[testInstanceNb:normMatSize, :], datingLabelVector[testInstanceNb:normMatSize], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabelVector[i]))
        if classifierResult != datingLabelVector[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(testInstanceNb)))

