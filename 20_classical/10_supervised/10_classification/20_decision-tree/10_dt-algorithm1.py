#!/usr/bin/python
# coding:utf-8

import operator
from math import log


def createDataSet():
    """DateSet 基础数据集

    Args:
        无需传入参数
    Returns:
        返回数据集和对应的label的list
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labelList = ['no surfacing', 'flippers']
    return dataSet, labelList


def calcShannonEnt(dataSet):
    """calcShannonEnt(calculate Shannon entropy 计算给定数据集的香农熵)

    Args:
        dataSet 数据集
    Returns:
        返回 一组数据集的信息熵
    """
    # 求list的长度，表示计算参与训练的数据量
    nbEntries = len(dataSet)
    # 计算分类标签label出现的次数
    labelCounts = {}
    # the number of unique elements and their occurance
    for entry in dataSet:
        # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = entry[-1]
        # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 对于label标签的占比，求出label标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率。
        prob = float(labelCounts[key])/nbEntries
        # 计算香农熵，以 2 为底求对数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, featureIndex, featureIndexValue):
    """如果index feature列的数据等于value，就要将instance划分到新建的数据集中
    Args:
        dataSet 数据集                     待划分的数据集
        index 表示每一行的index feature列   划分数据集的特征
        value 表示index列对应的value值      需要返回的特征的值。
    Returns:
        index列为value的所有instance集
    """
    returnDataSet = []
    for instance in dataSet:
        # 判断index列的值是否为value
        if instance[featureIndex] == featureIndexValue:
            # 删除该feature
            reducedFeatVec = instance[:featureIndex]
            reducedFeatVec.extend(instance[featureIndex + 1:])
            # [index+1:]表示从跳过index的index+1行，取接下来的数据
            returnDataSet.append(reducedFeatVec)
    return returnDataSet


def chooseBestFeatureToSplit(dataSet):
    """chooseBestFeatureToSplit(选择最好的特征)

    Args:
        dataSet 数据集
    Returns:
        bestFeature 最优的特征列
    """
    # feature总数, 最后一列是label，所以不算
    nbFeatures = len(dataSet[0]) - 1
    # label的信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值, 和最优的feature编号
    bestInfoGain, bestFeature = 0.0, -1
    for featureIndex in range(nbFeatures):
        # create a list of all the examples of this feature
        featureList = [example[featureIndex] for example in dataSet]
        # get a set of unique values of this feature
        featureValueSet = set(featureList)
        # 创建一个临时的信息熵
        newEntropy = 0.0
        # 遍历当前特征中的所有唯一值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并求熵求和
        for featureValue in featureValueSet:
            subDataSet = splitDataSet(dataSet, featureIndex, featureValue)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
        # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = featureIndex
    return bestFeature


def majorityCnt(classList):
    """majorityCnt(选择出现次数最多的一个结果)

    Args:
        classList label列的集合
    Returns:
        bestFeature 最优的特征列
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # print 'sortedClassCount:', sortedClassCount
    return sortedClassCount[0][0]


def createTree(dataSet, labelList):
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件: 所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件: 使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeature = chooseBestFeatureToSplit(dataSet)
    # 获取最优feature的label名称
    bestFeatureLabel = labelList[bestFeature]
    # 初始化myTree
    myTree = {bestFeatureLabel: {}}
    # 删除已用label
    del(labelList[bestFeature])
    # 取出最优列，然后它的branch做分类
    featureValueList = [example[bestFeature] for example in dataSet]
    featureValueSet = set(featureValueList)
    for featureValue in featureValueSet:
        # 求出剩余的标签label
        subLabels = labelList[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatureLabel][featureValue] = createTree(splitDataSet(dataSet, bestFeature, featureValue), subLabels)
    return myTree


def classify(inputTree, featureLabelList, testVec):
    """classify(给输入的节点，进行分类)

    Args:
        inputTree  决策树模型
        featLabels Feature标签对应的名称
        testVec    测试输入的数据
    Returns:
        classLabel 分类的结果值，需要映射label才能知道名称
    """
    # 获取tree的根节点对于的key值
    firstStr = inputTree.keys()[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featureLabelList.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featureLabelList, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


if __name__ == "__main__":
    # 创建数据和结果标签
    myDataSet, myLabelList = createDataSet()

    import copy
    myTree = createTree(myDataSet, copy.deepcopy(myLabelList))
    print(myTree)

    # 画图可视化展现
    # dtPlot.createPlot(myTree)

    # no surfacing and flippers is a fish
    print(classify(myTree, myLabelList, [1, 1]))
