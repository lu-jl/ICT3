#!/usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import *

"""
p(xy)=p(x|y)p(y)=p(y|x)p(x)
p(x|y)=p(y|x)p(x)/p(y)
"""


# 屏蔽社区留言板的侮辱性言论
def createDataSet():
    """
    创建数据集
    :return: 单词列表data set, 所属类别label set
    """
    dataSet = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labelSet = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return dataSet, labelSet


def createVocabList(dataSet):
    """
    获取所有单词的list
    :param dataSet: 数据集
    :return: 所有单词的list(即不含重复元素的单词列表)
    """
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 操作符 | 用于求两个集合的并集
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def createTermList(vocabList, inputSet):
    """
    遍历查看该单词是否出现，出现该单词则将该单词置1
    :param vocabList: 所有单词集合list
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    returnList = [0] * len(vocabList)  # [0,0......]
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnList[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnList


def createNBClassifier(trainMatrixArray, labelSetArray):
    """
    训练数据优化版本
    :param trainMatrixArray: 文件单词矩阵
    :param labelSetArray: 文件对应的类别
    :return:
    """
    # 总文件数
    numTrainDocs = len(trainMatrixArray)
    # 总单词数
    numWords = len(trainMatrixArray[0])
    # 侮辱性文件的出现概率
    pAbusive = sum(labelSetArray) / float(numTrainDocs)
    # 构造单词出现次数列表，p0Num 正常的统计，p1Num 侮辱的统计
    # 避免单词列表中的任何一个单词为0，而导致最后的乘积为0，所以将每个单词的出现次数初始化为 1
    p0Num = ones(numWords)
    p1Num = ones(numWords)

    # 整个数据集单词出现总数，2.0根据样本/实际调查结果调整分母的值（2主要是避免分母为0，当然值可以调整）
    # p0Denom 正常的统计，p1Denom 侮辱的统计
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if labelSetArray[i] == 1:
            # 累加辱骂词的频次
            p1Num += trainMatrixArray[i]
            # 对每篇文章的辱骂的频次进行统计汇总
            p1Denom += sum(trainMatrixArray[i])
        else:
            p0Num += trainMatrixArray[i]
            p0Denom += sum(trainMatrixArray[i])
    # 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    p1Vect = log(p1Num / p1Denom)
    # 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param pClass1: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    # 1. 加载数据集
    dataSet, labelSet = createDataSet()

    # 2. 创建单词集合
    myVocabList = createVocabList(dataSet)

    # 3. 计算单词是否出现并创建数据矩阵
    trainMatrix = []
    for postinDoc in dataSet:
        # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMatrix.append(createTermList(myVocabList, postinDoc))

    # 4. 训练数据
    p0V, p1V, pAb = createNBClassifier(array(trainMatrix), array(labelSet))

    # 5. 测试数据
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(createTermList(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(createTermList(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

