#!/usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import *


# 屏蔽社区留言板的侮辱性言论
def createDataSet():
    dataSet = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labelSet = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return dataSet, labelSet


# 获取所有单词不重复的list
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 操作符"|"用于求两个集合的并集
    return list(vocabSet)


# 创建单词是否出现list
def createTermList(vocabList, inputSet):
    returnList = [0] * len(vocabList)  # 创建一个和词汇表等长的list，并将其元素初始为0
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnList[vocabList.index(word)] = 1
    return returnList


def createNBClassifier(trainMatrixArray, labelSetArray):
    """
    p(c)：pAbusive
    p(x|c)：p0Num/p0Denom ...
    """
    numTrainDocs = len(trainMatrixArray)  # 总文件数
    numWords = len(trainMatrixArray[0])  # 每个文件的单词数
    pAbusive = sum(labelSetArray) / float(numTrainDocs)  # 侮辱性文件出现的概率

    # 拉普拉斯修正，将每个单词的出现次数初始化为 1
    p0Num = ones(numWords)  # 单词在标签为正常的文件中出现的次数
    p1Num = ones(numWords)  # 单词在标签为侮辱的文件中出现的次数

    # 拉普拉斯修正，整个文件单词总数
    p0Denom = 2.0  # 正常单词总数
    p1Denom = 2.0  # 侮辱单词总数

    # 针对每篇文章
    for i in range(numTrainDocs):
        if labelSetArray[i] == 1:  # 如果该文章为侮辱文章
            p1Num += trainMatrixArray[i]  # 累加辱骂词的频次
            p1Denom += sum(trainMatrixArray[i])  # 对每篇文章的辱骂的频次进行统计汇总
        else:
            p0Num += trainMatrixArray[i]
            p0Denom += sum(trainMatrixArray[i])

    p0Vect = log(p0Num / p0Denom)  # 每个 x 的 P(x|C0)：类别0，即正常文档
    p1Vect = log(p1Num / p1Denom)  # 每个 x 的 P(x|C1)：类别1，即侮辱性文档
    return p0Vect, p1Vect, pAbusive  # P(x|C0)，P(x|C1)，P(C1)


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
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
    for doc in dataSet:
        # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMatrix.append(createTermList(myVocabList, doc))

    # 4. 训练数据
    p0V, p1V, pAb = createNBClassifier(array(trainMatrix), array(labelSet))

    # 5. 测试数据
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(createTermList(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(createTermList(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

