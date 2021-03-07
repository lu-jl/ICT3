#!/usr/bin/python
# coding: utf8


import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def createDataSet():
    # 导入数据
    dataSet = []
    labelSet = []
    with open("data/body.txt") as ifile:
        for line in ifile:
            # feature: 身高 体重   label:  胖瘦
            tokens = line.strip().split(' ')
            dataSet.append([float(tk) for tk in tokens[:-1]])
            labelSet.append(tokens[-1])
    # 特征数据
    dataSetArray = np.array(dataSet)
    # label分类的标签数据
    labelSet = np.array(labelSet)
    # 预估结果的标签数据
    labelSetArray = np.zeros(labelSet.shape)

    # label转换为0/1
    labelSetArray[labelSet == 'fat'] = 1
    return dataSetArray, labelSetArray


if __name__ == '__main__':
    dataSet, labelSet = createDataSet()

    # 拆分数据，80%做训练，20%做测试
    trainDataSet, testDataSet, trainLabelSet, testLabelSet = train_test_split(dataSet, labelSet, test_size=0.2)

    # 使用ID3决策树
    id3DecisionTree = tree.DecisionTreeClassifier(criterion='entropy')
    id3DecisionTree.fit(trainDataSet, trainLabelSet)

    # 系数反映每个feature的影响力，越大表示该feature在分类中起到的作用越大
    print('feature_importances_: %s' % id3DecisionTree.feature_importances_)

    # 得到决策树准确率
    acc_decision_tree = round(id3DecisionTree.score(trainDataSet, trainLabelSet), 6)
    print(u'score 准确率为 %.4lf' % acc_decision_tree)

    # 测试训练数据集
    predictTrainLabelSet = id3DecisionTree.predict(trainDataSet)
