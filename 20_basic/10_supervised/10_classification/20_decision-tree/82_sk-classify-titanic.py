#!/usr/bin/python
# coding:utf-8

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    # 数据加载
    trainDataSet = pd.read_csv('data/titanic/train.csv')
    testDataSet = pd.read_csv('data/titanic/test.csv')

    # 数据探索
    """
    print(trainDataSet.info())
    print('-'*30)
    print(trainDataSet.describe())
    print('-'*30)
    print(trainDataSet.describe(include=['O']))
    print('-'*30)
    print(trainDataSet.head())
    print('-'*30)
    print(trainDataSet.tail())
    print('-'*30)
    # 确认'Embarked'这个feature有几个value，每个value用了几次
    print(train_data['Embarked'].value_counts())
    """

    # 数据清洗
    # 使用平均年龄来填充年龄中的 nan 值
    trainDataSet['Age'].fillna(trainDataSet['Age'].mean(), inplace=True)
    testDataSet['Age'].fillna(testDataSet['Age'].mean(), inplace=True)

    # 使用票价的均值填充票价中的 nan 值
    trainDataSet['Fare'].fillna(trainDataSet['Fare'].mean(), inplace=True)
    testDataSet['Fare'].fillna(testDataSet['Fare'].mean(), inplace=True)

    # 使用登录最多的港口来填充登录港口的 nan 值
    trainDataSet['Embarked'].fillna('S', inplace=True)
    testDataSet['Embarked'].fillna('S', inplace=True)

    # 特征选择
    featureList = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    trainFeaturedDataSet = trainDataSet[featureList]
    trainLabelSet = trainDataSet['Survived']
    testFeaturedDataSet = testDataSet[featureList]

    vectorTransformer = DictVectorizer(sparse=False)
    trainFeaturedDataSet = vectorTransformer.fit_transform(trainFeaturedDataSet.to_dict(orient='record'))
    testFeaturedDataSet = vectorTransformer.transform(testFeaturedDataSet.to_dict(orient='record'))

    # 构造ID3决策树
    id3DecisionTree = DecisionTreeClassifier(criterion='entropy')

    # 训练决策树, trainFeaturedDataSet和trainLabelSet必须是array
    id3DecisionTree.fit(trainFeaturedDataSet, trainLabelSet)

    # 得到决策树准确率
    acc_decision_tree = round(id3DecisionTree.score(trainFeaturedDataSet, trainLabelSet), 6)
    print(u'score 准确率为 %.4lf' % acc_decision_tree)

    # 决策树预测
    testLabelSet = id3DecisionTree.predict(testFeaturedDataSet)
    # print('test label set is: ', testLabelSet)
