# -*- coding: utf-8 -*-
# 乳腺癌诊断分类，可以同时解决线性可分及线性不可分

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    data = pd.read_csv("dataset/data.csv")

    # 数据探索
    # 因为数据集中列比较多，我们需要把dataframe中的列全部显示出来
    pd.set_option('display.max_columns', None)
    print(data.columns)
    print(data.head(5))
    print(data.describe())

    # 将特征字段分成3组
    features_mean = list(data.columns[2:12])
    features_se = list(data.columns[12:22])
    features_worst = list(data.columns[22:32])

    # 数据清洗
    # ID列没有用，删除该列
    data.drop("id", axis=1, inplace=True)

    # 将B良性替换为0，M恶性替换为1
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # 将肿瘤诊断结果可视化
    # sns.countplot(data['diagnosis'], label="Count")
    # plt.show()

    # 用热力图呈现features_mean字段之间的相关性
    # corr = data[features_mean].corr()
    # plt.figure(figsize=(14, 14))

    # annot=True显示每个方格的数据
    # sns.heatmap(corr, annot=True)
    # plt.show()

    # 特征选择
    features_remain = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
                       'symmetry_mean', 'fractal_dimension_mean']

    # 抽取30%的数据作为测试集，其余作为训练集
    train, test = train_test_split(data, test_size=0.3)

    # 抽取特征选择的数值作为训练和测试数据
    trainDataSet = train[features_remain]
    trainLabelSet = train['diagnosis']
    testDataSet = test[features_remain]
    testLabelSet = test['diagnosis']

    # 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
    standardScaler = StandardScaler()
    trainDataSet = standardScaler.fit_transform(trainDataSet)
    testDataSet = standardScaler.transform(testDataSet)

    # 创建SVM分类器
    svmClassifier = svm.SVC()

    # 用训练集做训练
    svmClassifier.fit(trainDataSet, trainLabelSet)

    # 用测试集做预测
    predictTestLabelSet = svmClassifier.predict(testDataSet)
    print(u'score 准确率为 %.4lf' % metrics.accuracy_score(predictTestLabelSet, testLabelSet))
