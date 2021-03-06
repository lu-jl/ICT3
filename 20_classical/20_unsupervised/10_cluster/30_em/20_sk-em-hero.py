# -*- coding: utf-8 -*-
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabaz_score


if __name__ == '__main__':
    # 数据加载，避免中文乱码问题
    oriDataSet = pd.read_csv('data/heros.csv', encoding='gb18030')
    features = [u'最大生命', u'生命成长', u'初始生命',u'最大法力', u'法力成长',
                u'初始法力', u'最高物攻', u'物攻成长',u'初始物攻', u'最大物防',
                u'物防成长', u'初始物防', u'最大每5秒回血', u'每5秒回血成长',
                u'初始每5秒回血', u'最大每5秒回蓝', u'每5秒回蓝成长', u'初始每5秒回蓝',
                u'最大攻速', u'攻击范围']
    dataSet = oriDataSet[features]

    # 对英雄属性之间的关系进行可视化分析
    # 设置 plt 正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 用热力图呈现 features_mean 字段之间的相关性
    corr = dataSet[features].corr()
    plt.figure(figsize=(14, 14))
    sns.heatmap(corr, annot=True)  # annot=True 显示每个方格的数据
    plt.show()

    # 相关性大的属性保留一个，因此可以对属性进行降维
    selectedFeatures = [u'最大生命', u'初始生命', u'最大法力', u'最高物攻', u'初始物攻',
                        u'最大物防', u'初始物防', u'最大每5秒回血', u'最大每5秒回蓝',
                        u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
    dataSet = oriDataSet[selectedFeatures]
    dataSet[u'最大攻速'] = dataSet[u'最大攻速'].apply(lambda x: float(x.strip('%')) / 100)
    dataSet[u'攻击范围'] = dataSet[u'攻击范围'].map({'远程':1, '近战':0})

    # 采用 Z-Score 规范化数据，保证每个特征维度的数据均值为 0，方差为 1
    ss = StandardScaler()
    dataSet = ss.fit_transform(dataSet)

    # 构造 GMM 聚类
    gmm = GaussianMixture(n_components=30, covariance_type='full')

    # 训练数据
    gmm.fit(dataSet)

    # 产生聚类
    predictCluster = gmm.predict(dataSet)
    print("predictCluster is: ", predictCluster)

    # 将分组结果输出到 CSV 文件中
    # oriDataSet.insert(0, '分组', predictCluster)
    # oriDataSet.to_csv('./hero_out.csv', index=False, sep=',')

    print("score is: ", calinski_harabaz_score(dataSet, predictCluster))
