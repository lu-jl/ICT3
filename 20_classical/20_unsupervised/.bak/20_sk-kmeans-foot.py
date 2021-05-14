# coding: utf-8
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # 输入数据
    dataSet = pd.read_csv('dataset/data.csv', encoding='gbk')
    trainDataSet = dataSet[['2019', '2018', '2015']]
    kmeans = KMeans(n_clusters=3)

    # 规范化到[0,1]空间
    min_max_scaler = preprocessing.MinMaxScaler()
    trainDataSet = min_max_scaler.fit_transform(trainDataSet)

    # kmeans算法
    kmeans.fit(trainDataSet)
    predictTrainLabelSet = kmeans.predict(trainDataSet)

    # 合并聚类结果，插入到原数据中
    result = pd.concat((dataSet, pd.DataFrame(predictTrainLabelSet)), axis=1)
    result.rename({0: u'聚类'}, axis=1, inplace=True)
    print(result)
