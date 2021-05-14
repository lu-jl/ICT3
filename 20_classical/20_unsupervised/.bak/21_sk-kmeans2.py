# -*- coding:UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

if __name__ == '__main__':
    # 加载数据集
    dataSet = []
    fr = open("dataset/testSet.txt")
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)    # 映射所有的元素为 float（浮点数）类型
        dataSet.append(fltLine)

    # 训练模型
    km = KMeans(n_clusters=4)
    km.fit(dataSet)
    km_pred = km.predict(dataSet)
    centers = km.cluster_centers_

    # 可视化结果
    plt.scatter(np.array(dataSet)[:, 1], np.array(dataSet)[:, 0], c=km_pred)
    plt.scatter(centers[:, 1], centers[:, 0], c="r")
    plt.show()
