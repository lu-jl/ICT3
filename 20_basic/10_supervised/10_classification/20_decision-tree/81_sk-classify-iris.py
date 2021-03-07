#!/usr/bin/python
# coding:utf-8


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    # 参数
    n_classes = 3
    plot_colors = "bry"
    plot_step = 0.02

    # 加载数据
    irisSet = load_iris()

    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
        # 只选用iris DataSet中的2个feature
        dataSet = irisSet.data[:, pair]
        labelSet = irisSet.target

        # 训练Decision Tree
        decisionTree = DecisionTreeClassifier().fit(dataSet, labelSet)

        # 绘制决策边界
        plt.subplot(2, 3, pairidx + 1)

        x_min, x_max = dataSet[:, 0].min() - 1, dataSet[:, 0].max() + 1
        y_min, y_max = dataSet[:, 1].min() - 1, dataSet[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        print(np.c_[xx.ravel(), yy.ravel()])
        Z = decisionTree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

        plt.xlabel(irisSet.feature_names[pair[0]])
        plt.ylabel(irisSet.feature_names[pair[1]])
        plt.axis("tight")

        # 绘制训练点
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(labelSet == i)
            plt.scatter(dataSet[idx, 0], dataSet[idx, 1], c=color, label=irisSet.target_names[i],
                        cmap=plt.cm.Paired)

        plt.axis("tight")

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend()
    plt.show()
