#!/usr/bin/python
# coding:utf8

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


if __name__ == "__main__":
    randomState = np.random.RandomState(1)
    dataSet = np.sort(5 * randomState.rand(80, 1), axis=0)
    labelSet = np.sin(dataSet).ravel()
    labelSet[::5] += 3 * (0.5 - randomState.rand(16))

    # 拟合回归模型
    dtRegressor1 = DecisionTreeRegressor(max_depth=2)
    dtRegressor1.fit(dataSet, labelSet)
    dtRegressor2 = DecisionTreeRegressor(max_depth=5)
    dtRegressor2 = DecisionTreeRegressor(min_samples_leaf=6)
    dtRegressor2.fit(dataSet, labelSet)
    dtRegressor3 = DecisionTreeRegressor(max_depth=3)
    dtRegressor3.fit(dataSet, labelSet)

    # 预测
    testDataSet = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    predictTestLabelSet1 = dtRegressor1.predict(testDataSet)
    predictTestLabelSet2 = dtRegressor2.predict(testDataSet)
    predictTestLabelSet3 = dtRegressor3.predict(testDataSet)

    # 绘制结果
    plt.figure()
    plt.scatter(dataSet, labelSet, c="darkorange", label="data")
    plt.plot(testDataSet, predictTestLabelSet1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(testDataSet, predictTestLabelSet2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.plot(testDataSet, predictTestLabelSet3, color="red", label="max_depth=3", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()
