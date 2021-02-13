#!/usr/bin/python
# coding:utf8


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state


# Isotonic Regression 等式回归
if __name__ == "__main__":
    n = 100
    dataSet = np.arange(n)
    randomState = check_random_state(0)
    labelSet = randomState.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))

    isotonicRegressor = IsotonicRegression()
    isotonicFitDataSet = isotonicRegressor.fit_transform(dataSet, labelSet)

    linearRegressor = LinearRegression()
    linearRegressor.fit(dataSet[:, np.newaxis], labelSet)  # 线性回归的dataSet需要为2d

    fig = plt.figure()
    plt.plot(dataSet, labelSet, 'r.', markersize=12)
    plt.plot(dataSet, isotonicFitDataSet, 'g.-', markersize=12)
    plt.plot(dataSet, linearRegressor.predict(dataSet[:, np.newaxis]), 'b-')
    plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
    plt.title('Isotonic & Linear Regression')
    plt.show()
