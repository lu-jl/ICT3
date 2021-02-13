#!/usr/bin/python
# coding:utf8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


if __name__ == "__main__":
    randomState = np.random.RandomState(1)
    dataSet = np.linspace(0, 6, 100)[:, np.newaxis]
    labelSet = np.sin(dataSet).ravel() + np.sin(6 * dataSet).ravel() + randomState.normal(0, 0.1, dataSet.shape[0])

    dtRegressor = DecisionTreeRegressor(max_depth=4)
    adaRegressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                     n_estimators=300, random_state=randomState)

    dtRegressor.fit(dataSet, labelSet)
    adaRegressor.fit(dataSet, labelSet)

    # Predict
    dtPredictLabelSet = dtRegressor.predict(dataSet)
    adaPredictLabelSet = adaRegressor.predict(dataSet)

    # Plot the results
    plt.figure()
    plt.scatter(dataSet, labelSet, c="k", label="training samples")
    plt.plot(dataSet, dtPredictLabelSet, c="g", label="n_estimators=1", linewidth=2)
    plt.plot(dataSet, adaPredictLabelSet, c="r", label="n_estimators=300", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree & AdaBoost Regression")
    plt.legend()
    plt.show()
