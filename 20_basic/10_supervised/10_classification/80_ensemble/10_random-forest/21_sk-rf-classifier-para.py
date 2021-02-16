# -*- coding: utf-8 -*-
# 使用RandomForest对IRIS数据集进行分类
# 利用GridSearchCV寻找最优参数

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris


if __name__ == "__main__":
    rfClassifier = RandomForestClassifier()
    parameters = {"n_estimators": range(1, 11)}
    iris = load_iris()

    # 使用GridSearchCV进行参数调优
    paraEstimator = GridSearchCV(estimator=rfClassifier, param_grid=parameters)

    # 对iris数据集进行分类
    paraEstimator.fit(iris.data, iris.target)
    print("Best score is: ", paraEstimator.best_score_)
    print("Best parameter is: ", paraEstimator.best_params_)
