# -*- coding: utf-8 -*-

# 使用RandomForest对IRIS数据集进行分类
# 利用GridSearchCV寻找最优参数,使用Pipeline进行流水作业

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


if __name__ == "__main__":
    rfClassifier = RandomForestClassifier()
    parameters = {"rfClassifier__n_estimators": range(1, 11)}
    iris = load_iris()
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rfClassifier', rfClassifier)
    ])

    # 使用GridSearchCV进行参数调优
    paraEstimatorPipeline = GridSearchCV(estimator=pipeline, param_grid=parameters)

    # 对iris数据集进行分类
    paraEstimatorPipeline.fit(iris.data, iris.target)
    print("Best Score is: ", paraEstimatorPipeline.best_score_)
    print("Best Parameter is: ", paraEstimatorPipeline.best_params_)
    # 运行结果：
    # 最优分数： 0.9667
    # 最优参数： {'n_estimators': 9}