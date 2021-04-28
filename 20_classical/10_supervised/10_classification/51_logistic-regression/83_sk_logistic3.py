#!/usr/bin/python
# coding: utf8

'''
Created on Oct 27, 2010
Update  on 2017-05-18
Logistic Regression Working Module
Author: 小瑶
GitHub: https://github.com/apachecn/AiLearning
scikit-learn的例子地址: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
'''

# 绘制多项式和一对二的逻辑回归 Plot multinomial and One-vs-Rest Logistic Regression
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

# 制作 3 类数据集进行分类
centers = [[-5, 0], [0, 1.5], [5, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)
transformation = [[0.4, 0.2], [-0.4, 1.2]]
X = np.dot(X, transformation)

for multi_class in ('multinomial', 'ovr'):
    clf = LogisticRegression(solver='sag', max_iter=100, random_state=42,
                             multi_class=multi_class).fit(X, y)

    # 打印训练分数
    print("training score : %.3f (%s)" % (clf.score(X, y), multi_class))

    # 创建一个网格来绘制
    h = .02  # 网格中的步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 绘制决策边界。为此，我们将为网格 [x_min, x_max]x[y_min, y_max]中的每个点分配一个颜色。
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # 将结果放入彩色图
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.title("Decision surface of LogisticRegression (%s)" % multi_class)
    plt.axis('tight')

    # 将训练点也绘制进入
    colors = "bry"
    for i, color in zip(clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)

    # 绘制三个一对数分类器
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    coef = clf.coef_
    intercept = clf.intercept_

    def plot_hyperplane(c, color):
        def line(x0):
            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
        plt.plot([xmin, xmax], [line(xmin), line(xmax)],
                 ls="--", color=color)

    for i, color in zip(clf.classes_, colors):
        plot_hyperplane(i, color)

plt.show()



