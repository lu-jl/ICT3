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


from __future__ import print_function

# Logistic Regression 3-class Classifier 逻辑回归 3-类 分类器

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# 引入一些数据来玩
iris = datasets.load_iris()
# 我们只采用样本数据的前两个feature
X = iris.data[:, :2]
Y = iris.target

h = .02  # 网格中的步长

logreg = linear_model.LogisticRegression(C=1e5)

# 我们创建了一个 Neighbours Classifier 的实例，并拟合数据。
logreg.fit(X, Y)

# 绘制决策边界。为此我们将为网格 [x_min, x_max]x[y_min, y_max] 中的每个点分配一个颜色。
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# 将结果放入彩色图中
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# 将训练点也同样放入彩色图中
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()

