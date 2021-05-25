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


# Logistic function 逻辑回归函数
# 这个类似于咱们之前讲解 logistic 回归的 Sigmoid 函数，模拟的阶跃函数

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

# 这是我们的测试集，它只是一条直线，带有一些高斯噪声。
xmin, xmax = -5, 5
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)
y = (X > 0).astype(np.float)
X[X > 0] *= 4
X += .3 * np.random.normal(size=n_samples)

X = X[:, np.newaxis]
# 运行分类器
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)

# 并且画出我们的结果
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X.ravel(), y, color='black', zorder=20)
X_test = np.linspace(-5, 10, 300)


def model(x):
    return 1 / (1 + np.exp(-x))
loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, color='red', linewidth=3)

ols = linear_model.LinearRegression()
ols.fit(X, y)
plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')

plt.ylabel('y')
plt.xlabel('X')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='small')
plt.show()

