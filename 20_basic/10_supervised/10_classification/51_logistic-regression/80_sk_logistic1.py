#!/usr/bin/python
# coding: utf8


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    digits = datasets.load_digits()

    dataSet, labelSet = digits.data, digits.target
    dataSet = StandardScaler().fit_transform(dataSet)

    # 将大小数字分类为小
    labelSet = (labelSet > 4).astype(np.int)

    # 设置正则化参数
    for i, C in enumerate((100, 1, 0.01)):
        # 减少训练时间短的容忍度
        clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
        clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
        clf_l1_LR.fit(dataSet, labelSet)
        clf_l2_LR.fit(dataSet, labelSet)

        coef_l1_LR = clf_l1_LR.coef_.ravel()
        coef_l2_LR = clf_l2_LR.coef_.ravel()

        # coef_l1_LR contains zeros due to the
        # L1 sparsity inducing norm
        # 由于 L1 稀疏诱导规范，coef_l1_LR 包含零

        sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
        sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100

        print("C=%.2f" % C)
        print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR)
        print("score with L1 penalty: %.4f" % clf_l1_LR.score(dataSet, labelSet))
        print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR)
        print("score with L2 penalty: %.4f" % clf_l2_LR.score(dataSet, labelSet))

        l1_plot = plt.subplot(3, 2, 2 * i + 1)
        l2_plot = plt.subplot(3, 2, 2 * (i + 1))
        if i == 0:
            l1_plot.set_title("L1 penalty")
            l2_plot.set_title("L2 penalty")

        l1_plot.imshow(np.abs(coef_l1_LR.reshape(8, 8)), interpolation='nearest',
                       cmap='binary', vmax=1, vmin=0)
        l2_plot.imshow(np.abs(coef_l2_LR.reshape(8, 8)), interpolation='nearest',
                       cmap='binary', vmax=1, vmin=0)
        plt.text(-8, 3, "C = %.2f" % C)

        l1_plot.set_xticks(())
        l1_plot.set_yticks(())
        l2_plot.set_xticks(())
        l2_plot.set_yticks(())

    plt.show()
