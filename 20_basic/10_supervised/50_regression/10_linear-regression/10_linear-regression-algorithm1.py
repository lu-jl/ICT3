#!/usr/bin/python
# coding:utf8

from numpy import *
import matplotlib.pylab as plt


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataSet = []
    labelSet = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataSet.append(lineArr)
        labelSet.append(float(curLine[-1]))
    return dataSet, labelSet


def linearRegression(dataSet, labelSet):
    """
    Description: 
        线性回归
    Args:
        dataSet : 输入的样本数据，包含每个样本数据的 feature
        labelSet : 对应于输入数据的类别标签，也就是每个样本对应的目标变量
    Returns:
        ws: 回归系数
    """
    dataSetMatrix = mat(dataSet)
    # labelSet转换为矩阵 mat().T 代表的是对矩阵进行转置操作
    labelSetMatrixT = mat(labelSet).T
    # 矩阵乘法的条件是左矩阵的列数等于右矩阵的行数
    xTx = dataSetMatrix.T * dataSetMatrix
    # 因为要用到xTx的逆矩阵，所以事先需要确定计算得到的xTx是否可逆，条件是矩阵的行列式不为0
    # linalg.det() 函数是用来求得矩阵的行列式的，如果矩阵的行列式为0，则这个矩阵是不可逆的，就无法继续运算
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 最小二乘法，求得w的最优解
    ws = xTx.I * (dataSetMatrix.T * labelSetMatrixT)
    return ws


if __name__ == "__main__":
    dataSet, labelSet = loadDataSet("data/data.txt")
    dataSetMatrix = mat(dataSet)
    labelSetMatrix = mat(labelSet)
    ws = linearRegression(dataSet, labelSet)
    fig = plt.figure()
    # add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
    ax = fig.add_subplot(111)
    ax.scatter(
        [dataSetMatrix[:, 1].flatten()],
        [labelSetMatrix.T[:, 0].flatten().A[0]]) 
    xCopy = dataSetMatrix.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

