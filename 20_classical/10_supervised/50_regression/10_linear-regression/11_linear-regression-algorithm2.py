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


# 局部加权线性回归
def lwlr(testPoint, dataSet, labelSet, k=1.0):
    """
    Description:
        局部加权线性回归，在待预测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归。
    Args:
        testPoint：样本点
        dataSet：样本的特征数据
        labelSet：每个样本对应的类别标签
        k：关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关
    Returns:
        testPoint * ws: 数据点与具有权重的系数相乘得到的预测点
    Notes:
        这其中会用到计算权重的公式，w = e^((x^((i))-x) / -2k^2)
        理解: x为某个预测点，x^((i))为样本点，样本点距离预测点越近，贡献的误差越大（权值越大），越远则贡献的误差越小（权值越小）。
        关于预测点的选取，在我的代码中取的是样本点。其中k是带宽参数，控制w（钟形函数）的宽窄程度，类似于高斯函数的标准差。
        算法思路: 假设预测点取样本点中的第i个样本点（共m个样本点），遍历1到m个样本点（含第i个），算出每一个样本点与预测点的距离，
        也就可以计算出每个样本贡献误差的权值，可以看出w是一个有m个元素的向量（写成对角阵形式）。
    """
    dataSetMatrix = mat(dataSet)
    yMat = mat(labelSet).T
    # 获得xMat矩阵的行数
    m = shape(dataSetMatrix)[0]
    # eye()返回一个对角线元素为1，其他元素为0的二维数组，创建权重矩阵weights，该矩阵为每个样本点初始化了一个权重                   
    weights = mat(eye(m))
    for j in range(m):
        # testPoint 的形式是 一个行向量的形式
        # 计算 testPoint 与输入样本点之间的距离，然后下面计算出每个样本贡献误差的权值
        diffMat = testPoint - dataSetMatrix[j, :]
        # k控制衰减的速度
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k**2))
    # 根据矩阵乘法计算 xTx ，其中的 weights 矩阵是样本点对应的权重矩阵
    xTx = dataSetMatrix.T * (weights * dataSetMatrix)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 计算出回归系数的一个估计
    ws = xTx.I * (dataSetMatrix.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, dataSet, labelSet, k=1.0):
    """
    Description:
        测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
    Args:
        testArr: 测试所用的所有样本点
        dataSet: 样本的特征数据，即 feature
        labelSet: 每个样本对应的类别标签，即目标变量
        k: 控制核函数的衰减速率
    Returns:
        yHat: 预测点的估计值
    """
    # 得到样本点的总数
    m = shape(testArr)[0]
    # 构建一个全部都是 0 的 1 * m 的矩阵
    yHat = zeros(m)
    # 循环所有的数据点，并将lwlr运用于所有的数据点 
    for i in range(m):
        yHat[i] = lwlr(testArr[i], dataSet, labelSet, k)
    # 返回估计值
    return yHat


if __name__ == "__main__":
    dataSet, labelSet = loadDataSet("data/data.txt")
    yHat = lwlrTest(dataSet, dataSet, labelSet, 0.003)
    dataSetMatrix = mat(dataSet)

    # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    srtInd = dataSetMatrix[:, 1].argsort(0)
    xSort = dataSetMatrix[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(
        [dataSetMatrix[:, 1].flatten().A[0]], [mat(labelSet).T.flatten().A[0]],
        s=2,
        c='red')
    plt.show()
