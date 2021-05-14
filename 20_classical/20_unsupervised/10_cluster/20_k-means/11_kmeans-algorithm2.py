#!/usr/bin/python
# coding:utf8


from numpy import *


def loadDataSet(fileName):
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataSet.append(fltLine)
    return dataSet


# 计算两个向量的欧式距离
def eulerDistance(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


# 为给定数据集构建一个包含k个随机质心的集合。
# 随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。
# 然后生成 0~1 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
def randomCentroids(dataSet, k):
    n = shape(dataSet)[1]  # 列数
    centroids = mat(zeros((k, n)))  # 创建k个质心矩阵
    for j in range(n):  # 创建随机簇质心，并且在每一维的边界内
        minJ = min(dataSet[:, j])  # 最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)  # 范围 = 最大值 - 最小值
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))  # 随机生成
    return centroids


# k-means 聚类算法
# 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
# 这个过程重复数次，知道数据点的簇分配结果不再改变位置。
def kMeans(dataSet, k, distMeas=eulerDistance, createCent=randomCentroids):
    m = shape(dataSet)[0]  # 行数
    clusterAssment = mat(zeros(
        (m, 2)))  # 创建一个与 dataSet 行数一样，但是有两列的矩阵，用来保存簇分配结果
    centroids = createCent(dataSet, k)  # 创建质心，随机k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 循环每一个数据点并分配到最近的质心中去
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :],
                                  dataSet[i, :])  # 计算数据点到质心的距离
                if distJI < minDist:  # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 簇分配结果改变
                clusterChanged = True  # 簇改变
                clusterAssment[
                    i, :] = minIndex, minDist**2  # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
        print(centroids)
        for cent in range(k):  # 更新质心
            ptsInClust = dataSet[nonzero(
                clusterAssment[:, 0].A == cent)[0]]  # 获取该簇中的所有点
            centroids[cent, :] = mean(
                ptsInClust, axis=0)  # 将质心修改为簇中所有点的平均值，mean 就是求平均值的
    return centroids, clusterAssment


# 二分 KMeans 聚类算法, 基于 kMeans 基础之上的优化，以避免陷入局部最小值
def biKMeans(dataMat, k, distMeas=eulerDistance):
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m, 2)))  # 保存每个数据点的簇分配结果和平方误差
    centroid0 = mean(dataMat, axis=0).tolist()[0]  # 质心初始化为所有数据点的均值
    centList = [centroid0]  # 初始化只有 1 个质心的 list
    for j in range(m):  # 计算所有数据点到初始质心的距离平方误差
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataMat[j, :])**2
    while (len(centList) < k):  # 当质心数量小于 k 时
        lowestSSE = inf
        for i in range(len(centList)):  # 对每一个质心
            ptsInCurrCluster = dataMat[nonzero(
                clusterAssment[:, 0].A == i)[0], :]  # 获取当前簇 i 下的所有数据点
            centroidMat, splitClustAss = kMeans(
                ptsInCurrCluster, 2, distMeas)  # 将当前簇 i 进行二分 kMeans 处理
            sseSplit = sum(splitClustAss[:, 1])  # 将二分 kMeans 结果中的平方和的距离进行求和
            sseNotSplit = sum(
                clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0],
                               1])  # 将未参与二分 kMeans 分配结果中的平方和的距离进行求和
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 找出最好的簇分配结果    
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(
            centList)  # 调用二分 kMeans 的结果，默认簇是 0,1. 当然也可以改成其它的数字
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0],
                     0] = bestCentToSplit  # 更新为最佳质心
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 更新质心列表
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[
            0]  # 更新原质心 list 中的第 i 个质心为使用二分 kMeans 后 bestNewCents 的第一个质心
        centList.append(
            bestNewCents[1, :].tolist()[0])  # 添加 bestNewCents 的第二个质心
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[
            0], :] = bestClustAss  # 重新分配最好簇下的数据（质心）以及SSE
    return mat(centList), clusterAssment


if __name__ == "__main__":
    # 加载测试数据集
    dataSet = mat(loadDataSet('dataset/testSet2.txt'))
    centList, myNewAssments = biKMeans(dataSet, 3)
    print('centList = ', centList)
