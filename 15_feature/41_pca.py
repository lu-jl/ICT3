from numpy import *
import matplotlib.pyplot as plt
import os
import sys

basepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(basepath, '../'))
import dataset
testset_path = os.path.join(dataset.pca_path, 'testSet.txt')


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topFeatNb=9999999):
    """pca

    Args:
        dataMat   原数据集矩阵
        topFeatNb  应用的N个特征
    Returns:
        lowDDataMat  降维后数据集
        reconMat     新的数据集空间
    """

    meanVals = mean(dataMat, axis=0)  # 计算每一列的均值

    meanRemoved = dataMat - meanVals  # 每个向量同时都减去均值

    # cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)+]/(n-1)
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))  # eigVals为特征值， eigVects为特征向量
    eigValInd = argsort(eigVals)  # 对特征值，进行从小到大的排序，返回从小到大的index序号

    # -1表示倒序，返回topN的特征值[-1 到 -(topNfeat+1) 但是不包括-(topNfeat+1)本身的倒叙]
    eigValInd = eigValInd[:-(topFeatNb + 1):-1]  # 特征值的逆序就可以得到topNfeat个最大的特征向量
    redEigVects = eigVects[:, eigValInd]  # 重组 eigVects 最大到最小

    # 将数据转换到新空间
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def show_picture(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0],
               dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:,
                                                       1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()


if __name__ == "__main__":
    dataMat = loadDataSet(testset_path)
    lowDDataMat, reconMat = pca(dataMat, 1)  # 只需要1个特征向量
    show_picture(dataMat, reconMat)
