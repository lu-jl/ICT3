from numpy import *
import matplotlib.pyplot as plt
import os, sys

basepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(basepath, '../'))
import dataset
secom_path = os.path.join(dataset.pca_path, 'secom.data')



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


def replaceNanWithMean():
    datMat = loadDataSet(secom_path, ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # 对value不为NaN的求均值
        # .A 返回矩阵基于的数组
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        # 将value为NaN的值赋值为均值
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat


def show_picture(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0],
               dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:,
                                                       1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()


def analyse_data(dataMat):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat-meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigvals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigvals)
    topFeatNb = 20
    eigValInd = eigValInd[:-(topFeatNb+1):-1]
    cov_all_score = float(sum(eigvals))
    sum_cov_score = 0
    for i in range(0, len(eigValInd)):
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score
        print('主成分: %s, 方差占比: %s%%, 累积方差占比: %s%%' % (format(i+1, '2.0f'), format(line_cov_score /
                                                                                 cov_all_score*100, '4.2f'), format(sum_cov_score/cov_all_score*100, '4.1f')))


if __name__ == "__main__":
    # 利用PCA对半导体制造数据降维
    dataMat = replaceNanWithMean()
    print(shape(dataMat))
    analyse_data(dataMat)  # 数据探索，发现前20已经占了大部分比例
    lowDDataMat, reconMat = pca(dataMat, 20)  # 从590维降到20维
    print(shape(lowDDataMat))
    show_picture(dataMat, reconMat)
