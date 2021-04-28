#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot
from numpy import array, zeros
from matplotlib.colors import ListedColormap


def file2matrix(fileName):
    """
    导入训练数据
    :param fileName: 数据文件路径
    :return: 数据矩阵
    """
    fr = open(fileName)
    # 获得文件中的数据行的行数
    fileSize = len(fr.readlines())
    # 生成对应的空矩阵
    instanceMatrix = zeros((fileSize, 3))
    labelVector = []
    fr = open(fileName)
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) 返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
        # 每列的属性数据
        instanceMatrix[index, :] = listFromLine[0:3]
        # 每列的类别数据，就是 label 标签数据
        labelVector.append(int(listFromLine[-1]))
        index += 1
    return instanceMatrix, labelVector


if __name__ == '__main__':
    # 从文件中加载数据
    datingInstanceMatrix, datingLabelVector = file2matrix('data/datingTestSet.txt')
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingInstanceMatrix[:, 0], datingInstanceMatrix[:, 1],
              s=15, c=15.0 * array(datingLabelVector))
    pyplot.show()

