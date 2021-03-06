# 聚类

聚类分析（Cluster Analysis）是无监督学习的一种方法，利用算法进行自动归类。分类算法主要解决如何将一个数据分到几个确定类别中的一类里去，它通常需要样本数据训练模型，再利用模型进行数据分类，所以是一种监督学习。聚类利用算法进行自动归类，通过聚类分析可以发现事物的内在规律，所以是无监督学习，不需要预先标注好的训练集。聚类其实就是将一个庞杂数据集中具有相似特征的数据自动归类到一起（称为一个簇），簇内的对象越相似，聚类的效果越好。聚类与分类最大的区别就是分类的目标事先已知，例如猫狗识别，在分类之前已经预先知道要将它分为猫、狗两个种类。而在聚类之前，目标是未知的。同样以动物为例，对于一个动物集来说，并不清楚这个数据集内部有多少种类的动物，能做的只是利用聚类方法自动按照特征分为多类，然后人为给出这个聚类结果的定义（即簇识别）。例如，将一个动物集分为了三簇（类），然后通过观察这三类动物的特征，为每一个簇起一个名字，如大象、狗、猫等，这就是聚类的基本思想。

## 性能度量

聚类的性能度量也称为有效性度量，用于评估聚类结果的好坏。其目的是使同一簇的样本尽可能彼此相似，不同簇的样本尽可能不同，也就是使簇内相似度高而簇间相似度低。

### 外部指标

将聚类结果与某个参考模型进行比较。

- Jaccard系数
- FM指数
- Rand指数
- DB指数
- Dunn指数

### 内部指标

直接参考聚类结果而不使用任何参考模型。

## 距离计算

至于“相似”这一概念，是利用距离这个评价标准来衡量的，通过计算对象与对象之间的特征距离远近来判断它们是否属于同一类别，即是否是同一个簇。至于距离如何计算，提出了许多种距离的计算方法，最常用的是闵可夫斯基距离（Minkowski Distance）：$dist_{mk}=(\displaystyle \sum_{u=1}^{n}|x_{iu}-x_{ju}|^p)^\frac{1}{p}$。

- 当 p=2 时，闵可夫斯基距离即欧式距离：$ d(x,y) ={\sqrt{ (x_{1}-y_{1})^{2}+(x_{2}-y_{2})^{2} + \cdots +(x_{n}-y_{n})^{2} }} ={\sqrt{ \sum_{ {i=1} }^{n}(x_{i}-y_{i})^{2} }} $
- 当 p=1 时，闵可夫斯基距离即曼哈顿距离：$dist_{man}(x_i,x_j)=||x_i-x_j||_1=\displaystyle \sum_{u=1}^n|x_{iu}-x_{ju}|$
- 不同属性的重要性不同时，可使用加权闵可夫斯基距离：$dist_{mk}=(\displaystyle \sum_{u=1}^{n}w_u|x_{iu}-x_{ju}|^p)^\frac{1}{p}$

## 具体算法

- [K-means K均聚类](20_k-means/README.md)
- [Birch](22_birch/README.md)
- [DBSCAN](26_dbscan/README.md)







