# 监督学习

## 简介

### 生成 vs. 判别

- 判别方法：由数据直接学习决策函数或条件概率分布。其基本思想是在有限样本条件下建立判别函数，不考虑样本的产生模型。不能反映训练数据本身的特性，但它可以寻找不同类别之间的最优分类面，反映的是异类数据之间的差异。
- 生成方法：从数据学习“联合概率密度分布”，然后求出条件概率分布作为预测模型，这类方法需要样本非常多时才能很好的描述数据的真正分布。之所以成为生成方法，是因为模型表示了给定输入 X 产生输出 Y 的生成关系。它从统计学的角度表示数据的分布情况，能够反映同类数据本身的相似度，但不关心划分各类的边界在哪里。

### 常用算法

分类（Classification）与回归最大的区别在于，不同的分类之前没有任何关联。

- 逻辑回归：常用
- 朴素贝叶斯（Naive Bayes）：用于 NLP
- 支持向量机 SVM（Support Vector Classifier）：中小型数据集表现好
- 决策树（Decision Tree Classifier）：常用
- 随机森林（Random Forest）：
- K-近邻算法 KNN（K-Nearest Neighbors）：较少用

## 原理

<img src="figures/image-20201115113106223.png" alt="image-20201115113106223" style="zoom:50%;" />



## 