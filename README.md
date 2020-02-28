# 人工智能

## 简介
模型训练方式不同可以分为监督学习（Supervised Learning），无监督学习（Unsupervised Learning）、半监督学习（Semi-supervised Learning）和强化学习（Reinforcement Learning）四大类。


## 数学基础
- [数学基础](10_math/README.md)


## 机器学习基础
- [机器学习基础](20_basics/README.md)

## 模型训练方式


#### 线性分类器（Linear Classifier）

##### Fisher的线性判别（Fisher’s Linear Discriminant）


##### 多项逻辑回归（Multionmial Logistic Regression）

##### 感知（Perception）



### 无监督学习



#### 关联规则学习（Association Rule Learning）
##### 先验算法（Apriori Algorithm）

##### Eclat算法（Eclat Algorithm）

##### FP-Growth算法

#### 分层聚类算法（Hierarchical Clustering）
##### 单连锁聚类（Single-linkage Clustering）

##### 概念聚类（Conceptual Clustering）



#### 异常检测（Anomaly detection）
##### K最邻近（K-nearest Neighbor，KNN）算法

##### 局部异常因子算法（Local Outlier Factor，LOF）

### 半监督学习
#### 生成模型（Generative Models）

#### 低密度分离（Low-density Separation）

#### 基于图形的方法（Graph-based Methods）

#### 联合训练（Co-training）

### 强化学习
#### Q学习（Q-learning）

#### 状态-行动-奖励-状态-行动（State-Action-Reward-State-Action，SARSA）

#### DQN（Deep Q Network）

#### 策略梯度算法（Policy Gradients）

#### 基于模型强化学习（Model Based RL）

#### 时序差分学习（Temporal Different Learning）

## 解决任务方式
### 二分类（Two-class Classification）
#### 二分类支持向量机（Two-class SVM）
适用于数据特征较多、线性模型的场景。

#### 二分类平均感知器（Two-class Average Perceptron）
适用于训练时间短、线性模型的场景。

#### 二分类逻辑回归（Two-class Logistic Regression）
适用于训练时间短、线性模型的场景。

#### 二分类贝叶斯点机（Two-class Bayes Point Machine）
适用于训练时间短、线性模型的场景。

#### 二分类决策森林（Two-class Decision Forest）
适用于训练时间短、精准的场景。

#### 二分类提升决策树（Two-class Boosted Decision Tree）
适用于训练时间短、精准度高、内存占用量大的场景

#### 二分类决策丛林（Two-class Decision Jungle）
适用于训练时间短、精确度高、内存占用量小的场景。

#### 二分类局部深度支持向量机（Two-class Locally Deep SVM）
适用于数据特征较多的场景。

#### 二分类神经网络（Two-class Neural Network）
适用于精准度高、训练时间较长的场景。

### 多分类算法（Multi-class Classification）
#### 多分类逻辑回归（Multiclass Logistic Regression）
适用训练时间短、线性模型的场景。

#### 多分类神经网络（Multiclass Neural Network）
适用于精准度高、训练时间较长的场景。

#### 多分类决策森林（Multiclass Decision Forest）
适用于精准度高，训练时间短的场景。

#### 多分类决策丛林（Multiclass Decision Jungle）
适用于精准度高，内存占用较小的场景。

#### “一对多”多分类（One-vs-all Multiclass）
取决于二分类器效果。

### 回归算法（Regression）
回归问题通常被用来预测具体的数值而非分类。除了返回的结果不同，其他方法与分类问题类似。我们将定量输出，或者连续变量预测称为回归；将定性输出，或者离散变量预测称为分类。
#### 排序回归（Ordinal Regression）
适用于对数据进行分类排序的场景。

#### 泊松回归（Poission Regression）
适用于预测事件次数的场景。

#### 快速森林分位数回归（Fast Forest Quantile Regression）
适用于预测分布的场景。


#### 神经网络回归（Neural Network Regression）
适用于精准度高、训练时间较长的场景。

#### 决策森林回归（Decision Forest Regression）
适用于精准度高、训练时间短的场景。

#### 提升决策树回归（Boosted Decision Tree Regression）
适用于精确度高、训练时间短、内存占用较大的场景。

### 聚类算法（Clustering）
聚类的目标是发现数据的潜在规律和结构。聚类通常被用做描述和衡量不同数据源间的相似性，并把数据源分类到不同的簇中。
#### 层次聚类（Hierarchical Clustering）
适用于训练时间短、大数据量的场景。

#### K-means算法
适用于精准度高、训练时间短的场景。

#### 模糊聚类FCM算法（Fuzzy C-means，FCM）
适用于精确度高、训练时间短的场景。

#### SOM神经网络（Self-organizing Feature Map，SOM）
适用于运行时间较长的场景。

### 异常检测（Anomaly Detection）
异常检测是指对数据中存在的不正常或非典型的分体进行检测和标志，有时也称为偏差检测。
异常检测看起来和监督学习问题非常相似，都是分类问题。都是对样本的标签进行预测和判断，但是实际上两者的区别非常大，因为异常检测中的正样本（异常点）非常小。

#### 一分类支持向量机（One-class SVM）
适用于数据特征较多的场景。

#### 基于PCA的异常检测（PCA-based Anomaly Detection）
适用于训练时间短的场景。

