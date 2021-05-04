# Bagging

## 简介

Bagging（Bootstrap aggregating）的核心思想是民主，所有基础模型都一致对待，每个基础模型手里都只有一票，然后使用民主投票的方式得到最终的结果。

Bagging refers to averaging slightly different versions of the same model as a means to improve the predictive power.

### 原理

Bagging 的做法是对样本进程重采样，产生出若干个不同的子集，再从每个子集中训练出一个个体学习器。

- 从原始样本集中抽取训练集：每轮从原始样本集中使用 Bootstraping 的方法抽取 n 个训练样本（在训练集中，有些样本可能被多次抽取到，而有些样本可能一次都没有被抽中）。共进行 k 轮抽取，得到 k 个训练集。（k 个训练集之间是相互独立的）
- 每次使用一个训练集得到一个模型，k 个训练集共得到 k 个个体学习器。（注：这里并没有具体的分类算法或回归方法，可以根据具体问题采用不同的分类或回归方法，如决策树、感知器等）
- 简单投票法：在对预测结果进行结合时，Bagging 通常采用投票法汇总 k 个个体学习器的结果，所有模型的重要性相同。
  - 对分类问题：将上步得到的 k 个模型采用投票的方式得到分类结果；
  - 对回归问题：计算上述模型的均值作为最后的结果。

<img src="../figures/image-20200321125959179.png" alt="image-20200321125959179" style="zoom:33%;" />

### 偏差-方差

大部分情况下，经过 bagging 得到的结果方差（variance）更小。也就是说，bagging 在保持低 bias 的情况下，同时会减小 variances。

By making slightly different or let say randomized models, bagging ensures that the predictions do not read very high variance. They're generally more generalizable, bagging doesn't over exhaust the information in the training data. 

## 采样

Bagging 往往采用自助采样（Bootstrap Sampling），给定包含 m 个样本的数据集，随机取出一个样本放入采样集中，再把该样本放回初始数据集，使得下次该样本仍有可能被选中。经过 m 次随机操作，得到喊 m 个样本的采样集。初始训练集中有的样本在采样集中多次出现，有的从未出现，平均约有 63.2% 的样本出现在采样集中。

剩下的 36.8% 的样本可用作验证集来对泛化性能进程“包外评估”。

### 算法

- [随机森林 = Bagging + 决策树](10_random-forest.md)

- 

