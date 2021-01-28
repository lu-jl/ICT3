# 朴素贝叶斯方法

## 简介

朴素贝叶斯（Naive Bayes Classifier）被称为朴素，因为它假设每个输入变量是独立的。这是一个强有力的假设，对于实际数据是不现实的，然而，该技术对于大范围的复杂问题非常有效。

## 模型训练

### Model

- 采用贝叶斯：<img src="figures/image-20201116093005116.png" alt="image-20201116093005116" style="zoom:50%;" />
- P(C1)、P(C2)：可以直接通过training set统计得出
- P(x|C1）、P(x|C2）：假设满足高斯分布
  - 高斯分布：<img src="figures/image-20201116091725995.png" alt="image-20201116091725995" style="zoom: 50%;" />

### Goodness Function

- Lost Function：假设 P(x|C1）、P(x|C2）是样本的最大似然数
  - Likehood：<img src="figures/image-20201116092418517.png" alt="image-20201116092418517" style="zoom:50%;" /> 
  - 然后通过样本估计最大似然高斯分布可以确定 P(x|C1）、P(x|C2），从而确定他们高斯分布的 mean μ 和 covariance Σ：<img src="figures/image-20201116092515804.png" alt="image-20201116092515804" style="zoom:50%;" />

### Best Function

- 最大似然其实就是lost function的最小优化，最后通过获取的 P(x|C1）、P(x|C2）在<img src="figures/image-20201116091402029.png" alt="image-20201116091402029" style="zoom:50%;" /> 中判断



### 例子

贝叶斯分类的一个典型的应用场合是垃圾邮件分类，通过对样本邮件的统计，我们知道每个词在邮件中出现的概率 P(Ai)，我们也知道正常邮件概率 P(B0) 和垃圾邮件的概率 P(B1)，还可以统计出垃圾邮件中各个词的出现概率  P(Ai∣B1)，那么现在一封新邮件到来，我们就可以根据邮件中出现的词，计算  ``P(B1∣Ai) = P(B1) * P(Ai∣B1) / P(Ai)``，即得到这些词出现情况下，邮件为垃圾邮件的概率，进而判断邮件是否为垃圾邮件。






