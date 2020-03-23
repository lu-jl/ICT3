# 人工智能

## 简介
模型训练方式不同可以分为监督学习（Supervised Learning），无监督学习（Unsupervised Learning）、半监督学习（Semi-supervised Learning）和强化学习（Reinforcement Learning）四大类。

- 监督学习：基于已知类别的训练数据进行学习。监督学习假定训练数据满足独立同分布的条件，并根据训练数据学习出一个由输入到输出的映射模型。反映这一映射关系的模型可能有无数种，所有模型共同构成了假设空间。监督学习的任务就是在假设空间中根据特定的误差准则找到最优的模型。
  - 生成方法是根据输入数据和输出数据之间的联合概率分布确定条件概率分布 P(Y∣X)，这种方法表示了输入 X 与输出 Y 之间的生成关系；
  - 判别方法则直接学习条件概率分布 P(Y∣X) 或决策函数 f(X)，这种方法表示了根据输入 X 得出输出 Y 的预测方法。
- 无监督学习：基于未知类别的训练数据进行学习
- 半监督学习：同时使用已知类别和未知类别的训练数据进行学习
- 强化学习：一系列决策，前面决策的结果会影响后面的决策

### 流程

- 收集数据
- 数据准备：涉及到数据清洗等工作。当数据本身没有什么问题后，我们将数据分成3个部分：训练集（60%）、验证集（20%）、测试集（20%），用于后面的验证和评估工作。
- 选择一个模型
- 训练
- 评估：一旦训练完成，就可以评估模型是否有用。这是我们之前预留的验证集和测试集发挥作用的地方。评估的指标主要有 准确率、召回率、F值，这个过程可以让我们看到模型如何对尚未看到的数是如何做预测的。这意味着代表模型在现实世界中的表现。
- 参数调整：完成评估后，希望了解是否可以以任何方式进一步改进训练。我们可以通过调整参数来做到这一点。当我们进行训练时，我们隐含地假设了一些参数，我们可以通过认为的调整这些参数让模型表现的更出色。
- 预测：开始使用


### 训练集、验证集、测试集

- 训练集：相当于上课学知识

- 验证集：相当于课后的的练习题，用来纠正和强化学到的知识

- 测试集：相当于期末考试，用来最终评估学习效果




## 数学基础
- [数学基础](10_math/README.md)


## 机器学习基础
- [机器学习基础](40_shallow-learning/README.md)

## 浅层学习

- [浅层学习](40_shallow-learning/README.md)

## 深度学习

- [深度学习](60_deep-learning/README.md)

