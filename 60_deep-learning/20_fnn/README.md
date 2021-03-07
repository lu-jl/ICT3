# FNN

## 简介

FNN 前馈神经网络（Feedforward Neural Network）是一种最简单的神经网络，各神经元分层排列。每个神经元只与前一层的神经元相连，接收前一层的输出，并输出给下一层。由于从输入到输出的过程中不存在与模型自身的反馈连接，因此被称为“前馈”。

- 多层神经元网络：单层感知器只能学习线性可分离的模式，而多层感知器则可以学习数据之间的非线性的关系
- 每一层是全连接的：层中的每个神经元都与下一层中的所有其他神经元相连

<img src="figures/image-20201117192325475.png" alt="image-20201117192325475" style="zoom: 33%;" />



##### Multi-classifier

在最后的 output layer 实现 multi-classifier

##### 类型

- 多层感知机（multilayer perceptron，MLP）
- 自编码器
- 限制玻尔兹曼机