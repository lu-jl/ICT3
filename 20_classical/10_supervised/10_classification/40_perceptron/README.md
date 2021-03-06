# 感知机

## 简介

感知机（Perceptron）是一种比较简单的二分类模型，它就是将线性回归的思想用在分类中，它将输入特征分类为 +1（good）、-1（bad）两类。例如，二维平面上的点只有两个输入特征（横轴坐标和纵轴坐标），一条直线就可以分类。如果输入数据有更多维度的特征，那么就需要建立同样多维度的模型，高维度上的分类模型也被称为超平面。

感知机的思想很简单，比如在一个平台上有很多的男孩、女孩，感知机的模型就是尝试找到一条直线，能够把所有的男孩和女孩隔离开。放到三维空间或者更高维的空间，感知机的模型就是尝试找到一个超平面，能够把所有的二元类别隔离开。当然，如果找不到这么一条直线的话怎么办？找不到的话那就意味着类别线性不可分，也就意味着感知机模型不适合该类数据的分类。使用感知机一个最大的前提，就是数据是线性可分的，这严重限制了感知机的使用场景。

### 优缺点

- 缺点：感知机是单个的，只能解决线性可分问题。当需要解决非线性可分问题，需要考虑使用多层神经网络，也就是多层感知机。

## 算法

### 数据集

- $X$：
- $Y$：$\{+1(good), -1(bad)\}$
- 前提：所有的样本线性可分。

### 表示：假设

$h(x)=sign((\sum^{d}_{i=1}w_ix_i)-threshold)$

以上 h 也可以被表示为 3 层感知机模型：

- 输入处理层：接收外部信号后做线性叠加（线性回归）
- 激活函数层：阶跃函数 $sign(x)$ <img src="figures/image-20200220134302690.png" alt="image-20200220134302690" style="zoom:25%;" />。不同于逻辑回归，感知机采用不连续函数作为激活函数。
- 阈值判断层：$=1$ 为类 1，$=-1$ 为类 0。 

感知机具体数学模型如下：$f(x)=sign(wx+b)=sign(\theta X)$，其中 x 代表输入的特征空间向量，输出空间是{-1, +1}，w 为权值向量，b 叫作偏置，$\theta$ 是 w 与 b 的组合，**因此 $\theta$ 也就代码了 H 空间中的 h**。

H 的直观解释就是，就是当感知机输出为 +1 表示输入值在超平面的上方，当感知机输出为 -1 表示输入值在超平面的下方。训练感知机模型就是要计算出 $\theta$（w 和 b 的值），当有新的数据需要分类的时候，输入感知机模型就可以计算出 +1 或者 -1 从而进行分类。

<img src="figures/image-20210221095157678.png" alt="image-20210221095157678" style="zoom: 25%;" />

### 评估

#### 损失函数：0/1 Error

感知机的损失函数是 0/1 Error：$err(\hat y,y)=P[\hat y \neq y]$，也就是针对每个样本或则分类正确，或则分类错误。

#### 代价函数：偏差之和

偏差之和就是感知机的代价函数（大部分情况下也叫损失函数）：$L(w,b)=-\sum_{{x_i}\in M}y_i(wx_i+b)$。其中 M 为误分类点集合，误分类点越少，损失函数的值越小。如果没有误分类点，损失函数值为 0。

为了后面便于定义损失函数，将满足 $𝜃𝑥>0$ 的样本类别输出值取为 1，满足 $𝜃𝑥<0$ 的样本类别输出值取为 -1，这样取 y 的值有一个好处，就是方便定义损失函数。因为正确分类的样本满足 $𝑦𝜃𝑥>0$，而错误分类的样本满足 $𝑦𝜃𝑥<0$。损失函数的优化目标，就是期望使误分类的所有样本，到超平面的距离之和最小。

由于 $𝑦𝜃𝑥<0$，所以对于每一个误分类的样本 𝑖，它到超平面的距离是：$−𝑦(𝑖)𝜃𝑥(𝑖)/||𝜃||_2$。其中 $||𝜃||_2$ 为 L2 范数。假设所有误分类的点的集合为 M，则所有误分类的样本到超平面的距离之和为：$-\sum_{𝑥_𝑖∈𝑀}𝑦(𝑖)𝜃𝑥(𝑖)/||𝜃||_2$，这样就得到了初步的感知机模型的损失函数。

可以发现，分子和分母都含有 𝜃，当分子的 𝜃 扩大 N 倍时，分母的 L2 范数也会扩大 N 倍。也就是说，分子和分母有固定的倍数关系。那么可以固定分子或分母为 1，然后求另一个即分子自己或者分母的倒数的最小化作为损失函数，这样可以简化损失函数。在感知机模型中，采用的是保留分子，即最终感知机模型的损失函数简化为：$𝐽(𝜃)=−\sum_{𝑥_𝑖∈𝑀}𝑦(𝑖)𝜃𝑥(𝑖)$。

### 优化：学习算法

#### PLA

PLA（Perceptron Lineair Algorithm）：先从一个 $\theta_0$ 开始，每次修正，使获得最优函数 g。使用梯度下降更新 $\theta$（w 和 b），不断迭代使损失函数 $J(\theta)$ 不断减小，直到为 0，也就是没有误分类点。感知机算法的实现过程：

1. 选择初始值 $\theta_0$。
2. 在样本集合中选择样本数据 $(x_i,y_i)$。
3. 如果 $y_i𝜃𝑥_i<0$，表示 $y_i$ 为误分类点，那么 $\theta=\theta+\alpha y_ix_i$，在梯度方向校正 $\theta$。其中 $\alpha$ 为步长。步长选择要适当，步长太长会导致每次计算调整太大出现震荡，步长太短又会导致收敛速度慢、计算时间长。
4. 跳转回 2，直到样本集合中没有误分类点， 即全部样本数据 $y_i𝜃𝑥_i≥0$。

##### 梯度下降法

- 梯度：$-y_ix_i$
- 步长：𝛼
- 采样：随机梯度下降

损失函数求解可以用梯度下降法来。但是用普通的基于所有样本的批量梯度下降法（BGD）是行不通的，原因在于损失函数里面有限定，只有误分类的 M 集合里面的样本才能参与损失函数的优化。所以不能用最普通的批量梯度下降，只能采用随机梯度下降（SGD），每次仅仅需要使用一个误分类的点来更新梯度。

损失函数基于 𝜃，向量的的偏导数为：$\frac{∂}{∂𝜃}𝐽(𝜃)=−\sum_{𝑥_𝑖∈𝑀}𝑦(𝑖)𝑥(𝑖)$。

𝜃 的梯度下降迭代公式应该为：$𝜃=𝜃+𝛼\sum_{𝑥_𝑖∈𝑀}𝑦(𝑖)𝑥(𝑖)$。由于采用随机梯度下降，所以每次仅仅采用一个误分类的样本来计算梯度，假设采用第 i 个样本来更新梯度，则简化后的 𝜃 向量的梯度下降迭代公式为：$𝜃=𝜃+𝛼𝑦(𝑖)𝑥(𝑖)$。其中 𝛼 为步长，𝑦(𝑖) 为样本输出 1 或 -1，𝑥(𝑖) 为 (n+1)x1 的向量。 

##### 直观解释

PLA 的基本原理就是**逐点修正**，首先在超平面上随意取一条分类面，统计分类错误的点；然后随机对某个错误点就行修正，即变换直线的位置，使该错误点得以修正；接着再随机选择一个错误点进行纠正，分类面不断变化，直到所有的点都完全分类正确了，就得到了最佳的分类面。

利用二维平面例子来进行解释，第一种情况是错误地将正样本（y=1）分类为负样本（y=-1）。此时，$𝜃x<0$，即 𝜃 与 x 的夹角大于 90 度，分类线 ll 的两侧。修正的方法是让夹角变小，修正 w 值，使二者位于直线同侧：$𝜃=𝜃+yx$。修正过程示意图如下所示：

![这里写图片描述](figures/20180529091355719.jpeg)

第二种情况是错误地将负样本（y=-1）分类为正样本（y=1）。此时，$𝜃x>0$，即 𝜃 与 x 的夹角小于 90 度，分类线 ll 的同一侧。修正的方法是让夹角变大，修正 w 值，使二者位于直线两侧：$𝜃=𝜃+yx$，修正过程示意图如下所示：

![这里写图片描述](figures/20180529091445736.jpeg)

经过两种情况分析，我们发现 PLA 每次 𝜃 的更新表达式都是一样的：$𝜃=𝜃+yx$。掌握了每次 𝜃 的优化表达式，那么 PLA 就能不断地将所有错误的分类样本纠正并分类正确。

#### Pocket PLA

数据集：无线性可分假设：对于要分类的点，找出误分类点最少的线，无线性可分假设。

评估：Pocket PLA 与 PLA 最大的区别在于损失函数，Pocket 只看最少犯错，而不是完全不犯错。

原理：遍历所有样本，将最好的超平面放在口袋里。当数据不是线性可分时，可以选择犯错最少的那个线作为最优函数 g。其公式可以表达为：$g \Leftarrow argmin\sum^{N}_{n=1}[y_n\neq \theta X_n]$。该公式已被证明是个 NP-hard 问题，因此只能采用以下迭代方式。

Pocket PLA 算法基于贪心的思想，它的原理是让遇到的最好的线放在 Pocket 里。 

- 首先口袋里有一条分割线，发现它在数据点上面犯了错误，那就纠正这个分割线。
- 比较口袋中的线与纠正后的线，选取犯错做少的那根放口袋里。
- 遍历所有的数据，最后留在口袋中的就是最佳函数 g。
- 何时停止：规定迭代的次数。由于 Pocket 算法得到的线越来越好（PLA 就不一定了，PLA 是最终结果最好，其他情况就说不准了），所以就自己规定迭代的次数 。 

## Lab

