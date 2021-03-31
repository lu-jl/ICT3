# SVM

## 简介

对于要分类的点，找出分类间隔最大的那条钱，但假设前提是要分类的点线性可分。

SVM（Support Vector Machine）支持向量机用于超平面是分割输入特征空间。在SVM中，选择超平面以最佳地将输入变量空间中的点与它们的类（0级或1级）分开，所以主要是针对二分类。在二维中，可以将其视为一条线，并假设我们的所有输入点都可以被这条线完全分开。支持向量机（SVM）是一种监督学习算法，主要用于分类任务，但也适用于回归任务。

### 术语

- 支持向量（Support Vector）就是离分隔超平面最近的那些点。
- 机（Machine）就是表示一种算法，而不是表示机器。
- 分类间隔：在保证决策面不变，且分类不产生错误的情况下，我们可以移动决策面 C，直到产生两个极限的位置：如图中的决策面 A 和决策面 B。极限的位置是指，如果越过了这个位置，就会产生分类错误。这样的话，两个极限位置 A 和 B 之间的分界线 C 就是最优决策面。极限位置到最优决策面 C 之间的距离，就是“分类间隔”，英文叫做 margin。如果我们转动这个最优决策面，你会发现可能存在多个最优决策面，它们都能把数据集正确分开，这些最优决策面的分类间隔可能是不同的，而那个拥有“最大间隔”（max margin）的决策面就是 SVM 要找的最优解。

### 原理

SVM 通过绘制决策边界来区分类。在创建决策边界之前，将每个观察值（或数据点）绘制在 n 维空间中（n 是所使用特征的数量）。 例如，如果使用"长度"和"宽度"对不同的"单元格"进行分类，则观察结果将绘制在二维空间中，并且决策边界为一条线。如果使用 3 个特征，则决策边界是 3 维空间中的平面。 如果使用 3 个以上的特征，则决策边界将变成一个很难可视化的超平面。

SVM学习算法找到导致超平面最好地分离类的系数。决策边界以与支持向量的距离最大的方式绘制。 如果决策边界距离支持向量太近，它将对噪声高度敏感并且不能很好地泛化。 即使自变量的很小变化也可能导致分类错误。超平面与最近数据点之间的距离称为边距。可以将两个类分开的最佳或最佳超平面是具有最大边距的线，只有这些点与定义超平面和分类器的构造有关，这些点称为支持向量。它们支持或定义超平面。对于 SVM 来说，它是最大化两个类别边距的那种方式，换句话说：超平面（在本例中是一条线）对每个类别最近的元素距离最远。

<img src="../figures/image-20200321122333643.png" alt="image-20200321122333643" style="zoom:33%;" />

SVM 就是帮我们找到一个超平面，这个超平面能将不同的样本划分开，同时使得样本集中的点到这个分类超平面的最小距离（即分类间隔）最大化。在这个过程中，支持向量就是离分类超平面最近的样本点，实际上如果确定了支持向量也就确定了这个超平面。所以支持向量决定了分类间隔到底是多少，而在最大间隔以外的样本点，其实对分类都没有意义。

<img src="figures/image-20210208150041196.png" alt="image-20210208150041196" style="zoom: 25%;" />

## 算法

### 模型

假设超平面 $w^Tx+b=0$ 能够将训练样本正确分类，样本空间任一点 x 到超平面的距离为：$r=\frac{|w^Tx+b|}{|w|}$。

如果超平面能正确二分类，则对于 $(x_i,y_i)\in D$，若 $y_i=+1$ 则有 $w^Tx_i+b>0$；若 $y_i=-1$，则有 $w^Tx_i+b<0$。

令 $$\begin{cases}
w^Tx_i+b\geq+1,& y_i=+1\\
w^Tx_i+b\leq-1,& y_i=-1
\end{cases}$$

距离超平面最近的样本，也就是使得等号成立的样本，它们被称为“支持向量”，两个异类支持向量到超平面的距离之和被称为“间隔” $r=\frac{2}{|w|}$。

### 模型求解

SVM 支持向量机就是为了找到具有最大间隔的划分超平面，也就是找到w、b，使得间隔最大 $max_{w,b}\frac{2}{|w|}$，因此可以推出支持向量机的模型 $min_{w,b}\frac{1}{2}|w|^2$。

### 软间隔



## 核函数

### 原理

数据点并非总是线性可分离。在这些情况下，SVM 使用内核技巧来测量较高维空间中数据点的相似性（或接近度），以使它们线性可分离。它将样本从原始空间映射到一个更高维的特征空间中，使得样本在新的空间中线性可分。这样就可以使用原来的推导来进行计算，只是所有的推导是在新的空间，而不是在原来的空间中进行。如果原本的空间是有限维，那么就一定存在一个高维特征空间使样本可分。

<!--内核功能是一种相似性度量，输入是原始要素，输出是新要素空间中的相似性度量，这里的相似度表示紧密度，实际上将数据点转换为高维特征空间是一项昂贵的操作，该算法实际上并未将数据点转换为新的高维特征空间。内核化SVM无需实际进行变换就可以根据高维特征空间中的相似性度量来计算决策边界。在维数大于样本数的情况下，SVM特别有效。 找到决策边界时，SVM使用训练点的子集而不是所有点，从而提高了存储效率。 另一方面，大型数据集的训练时间会增加，这会对性能产生负面影响。-->

所以在非线性 SVM 中，核函数的选择就是影响 SVM 最大的变量。最常用的核函数有线性核、多项式核、高斯核、拉普拉斯核、sigmoid 核，或是这些核函数的组合。这些函数的区别在于映射方式的不同。通过这些核函数，就可以把样本空间投射到新的高维空间中。

### 模型

令 $\phi(x)$ 表示将 x 映射后的特征向量，于是新的超平面可以表示为：$f(x)=w^T\phi(x)+b$。

## 支持向量回归

支持向量回归 SVR（Support Vector Regression）假设能容忍 $f(x)$ 与 y 之间有最多 $\epsilon$ 的偏差，即仅当 $f(x)$ 与 y 之间的差别绝对值大于 $\epsilon$ 时才计算损失。
