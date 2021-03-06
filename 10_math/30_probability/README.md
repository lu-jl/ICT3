

# 概率论

## 简介

### 函数 vs. 概率分布

机器学习的任务是从属性 X 预测标记 Y，也就是 X 到 Y 的映射，函数和概率分布都可以用来表示 $X\rightarrow Y$的映射。

- 函数：代表从空间 X 到空间 Y 的映射关系是确定的，$y=f(x)$ 表示 x 对应的唯一 y 值。例如 $y=f(x)=2x$，则表示$(x,y)=(1,2)$ 势必会出现。
- 概率分布：代表从空间 X 到空间 Y 的映射关系是不确定的，也就是一个 x 值可以对应多个 y 值。概率 $P(y|x)$ 表示出现$(x,y)$ 的概率，表示的是映射关系的概率，常用正态分布来表示映射的概率。例如，在概率分布情况下对于 $(x,y)$，可能有20%会出现（1,1.8）、50%会出现 (1,2)，30%会出现 (1,2.2)。
  - 条件概率：对于离散的 X，一般采用概率来表示映射关系，此条件概率满足某种分布
  - 密度函数：对于连续的 X，一般采用密度函数来表示映射关系，此密度函数也满足某种分布

数据 X 和标签 Y 的映射关系可以用函数直观地表示，也可以用概率分布“笼统”地表示。函数是概率的一种特殊情况，即映射关系为 $y=f(x)$ 一一对应，其出现的概率为 100%。之后提到的“最大似然估计”法是在概率分布的前提下，表示当前观测到的训练集中 X、Y 的映射关系是训练集中的 X、Y 的概率最大。

### 频率（古典）学派 vs. 贝叶斯学派

对于样本分布 F(X,O)，其中 X 是样本、O 是概率分布。要对其中未知的概率分布 O 进行估计，频率学派与贝叶斯学派的区别在于：

- 频率学派：认为对于一批样本 X，其总体概率分布 O 的参数是客观存在的确定值，只不过未知，因此可以通过优化似然函数等准则来确定参数值。频率学派认为概率即是频率，某次得到的样本 X 只是无数次可能的试验结果的一个具体实现。样本中未出现的结果不是不可能出现，只是这次抽样没有出现而已。因此综合考虑已抽取到的样本 X 以及未被抽取的结果，可以认为总体概率分布是确定的，不过 O 未知。因为样本来自于总体，故其样本分布 F(X,O) 也同样的特点，因此可以根据 X 的分布去推断 O 的参数。
- 贝叶斯学派：否定了频率学派的观点，认为 O 的参数也是随机变量，本身也可有分布。因此假设 O 的参数服从一个先验分布，然后基于观测到的数据来计算参数的后验分布。贝叶斯学派反对把样本 X 放到“无限多可能值之一”背景下去考虑，既然只得到了样本 X，那么就只能依靠它去做推断，而不能考虑那些有可能出现而未出现的结果。与此同时，贝叶斯学派引入了主观概率的概念，认为一个事件在发生之前，人们应该对它是有所认知的，即 F(X,O) 中的 O 不是固定的，而是一个随机变量，并且服从分布 H(O )，该分布称为“先验分布”（指抽样之前得到的分布）。当得到样本X后，对 O 的分布则有了新的认识，此时 H(O) 有了更新，这样就得到了“后验分布”（指抽样之后得到的分布）。此时可以再对 O 做点估计、区间估计，此时的估计不再依赖样本，完全只依赖 O 的后验分布了。

在贝叶斯学派眼中，概率描述的是随机事件的可信程度。如果手机里的天气预报应用给出明天下雨的概率是 85%，这就不能从频率的角度来解释了，而是意味着明天下雨这个事件的可信度是 85%。频率学派认为假设是客观存在且不会改变的，即存在固定的先验分布，只是作为观察者的我们无从知晓。因而在计算具体事件的概率时，要先确定概率分布的类型和参数，以此为基础进行概率推演。相比之下，贝叶斯学派则认为固定的先验分布是不存在的，参数本身也是随机数。换言之，假设本身取决于观察结果，是不确定并且可以修正的。数据的作用就是对假设做出不断的修正，使观察者对概率的主观认识更加接近客观实际。


## 频率学派

概率是对随机事件发生的可能性进行规范的数学描述。

- 古典概率模型：假设所有基本事件的数目为 n，待观察的随机事件 A 中包含的基本事件数目为 k，则概率的计算公式为：P(A)=k/n。
- 联合概率（joint probability）：P(AB) 表示 A 和 B 两个事件共同发生的概率。如果联合概率等于两个事件各自概率的乘积，即 P(AB)=P(A)⋅P(B)，说明这两个事件的发生互不影响，即两者相互独立。
- 条件概率（conditional probability）：根据已有信息对样本空间进行调整后得到的新的概率分布。假定有两个随机事件 A 和 B，条件概率就是指在事件 B 已经发生的条件下事件 A 发生的概率：P(A∣B)=P(AB)/P(B)。P(A∣B) 可以看成 B-->A 的转换率，例如到车站 B 有40 人，其中有 10 人会再到车站 A，所以P(A∣B)=25%。对于相互独立的事件，条件概率就是自身的概率是 P(A∣B)=P(A)。
- 全概率公式（law of total probability）：将复杂事件的概率求解转化为在不同情况下发生的简单事件的概率求和，也就是有多条路径 $B_i$ 到达 A 的概率是每条路径到 A 的概率总和：$P(A)=\displaystyle \sum_{i=1}^NP(A|B_i)P(B_i), \displaystyle \sum_{i=1}^NP(B_i)=1$。 

### MLE 最大似然估计法

似然：等价于概率，表示某数据集 D 出现的概率。

最大似然估计法 MLE（Maximum Likelihood Estimation）：假设数据集 D 出现的概率为最大，从而估算出该整体数据空间的分布。

就是把训练数据的分布作为整体分布，其思想是使训练数据出现的概率最大化，依此确定概率分布中的未知参数，估计出的概率分布也就最符合训练数据的分布，所以最大似然估计法只需要使用训练数据。

##贝叶斯学派

贝叶斯（Bayesian）是一种基于条件概率的分类算法，其目的是**求后验概率**。如果已经知道 c 和 x 的发生概率，并且知道了 c 发生情况下 x 发生的概率（似然函数），则可以用贝叶斯公式计算 x 发生的情况下 c 发生的概率（后验概率）。事实上，可以根据数据集 x 的情况判断 c 的概率（即 c 的可能性），进而进行分类。

### 术语

- 假设分类 $c$：也就是原因。
- 观察结果 $x$：是观测事件样本，也就是结果。
- 证据因子 $p(x)$： 表示观测事件发生的概率。
- 先验概率 $p(c)$：在没有观测结果 x 的情况下，c 预先设定的假设成立的概率。
- 似然函数 $p(x|c)$：似然函数也就是似然概率，从“因到果”的条件函数/概率，即在已经知道分类 c 的情况下，观测结果 x 的分布情况。在 c 中观测到的假设成立的的比例，也就是在每个分类（假设）中观测到的概率。
- 后验概率 $p(c|x)$：在观测结果 x 的情况下假设 c 成立的概率，从“果到因”、“从观测到结果”的条件函数/概率，也就是从训练集找出真正的分类。

在机器学习时，所谓的“因”实际上是参数。因为机器学习的任务就是把参数当成“因”，把训练数据当成“果”，通过训练数据来学习参数。而参数并不是事件，不符合概率的严格定义，因此对于某一参数产生实际数据情况的可能性，只能称之为“似然”，也就是“像”的意思。

例如西瓜书中的“瓜熟蒂落”的例子，需要判断的假设分类是“是否瓜熟”（c-因），而给出的观察结果为“是否落地”（x-果）。贝叶斯中求后验概率也就是一种从“果”来判断“因”的过程。

### 公式

- 贝叶斯公式（逆概率）：在事件结果已经确定的条件下 P(A)，推断各种假设发生的可能性 $P(c∣x)$，也就是由 x 到 c 的概率：$P(c|x)=\frac{P(x|c)P(c)}{P(x)}$。贝叶斯公式等号右边的概率，可以通过对数据的统计获得，当有新的数据到来的时候，就可以带入上面的贝叶斯公式计算其概率。而如果设定概率超过某个值就认为其会发生，那么就对这个数据进行了分类和预测。但以上公式只对当前训练样本有效。
- 贝叶斯定理（Bayes' theorem）：也就是之前贝叶斯学派所说的观点，任何事物的概率都不是固定的，根据新的观测结果可以更新原先的概率，也就是在原先的概率中根据观测结果会有不成比例的限制区域。新的概率 P(c|x)（即观测到 x 的情况下 c 的新概率）为在原先 c 中观测/发生到 x 的概率除以所有情况下 x 的概率（即发生 c 和不发生 c 的情况下 x 的概率）。<img src="figures/image-20200305084202851.png" alt="image-20200305084202851" style="zoom:15%;" />

### 例子

在看不见路况的情况下预测是否到了十字路口

- P(c)：车子开到十字路口的概率
- P(x)：车子打右转灯的概率
- P(x|c)：车子到了十字路口并且打右转灯的概率
- P(c|x)：后面的车打右转灯而正巧在十字路口的概率

P(c)=5%、P(x)=2%、P(x|c)=25% --> P(c|x) = 62.5%

### 最大后验概率法

最大后验概率法（maximum a posteriori estimation）的思想则是根据训练数据和已知的其他条件，使未知参数出现的可能性最大化，并选取最可能的未知参数取值作为估计值。

最大后验概率法除了数据外还需要额外的信息，就是贝叶斯公式中的先验概率。

## 随机变量

### 离散型随机变量（discrete random variable）

- 质量函数（probability mass function）：离散变量的每个可能的取值都具有大于 0 的概率，取值和概率之间一一对应的关系就是离散型随机变量的分布律，也叫概率质量函数。
- 两点分布（Bernoulli  distribution）：适用于随机试验的结果是二进制的情形，事件发生 / 不发生的概率分别为  p/(1−p)。任何只有两个结果的随机试验都可以用两点分布描述，抛掷一次硬币的结果就可以视为等概率的两点分布。
- 二项分布（Binomial  distribution）：将满足参数为 p 的两点分布的随机试验独立重复 n 次，事件发生的次数即满足参数为 (n,p)  的二项分布。二项分布的表达式可以写成 P(X=k)=Ckn⋅pk⋅(1−p)(n−k),0≤k≤n。
- 泊松分布（Poisson  distribution）：放射性物质在规定时间内释放出的粒子数所满足的分布，参数为 λ 的泊松分布表达式为  P(X=k)=λk⋅e−λ/(k!)。当二项分布中的 n 很大且 p 很小时，其概率值可以由参数为 λ=np 的泊松分布的概率值近似。



### 连续型随机变量（continuous random variable）

- 密度函数（probability density function）：概率质量函数在连续型随机变量上的对应就是概率密度函数。概率密度函数体现的并非连续型随机变量的真实概率，而是不同取值可能性之间的相对关系，对概率密度函数进行积分，得到的才是连续型随机变量的取值落在某个区间内的概率。

- 均匀分布（uniform distribution）：在区间 (a, b) 上满足均匀分布的连续型随机变量，其概率密度函数为 1 / (b - a)，这个变量落在区间 (a, b) 内任意等长度的子区间内的可能性是相同的。

- 指数分布（exponential distribution）：满足参数为 θ 指数分布的随机变量只能取正值，其概率密度函数为 e−x/θ/θ,x>0。指数分布的一个重要特征是无记忆性：即 P(X > s + t | X > s) = P(X > t)。

- 正态分布（normal distribution）：参数为正态分布的概率密度函数为：<img src="figures/image-20200209092852269.png" alt="image-20200209092852269" style="zoom:33%;" />

  当 μ=0,σ=1 时，上式称为标准正态分布。正态分布是最常见最重要的一种分布，自然界中的很多现象都近似地服从正态分布。

### 特性的常数

- 数学期望（expected value）：体现的是随机变量可能取值的加权平均，即根据每个取值出现的概率描述作为一个整体的随机变量的规律。
- 方差（variance）：表示的则是随机变量的取值与其数学期望的偏离程度。方差较小意味着随机变量的取值集中在数学期望附近，方差较大则意味着随机变量的取值比较分散。
- 协方差（covariance）：度量了两个随机变量之间的线性相关性