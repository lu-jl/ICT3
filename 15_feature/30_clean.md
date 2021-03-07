# 数据清洗

好的数据分析师必定是一名数据清洗高手，要知道在整个数据分析过程中，不论是在时间还是功夫上，数据清洗大概都占到了 80%。

质量准则：

- 完整性：单条数据是否存在空值，统计的字段是否完善。
- 全面性：观察某一列的全部数值，比如在 Excel  表中，我们选中一列，可以看到该列的平均值、最大值、最小值。我们可以通过常识来判断该列是否有问题，比如：数据定义、单位标识、数值本身。
- 合法性：数据的类型、内容、大小的合法性。比如数据中存在非 ASCII 字符，性别存在了未知，年龄超过了 150  岁等。
- 唯一性：数据是否存在重复记录，因为数据通常来自不同渠道的汇总，重复的情况是常见的。行数据、列数据都需要是唯一的，比如一个人不能重复记录多次，且一个人的体重也不能在列指标中重复记录多次。

按照以上的原则，我们能解决数据清理中遇到的大部分问题，使得数据标准、干净、连续，为后续数据统计、数据挖掘做好准备。

### 完整性

#### 缺失值

在数据中有些年龄、体重数值是缺失的，这往往是因为数据量较大，在过程中，有些数值没有采集到。通常我们可以采用以下三种方法：

- 删除：删除数据缺失的记录；
- 均值：使用当前列的均值；
- 高频：使用当前列出现频率最高的数据。

比如我们想对 df[‘Age’]中缺失的数值用平均年龄进行填充，可以这样写：

```python
df['Age'].fillna(df['Age'].mean(), inplace=True)
```

#### 空行

我们发现数据中有一个空行，除了 index 之外，全部的值都是 NaN。Pandas 的 read_csv() 并没有可选参数来忽略空行，这样，我们就需要在数据被读入之后再使用 dropna() 进行处理，删除空行。

```python
df.dropna(how='all',inplace=True) 
```

### 全面性

#### 列数据单位不统一

```python
# 获取 weight 数据列中单位为 lbs 的数据
rows_with_lbs = df['weight'].str.contains('lbs').fillna(False)
print df[rows_with_lbs]
# 将 lbs转换为 kgs, 2.2lbs=1kgs
for i,lbs_row in df[rows_with_lbs].iterrows():
# 截取从头开始到倒数第三个字符之前，即去掉lbs。
weight = int(float(lbs_row['weight'][:-3])/2.2)
df.at[i,'weight'] = '{}kgs'.format(weight) 
```

### 合理性

#### 非 ASCII 字符

可以采用删除或者替换的方式来解决非 ASCII 问题，这里我们使用删除方法：

```python
df['first_name'].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df['last_name'].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
```

### 唯一性

#### 一列有多个参数

在数据中不难发现，姓名列（Name）包含了两个参数 Firstname 和 Lastname。为了达到数据整洁目的，我们将 Name 列拆分成 Firstname 和 Lastname  两个字段。我们使用 Python 的 split 方法，str.split(expand=True)，将列表拆成新的列，再将原来的 Name  列删除。

```python
df[['first_name','last_name']] = df['name'].str.split(expand=True)
df.drop('name', axis=1, inplace=True)
```

#### 重复数据

校验一下数据中是否存在重复记录。如果存在重复记录，就使用 Pandas 提供的 drop_duplicates() 来删除重复数据。

```python
# 删除重复数据行
df.drop_duplicates(['first_name','last_name'],inplace=True)
```

没有高质量的数据，就没有高质量的数据挖掘，而数据清洗是高质量数据的一道保障。当你从事这方面工作的时候，你会发现养成数据审核的习惯非常重要。而且越是优秀的数据挖掘人员，越会有“数据审核”的“职业病”。这就好比编辑非常在意文章中的错别字、语法一样。数据的规范性，就像是你的作品一样，通过清洗之后，会变得非常干净、标准。



- 异常值
  - 异常值原因
    - 小概率事件(valid)
    - 数据问题(invalid)
  - 影响：模型权重，MSE计算偏重， boosting
  - 解决方法
    - Valid or not
    - 去掉异常值
    - 变换：log、开方、binning
    - 设置可取值范围
    - 选择更robust loss function，e.g. MAE
    - 对异常值不敏感的模型：tree-based model

- 缺失值
  - 缺失数据的原因：MCAR、MAR、NMAR
  - 解决方法：
    - 去掉不用：MAR或者数据足够多
    - Imputation：
      - Ad-hoc：均值，众数, 0， last obs carry forward
      - 用模型预测：kNN, multiple imputation
    - 加入新特征：是否缺失
    - 使用对缺失不敏感的模型：e.g. tree-based model