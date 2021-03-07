# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 读取数据
trainDataSet = pd.read_csv('data/kc_train.csv')
trainLabelSet = pd.read_csv('data/kc_train2.csv')  # 销售价格
testDataSet = pd.read_csv('data/kc_test.csv')      # 测试数据

# 数据预处理
trainDataSet.info()    # 查看是否有缺失值

# 特征缩放
minMaxScaler = MinMaxScaler()
minMaxScaler.fit(trainDataSet)   # 进行内部拟合，内部参数会发生变化
scaledTrainDataSet = minMaxScaler.transform(trainDataSet)
scaledTrainDataSet = pd.DataFrame(scaledTrainDataSet, columns=trainDataSet.columns)
minMaxScaler2 = MinMaxScaler()
minMaxScaler2.fit(testDataSet)
scaledTestDataSet = minMaxScaler2.transform(testDataSet)
scaledTestDataSet = pd.DataFrame(scaledTestDataSet, columns=testDataSet.columns)

# 选择基于梯度下降的线性回归模型
linearRegressor = LinearRegression()
linearRegressor.fit(scaledTrainDataSet, trainLabelSet)

# 使用均方误差用于评价模型好坏
predictTrainLabelSet = linearRegressor.predict(scaledTrainDataSet)   # 输入数据进行预测得到结果
mse = mean_squared_error(predictTrainLabelSet, trainLabelSet)   # 使用均方误差来评价模型好坏，可以输出mse进行查看评价值

# 绘图进行比较
plot.figure(figsize=(10, 7))  # 画布大小
num = 100
x = np.arange(1, num+1)  # 取100个点进行比较
plot.plot(x, trainLabelSet[:num], label='target')  # 目标取值
plot.plot(x, predictTrainLabelSet[:num], label='predict')  # 预测取值
plot.legend(loc='upper right')  # 线条显示位置
plot.show()

# 输出测试数据
predictTestLabelSet = linearRegressor.predict(scaledTestDataSet)
formedPredictTestLabelSet = pd.DataFrame(predictTestLabelSet)
formedPredictTestLabelSet.to_csv("result.csv")
