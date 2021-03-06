# encoding:utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


if __name__ == '__main__':
    # 原始数据
    X = [[13, 2, 3, 4], [11, 2, 3, 4], [63, 2, 3, 4], [71, 2, 3, 4], [22, 2, 3, 4], [12, 2, 3, 4], [11, 2, 3, 4], [12, 2, 3, 4],
         [22, 2, 3, 4], [41, 2, 3, 4], [52, 2, 3, 4], [81, 2, 3, 4], [41, 2, 3, 4], [12, 2, 3, 4], [12, 2, 3, 4], [11, 2, 3, 4],
         [11, 2, 3, 4], [11, 2, 3, 4], [24, 2, 3, 4], [12, 2, 3, 4], [41, 2, 3, 4], [33, 2, 3, 4], [13, 2, 3, 4], [21, 2, 3, 4],
         [13, 2, 3, 4], [23, 2, 3, 4], [21, 2, 3, 4], [13, 2, 3, 4], [14, 2, 3, 4], [13, 2, 3, 4], [11, 2, 3, 4], [31, 2, 3, 4],
         [51, 2, 3, 4], [61, 2, 3, 4], [71, 2, 3, 4], [41, 2, 3, 4], [71, 2, 3, 4], [81, 2, 3, 4], [61, 2, 3, 4], [41, 2, 3, 4],
         [11, 2, 3, 5], [61, 2, 3, 5], [10, 2, 3, 5], [22, 2, 3, 5], [21, 2, 3, 5], [27, 2, 3, 5], [32, 2, 3, 5], [42, 2, 3, 5],
         [21, 2, 3, 5], [71, 2, 3, 5], [12, 2, 3, 5], [19, 2, 3, 5], [26, 2, 3, 5], [28, 2, 3, 5], [33, 2, 3, 5], [41, 2, 3, 5],
         [31, 2, 3, 5], [81, 2, 3, 5], [13, 2, 3, 5], [18, 2, 3, 5], [23, 2, 3, 5], [29, 2, 3, 5], [34, 2, 3, 5], [39, 2, 3, 5],
         [41, 2, 3, 5], [81, 2, 3, 5], [14, 2, 3, 5], [17, 2, 3, 5], [24, 2, 3, 5], [30, 2, 3, 5], [35, 2, 3, 5], [38, 2, 3, 5],
         [51, 2, 3, 5], [91, 2, 3, 5], [15, 2, 3, 5], [16, 2, 3, 5], [25, 2, 3, 5], [31, 2, 3, 5], [36, 2, 3, 5], [37, 2, 3, 5],
         [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6],
         [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6],
         [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6],
         [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6],
         [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6]]

    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    #
    # n_estimators = 10,               弱学习器的最大迭代次数，太小，容易欠拟合，太大，又容易过拟合
    # criterion = "gini",              衡量分裂质量的性能函数，默认是基尼不纯度，熵达到峰值的过程要相对慢一些。
    # max_depth = None,                ★ 决策树最大深度，如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，
    #                                  具体的取值取决于数据的分布。常用的可以取值10-100之间
    # min_samples_split = 2,           ★ 内部节点再划分所需最小样本数。如果某节点的样本数少于min_samples_split，
    #                                  则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。
    #                                  如果样本量数量级非常大，则推荐增大这个值。
    # min_samples_leaf = 1,            ★ 叶子节点最少样本数。如果样本量数量级非常大，则推荐增大这个值。
    # min_weight_fraction_leaf = 0.,   叶子节点最小的样本权重和。
    #                                  默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。
    # max_features = "auto",           ★ 最大特征数，
    # max_leaf_nodes = None,           最大叶子节点数。限制最大叶子节点数，可以防止过拟合。
    # min_impurity_decrease = 0.,      如果节点的分裂导致的不纯度的下降程度大于或者等于这个节点的值，那么这个节点将会被分裂
    # min_impurity_split = None,       节点划分最小不纯度。
    # bootstrap = True,                建立决策树时，是否使用有放回抽样
    # oob_score = False,               建议用True，袋外分数反应了一个模型拟合后的泛化能力。
    # n_jobs = 1,                      用于拟合和预测的并行运行的工作（作业）数量。如果值为-1，那么工作数量被设置为核的数量
    # random_state = None,             是随机数生成器使用的种子; 如果是RandomState实例，random_state就是随机数生成器;
    #                                  如果为None，则随机数生成器是np.random使用的RandomState实例
    # verbose = 0,                     控制决策树建立过程的冗余度。
    # warm_start = False,              当被设置为True时，重新使用之前呼叫的解决方案，用来给全体拟合和添加更多的估计器，
    #                                  反之，仅仅只是为了拟合一个全新的森林。
    # class_weight = None

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    # clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
    clf.fit(X, y)
    print(clf.predict([[33331, 2, 3, 4]]))
    print(clf.predict([[3222, 2, 3, 5]]))
    print(clf.predict([[3121, 2, 3, 6]]))
    print(clf.predict([[331, 2, 3, 7.5]]))
