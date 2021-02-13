from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


if __name__ == "__main__":
    data = load_boston()

    trainDataSet, testDataSet, trainLabelSet, testLabelSet = train_test_split(
        data.data, data.target, test_size=0.25, random_state=33)

    # decision tree
    dtRegressor = DecisionTreeRegressor()
    dtRegressor.fit(trainDataSet, trainLabelSet)
    dtPredictTestLabelSet = dtRegressor.predict(testDataSet)
    dtMse = mean_squared_error(testLabelSet, dtPredictTestLabelSet)
    print("decision tree mean squared error is: ", round(dtMse, 2))


    # KNN
    knnRegressor = KNeighborsRegressor()
    knnRegressor.fit(trainDataSet, trainLabelSet)
    knnPredictTestLabelSet = knnRegressor.predict(testDataSet)
    knnMse = mean_squared_error(testLabelSet, knnPredictTestLabelSet)
    print("KNN mean squared error is: ", round(knnMse, 2))


    # adaboost
    adaRegressor = AdaBoostRegressor()
    adaRegressor.fit(trainDataSet, trainLabelSet)
    predictTestLabelSet = adaRegressor.predict(testDataSet)
    meanSquaredError = mean_squared_error(testLabelSet, predictTestLabelSet)
    # print("the predict test label set is: ", predictTestLabelSet)
    print("adaBoost mean squared error is: ", round(meanSquaredError, 2))
