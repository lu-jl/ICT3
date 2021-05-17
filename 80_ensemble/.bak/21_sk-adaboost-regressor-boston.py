from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor


if __name__ == "__main__":
    data = load_boston()

    trainDataSet, testDataSet, trainLabelSet, testLabelSet = train_test_split(
        data.data, data.target, test_size=0.25, random_state=33)

    adaRegressor = AdaBoostRegressor()
    adaRegressor.fit(trainDataSet, trainLabelSet)
    predictTestLabelSet = adaRegressor.predict(testDataSet)
    meanSquaredError = mean_squared_error(testLabelSet, predictTestLabelSet)
    print("the predict test label set is: ", predictTestLabelSet)
    print("mean squared error is: ", round(meanSquaredError, 2))
