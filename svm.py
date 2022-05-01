# svm.py
import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv
import statsmodels.api as sm  # for finding the p-value
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import os

# >> FEATURE SELECTION << #
def remove_correlated_features(X):
    corr_threshold = 0.9
    # corr_threshold = 0.8
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped


def remove_less_significant_features(X, Y):
    sl = 0.05
    # sl = 0.1
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped


# reg_strength = 10000 # C: regularization strength
# learning_rate = 0.000001
reg_strength = 1000000  # 10000 # C: regularization strength
learning_rate = 0.000001


# >> MODEL TRAINING << #
def compute_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost


def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    # print(type(Y_batch))
    if type(Y_batch) == np.int64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    # print(distance)
    dw = np.zeros(len(W))
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (reg_strength * Y_batch[ind] * X_batch[ind])
        dw += di
    dw = dw/len(Y_batch)  # average
    return dw


def sgd(features, outputs):
    max_epochs = 5000
    # max_epochs = 2000
    weights = np.zeros(features.shape[1])
    # Need to define learning rate here first:
    # learning_rate = 0.000001
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)

    return weights


def init():
    data = pd.read_csv('./train.csv')
    # drop unnecessary first column (id)
    data.drop(data.columns[[0]], axis=1, inplace=True)
    # SVM only accepts numerical values.
    # Therefore, we will transform the categories above mean and below mean into
    # values 1 and -1 (or -1 and 1), respectively. (pawpularity mean = 38.04)
    # print(data['Pawpularity'].mean())
    category = pd.cut(data.Pawpularity,bins=[0,38,101],labels=[-1,1])
    data.insert(12,'Pawpularity Group', category)
    # Now, we drop the original Pawpularity score after adding classified scores:
    data.drop(data.columns[[-1]], axis=1, inplace=True)
    # print(data.head(10))

    # Put features and output in different DataFrames:
    Y = data.loc[:, 'Pawpularity Group']
    X = data.iloc[:,:-1]
    # filter features
    # print(remove_correlated_features(X))
    # print(remove_less_significant_features(X, Y))
    print(X.head(10))
    print(Y.head(10))

    # first insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='Intercept', value=1)
    print(len(X))
    print(len(Y))
    # test_size is the portion of data that will go into test set
    # random_state is the seed used by the random number generator
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)
    # print(X_train.head(10))
    # print(y_train.head(10))

    # train the model
    print("training started...")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))

    # accuracy of training set
    y_train_predicted = np.array([])
    for i in range(X_train.shape[0]):
        yp = np.sign(np.dot(W, X_train.to_numpy()[i]))  # model
        y_train_predicted = np.append(y_train_predicted, yp)
    print("accuracy on training dataset: {}".format(accuracy_score(y_train.to_numpy(), y_train_predicted)))

    # testing the model on test set
    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.to_numpy()[i]))  # model
        y_test_predicted = np.append(y_test_predicted, yp)
    print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))



def hog_init():
    data = pd.read_csv('./train.csv')
    # drop unnecessary first column (id)
    data.drop(data.columns[[0]], axis=1, inplace=True)
    # SVM only accepts numerical values.
    # Therefore, we will transform the categories above mean and below mean into
    # values 1 and -1 (or -1 and 1), respectively. (pawpularity mean = 38.04)
    # print(data['Pawpularity'].mean())
    category = pd.cut(data.Pawpularity,bins=[0,38,101],labels=[-1,1])
    data.insert(12,'Pawpularity Group', category)
    # Now, we drop the original Pawpularity score after adding classified scores:
    data.drop(data.columns[[-1]], axis=1, inplace=True)
    # print(data.head(10))

    # Put features and output in different DataFrames:
    Y = data.loc[:, 'Pawpularity Group']
    N = 9912  # number of images
    X = np.empty((N, 3781))

    # first insert 1 in every row for intercept term b
    for i in range(0, N):
        X[i][3780] = 1
    # print(len(X[0]))
    # print(X[0])
    # print(X[9911])
    # print(Y.head(10))
    # creating hog features
    counter = 0
    print("extracting hog features...")
    for filename in os.listdir('train'):
        filename = filename[-36:]
        filename = 'train/' + filename
        # print(filename)
        img = imread(filename)
        resized_img = resize(img, (128, 64))
        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, channel_axis=2)
        X[counter][0:3780] = fd
        counter = counter + 1
        # print(fd)
        # print(X)


    # test_size is the portion of data that will go into test set
    # random_state is the seed used by the random number generator
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(X, Y.to_numpy(), test_size=0.2, random_state=42)
    # print(X_train.head(10))
    # print(y_train.head(10))

    # train the model
    print("training started...")
    W = sgd(X_train, y_train)
    print("training finished.")
    print("weights are: {}".format(W))

    # accuracy of training set
    y_train_predicted = np.array([])
    for i in range(X_train.shape[0]):
        yp = np.sign(np.dot(W, X_train[i]))  # model
        y_train_predicted = np.append(y_train_predicted, yp)
    print("accuracy on training dataset: {}".format(accuracy_score(y_train, y_train_predicted)))

    # testing the model on test set
    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test[i]))  # model
        y_test_predicted = np.append(y_test_predicted, yp)
    print("accuracy on test dataset: {}".format(accuracy_score(y_test, y_test_predicted)))


if __name__ == '__main__':
    print(reg_strength)
    init()  # svm metadata implementation
    # hog_init()
