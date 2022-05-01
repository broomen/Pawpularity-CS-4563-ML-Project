import numpy as np
import pandas as pd  # for reading data from csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# takes a numpy array as input and returns activated array
def relu(z):
    a = np.maximum(0,z)
    return a


# takes a numpy array as input and returns activated array
def sigmoid(z):
    return 1 / (1+np.exp(-z))


# takes a list of the layer sizes as input and returns initialized parameters
def initialize_params(layer_sizes):
    params = {}
    for i in range(1, len(layer_sizes)):
        params['W' + str(i)] = np.random.randn(layer_sizes[i], layer_sizes[i-1])*0.01
        params['B' + str(i)] = np.random.randn(layer_sizes[i], 1)*0.01
    return params


# takes input training features and parameters as input and returns a dictionary containining the numpy arrays
# of activations of all layers
def forward_propagation(X_train, params):
    layers = len(params)//2
    values = {}
    for i in range(1, layers+1):
        if i==1:
            values['Z' + str(i)] = np.dot(params['W' + str(i)], X_train) + params['B' + str(i)]
            values['A' + str(i)] = relu(values['Z' + str(i)])
        else:
            values['Z' + str(i)] = np.dot(params['W' + str(i)], values['A' + str(i-1)]) + params['B' + str(i)]
            if i==layers:
                values['A' + str(i)] = values['Z' + str(i)]
            else:
                values['A' + str(i)] = relu(values['Z' + str(i)])
    return values


# takes true values and dictionary having activations of all layers as input and returns cost
def compute_cost(values, Y_train):
    layers = len(values)//2
    Y_pred = values['A' + str(layers)]
    # print(len(Y_pred[1]))
    # print(len(Y_train))
    cost = 1/(2*len(Y_train)) * np.sum(np.square(Y_pred - Y_train))
    return cost


# takes parameters, activations, training set as input and returns gradients wrt parameters
def backward_propagation(params, values, X_train, Y_train):
    layers = len(params)//2
    # print(layers)
    # print(len(values['A3'][0]))
    m = len(Y_train)
    # print(m)
    grads = {}
    for i in range(layers,0,-1):
        if i==layers:
            dA = 1/m * (values['A' + str(i)] - Y_train)
            dZ = dA
        else:
            dA = np.dot(params['W' + str(i+1)].T, dZ)
            dZ = np.multiply(dA, np.where(values['A' + str(i)]>=0, 1, 0))
        if i==1:
            grads['W' + str(i)] = 1/m * np.dot(dZ, X_train.T)
            grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        else:
            grads['W' + str(i)] = 1/m * np.dot(dZ,values['A' + str(i-1)].T)
            grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    return grads


# takes parameters, gradients and learning rate as input and returns updated parameters
def update_params(params, grads, learning_rate):
    layers = len(params)//2
    params_updated = {}
    for i in range(1, layers+1):
        params_updated['W' + str(i)] = params['W' + str(i)] - learning_rate * grads['W' + str(i)]
        params_updated['W' + str(i)] = params['W' + str(i)] - learning_rate * grads['W' + str(i)]
        params_updated['B' + str(i)] = params['B' + str(i)] - learning_rate * grads['B' + str(i)]
    return params_updated


# trains the model
def model(X_train, Y_train, layer_sizes, num_iters, learning_rate):
    params = initialize_params(layer_sizes)
    # print(len(params))
    # print(params)
    for i in range(num_iters):
        values = forward_propagation(X_train.T, params)
        cost = compute_cost(values, Y_train.T)
        grads = backward_propagation(params, values, X_train.T, Y_train.T)
        params = update_params(params, grads, learning_rate)
        print('Cost at iteration ' + str(i+1) + ' = ' + str(cost) + '\n')
    return params


# compute accuracy on test and training data given learnt parameters
def compute_accuracy(X_train, X_test, Y_train, Y_test, params):
    values_train = forward_propagation(X_train.T, params)
    values_test = forward_propagation(X_test.T, params)
    train_acc = np.sqrt(mean_squared_error(Y_train, values_train['A' + str(len(layer_sizes)-1)].T))
    test_acc = np.sqrt(mean_squared_error(Y_test, values_test['A' + str(len(layer_sizes)-1)].T))
    return train_acc, test_acc


# predict on new array X given learnt parameters
def predict(X, params):
    values = forward_propagation(X.T, params)
    predictions = values['A' + str(len(values)//2)].T
    return predictions


data = pd.read_csv('./train.csv')                                                   # load dataset
data.drop(data.columns[[0]], axis=1, inplace=True)                                  # drop unnecessary first column (id)
# attempt dropping other columns:
X = data
Y = data['Pawpularity']                                                             # separate data into input and output features
X.drop(X.columns[[-1]], axis=1, inplace=True)                                       # drop target variable from feature vector X
# # Drop statistically insignificant features from feature matrix X:
# for i in range(0, 7):
#     X.drop(X.columns[[-2]], axis=1, inplace=True)
# X.drop(X.columns[[0]], axis=1, inplace=True)
# X.drop(X.columns[[1]], axis=1, inplace=True)
# # Add value to boolean features of higher statistical relevance:
# X['Eyes'] = X['Eyes'].replace(1, 3)
# X['Blur'] = X['Blur'].replace(1, 5)
print(X.head(10))
print(Y.head(10))
# split data into train and test sets in 80-20 ratio
X_train, X_test, Y_train, Y_test = train_test_split(pd.DataFrame(X).to_numpy(), pd.DataFrame(Y).to_numpy(), test_size=0.2)
# print(Y_test)
layer_sizes = [3, 5, 5, 1]                                                          # set layer sizes, do not change the size of the last layer
num_iters = 4                                                                       # set number of iterations over the training set
learning_rate = 0.0001                                                              # set learning rate for gradient descent
params = model(X_train, Y_train, layer_sizes, num_iters, learning_rate)             # train the model
train_acc, test_acc = compute_accuracy(X_train, X_test, Y_train, Y_test, params)    # get training and test accuracy
print('Root Mean Squared Error on Training Data = ' + str(train_acc))
print('Root Mean Squared Error on Test Data = ' + str(test_acc))
