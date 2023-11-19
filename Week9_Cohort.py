# WEEK 9 PROBLEM SET - COHORT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

# CS0. Plot: Read data for Boston Housing Prices and write a function get_features_targets() to get the columns 
# for the features and the targets from the input argument data frame. The function should take in Pandas' dataframe and two lists. 
# The first list is for the feature names and the other list is for the target names.

# We will use the following columns for our test cases:

# x data: RM column - use z normalization (standardization)
# y data: MEDV column
# Make sure you return a new data frame for both the features and the targets.

# We will normalize the feature using z normalization. Plot the data using scatter plot.

# z-normalization = (data - mean) / standard deviation
def normalize_z(dfin, columns_means=None, columns_stds=None):
    if columns_means == None:
        columns_means = dfin.mean(axis=0)
    if columns_stds == None:
        columns_stds = dfin.std(axis=0)
    print(columns_means, columns_stds)
    dfout = (dfin - columns_means) / columns_stds
    return dfout, columns_means, columns_stds

def get_features_targets(df, feature_names, target_names):
    df_feature = df[feature_names]
    df_target = df[target_names]
    return df_feature, df_target

df = pd.read_csv("housing_processed.csv")
df_feature, df_target = get_features_targets(df, ["RM"], ["MEDV"])
print(df_feature)
df_feature,_,_ = normalize_z(df_feature)
#print(df_feature)
print(df_feature.mean())

# test cases
df = pd.read_csv("housing_processed.csv")
df_feature, df_target = get_features_targets(df,["RM"],["MEDV"])
df_feature,_,_ = normalize_z(df_feature)

assert isinstance(df_feature, pd.DataFrame)
assert isinstance(df_target, pd.DataFrame)
assert np.isclose(df_feature.mean(), 0.0)
assert np.isclose(df_feature.std(), 1.0)
assert np.isclose(df_target.mean(), 22.532806)
assert np.isclose(df_target.std(), 9.1971)

sns.set()
plt.scatter(df_feature, df_target)

# CS1. Cost Function: Write a function compute_cost_linreg() to compute the cost function of a linear regression model. 
# The function should take in two 2-D numpy arrays. The first one is the matrix of the linear equation and the second one is the actual target value.

# Recall that:

# 𝐽(𝛽̂ 0,𝛽̂ 1)=12𝑚Σ𝑚𝑖=1(𝑦̂ (𝑥𝑖)−𝑦𝑖)2
 
# where

# 𝑦̂ (𝑥𝑖)=𝛽̂ 0+𝛽̂ 1𝑥𝑖
 
# The function should receive a numpy array, so we will need to convert to numpy array and change the shape. 
# To do this, we will create three other functions:

# calc_linreg(X, b): which is used to calculate the  𝑦̂ =𝑋𝑏 vector.
# prepare_feature(df): which takes in a data frame or two-dimensional numpy array for the feature. 
#                      If the input is a data frame, the function should convert the data frame to a numpy array and change it into a column vector. 
#                      The function should also add a column of constant 1s in the first column.
# prepare_target(df): which takes in a data frame or two-dimensional numpy array for the target. 
#                     If the input is a data frame, the function should simply convert the data frame to a numpy array and change it into column vectors. 
#                     The function should be able to handle if the data frame or the numpy array have more than one column.

# You can use the following methods in your code:
# df.to_numpy(): which is to convert a Pandas data frame to Numpy array.
# np.reshape(row, col): which is to reshape the numpy array to a particular shape.
# np.concatenate((array1, array2), axis): which is to join a sequence of arrays along an existing axis.
# np.matmul(array1, array2): which is to do matrix multiplication on two Numpy arrays.

def calc_linreg(X, beta):
    Y_hat = np.matmul(X, beta)
    return Y_hat
    pass

def compute_cost_linreg(X, y, beta):
    J = 0
    # 1. calculate error square
    error = calc_linreg(X, beta) - y
    error_square = np.matmul(error.T, error)
    # 2. calculate cost function J
    m = X.shape[0]
    J = 1/(2 * m) * error_square
    # 3. make J numeric
    J = J[0][0]
    return J

def prepare_feature(df_feature):
    # 1. get column number
    cols = df_feature.shape[1]
    # convert dataframe into numpy feature
    if type(df_feature) == pd.DataFrame:
        np_feature = df_feature.to_numpy()
    else:
        np_feature = df_feature
    # 2. reshape feature
    feature = np_feature.reshape(-1, cols)
    # concatenate 1 to get Matrix X
    X = np.concatenate((np.ones((feature.shape[0], 1)), feature), axis=1)
    # return
    return X
    pass

def prepare_target(df_target):
    # 1. get column number
    cols = df_target.shape[1]
    # 2. covert dataframe into numpy array
    if type(df_target) == pd.DataFrame:
        np_target = df_target.to_numpy()
    else:
        np_target = df_target
    # 3. reshape target and return
    target= np_target.reshape(-1, cols)
    return target
    pass

# test cases
X = prepare_feature(df_feature)
target = prepare_target(df_target)

assert isinstance(X, np.ndarray)
assert isinstance(target, np.ndarray)
assert X.shape == (506, 2)
assert target.shape == (506, 1)

# print(X)
beta = np.zeros((2,1))
J = compute_cost_linreg(X, target, beta)
print(J)
assert np.isclose(J, 296.0735)

beta = np.ones((2,1))
J = compute_cost_linreg(X, target, beta)
print(J)
assert np.isclose(J, 268.157)

beta = np.array([-1, 2]).reshape((2,1))
J = compute_cost_linreg(X, target, beta)
print(J)
assert np.isclose(J, 308.337)

# CS2. Gradient Descent: Write a function called gradient_descent_linreg() that takes in these parameters:

# X: is a 2-D numpy array for the features
# y: is a vector array for the target
# beta: is a column vector for the initial guess of the parameters
# alpha: is the learning rate
# num_iters: is the number of iteration to perform
# The function should return two numpy arrays:

# beta: is coefficient at the end of the iteration
# J_storage: is the array that stores the cost value at each iteration
# You can use some of the following functions: 

# calc_linreg(X, b): which is used to calculate  𝑦=𝑋𝑏
#   vector.
# np.matmul(array1, array2): which is to do matrix multiplication on two Numpy arrays.
# compute_cost_linreg(): which the function you created in the previous problem set to compute the cost.

def gradient_descent_linreg(X, y, beta, alpha, num_iters):
    # 1. calculate m 
    m = X.shape[0]
    # 2. initialize J_storage
    J_storage = np.zeros((num_iters, 1))
    # 3. update beta, J_storage in a for loop
    for n in range(num_iters):
        deriv = np.matmul(X.T, (calc_linreg(X, beta) - y))
        beta = beta - alpha * 1/m * deriv
        J_storage[n] = compute_cost_linreg(X, y, beta)
    return beta, J_storage

# test cases
iterations = 1500
alpha = 0.01
beta = np.zeros((2,1))

beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)
print(beta)
assert np.isclose(beta[0], 22.5328)
assert np.isclose(beta[1], 6.3953)

plt.plot(J_storage)

# CS3. Predict: Write the function predict_linreg() that calculates the straight line equation given the features and its coefficient.

# predict_linreg(): this function should standardize the feature using z normalization, change it to a Numpy array, and add a column of constant 1s. 
#                   You should use prepare_feature() for this purpose. Lastly, this function should also call calc_linreg() to get the predicted y values.

# You can use some of the following functions:
# calc_linreg(X, beta): which is used to calculate the predicted y after X has been normalized and added by a constant.
# np.matmul(array1, array2): which is to do matrix multiplication on two Numpy arrays.
# normalize_z(df): which is to do z normalization on the data frame.

def predict_linreg(df_feature, beta, means=None, stds=None):
    # normalize feature
    feature,_,_ = normalize_z(df_feature, means, stds)
    # prepare feature
    X = prepare_feature(feature)
    # predict values
    Y = calc_linreg(X, beta)
    return Y
    pass

# test cases
df_feature, buf = get_features_targets(df, ["RM"], [])
beta = [[22.53279993],[ 6.39529594]] # from previous result
pred = predict_linreg(df_feature, beta)

assert isinstance(pred, np.ndarray)
assert pred.shape == (506, 1)
assert np.isclose(pred.mean(), 22.5328)
assert np.isclose(pred.std(), 6.38897)

means = [6.284634]
stds = [0.702617]
beta = [[22.53279993],[ 6.39529594]] # from previous result
input_1row = np.array([[6.593]])
pred_1row = predict_linreg(input_1row, beta, means, stds)
assert np.isclose(pred_1row[0][0], 25.33958)

plt.plot(df_feature["RM"],target,'o')
plt.plot(df_feature["RM"],pred,'-')

# CS4. Splitting data: Do the following tasks:

# Read RM as the feature and MEDV as the target from the data frame.
# Use Week 9's function split_data() to split the data into train and test using random_state=100 and test_size=0.3.
# Normalize and prepare the features and the target.
# Use the training data set and call gradient_descent_linreg() to obtain the theta.
# Use the test data set to get the predicted values.
# You need to replace the None in the code below with other a function call or any other Python expressions.

def split_data(df_feature, df_target, random_state=None, test_size=0.5):
    indexes = df_feature.index
    if random_state != None:
        np.random.seed(random_state)
    k = int(test_size * len(indexes))
    test_index = np.random.choice(indexes, k, replace=False)
    indexes = set(indexes)
    test_index = set(test_index)
    train_index = indexes - test_index
    
    df_feature_train = df_feature.loc[train_index, :]
    df_feature_test = df_feature.loc[test_index, :]
    df_target_train = df_target.loc[train_index, :]
    df_target_test = df_target.loc[test_index, :]
    return df_feature_train, df_feature_test, df_target_train, df_target_test

# get features and targets from data frame
df_feature, df_target = get_features_targets(df, ["RM"], ["MEDV"])

# split the data into training and test data sets
df_feature_train, df_feature_test, df_target_train, df_target_test = split_data(df_feature, df_target, random_state=100, test_size=0.3)

# normalize the feature using z normalization
df_feature_train_z,_,_ = normalize_z(df_feature_train)

# prepare feature and target
feature = prepare_feature(df_feature_train_z)
target = prepare_target(df_target_train)

# initialize parameters
iterations = 1500
alpha = 0.01
beta = np.zeros((2,1))

# call the gradient_descent function
beta, J_storage = gradient_descent_linreg(feature, target, beta, alpha, iterations)

# call the predict method to get the predicted values
pred = predict_linreg(df_feature_test, beta)

# test cases 
assert isinstance(pred, np.ndarray)
assert pred.shape == (151, 1)
assert np.isclose(pred.mean(), 22.66816)
assert np.isclose(pred.std(), 6.257265)

plt.scatter(df_feature_test, df_target_test)
plt.plot(df_feature_test, pred, color="orange")

# CS5. R2 Coefficient of Determination: Write a function to calculate the coefficient of determination as given by the following equations.

# 𝑟^2 = 1 − (𝑆𝑆𝑟𝑒𝑠 / 𝑆𝑆𝑡𝑜𝑡)
 
# where

# 𝑆𝑆𝑟𝑒𝑠 = Σ𝑛𝑖=1 (𝑦𝑖−𝑦̂ 𝑖)^2
 
# where  𝑦𝑖 is the actual target value and  𝑦̂ 𝑖 is the predicted target value.

# 𝑆𝑆𝑡𝑜𝑡 = Σ𝑛𝑖=1 (𝑦𝑖−𝑦⎯⎯⎯)^2
 
# where 𝑦⎯⎯⎯ = 1 / 𝑛 (Σ𝑛𝑖 = 1 𝑦𝑖)
 
# and  𝑛 is the number of target values.

# You can use the following functions in your code:
# np.mean(array): which is to get the mean of the array. You can also call it using array.mean().
# np.sum(array): which is to sum the array along a default axis. You can specify which axis to perform the summation.

def r2_score(y, ypred):
    ymean = np.mean(y)
    diff = y - ymean
    error = y - ypred
    # sstot = np.matmul(diff.T, diff)
    # ssres = np.matmul(error.T, error)
    # r2 = 1 - ssres / sstot[0][0]
    sstot = np.sum(diff**2)
    ssres = np.sum(error**2)
    r2 = 1 - (ssres/sstot)
    return r2
    ###
    ### YOUR CODE HERE
    ###
    pass

# test cases
target = prepare_target(df_target_test)
r2 = r2_score(target, pred)
print(r2)
assert np.isclose(r2, 0.45398)

# CS6. Mean Squared Error: Create a function to calculate the MSE as given below.

# 𝑀𝑆𝐸 = 1 / 𝑛 (Σ𝑛𝑖=1 (𝑦𝑖−𝑦̂ 𝑖)^2)

def mean_squared_error(target, pred):
    n = target.shape[0]
    mean_error = target - pred
    mean_squared_error = (1 / n) * np.sum(mean_error**2)
    return mean_squared_error
    pass

# test cases
mse = mean_squared_error(target, pred)
print(mse)
assert np.isclose(mse, 53.6375)

# CS7. Optional: Redo the above tasks using Sci-kit learn libraries. You will need to use the following:

# LinearRegression
# train_test_split
# r2_score
# mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Read the CSV and extract the features
df = pd.read_csv("housing_processed.csv")
df_feature, df_target = get_features_targets(df, ["RM"], ["MEDV"])
# normalize
df_feature,_,_ = normalize_z(df_feature)

# Split the data into training and test data set using scikit-learn function
df_feature_train, df_feature_test, df_target_train, df_target_test = train_test_split(df_feature, df_target, random_state=100, test_size=0.3, shuffle=True)

# Instantiate LinearRegression() object
model = LinearRegression()

# Call the fit() method
model.fit(df_feature_train, df_target_train)
pass

print(model.coef_, model.intercept_)
assert np.isclose(model.coef_,[6.05090511])
assert np.isclose(model.intercept_, 22.52999668)

# Call the predict() method
pred = model.predict(df_feature_test)

print(type(pred), pred.mean(), pred.std())
assert isinstance(pred, np.ndarray)
assert np.isclose(pred.mean(), 22.361699)
assert np.isclose(pred.std(), 5.7011267)

plt.scatter(df_feature_test, df_target_test)
plt.plot(df_feature_test, pred, color="orange")

r2 = r2_score(df_target_test, pred)
print(r2)
assert np.isclose(r2, 0.457647)

mse = mean_squared_error(df_target_test, pred)
print(mse)
assert np.isclose(mse, 54.93216)