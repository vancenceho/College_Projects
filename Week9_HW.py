# WEEK 9 PROBLEM SET - HOMEWORK

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

# HW0. Copy and paste some of the functions from you cohort sessions and previous exercises that you will need in this homework. \

# See below template and the list here:

# normalize_z()
# get_features_targets()
# calc_linreg()
# prepare_feature()
# prepare_target()
# predict_linreg()
# split_data()
# r2_score()
# mean_squared_error()

# Then do the following:

# Read the CSV file housing_processed.csv and extract the following columns:
# x data: RM, DIS, and INDUS columns
# y data: MEDV column
# Normalize the features using z normalization.
# Plot the data using scatter plot. Use the following columns:

def normalize_z(dfin, columns_means=None, columns_stds=None):
    if columns_means is None:
        columns_means = dfin.mean(axis=0)
    if columns_stds is None:
        columns_stds = dfin.std(axis=0)
    dfout = (dfin - columns_means) / columns_stds
    return dfout, columns_means, columns_stds

def get_features_targets(df, feature_names, target_names):
    df_feature = df[feature_names]
    df_target = df[target_names]
    return df_feature, df_target

def prepare_feature(df_feature):
    cols = df_feature.shape[1]
    if type(df_feature) == pd.DataFrame:
        np_feature = df_feature.to_numpy()
    else:
        np_feature = df_feature
    
    feature =  np_feature.reshape(-1, cols)
    X = np.concatenate((np.ones((feature.shape[0], 1)), feature), axis=1)
    return X

def prepare_target(df_target):
    cols = df_target.shape[1]
    if type(df_target) == pd.DataFrame:
        np_target = df_target.to_numpy()
    else:
        np_target = df_target
    
    target = np_target.reshape(-1, cols)
    return target

def predict_linreg(df_feature, beta, means=None, stds=None):
    feature,_,_ = normalize_z(df_feature, means, stds)
    X = prepare_feature(feature)
    Y = calc_linreg(X, beta)
    
    return Y

def calc_linreg(X, beta):
    Y_hat = np.matmul(X, beta)
    return Y_hat

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
  
def r2_score(y, ypred):
    ymean = np.mean(y)
    diff = y - ymean
    error = y - ypred
    sstot = np.sum(diff**2)
    ssres = np.sum(error**2)
    r2 = 1 - (ssres / sstot)
    return r2

def mean_squared_error(target, pred):
    n = target.shape[0]
    error = target - pred
    mse = (1 / n) * np.sum(error**2)
    return mse

# Read the CSV file
df = pd.read_csv("housing_processed.csv")

# Extract the features and the targets
feature = ["RM", "DIS", "INDUS"]
target = ["MEDV"]
df_features, df_target = get_features_targets(df, feature, target)

# Normalize using z normalization
df_features,_,_ = normalize_z(df_features)

# test cases
display(df_features.describe())
display(df_target.describe())
assert np.isclose(df_features['RM'].mean(), 0)
assert np.isclose(df_features['DIS'].mean(), 0)
assert np.isclose(df_features['INDUS'].mean(), 0)

assert np.isclose(df_features['RM'].std(), 1)
assert np.isclose(df_features['DIS'].std(), 1)
assert np.isclose(df_features['INDUS'].std(), 1)

assert np.isclose(df_target['MEDV'].mean(), 22.532806)
assert np.isclose(df_target['MEDV'].std(), 9.197104)

assert np.isclose(df_features['RM'].median(), -0.1083583)
assert np.isclose(df_features['DIS'].median(), -0.2790473)
assert np.isclose(df_features['INDUS'].median(), -0.2108898)

sns.set()
plt.scatter(df_features["RM"], df_target)

plt.scatter(df_features["DIS"], df_target)

plt.scatter(df_features["INDUS"], df_target)

# HW1. Multiple variables cost function: Write a function compute_cost_linreg() to compute the cost function of a linear regression model. 
# The function should take in two 2-D numpy arrays. The first one is the matrix of the linear equation and the second one is the actual target value.

# Recall that:

# 𝐽(𝛽̂ 0,𝛽̂ 1) = 1/2𝑚 * (Σ𝑚𝑖=1(𝑦̂ (𝑥𝑖)−𝑦𝑖)2)
 
# where

# 𝑦̂(𝑥) = 𝛽̂0 + 𝛽̂1𝑥1 + 𝛽̂2𝑥2 + … + 𝛽̂𝑛𝑥𝑛
 
# The function should receive three Numpy arrays:
# X: is the feature 2D Numpy array
# y: is the target 2D Numpy array
# beta: is the parameter 2D Numpy array
# The function should return the cost which is a float.

# You can use the following function in your code:
# np.matmul(array1, array2)
# Note that if you wrote your Cohort session's compute_cost_linreg() using proper Matrix operations to do the square and the summation, 
# the code will be exactly the same here and you just need to copy and paste it here.

def compute_cost_linreg(X, y, beta):
    J = 0
    error = calc_linreg(X, beta) - y
    error_square = np.matmul(error.T, error)
    m = X.shape[0]
    J = 1/(2 * m) * error_square
    J = J[0][0]
    return J

# test cases
X = prepare_feature(df_features)
target = prepare_target(df_target)

assert isinstance(X, np.ndarray)
assert isinstance(target, np.ndarray)
assert X.shape == (506, 4)
assert target.shape == (506, 1)

beta = np.zeros((4,1))
J = compute_cost_linreg(X, target, beta)
print(J)
assert np.isclose(J, 296.0734)

beta = np.ones((4,1))
J = compute_cost_linreg(X, target, beta)
print(J)
assert np.isclose(J, 270.4083)

beta = np.array([-1, 2, 1, 2]).reshape((4,1))
J = compute_cost_linreg(X, target, beta)
print(J)
assert np.isclose(J, 314.8510)

# HW2. Gradient Descent: Write a function called gradient_descent_linreg() that takes in four parameters:

# X: is a 2-D numpy array for the features
# y: is a vector array for the target
# alpha: is the learning rate
# num_iters: is the number of iteration to perform
# The function should return two arrays:

# beta: is coefficient at the end of the iteration
# J_storage: is the array that stores the cost value at each iteration
# You can use some of the following functions:

# np.matmul(array1, array2): which is to do matrix multiplication on two Numpy arrays.
# compute_cost_linreg(): which the function you created in the previous problem set to compute the cost.
# Note that if you use proper matrix operations in your cohort sessions for the gradient descent function, the code will be the same here.

def gradient_descent_linreg(X, y, beta, alpha, num_iters):
    m = X.shape[0]
    J_storage = np.zeros((num_iters, 1))
    for n in range(num_iters):
        deriv = np.matmul(X.T, (calc_linreg(X, beta) - y))
        beta = beta - alpha * 1/m * deriv
        J_storage[n] = compute_cost_linreg(X, y, beta)
    return beta, J_storage

# test cases
iterations = 1500
alpha = 0.01
beta = np.zeros((4,1))

beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)
print(beta)
assert np.isclose(beta[0], 22.5328)
assert np.isclose(beta[1], 5.4239)
assert np.isclose(beta[2], -0.90367)
assert np.isclose(beta[3], -2.95818)

plt.plot(J_storage)

# HW3. Do the following tasks:

# Get the features and the targets.
# features: RM, DIS, INDUS
# target: MEDV
# Use the previous functions called predict() to calculated the predicted values given the features and the model.
# Create a target numpy array from the data frame.

# This is from the previous result
beta = np.array([[22.53279993],
       [ 5.42386374],
       [-0.90367449],
       [-2.95818095]])

# Extract the feature and the target
feature = ["RM", "DIS", "INDUS"]
target = ["MEDV"]
df_features, df_target = get_features_targets(df, feature, target)

# Call predict()
pred = predict_linreg(df_features, beta)

# Change target to numpy array
target = df_target.to_numpy()

# test cases
assert isinstance(pred, np.ndarray)
assert np.isclose(pred.mean(), 22.5328)
assert np.isclose(pred.std(), 6.7577)

plt.scatter(df_features["RM"],target)
plt.scatter(df_features["RM"],pred)

plt.scatter(df_features["DIS"],target)
plt.scatter(df_features["DIS"],pred)

plt.scatter(df_features["INDUS"],target)
plt.scatter(df_features["INDUS"],pred)

# HW4. Splitting data: Do the following tasks:

# Extract the following:
# features: RM, DIS, and INDUS
# target: MEDV
# Use Week 9's function split_data() to split the data into train and test using random_state=100 and test_size=0.3.
# Normalize and prepare the features and the target.
# Use the training data set and call gradient_descent_linreg() to obtain the theta.
# Use the test data set to get the predicted values.
# You need to replace the None in the code below with other a function call or any other Python expressions.

features = ["RM", "DIS", "INDUS"]
target = ["MEDV"]
# Extract the features and the target
df_features, df_target = get_features_targets(df, features, target)

# Split the data set into training and test
df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, random_state=100, test_size=0.3)

# Normalize the features using z normalization
df_features_train_z,_,_ = normalize_z(df_features_train)

# Change the features and the target to numpy array using the prepare functions
X = prepare_feature(df_features_train_z)
target = prepare_target(df_target_train)

iterations = 1500
alpha = 0.01
beta = np.zeros((4,1))

# Call the gradient_descent function
beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)

# call the predict() method
pred = predict_linreg(df_features_test, beta)

# test cases
assert isinstance(pred, np.ndarray)
assert pred.shape == (151, 1)
assert np.isclose(pred.mean(), 22.66816)
assert np.isclose(pred.std(), 6.67324)

plt.scatter(df_features_test["RM"], df_target_test)
plt.scatter(df_features_test["RM"], pred)

plt.scatter(df_features_test["DIS"], df_target_test)
plt.scatter(df_features_test["DIS"], pred)

plt.scatter(df_features_test["INDUS"], df_target_test)
plt.scatter(df_features_test["INDUS"], pred)

# HW5. Calculate the coefficient of determination,  𝑟2.

# change target test set to a numpy array
target = prepare_target(df_target_test)

# Calculate r2 score by calling a function
r2 = r2_score(target, pred)

print(r2)

# test cases
assert np.isclose(r2, 0.47713)

# HW6. Calculate the mean squared error.

# Calculate the mse
mse = mean_squared_error(target, pred)

print(mse)

# test cases
assert np.isclose(mse, 51.363)

# HW7. Polynomial Transformation: Redo the steps for breast cancer data but this time we will use quadratic model. 
# Use the following columns:
# x data: radius_mean
# y data: area_mean
# We will create a quadratic hypothesis for this x and y data. 
# To do that write a function transform_features(df, colname, colname_transformed) that takes in a dataframe for the features, 
# the original column name, and the transformed column name. The function should add another column with the name colname_transformed with the value of column 
# in colname transformed to its quadratic value.

# Read from breast_cancer_data.csv file
df = pd.read_csv("breast_cancer_data.csv")

# Extract feature and target
df_feature, df_target = get_features_targets(df, ["radius_mean"], ["area_mean"])

plt.scatter(df_feature, df_target)

# write your function to create a quadratic feature of x

def transform_features(df_feature, colname, colname_transformed):
    df_feature[colname_transformed] = df_feature[colname] ** 2
    return df_feature

df_features = transform_features(df_feature, "radius_mean", "radius_mean^2")

assert np.allclose(df_features.loc[:,"radius_mean^2"], df_features.loc[:,"radius_mean"] ** 2)

# split data using random_state = 100 and 30% test size
df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, random_state=100, test_size=0.3)

# normalize features
df_features_train_z,_,_ = normalize_z(df_features_train)

# change to numpy array and append column for feature

X = prepare_feature(df_features_train_z)
target = prepare_target(df_target_train)

iterations = 1500
alpha = 0.01
beta = np.zeros((3,1))

# call gradient_descent() function
print(X.shape)
print(beta.shape)
beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)

# test cases
assert np.isclose(beta[0], 646.0787)
assert np.isclose(beta[1], 146.4801)
assert np.isclose(beta[2], 201.9803)

plt.plot(J_storage)

# change target to numpy array
beta = np.array([[646.0787641 ], [146.4800792 ], [201.98031254]])

target = prepare_target(df_target_test)

# get predicted values
pred = predict_linreg(df_features_test, beta)

plt.scatter(df_features_test["radius_mean"], target)
plt.scatter(df_features_test["radius_mean"], pred)

target = prepare_target(df_target_test)
r2 = r2_score(target, pred)
print(r2)
assert np.isclose(r2, 0.985095)

target = prepare_target(df_target_test)
mse = mean_squared_error(target, pred)
print(mse)
assert np.isclose(mse, 1919.164)

# HW8. Optional: Redo the above tasks using Sci-kit learn libraries. 

# You will need to use the following:
# LinearRegression
# train_test_split
# r2_score
# mean_squared_error
# PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Redo HW 4 using Scikit Learn
# Read the housing_processed.csv file
df = pd.read_csv("housing_processed.csv")

# extract the features from ["RM", "DIS", "INDUS"] and target from []"MEDV"]
columns = ["RM", "DIS", "INDUS"]
df_features, df_target = get_features_targets(df, columns, ["MEDV"])
# normalize
df_features,_,_ = normalize_z(df_features)

display(df_features)
display(df_target)

# Split the data into training and test data set using scikit-learn function
df_features_train, df_features_test, df_target_train, df_target_test = train_test_split(df_features, df_target, random_state=100, test_size=0.3, shuffle=True)

# Instantiate LinearRegression() object
model = LinearRegression()

# Call the fit() method
model.fit(df_features_train, df_target_train)

print(model.coef_, model.intercept_)
assert np.isclose(model.coef_, [ 5.01417104, -1.00878266, -3.27301726]).all()
assert np.isclose(model.intercept_, 22.45962454)

# Call the predict() method
pred = model.predict(df_features_test)

plt.scatter(df_features_test["RM"], df_target_test)
plt.scatter(df_features_test["RM"], pred)

plt.scatter(df_features_test["DIS"], df_target_test)
plt.scatter(df_features_test["DIS"], pred)

plt.scatter(df_features_test["INDUS"], df_target_test)
plt.scatter(df_features_test["INDUS"], pred)

r2 = r2_score(df_target_test, pred)
print(r2)
assert np.isclose(r2, 0.48250)

mse = mean_squared_error(df_target_test, pred)
print(mse)
assert np.isclose(mse, 52.41451)

# Redo HW7 Using Scikit Learn
# Read the file breast_cancer_data.csv
df = pd.read_csv("breast_cancer_data.csv")
# extract feature from "radius_mean" and target from "area_mean"
df_feature, df_target = get_features_targets(df, ["radius_mean"], ["area_mean"])

# instantiate a PolynomialFeatures object with degree = 2
poly = PolynomialFeatures(degree=2)

# call its fit_transform() method
df_features = poly.fit_transform(df_feature)

# call train_test_split() to split the data
df_features_train, df_features_test, df_target_train, df_target_test = train_test_split(df_features, df_target, random_state=100, test_size=0.3, shuffle=True)

# instantiate LinearRegression() object
model = LinearRegression()

# call its fit() method
model.fit(df_features_train, df_target_train)
pass

# test cases
print(model.coef_, model.intercept_)
assert np.isclose(model.coef_, [0., 3.69735512, 2.9925278 ]).all()
assert np.isclose(model.intercept_, -32.3684598)

# Call the predict() method
pred = model.predict(df_features_test)

# test cases
print(type(pred), pred.mean(), pred.std())
assert isinstance(pred, np.ndarray)
assert np.isclose(pred.mean(), 672.508465)
assert np.isclose(pred.std(), 351.50271)

plt.scatter(df_features_test[:,1], df_target_test)
plt.scatter(df_features_test[:,1], pred)

r2 = r2_score(df_target_test, pred)
print(r2)
assert np.isclose(r2, 0.99729)

mse = mean_squared_error(df_target_test, pred)
print(mse)
assert np.isclose(mse, 346.79479)