# WEEK 10 PROBLEM SET - COHORT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# CS0. Do the following tasks before you start with the first cohort session.

# Task 1. Paste the following functions from your previous work:

# get_features_targets()
# normalize_z()
# prepare_feature()
# prepare_target()
# split_data()

# Deal with data which are represented differently with respect to the original data.
def normalize_z(dfin, columns_means=None, columns_stds=None):
    if columns_means is None:
        columns_means = dfin.mean(axis=0)
    if columns_stds is None:
        columns_stds = dfin.std(axis=0)
    dfout = (dfin - np.array(columns_means)) / np.array(columns_stds)
    return dfout, columns_means, columns_stds

def get_features_targets(df, feature_names, target_names):
    df_feature = df[feature_names]
    df_target = df[target_names]
    return df_feature, df_target

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
 
# Task 2. Load the breast cancer data from breast_cancer_data.csv into a Data Frame.

# read breast_cancer_data.csv
df = pd.read_csv("breast_cancer_data.csv")

df

# Task 3. Do the following tasks.

# Read the following columns
# feature: radius_mean
# target: diagnosis
# Normalize the feature column using z normalization.

# extract the feature and the target
df_feature, df_target = get_features_targets(df, ["radius_mean"], ["diagnosis"])

# normalize the feature
df_feature,_,_ = normalize_z(df_feature) 

# Task 4. Write a function replace_target() to replace the diagnosis column with the following mapping: 
# - M: 1, this means that malignant cell are indicated as 1 in our new column. 
# - B: 0, this means that benign cell are indicated as 0 in our new column.

# The function should takes in the following:
# df_target: the target data frame
# target_name: which is the column name of the target data frame
# map: which is a dictionary containing the map

# It should returns a new data frame with the same column name but with its values changed according to the mapping.

def replace_target(df_target, target_name, map_vals):
    df_out = df_target.copy() # make a copy of the original target df that contains "m" and "b" as values
    # replace the column matching target_name, e.g: "diagnosis", and apply a function over its elements. 
    # x represents every element in diagnosis column, which is x can be "M" or x can be "B".
    df_out.loc[:, target_name] = df_target[target_name].apply(lambda x: map_vals[x])
    
    return df_out

# use the function above
# initially, df_target contains column called diagnosis, with rows of "M" and "B" each row.

df_target = replace_target(df_target, "diagnosis", {'M': 1, 'B': 0})
df_target

# Task 5. Do the following tasks.

# Change feature to Numpy array and append constant 1 column.
# Change target to Numpy array

# get the big X matrix
# change feature data frame to numpy array and append column 1
feature = prepare_feature(df_feature)

# get the y vector
# change target data frame to numpy array
target = prepare_target(df_target)

# CS1. Logistic function: Write a function to calculate the hypothesis using a logistic function. 
# Recall that the hypothesis for a logistic regression model is written as:

# 𝐩(𝑥)=1 / (1+𝑒−𝐗𝐛)
 
# The shape of the input is as follows:
# 𝐛 : is a column vector for the parameters
# 𝐗 : is a matrix where the number of rows are the number of data points and the the number of columns must the same as the number of parameters in 𝐛.
# Note that you need to ensure that the output is a column vector.

# You can use the following functions:
# np.matmul(array1, array2): which is to perform matrix multiplication on the two numpy arrays.
# np.exp(): which is to calculate the function 𝑒𝑥

# argument types must be np array. 
# X is a matrix of size # samples + # of parameters
# beta is a vector of size # of parameters
# no-of-params is as many as # features + 1
# output must be a column vector
# e.g.: [[1], [2], [3]]
def calc_logreg(X, beta):
    return 1 / (1 + np.exp(np.matmul(X, -beta)))

# test cases
beta = np.array([0])
x = np.array([0])
ans = calc_logreg(x, beta)
assert ans == 0.5

beta = np.array([2])
x = np.array([40])
ans = calc_logreg(x, beta)
assert np.isclose(ans, 1.0)

beta = np.array([2])
x = np.array([-40])
ans = calc_logreg(x, beta)
assert np.isclose(ans, 0.0)

beta = np.array([[1, 2, 3]])
x = np.array([[3, 2, 1]])
ans = calc_logreg(x, beta.T)
assert np.isclose(ans.all(), 1.0)

beta = np.array([[1, 2, 3]])
x = np.array([[3, 2, 1], [3, 2, 1]])
ans = calc_logreg(x, beta.T)
assert ans.shape == (2, 1)
assert np.isclose(ans.all(), 1.0)

# CS2. Cost Function: Write a function to calculate the cost function for logistic regression. 
# Recall that the cost function for logistic regression is given by:

# 𝐽(𝛽) = −1/𝑚 * [Σ𝑚𝑖=1𝑦𝑖log(𝑝(𝑥𝑖))+(1−𝑦𝑖)log(1−𝑝(𝑥𝑖))]
 
# You can use the following function in your code:
# np.where(condition, then_expression, else_expression)

def compute_cost_logreg(beta, X, y):
    #np.seterr(divide = 'ignore')
    # input X is a matrix of size # samples x # parameters
    # input beta is a column vector of size # parameters
    # input y is an actual target, a column vector of size # samples 
    # should return a scalar
    number_of_samples = len(y)
    J = -(1 / number_of_samples) * np.sum(
        # np.where receives 3 arguments, first is the condition, then the true expression,
        # then the false expression
        np.where(y==1, np.log(calc_logreg(X, beta)), np.log(1-calc_logreg(X, beta))))
    #np.seterr(divide = 'warn')
    return J

# test cases
y = np.array([[1]])
X = np.array([[10, 40]])
beta = np.array([[1, 1]]).T
ans = compute_cost_logreg(beta, X, y)
print(ans)
assert np.isclose(ans, 0)

y = np.array([[0]])
X = np.array([[10, 40]])
beta = np.array([[-1, -1]]).T
ans = compute_cost_logreg(beta, X, y)
print(ans)
assert np.isclose(ans, 0)

# CS3. Gradient Descent: Recall that the update functions can be written as a matrix multiplication.

# 𝐛 = 𝐛 − 𝛼 * (1/𝑚 * (𝐗.𝑇 * (𝐩 − 𝐲)))
 
# Write a function called gradient_descent_logreg() that takes in four parameters:

# X: is a 2-D numpy array for the features
# y: is a vector array for the target
# beta: is a column vector for the initial guess of the parameters
# alpha: is the learning rate
# num_iters: is the number of iteration to perform

# The function should return two arrays:
# beta: is coefficient at the end of the iteration
# J_storage: is the array that stores the cost value at each iteration

# The solution is similar to Linear Regression gradient descent function with two differences:
# you need to use log_regression() to calculate the hypothesis
# you need to use compute_cost_logreg() to calculate the cost

# input parameters are X matrix, np.array, then beta parameters (column vector),
# alpha and num_iters are float and int respectively
# y is the actual target, which is also a column vector
# return a tuple of two things: beta vector, 
# and the 1D array of J_storage storing the error at each iteration of gradient descent
def gradient_descent_logreg(X, y, beta, alpha, num_iters):
    number_of_samples = X.shape[0]
    # prepare an array to store errors
    # an array of zeros, with dimension of # iters times 1 (it's a 1D array)
    J_storage = np.zeros((num_iters, 1))
    for n in range(num_iters):
        # update beta repeatedly
        # compute derivative of error w.r.t current beta
        derivative = np.matmul(X.T, (calc_logreg(X, beta) - y))
        # update beta 
        beta = beta - alpha * (1 / number_of_samples) * derivative
        # store the current error
        J_storage[n] = compute_cost_logreg(beta, X, y)
    
    return beta, J_storage 

# test cases
iterations = 1500
alpha = 0.01
beta = np.zeros((2,1))
beta, J_storage = gradient_descent_logreg(feature, target, beta, alpha, iterations)

print(beta)
assert beta.shape == (2, 1)
assert np.isclose(beta[0][0], -0.56630)
assert np.isclose(beta[1][0], 1.93764)

plt.plot(J_storage)

# CS4. Predict: Write two functions predict_logreg() and predict_norm() that calculate the straight line equation given the features and its coefficient.

# predict_logreg(): this function should standardize the feature using z normalization, change it to a Numpy array, and add a column of constant 1s. 
#                   You should use prepare_feature() for this purpose. Lastly, this function should also call predict_norm() to get the predicted y values.
# predict_norm(): this function should calculate the hypothesis or its probability using calc_logreg() and categorize it to either 0 or 1 based on its probability. 
#                 If the probability is greater or equal to 1, it should be classified as class 1. Otherwise, it is classified as 0.

# You can use the following function in your code:
# np.where()

def predict_norm(X, beta):
    # p is a vector containing values between 0 to 1, representing for each sample,
    # what the probability of the occurence of the diagnosis (or some event you're computing)
    p = calc_logreg(X, beta)
    # convert each element in p into value 0 or 1, depending on whether p is < or > 0.5
    # also returns a column vector, but now with values clamped into 0 or 1
    return np.where(p >= 0.5, 1, 0)
    pass

# runs result of trained beta and compute the output of logistic regression,
# should return a vector of 1s or 0s to classify the input feature
def predict_logreg(df_feature, beta, means=None, stds=None):
    # normalize the features first, so that we dont run out into floating point error
    norm_data,_,_ = normalize_z(df_feature, means, stds)
    # compute the big X matrix, which appends the columns of 1 to the feature matrix
    # uses the predit_norm function to obtain binary predictions based on logistic regression probabilities
    X = prepare_feature(norm_data)
    return predict_norm(X, beta)
    pass

# test cases
pred = predict_logreg(df_feature, beta)
print(pred.mean(), pred.std())
assert isinstance(pred, np.ndarray)
assert np.isclose(pred.mean(), 0.28998)
assert np.isclose(pred.std(), 0.45375)
means = [0]
stds = [1]
beta =np.array([[-0.56630289], [ 1.93763591]])
input_1row = np.array([[2.109139]])
pred_1row = predict_logreg(input_1row, beta, means, stds)
assert pred_1row[0][0] == 1

plt.scatter(df_feature, df_target)
plt.scatter(df_feature, pred)

# CS5. Multiple features and splitting of data set:

# Do the following task in the code below:

# Read the following column names as the features: "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean"
# Read the column diagnosis as the target. Change the value from M and B to 1 and 0 respectively.
# Split the data set with 30% test size and random_state = 100.
# Normalize the training feature data set using normalize_z() function.
# Convert to numpy array both the target and the features using prepare_feature() and prepare_target() functions.
# Call gradient_descent() function to get the parameters using the training data set.
# Call predict() function on the test data set to get the predicted values.

columns = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean"]

# extract the features and the target columns
df_features, df_target = get_features_targets(df, columns, ["diagnosis"])

# replace the target values using from string to integer 0 and 1
df_target = replace_target(df_target, "diagnosis", {"M": 1, "B": 0})

# split the data with random_state = 100 and 30% test size
df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, random_state=100, test_size=0.3)

# normalize the features so we dont run into floating point error
df_features_train_z, means, stds = normalize_z(df_features_train)

# change the feature columns to numpy array and append column of 1s
features = prepare_feature(df_features_train_z)

# change the target column to numpy array
target = prepare_target(df_target_train)

iterations = 1500
alpha = 0.01

# provide initial guess for beta
beta = np.zeros((features.shape[1], 1))

# call the gradient descent method
beta, J_storage = gradient_descent_logreg(features, target, beta, alpha, iterations)

print(beta)

# test cases
assert beta.shape == (8, 1)
ans = np.array([[-0.6139379 ], 
                [ 0.82529554],
                [ 0.72746485],
                [ 0.8236603 ],
                [ 0.81647937],
                [ 0.5057749 ],
                [ 0.44176466],
                [ 0.78736842]])
assert np.isclose(beta, ans).all()

plt.plot(J_storage)

# call predict() on one record to get the predicted values
# use the variable 'means' and 'stds' to normalize
input_1row = np.array([[12.22, 20.04, 79.47, 453.1, 0.10960, 0.11520, 0.08175]])
pred_1row = predict_logreg(input_1row, beta, means, stds)

print(pred_1row)

assert pred_1row[0][0] == 0

# call predict() on df_features test dataset to get the predicted values
pred = predict_logreg(df_features_test, beta)

plt.scatter(df_features_test["radius_mean"], df_target_test)
plt.scatter(df_features_test["radius_mean"], pred)

plt.scatter(df_features_test["texture_mean"], df_target_test)
plt.scatter(df_features_test["texture_mean"], pred)

plt.scatter(df_features_test["perimeter_mean"], df_target_test)
plt.scatter(df_features_test["perimeter_mean"], pred)

# CS6. Confusion Matrix: Write a function confusion_matrix() that takes in:

# ytrue: which is the true target values
# ypred: which is the predicted target values
# labels: which is a list of the category. In the above case it will be [1, 0]. Put the positive case as the first element of the list.
# The function should return a dictionary containing the matrix with the following format.

# predicted positive (1)	predicted negative (0)
# actual positive (1)	correct positive (1, 1)	false negative (1, 0)
# actual negative (0)	false positive (0, 1)	correct negative (0, 0)
# The keys to the dictionary are the indices: (0, 0), (0, 1), (1, 0), (1, 1).

# You can use the following function in your code:

# itertools.product(): this is to create a combination of all the labels.

import itertools
def confusion_matrix(ytrue, ypred, labels):
    # initialize the output dictionary
    output = {}
    # e.g.: if labels is [0,1], produce a combination of these labels
    # autogenerate the tuple of keys
    # keys = itertools.product(labels, repeat=2)
    
    # for j in keys:
    #     output[j] = 0
    
    # one-liner to show you're the best
    output = {key: 0 for key in itertools.product(labels, repeat=2)}
    
    # loop through each sample, find out the value of ytrue and ypred for that sample,
    # and add it to the count inside the confusion matrix output
    for idx in range(ytrue.shape[0]):
        output[(ytrue[idx,0], ypred[idx, 0])] += 1
    return output

# test cases
result = confusion_matrix(df_target_test.values, pred, [1,0])
print(result)
assert result == {(0, 0): 100, (0, 1): 1, (1, 0): 12, (1, 1): 57}

# CS7. Metrics: Write a function calc_accuracy() that takes in a Confusion Matrix array and output a dictionary with the following keys and values:

# accuracy: total number of correct predictions / total number of records
# sensitivity: total correct positive cases / total positive cases
# specificity: total true negatives / total negative cases
# precision: total of correct positive cases / total predicted positive cases

def calc_accuracy(cm):
    negneg, pospos, negpos, posneg = cm[(0, 0)], cm[(1, 1)], cm[(0, 1)], cm[(1, 0)]
    
    return {
        "accuracy": (negneg + pospos) / np.sum(list(cm.values())),
        "sensitivity": pospos / (pospos + posneg),
        "specificity": negneg / (negneg + negpos),
        "precision": pospos / (pospos + negpos),
    }

# test cases
ans = calc_accuracy(result)
expected = {'accuracy': 0.9235294117647059, 'sensitivity': 0.8260869565217391, 'specificity': 0.9900990099009901, 'precision': 0.9827586206896551}
assert np.isclose(ans['accuracy'], expected['accuracy'])
assert np.isclose(ans['sensitivity'], expected['sensitivity'])
assert np.isclose(ans['specificity'], expected['specificity'])
assert np.isclose(ans['precision'], expected['precision'])
 
# CS8. Optional: Redo the above tasks using Scikit Learn libraries. You will need to use the following:
# LogisticRegression
# train_test_split
# confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

columns = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean"]
# get the features and the columns
df_features, df_target = get_features_targets(df, columns, ["diagnosis"])

# replace target values with 0 and 1
df_target = replace_target(df_target, "diagnosis", {"M": 1, "B": 0})

# split data set using random_state = 100 and 30% test size
df_features_train, df_features_test, df_target_train, df_target_test = train_test_split(df_features, df_target, random_state=100, test_size=0.3, shuffle=True)

# change feature to numpy array and append column of 1s
feature = prepare_feature(df_features_train)
test = prepare_feature(df_features_test)

# change target to numpy array
target = prepare_target(df_target_train)

# create LogisticRegression object instance, use newton-cg solver
model = LogisticRegression(solver='newton-cg')

# build model
model.fit(feature, target)

print(feature.shape)
print(target.shape)
print(test.shape)
# get predicted value
pred = model.predict(test)

# calculate confusion matrix
cm = confusion_matrix(df_target_test, pred, labels=[1,0])
print(cm)

# test cases
expected = np.array([[58,  11], [6, 96]])
assert (cm == expected).all()

plt.scatter(df_features_test["radius_mean"], df_target_test)
plt.scatter(df_features_test["radius_mean"], pred)

plt.scatter(df_features_test["texture_mean"], df_target_test)
plt.scatter(df_features_test["texture_mean"], pred)

plt.scatter(df_features_test["perimeter_mean"], df_target_test)
plt.scatter(df_features_test["perimeter_mean"], pred)

