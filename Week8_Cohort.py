# WEEK 8 PROBLEM SET - COHORT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display

# CS1. Reading Data: Read CSV file for Boston Housing prices.

# Task 1: Read the data set. 
# Hint: 
# Pandas read_csv
# Boston's housing data set (filename: housing_processed.csv):
# Boston's housing data description.

# Task 1
# read CSV file, replace the None
df = pd.read_csv("housing_processed.csv")

display(df)

# test cases
assert isinstance(df, pd.DataFrame)
assert df.shape == (506, 14)
assert df.columns[0] == 'CRIM' and df.columns[-1] == 'MEDV'

# Task 2: Display the number of rows and columns. 
# Hint: you can use df.shape to get the number of rows and columns.

# Task 2
# get the shape from the data frame, replace the None
shape = df.shape

# use the 'shape' variable to get the row and the column
# replace the None
row = shape[0]
col = shape[1]

print(shape)
print(row, col)

# test cases
assert shape == (506, 14)
assert row == 506
assert col == 14

# Task 3: Display the name of all the columns. 
# Hint:
# you can use df.columns to get the name of all the columns
# check the meaning of each column using the link above

# Task 3
# display column names, replace the None
names = df.columns

display(names)

# test cases
assert isinstance(names, pd.Index)
assert np.all(names == pd.Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT', 'MEDV']))

# Task 4: Do the following:
# Create a subset data set containing only the following columns: "RM", "DIS", "INDUS" for the features. Make sure it is of pd.DataFrame type.
# Create a subset data set containing only "MEDV" for the target. Make sure it is of pd.DataFrame type.

# Task 4
# Specify the columns you want to extract into a list
# replace the None
columns = ["RM", "DIS", "INDUS"]

# extract the respective columns from the data frame
# replace the None
df_feature = df[columns]
MEDV = df["MEDV"]
df_target = pd.DataFrame(MEDV)

print(type(df_feature))
print(type(df_target))

display(df_feature)
display(df_target)

# test cases
assert isinstance(df_feature, pd.DataFrame)
assert isinstance(df_target, pd.DataFrame)
assert df_feature.shape == (506, 3)
assert df_target.shape == (506, 1)
assert np.all(df_feature.columns == pd.Index(['RM', 'DIS', 'INDUS']))
assert df_target.columns == pd.Index(['MEDV'])

# CS2. Data Frame Operation:

# Reference:
# Indexing and Selecting Data
# Create separate and new data frame for the columns: "RM", "DIS", "INDUS", "MEDV" that satisfies each of the following condition:

# Task 1: All records with weighted distances to ﬁve Boston employment centers between 0 to 3.

# specify the columns of interest, replace the None
columns = ["RM", "DIS", "INDUS", "MEDV"]

# use conditions for row selector and column selector
# replace the None
df_1 = df.loc[(df["DIS"] >= 0) & (df["DIS"] <=3), columns]

print(df_1)

myplot = sns.pairplot(data=df_1)

# Task 2: All records with average number of room between 5 to 8.

# specify the columns of interest, replace the None
columns = ["RM", "DIS", "INDUS", "MEDV"]

# use conditions for row selector and column selector
# replace the None
df_2 = df.loc[(df["RM"] >= 5) & (df["RM"] <=8), columns]

myplot = sns.pairplot(data=df_2)

# Task 3: The first 15 records in the table.

# specify the columns of interest, replace the None
columns = ["RM", "DIS", "INDUS", "MEDV"]

# use conditions for row selector and column selector
# replace the None
df_3 = df.loc[0:14, columns]

display(df_3)

# Task 4: The last 15 records in the table.

# specify the columns of interest, replace the None
columns = ["RM", "DIS", "INDUS", "MEDV"]

# use conditions for row selector and column selector
df_specific = df[columns]
# replace the None
df_4 = df_specific.iloc[-15:, ]

display(df_4)

# Task 5: All records with even index numbers, i.e. index 0, 2, 4, ... .

# specify the columns of interest, replace the None
columns = ["RM", "DIS", "INDUS", "MEDV"]

# use conditions for row selector and column selector
# replace the None
df_specific = df[columns]
df_5 = df_specific.loc[::2]

display(df_5)

# CS3. Histogram and Box plot: Plot the histogram for the median value in $1000 for the Boston's housing price.

# Reference:
# Histogram
# Box plot

# Task 1: Plot the histogram with default bin values.

# plot histogram for MEDV, replace the None
myplot = sns.histplot(x="MEDV", data=df)

# set the x label, write the code below
myplot.set_xlabel("Median value for Boston housing price in $1000", fontsize=16)

# set the y label, write the code below
myplot.set_ylabel("Count", fontsize=16)

# Task 2: Plot the histogram with 5 bins only. 

# plot histogram for MEDV, replace the None
myplot = sns.histplot(x="MEDV", data=df, bins=5)

# set the x label, write the code below
myplot.set_xlabel("Median value of Boston housing price in $1000", fontsize=16)

# set the y label, write the code below
myplot.set_ylabel("Count", fontsize=16)

# Task 3: Plot the histogram with the following bin edges 0, 10, 20, 30, 40, 50.

# plot histogram for MEDV, replace the None
myplot = sns.histplot(x="MEDV", data=df, bins=[0, 10, 20, 30, 40, 50])

# set the x label, write the code below
myplot.set_xlabel("Median value for Boston housing price in $1000")

# set the y label, write the code below
myplot.set_ylabel("Count")

# Task 4: Plot the same data using a box plot in a horizontal manner.

# plot boxplot for MEDV, replace the None
myplot = sns.boxplot(x="MEDV", data=df)

# set the x label, write the code below
myplot.set_xlabel("Median value for Boston housing in $1000")

# CS4. Scatter plot: Do the following plots.
# Task 1: Display scatter plot of "RM" versus "MEDV". Hint:
# Use Seaborn default theme instead of matplotlib
# Seaborn scatter plot
# Use "RM" as x data
# Use "MEDV" as y data

# Set the theme parameter to Seaborn's default
sns.set()

# Task 5
# plot scatter plot, replace the None 
myplot = sns.scatterplot(x="RM", y="MEDV", data=df)

# Task 2: Display a scatter plot "weighted distances to ﬁve Boston employment centers" versus "Median value of owner-occupied homes in $1000s". 
# Use average number of rooms per dwelling as the hue data.

# plot using scatter plot
# replace the None
myplot = sns.scatterplot(x="DIS", y="MEDV", hue="RM", data=df)

# Task 3: Display a scatter plot "proportion of non-retail business acres per town" versus "Median value of owner-occupied homes in $1000s". 
# Use "proportion of residential land zoned for lots over 25,000 sq.ft." as the hue data.

# plot using scatter plot and use "hue"
# replace the None
myplot = sns.scatterplot(x="INDUS", y="MEDV", hue="ZN", data=df)

# CS5. Splitting Data Randomly: Create a function to split the Data Frame randomly. 

# The function should have the following arguments:

# df_feature: which is the data frame for the features.
# df_target: which is the data frame for the target.
# random_state: which is the seed used to split randomly.
# test_size: which is the fraction for the test data set (0 to 1), by default is set to 0.5
# The output of the function is a tuple of four items:

# df_feature_train: which is the train set for the features data frame
# df_feature_test: which is the test set for the features data frame
# df_target_train: which is the train set for the target data frame
# df_target_test: which is the test set for the target data frame
# Hint: Use numpy.random.choice()

def split_data(df_feature, df_target, random_state=None, test_size=0.5):
    indexes = df_feature.index
    df_feature_rows = df.shape[0]
    k = int(df_feature_rows * test_size)
    
    if random_state != None:
        np.random.seed(random_state)
        
    indexes_test = np.random.choice(indexes, k, replace=False)
    indexes_train = set(indexes) - set(indexes_test)
    
    df_feature_train = df_feature.loc[indexes_train]
    df_feature_test = df_feature.loc[indexes_test]
    df_target_train = df_target.loc[indexes_train]
    df_target_test = df_target.loc[indexes_test]
        
    return df_feature_train, df_feature_test, df_target_train, df_target_test
    
df_feature_train, df_feature_test, df_target_train, df_target_test = split_data(df_feature, df_target, random_state=100, test_size=0.3)
display(df_feature_train.describe())

# test cases
df_feature_train, df_feature_test, df_target_train, df_target_test = split_data(df_feature, df_target, random_state=100, test_size=0.3)

display(df_feature_train.describe())
display(df_feature_test.describe())
display(df_target_train.describe())
display(df_target_test.describe())

assert np.all(df_feature_train.count() == 355)
assert np.all(df_feature_test.count() == 151)
assert np.all(df_target_train.count() == 355)
assert np.all(df_target_test.count() == 151)

assert np.isclose(df_feature_train["RM"].mean(), 6.2968)
assert np.isclose(df_feature_test["DIS"].std(), 2.2369)
assert np.isclose(df_target_train["MEDV"].median(), 21.40)
assert np.isclose(df_target_test["MEDV"].median(), 20.90)

# CS6. Standardization: Write a function that takes in data frame where all the column are the features and normalize each column according to the following formula. 
# The function should also take in two optional arguments containing the means of each column and the standard deviation for each column. 
# If these lists are not provided, the means and the standard deviation are computed from the input data frame.

# 𝑛𝑜𝑟𝑚𝑎𝑙𝑖𝑧𝑒𝑑 = (𝑑𝑎𝑡𝑎−𝜇) / 𝜎
# where 𝜇 is the mean of the data and  𝜎 is the standard deviation of the data. 
# You need to normalize for each column respectively. 
# The function should return a new data frame as well as two lists of mean and standard deviation that is used to normalized each column.

# Use the following functions from Pandas:
# df.mean(axis=0): This is to calculate the mean along the index axis.
# df.std(axis=0): This is to calculate the standard deviation along the index axis.

# Note:
# Your function should be able to handle a numpy array as well as Panda's data frame. 
# Hint: use axis=0 argument when calculating mean and standard deviation.
# Your function should be able to take in a single data point given as numpy array.

# dfin = data frame
# columns_means = optional arguments containing the means of each columns
# columns_stds = optional arguments containing the std of each columns

def normalize_z(dfin, columns_means=None, columns_stds=None):
    if columns_means == None:
        columns_means = dfin.mean(axis=0)
    if columns_stds == None:
        columns_stds = dfin.std(axis=0)
    dfout = (dfin - columns_means) / columns_stds
    return dfout, columns_means, columns_stds

# test cases
data_norm, columns_means, columns_stds = normalize_z(df_feature)
print(data_norm,columns_means, columns_stds)
stats = data_norm.describe()
display(stats)
assert np.isclose(stats.loc["mean", "RM"], 0.0) and \
       np.isclose(stats.loc["std", "RM"], 1.0, atol=1e-3)
assert np.isclose(stats.loc["mean", "DIS"], 0.0) and \
       np.isclose(stats.loc["std", "DIS"], 1.0, atol=1e-3)
assert np.isclose(stats.loc["mean", "INDUS"], 0.0) and \
       np.isclose(stats.loc["std", "INDUS"], 1.0, atol=1e-3)
assert np.isclose(columns_means["RM"], 6.2846)
assert np.isclose(columns_stds["INDUS"], 6.8604)

data_norm,_, _ = normalize_z(df_feature.to_numpy())
assert np.isclose(data_norm[:,0].mean(), 0.0) and \
       np.isclose(data_norm[:,0].std(), 1.0, atol=1e-3)
assert np.isclose(data_norm[:,1].mean(), 0.0) and \
       np.isclose(data_norm[:,1].std(), 1.0, atol=1e-3)
assert np.isclose(data_norm[:,2].mean(), 0.0) and \
       np.isclose(data_norm[:,2].std(), 1.0, atol=1e-3)

input_1row = np.array([6.593, 2.4786, 11.93])
means = [6.284634, 3.795043, 11.136779]
stds = [0.702617, 2.105710, 6.860353]
data_norm,_, _ = normalize_z(input_1row, means, stds)
print(data_norm)
assert np.isclose(data_norm[0], 0.43888, atol=1e-3)
assert np.isclose(data_norm[1], -0.625, atol=1e-3)
assert np.isclose(data_norm[2], 0.1156, atol=1e-3)