# WEEK 8 PROBLEM SET - HOMEWORK

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display

# HW1. Breast Cancer Data: Read Breast Cancer data using Pandas library:

# Task 1: Read the file: breast_cancer_data.csv:
# Download the CSV file
# Read Description of Data
# use comma "," as field separator
# use "utf-8" as encoding field

# Read file, replace the None
df = pd.read_csv("breast_cancer_data.csv", sep=None, encoding=None)

display(df)

# test cases
assert isinstance(df, pd.DataFrame)
assert df.shape == (569, 32)
assert df.columns[0] == 'id' and df.columns[-1] == 'fractal_dimension_worst'

# Task 2: Find the number of rows and columns.

# get the shape
shape = df.shape

# get rows and columns from shape
row = shape[0]
col = shape[1]

print(row,col)

# test cases
assert shape == (569, 32)
assert row == 569
assert col == 32

# Task 3: Find the name of all the columns.

# display the name of all the columns
names = df.columns

print(names)

# test cases
assert isinstance(names, pd.Index)
assert np.all(names == pd.Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']))

# Task 4: Create a subset data set containing only the following features:
# radius (mean of distances from center to points on the perimeter)
# texture (standard deviation of gray-scale values)
# perimeter
# area
# smoothness (local variation in radius lengths)
# concavity (severity of concave portions of the contour)
# Make sure the data type is pd.DataFrame.

# set the name of the columnbs for the subset of data
columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
           'concavity_mean']

# extract the columns, replace the None
df_features = df[columns]

df_features

# Task 5: Create a subset data set containing only the target from the column "diagnosis".
# Make sure the data type is pd.DataFrame.

# extract target
df_target = pd.DataFrame(df['diagnosis'])

df_target

# test cases
assert isinstance(df_target, pd.DataFrame)
assert df_target.shape == (569, 1)
assert df_target.columns[0] == 'diagnosis'

# Task 6: Create a new Data Frame from the column "diagnosis", called "diagnosis_int" which is the integer representation of the column diagnosis. 
# Copy the column into the original data frame so that it has a copy.
# if the value in "diagnosis" column is "M", the value in the column "diagnosis_int" should be set to 1
# otherwise, it should be set to 0
# Hint: use .apply() method.

# creating diagnosis_int
df_target["diagnosis_int"] = df["diagnosis"].apply(lambda x: 1 if x == "M" else 0)

# copy the new column into the original data frame
df["diagnosis_int"] = df_target["diagnosis_int"]

display(df_target)
display(df["diagnosis_int"])

# test cases
assert isinstance(df_target, pd.DataFrame) and isinstance(df, pd.DataFrame)
assert df_target.shape == (569, 2)
assert np.all(df_target.columns == ["diagnosis", "diagnosis_int"])
assert "diagnosis_int" in df.columns

# Task 7: Use scatter plot to see the relationship between "radius_mean" and "diagnosis_int".
# Use the "diagnosis" column as the hue for the scatter plot.
# Label the x axis as "Mean of Cell Radius"
# Label the y axis as "1: Malignant, 0: Benign"

# set the default theme to use Seaborn
sns.set()

# display using scatter plot
myplot = sns.scatterplot(x="radius_mean", y="diagnosis_int", hue="diagnosis", data=df)

# set the x label
myplot.set_xlabel("Mean of Cell Radius", fontsize=16)

# set the y label
myplot.set_ylabel("1:Malignant, 0:Benign", fontsize=16)

# Task 8: Use scatter plot to see the relationship between "concavity_mean" and "diagnosis_int".
# Use the "diagnosis" column as the hue for the scatter plot.
# Label the x axis as "Mean of Cell Concavity"
# Label the y axis as "1: Malignant, 0: Benign"

# set the default theme to use Seaborn
sns.set()

# display using scatter plot
myplot = sns.scatterplot(x="concavity_mean", y="diagnosis_int", hue="diagnosis", data=df)

# set the x label
myplot.set_xlabel("Mean of Cell Concavity", fontsize=16)

# set the y label
myplot.set_ylabel("1:Malignant, 0:Benign", fontsize=16)

# HW2. Count Plot: Create a function to count how many records are diagnosed as Malignant and Benign. 
# The function should return a tuple: (n_malignant, n_benign), where n_malignant is the number of records diagnosed as Malignant cell and n_benign is the number of records 
# diagnosed as Benign.

# Use Count plot to verify the answer.

# Reference:
# Count Plot

def count_cell(target):
    
    n_malignant = sum(target == "M")
    n_benign = sum(target == "B")

    return (n_malignant, n_benign)
        
# test cases
result = count_cell(df_target["diagnosis"])
assert result == (212, 357)

# write the code to plot the count of the two classes
myplot = sns.countplot(data=df, x="diagnosis")

# HW3. Normalization: Create a function that takes in Data Frame as the input and returns the normalized Data Frame as the output. 
# Each column is normalized separately using min-max normalization. The function should return a new data frame instead of modifying the input data frame. 
# The function should also take in two optional arguments containing the minimums and the maximus of each column. 
# If these lists are not provided, the minimums and the maximums are computed from the input data frame.

# 𝑛𝑜𝑟𝑚𝑎𝑙𝑖𝑧𝑒𝑑 = (𝑑𝑎𝑡𝑎−𝑚𝑖𝑛) / (𝑚𝑎𝑥−𝑚𝑖𝑛)
 
# Use the following functions from Pandas or Numpy:
# df.copy(): This is to create a new copy of the data frame.
# df.min(axis=0): This is to get the minimum along the index axis.
# df.max(axis=0): This is to get the maximum along the index axis.\

# Note:
# Your function should be able to handle a numpy array as well as Panda's data frame.
# Your function should be able to take in a single datqa point given as a numpy array.

display(df_features)
# dfin = data frame
# column_mins = optional arguments containing mins. of each column
# column_maxs = optional arguments containing maxs. of each column

def normalize_minmax(dfin, columns_mins=None, columns_maxs=None):
    if columns_mins is None:
        columns_mins = dfin.min(axis=0)
    if columns_maxs is None:
        columns_maxs = dfin.max(axis=0)
    dfout = (dfin - columns_mins) / (np.array(columns_maxs) - np.array(columns_mins))
    return dfout, columns_mins, columns_maxs

# test cases
data_norm, columns_mins, columns_maxs = normalize_minmax(df_features)
print(columns_mins, columns_maxs)
stats = data_norm.describe()
display(stats)
assert stats.loc["max", "radius_mean"] == 1.0 and \
       stats.loc["min", "radius_mean"] == 0 and \
       np.isclose(stats.loc["mean", "radius_mean"], 0.338222)
assert stats.loc["max", "texture_mean"] == 1.0 and \
       stats.loc["min", "texture_mean"] == 0 and\
       np.isclose(stats.loc["mean", "texture_mean"], 0.323965)
assert np.isclose(columns_mins["radius_mean"], 6.981)
assert np.isclose(columns_mins["perimeter_mean"], 43.79)
assert np.isclose(columns_mins["smoothness_mean"], 0.05263)
assert np.isclose(columns_maxs["radius_mean"], 28.11)
assert np.isclose(columns_maxs["perimeter_mean"], 188.5)
assert np.isclose(columns_maxs["smoothness_mean"], 0.1634)

data_norm,_,_ = normalize_minmax(df_features.to_numpy())
assert data_norm[:,0].max() == 1.0 and \
       data_norm[:,0].min() == 0 and \
       np.isclose(data_norm[:,0].mean(), 0.338222)
assert data_norm[:,1].max() == 1.0 and \
       data_norm[:,1].min() == 0 and\
       np.isclose(data_norm[:,1].mean(), 0.323965)

input_1row = np.array([21.56, 22.39, 142.00, 1479.0, 0.11100, 0.24390])
mins = [6.98100, 9.71000, 43.79000, 143.50000, 0.05263, 0.00000]
maxs = [28.1100, 39.2800, 188.5000, 2501.0000, 0.1634, 0.4268]
data_norm, _, _ = normalize_minmax(input_1row, mins, maxs)
print(data_norm)
assert np.isclose(data_norm[0], 0.689999)
assert np.isclose(data_norm[1], 0.428813)
assert np.isclose(data_norm[2], 0.678667)

# HW4. Splitting the Data: Use the function to split the breast cancer data set into a training data set and a testing data set. 
# Use random_state=100 and test_size=0.3 and the normalized features data set from the previous exercise.

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

# call normalize_minmax() to normalize df_features
data_norm, df_mins, df_maxs = normalize_minmax(df_features)
#print(data_norm, df_mins, df_maxs)

# call split_data() with seed of 100 and test size of 30%
datasets_tupple = split_data(data_norm, df_target, random_state=100, test_size=0.3)
print(datasets_tupple)

# test cases
df_features_train = datasets_tupple[0]
df_features_test = datasets_tupple[1]
df_target_train = datasets_tupple[2]
df_target_test = datasets_tupple[3]

display(df_features_train.describe())
display(df_features_test.describe())
display(df_target_train.describe())
display(df_target_test.describe())

assert isinstance(df_features_train, pd.DataFrame)
assert isinstance(df_features_test, pd.DataFrame)
assert isinstance(df_target_train, pd.DataFrame)
assert isinstance(df_target_test, pd.DataFrame)

assert df_features_train.shape == (399, 6)
assert df_features_test.shape == (170, 6)
assert df_target_train.shape == (399, 2)
assert df_target_test.shape == (170, 2)

assert np.isclose(df_features_train.mean().mean(), 0.29844) 
assert np.isclose(df_features_test.mean().mean(), 0.311966) 
assert np.isclose(df_target_train.mean().mean(), 0.358396) 
assert np.isclose(df_target_test.mean().mean(), 0.40588) 

# HW5. Pair Plot: Use pair plot to find out the relationship between different columns in df_features. 
# Ensure that similar relationship exists in both the training and the test datasets.

# write your code below to plot for df_features
myplot = sns.pairplot(data=df_features)

# write your code below to plot for df_features_train
myplot = sns.pairplot(data=df_features_train)

# write your code below to plot for df_features_test
myplot = sns.pairplot(data=df_features_test)