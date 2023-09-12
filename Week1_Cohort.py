# WEEK 1 PROBLEM SET

# CS0
# Implement factorial problem using iteration. The function should takes in an Integer input and returns 
# and Integer output which is the factorial of the input. Recall that:
# 𝑛!=𝑛×(𝑛−1)×(𝑛−2)×…×2×1
# You should consider the case when  𝑛 is zero and one as well.

# iteration
def factorial_iteration(n):
    result = None
    if n == 0 or n == 1:
        return 1
    elif n > 1:
        return (n * factorial_iteration(n - 1))
    else:
        print("The factorial does not exist. Negative Numbers.")

# Test case
print(factorial_iteration(0))
print(factorial_iteration(1))
print(factorial_iteration(5))
print(factorial_iteration(7))
print(factorial_iteration(11))
assert factorial_iteration(0) == 1
assert factorial_iteration(1) == 1
assert factorial_iteration(5) == 120
assert factorial_iteration(7) == 5040
assert factorial_iteration(11) == 39916800

# CS1-a. Enter the steps of the following pseudocode as a list to implement Bubble Sort version 1. 
# Enter your answer by creating a list with the right sequence. 
# Add "sub" if the subsequent steps are steps "under" (sub) the previous step. 
# Use "exit sub" before steps that is not under the overarching steps. 
# See example below for the first few steps of Bubble Sort pseudocode and enter the remaining steps.
steps = {'1': 'n = length of array',
         '2': 'For outer_index from 1 to n-1, do:',
         '2.1': 'For inner_index from 1 to n-1, do:',
         '2.1.1': 'first_number = array[inner_index - 1]',
         '2.1.2': 'second_number = array[inner_index]',
         '2.1.3': 'if first_number > second_number, do:',
         '2.1.3.1': 'swap(first_number, second_number)}'}

# replace the None with the right key from the steps variable above or "sub"/"exit sub"
answer = ['1', '2', 'sub', '2.1', 'sub', '2.1.1', '2.1.2', '2.1.3', 'sub', '2.1.3.1']

# CS1-b. 
# Create a function that implements Bubble Sort version 1 (from your Notes) to sort an array of integers. 
# The function should sort the input array in place. Refer to the above pseudocode.

# bubbleSort
def bubbleSort(arr):
    n = len(arr)
    for outer_index in range(1, n):
        for inner_index in range(1, n):
            first_number = arr[inner_index - 1]
            second_number = arr[inner_index]
            if first_number > second_number:
                arr[inner_index - 1], arr[inner_index] = arr[inner_index], arr[inner_index - 1]


# Test case
array = [10, 3, 8, 47, 1, 0, -39, 8, 4, 7, 6, -5]
bubbleSort(array)
print(array)
assert array == [-39, -5, 0, 1, 3, 4, 6, 7, 8, 8, 10, 47]

# CS2. Modify CS1, so that it returns the number of comparisons that are performed.
# Hint: To count the number of comparisons, you can create an integer variable which you increment just 
# before the if statement.

# bubbleSortv2
def bubbleSort2(arr):
    count = 0
    n = len(arr)
    for outer_index in range(1, n):
        for inner_index in range(1, n):
            first_number = arr[inner_index - 1]
            second_number = arr[inner_index]
            count += 1
            if first_number > second_number:
                arr[inner_index - 1], arr[inner_index] = arr[inner_index], arr[inner_index - 1]
    return count

# Test case
array = [10,3,8,47,1,0,-39,8,4,7,6,-5]
count = bubbleSort2(array)
print(array, count)
assert array == [-39, -5, 0, 1, 3, 4, 6, 7, 8, 8, 10, 47]
assert count == 121

#CS3-a. Sequence the steps below to have a correct pseudocode for Insertion sort.
steps = {'a': 'For outer_index in Range(from 1 to n-1), do:',
         'b': 'As long as (inner_index > 0) AND (array[inner_index] < array[inner_index - 1]), do:',
         'c': 'inner_index = inner_index - 1',
         'd': 'n = length of array',
         'e': 'inner_index = outer_index',
         'f': 'swap(array[inner_index - 1], array[inner_index])'}

# add "sub" or "exit sub" as necessary
answer = ['d', 'a', 'sub', 'e', 'b', 'sub', 'f', 'c']

# CS3-b. Create a function that implements Insertion Sort to sort an array of floats. 
# The function should sort the input array in place.

# insertionSort
def insertionSort(arr):
    n = len(arr)
    for outer_index in range(1, n):
        inner_index = outer_index
        while (inner_index > 0) and (arr[inner_index] < arr[inner_index - 1]):
            arr[inner_index - 1], arr[inner_index] = arr[inner_index], arr[inner_index - 1]
            inner_index = inner_index - 1

# Test case
array = [10.3,3.8,8.4,47.1,1.0,0,-39.8,8.4,4.7,7.6,-6.5,-5.0]
insertionSort(array)
print(array)
assert array == [-39.8, -6.5, -5.0, 0.0, 1.0, 3.8, 4.7, 7.6, 8.4, 8.4, 10.3, 47.1]

# CS4. Write a function gen_random_int that generates a list of integers from 0 to n - 1 
# with its sequence randomly shuffled. 
# The function should take in an integer n, denoting the number of integers to be generated.
# Hint:You can use random.shuffle(mylist) to shuffle the elements of a list called mylist.
# Refer to Python's Random module for more details on how to use shuffle().

import random

def gen_random_int(number, seed):
    result = None
    mylist = []
    for i in range(number):
        mylist.append(i)
    random.seed(seed)
    random.shuffle(mylist)
    return mylist

# Test case
output = gen_random_int(10, 100)
print(output)