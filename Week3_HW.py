# WEEK 3 PROBLEM SET - COHORT

# HW1. Fibonaci: Write a function to find the Fibonacci number given an index. These are the example of a the first few numbers in Fibonacci series.
# INDICES = 0 1 2 3 4 5 6 7 8 9
# THE SERIES = 0 1 1 2 3 5 8 13 21 34
# We can write that the Fibonacci number at index i is given by:
# 𝑓𝑖𝑏𝑜(𝑖)=𝑓𝑖𝑏𝑜(𝑖−1)+𝑓𝑖𝑏𝑜(𝑖−2)
# Use recursion for your implementation.

codes = {
    'a': 'return 0',
    'b': 'return 1',
    'c': 'return fibonacci(index - 1) + fibonacci(index - 2)',
    'd': 'return fibonacci(index) + fibonacci(index - 1)',
    'e': 'def fibonacci(index):',
    'f': 'else',
    'g': 'elif index == 1:',
    'h': 'if index == 0:'
}

# Enter your sequence of code and use "sub" or "exit sub" as necessary
answer = ['e', 'sub', 'h', 'sub', 'a', 'exit sub', 'g', 'sub', 'b', 'exit sub', 'f', 'sub', 'c']

# fibonacci(index)
def fibonacci(index):
    if index == 0:
        return 0
    elif index == 1:
        return 1
    else:
        return fibonacci(index - 1) + fibonacci(index - 2)
    
# test case
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(3) == 2
assert fibonacci(7) == 13
assert fibonacci(9) == 34

# HW2. Max-Heapify: Recall the algorithm for max-heapify in restoring the max-heap property of a binary heap tree. 
# In the previous implementation, we used iteration. Implement the same algorithm using recursion.
# Reorder the sequence of the code to implement max heapify function.

codes = {
        'a': 'largest = index',
        'b': 'largest = left_idx',
        'c': 'largest = right_idx',
        'd': 'def max_heapify(array, index, heap_size):',
        'e': '''left_idx = left_of(index)
               right_idx = right_of(index)''',
        'f': 'else:',
        'g': 'max_heapify(array, largest, heap_size)',
        'h': 'if largest != index:',
        'i': 'if (left_idx < heap_size) and (array[left_idx] > array[index]):',
        'j': 'if (right_idx < heap_size) and (array[right_idx] > array[largest]):',
        'k': 'array[index], array[largest] = array[largest], array[index]'
}

# enter your code sequence below and use "sub" or "exit sub" as necessary.
answer = ['d', 'sub', 'e', 'i', 'sub', 'b', 'exit sub', 'f', 'sub', 'a', 
          'exit sub', 'j', 'sub', 'c', 'exit sub', 'h', 'sub', 'k', 'g']

def left_of(index):
    return (index * 2) + 1

def right_of(index):
    return (index + 1) * 2

def max_heapify(array, index, heap_size):
    # initialize left_idx and right_idx
    left_idx = left_of(index)
    right_idx = right_of(index)
    if (left_idx < heap_size) and (array[left_idx] > array[index]):
        largest = left_idx
    else:
        largest = index
    if (right_idx < heap_size) and (array[right_idx] > array[largest]):
            largest = right_idx
    if largest != index:
        array[index], array[largest] = array[largest], array[index]
        max_heapify(array, largest, heap_size)

# test case
result = [16, 4, 10, 14, 7, 9, 3, 2, 8, 1]
max_heapify(result, 1, len(result))
print(result)
assert result == [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
result = [4, 1, 10, 14, 16, 9, 3, 2, 8, 7]
max_heapify(result, 1, len(result))
print(result)
assert result == [4, 16, 10, 14, 7, 9, 3, 2, 8, 1]

# HW3. String Permutation: Write two functions to return an 
# array of all the permutations of a string. 
# For example, if the input is abc, the output should be output = ["abc", "acb", "bac", "bca", "cab", "cba"]
# The first function only has one argument which is the input string, i.e. permutate(s). 
# The second function is the recursive function with two arguments String 1 (str1) and and String 2 (str2). 
# The first function calls the second method permutate("", s) at the beginning. 
# The second function should use a loop to move a character from str2 to str1 and recursively invokes it with a new str1 and str2.