# WEEK 1 PROBLEM SET HOMEWORK

# HW0-a. Reorder the code to check if a string is a palindrome.

steps = { 'a': 'return False',
          'b': 'return True',
          'c': 'if s[idx] != s[len(s)-idx-1]:',
          'd': 'for idx in range(len(s)//2):',
          'e': 'def palindrome(s):'
        }

answer = ['e', 'sub', 'd', 'sub', 'c', 'sub', 'a', 'exit sub', 'exit sub', 'b']

# HW0-b. Write a function palindrome(s)to check if the string s is a Palindrome. 
# Use iteration for this solution.

# palindrome function
def palindrome(s):
    for idx in range(len(s) // 2):
        if (s[idx] != s[len(s) - idx - 1]):
            return False
    return True

# Test case
print(palindrome("moon"))
print(palindrome("noon"))
print(palindrome("a a"))
print(palindrome("ada"))
print(palindrome("ad a"))

assert not palindrome("moon") 
assert palindrome("noon") 
assert palindrome("a a") 
assert palindrome("ada") 
assert not palindrome("ad a")

# HW-1-a. Reorder the correct sequence of code to implement bubble sort algorithm version 2. 
# You can check the pseudocode in Bubble Sort and Insertion Sort.

# The function should ...
# modify/mutate an existing array into a sorted one
# return the number of comparisons made
# Hint: To count the number of comparisons made you can simply increment an integer variable (counter)
# right before the if statement.

codes = {'a': 'swapped = False',
         'b': '''count = 0
                 n = len(array)
                 swapped = True
              ''',
         'c': 'count += 1',
         'd': 'return count',
         'e': 'while swapped:',
         'f': 'array[idx - 1], array[idx] = array[idx], array[idx - 1]',
         'g': 'if array[idx-1] > array[idx]:',
         'h': 'for idx in range(1, n):',
         'i': 'def bubble_sort(array):',
         'j': 'swapped = True'
        }

# enter "sub" or "exit sub" as necessary
# you can also use code more than once
answer = ['i', 'sub', 'b', 'e', 'sub', 'a', 'h', 'sub', 'c', 'g', 'sub', 'f', 'j', 'exit sub', 'exit sub', 'exit sub', 'd']

# HW1-b. Create a function to implement Bubble Sort Algorithm version 2.
def bubble_sort(array):
    count = 0
    n = len(array)
    swapped = True
    while swapped == True:
        swapped = False
        for idx in range(1, n):
            count += 1
            if (array[idx - 1] > array[idx]):
                array[idx - 1], array[idx] = array[idx], array[idx - 1]
                swapped = True
    return count

# Test case
array = [10,3,8,47,1,0,-39,8,4,7,6,-5]
count = bubble_sort(array)
print(array, count)
assert array == [-39, -5, 0, 1, 3, 4, 6, 7, 8, 8, 10, 47]
assert count == 121

# HW2. The solution in HW1 can be improved!
# The n-th pass places the n-th largest element into its final place 
# (i.e. after 1 cycle, the (1st) largest element will be in its correct position).
# Hence, we can reduce subsequent iterations by only considering the other (unsorted) n-1 elements in the array.
# Implement the algorithm of Bubble Sort version 4 under Bubble Sort and Insertion Sort.

# bubbleSortV4
def bubble_sortV4(arrary):
    count = 0
    n = len(array)
    swapped = True
    while swapped == True:
        swapped = False
        new_n = 0
        for inner_index in range(1, n):
            count += 1
            if (array[inner_index - 1] > array[inner_index]):
                array[inner_index - 1], array[inner_index] = array[inner_index], array[inner_index - 1]
                swapped = True
                new_n = inner_index
        n = new_n
    return count

# Test case
array = [10,3,8,47,1,0,-39,8,4,7,6,-5]
count = bubble_sortV4(array)
print(array, count)
assert array == [-39, -5, 0, 1, 3, 4, 6, 7, 8, 8, 10, 47]
assert count == 66