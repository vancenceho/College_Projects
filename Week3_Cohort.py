# WEEK 3 PROBLEM SET

# Exercise 1
def f(x):
    if x < 3:
        return 0
    else:
        return (x - 3) ** 2

print(f(4))

# Exercise 2
def g(x):
    if x == 0:
        return 0
    else: 
        return x + g(x - 1)

print(g(14))

# Exercise 3
def atoi(s):
    if len(s) == 0:
        return 0
    else:
        t = s[0:-1]
        return int(s[-1]) + 10 * atoi(t)

print(atoi("123"))

# Exercise 4 - Binary Search
def search(ls, start, end, target):
    if (end < start):
        return -1
    middle = (start + end) // 2 # can also use math library for floor()
    if (target < ls[middle]):
        return search(ls, start, middle - 1, target)
    elif (target == ls[middle]):
        return middle
    else: 
        return search(ls, middle + 1, end, target)

ls = [1, 3, 5, 6, 7, 9, 10]
print(search(ls, 0, 6, 9))

# CS1. You have implemented factorial using iteration in Problem Set 1. 
# Now, implement the factorial problem using a recursion. 
# The function should takes in an Integer input and returns and Integer output which is the factorial of the input. 
# Recall that:

# 𝑛!=𝑛×(𝑛−1)×(𝑛−2)×…×2×1
# You should consider the case when  𝑛 is zero and one as well.

codes = {
    'a': 'else',
    'b': 'def factorial_recursion(n):',
    'c': 'return n * factorial_recursion(n)',
    'd': 'return n * factorial_recursion(n - 1)',
    'e': 'return 0',
    'f': 'return 1',
    'g': 'if n == 1 or n == 0:'
}

# enter the sequence and use "sub" or "exit sub" as necessary
answer = ['b', 'sub', 'g', 'sub', 'f', 'exit sub', 'a', 'sub', 'd']

# recursion
def factorial_recursion(n):
    # base case
    if n == 1 or n == 0:
        return 1
    # recursive case
    else:
        return n * factorial_recursion(n - 1)

# test case
assert factorial_recursion(0) == 1
assert factorial_recursion(1) == 1
assert factorial_recursion(5) == 120
assert factorial_recursion(7) == 5040
assert factorial_recursion(11) == 39916800

# CS2. Helper Function: Write a function palindrome(s)to check if the string s is a Palindrome. 
# To do this, write another function is_palindrome(s, left, right) where left and right are indices 
# from the left and from the right to check the character in str. 
# Use recursion instead of iteration in this problem.

# palindrome()
# abstraction - the user of the function just needs
# to call palindrome()
# How you implement the algorithm, the user does not need to care.
def palindrome(s):
    end = len(s) - 1
    return is_palindrome(s, 0, end)

# is_palindrome() - helper function
def is_palindrome(s, start, end):
    if end <= start:
        return True
    elif s[start] != s[end]:
        return False
    else: 
        return is_palindrome(s, start + 1, end - 1)

def palindromeV2(s):
    if len(s) == 1 or len(s) == 0:
        return True
    else: 
        return s[0] == s[-1] and palindromeV2(s[1:-1])

# test case
assert not palindrome("moon") 
assert palindrome("noon") 
assert palindrome("a a") 
assert palindrome("ada") 
assert not palindrome("ad a")

# CS3. Towers of Hanoi: Write a function move_disks(n, from_tower, to_tower, aux_tower) 
# which returns an array of String for the movement of disks that solves the Towers of Hanoi problem.

# The first argument n is an Integer input that gives information on the number of disk.
# The second argument from_tower is a String which is the label of the origin tower.
# The third argument to_tower is a String which is the label of the destination tower.
# The last argument aux_tower is a String which is the label of the auxilary tower.

# Hanoi problem without an array of strings.
def hanoi(n, source, destination, auxilary, array):
    display = "Move disk {} from {} to {}."
    if (n == 1):
        array.append(display.format(1, source, destination))
    else:
        hanoi(n - 1, source, auxilary, destination, array)
        array.append(display.format(n, source, destination))
        hanoi(n - 1, auxilary, destination, source, array)

def move_disk(n, source, destination, auxilary):
    result = []
    hanoi(n, source, destination, auxilary, result)
    return result

ls = move_disk(3, "A", "B", "C")
print(ls)

# CS4. Merge Sort: Write functions to implement the Merge Sort algorithm. 
# The first function mergesort(array) should takes in an array of Integers in array. 
# The function should sort the array in place. 
# The second function merge(array, p, q, r) should implements the merge procedure. 
# This function takes in an array of Integers in. array, the starting index for the left array p, 
# the ending index for the left array q, and the ending index for the right array r. 
# You can use a helper function for your recursion if needed.

# merge(array, start, middle, end)
def merge(array, start, middle, end):
    # left_array stores the elements from the beginning of the index to the middle of the array.
    A_left = array[start:middle + 1]
    # n_left = the length of the left array.
    n_left = len(A_left)
    # initialize left arrow as 0 to the position of left array.
    left = 0
    # right_array stores the elements from the middle of the array to the end of the array.
    A_right = array[middle + 1:end + 1] 
    # n_right = the length of the right array.
    n_right = len(A_right)
    # initialize right arrow as 0 to the position of right array.
    right = 0
    # destination is referring to the position of the original array which consists of the total number of elements.
    dest = start

    # Comparing using the arrows and sorting. 
    while (left < n_left) and (right < n_right):
        if A_left[left] <= A_right[right]:
            array[dest] = A_left[left]
            left = left + 1
        else:
            array[dest] = A_right[right]
            right = right + 1
        dest = dest + 1
    # After comparing the 2 arrays left & right, if 1 of the arrays has been finished comparing and we still have 
    # left over indexes left over in an array, we thereafter just add the remaining the elements in the left over
    # array in the main array. 
    while (left < n_left):
        array[dest] = A_left[left]
        left = left + 1
        dest = dest + 1
    while (right < n_right):
        array[dest] = A_right[right]
        right = right + 1
        dest = dest + 1

# test case for merge(array, start, middle, end)
ls = [1, 7, 4, 5, 8, 9, 10]
merge(ls, 0, 1, 3)
print(ls)

# mergesort_recursive(array, start, end)
def mergesort_recursive(array, start, end):
    if (end - start > 0):
        middle = (start + end) // 2
        mergesort_recursive(array, start, middle)
        mergesort_recursive(array, middle + 1, end)
        merge(array, start, middle, end)

# test case for mergesort_recursive(array, start, end)
ls = [7, 1, 2, 5, 3, 9, 4, 8]
mergesort_recursive(ls, 0, 7)
print(ls)

# mergesort(array)
def mergesort(array):
    mergesort_recursive(array, 0, len(array) - 1)
    return array

ls = [7, 1, 2, 5, 3, 9, 4, 8]
print(mergesort(ls))