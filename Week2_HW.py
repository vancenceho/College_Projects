# WEEK 2 PROBLEM SET HW

# HW1. Min-Heap: Write the following function to implement min-heap. 
# A min-heap is a binary heap that satisfies the min-heap property. 
# This property can be described as:

# For all nodes except the root:

# A[left(i)] >= A[i]
# A[right(i)] >= A[i]

# - min_child(heap, index): which returns the index of the node's smallest child. 
#       The node you are referring has index of value index
# - min_heapify(array, index, size): which moves the node at index down the tree so as 
#       to satisfy the min-heap property. 
#       The argument index is the index of the node you want to start moving down in the array. 
#       The argument size is the size of the heap. This size may be the same or less than 
#       the number of elements in the array. Hint: You may need the min_child() function.
# - build_min_heap(array): which build a min-heap from an arbitrary array of integers. 
#       This function should make use of min_heapify(array, index).

def left_of(index):
    return (index * 2) + 1

def right_of(index):
    return (index + 1) * 2

def min_child(heap, index, heap_size):
    if right_of(index) >= heap_size:
        return left_of(index)
    if heap[left_of(index)] < heap[right_of(index)]:
        return left_of(index)
    else:
        right_of(index)
        
# test case 
minheap = [1, 2, 4, 3, 9, 7, 8, 10, 14, 16]
assert min_child(minheap, 0, len(minheap)) == 1
assert min_child(minheap, 2, len(minheap)) == 5
assert min_child(minheap, 3, len(minheap)) == 7
assert min_child(minheap, 1, len(minheap)) == 3
assert min_child(minheap, 4, len(minheap)) == 9

# Reorder the code for min heapify function. The function takes in the array, 
# the index indicating which node to start the heapifying process and the size of the heap. 
# It should modify the input array in such a way that it satisfies the min-heap property 
# starting from the index node.

# Input:

# array: binary tree to be restored to satisfy the min-heap property
# index: index of the node where the min-heapify property should be satisfied on its subtree
# size: number of elements of the binary tree
# Output:

# None, the function should modify the array in-place.

codes = {
    'a': 'array[min_child_idx], array[cur_idx] = array[cur_idx], array[min_child_idx]',
    'b': 'cur_idx = index',
    'c': 'cur_idx = min_child_idx',
    'd': 'min_child_idx = min_child(array, cur_idx, size)',
    'e': 'def min_heapify(array, index, size):',
    'f': 'if min_child_idx < size and array[min_child_idx] < array[cur_idx]:',
    'g': 'while cur_idx < size:'
}

# Enter the key sequence and add "sub" or "exit sub" as necessary
answers = ['e', 'sub', 'b', 'g', 'sub', 'd', 'f', 'sub', 'a', 'exit sub', 'c']

def min_heapify(array, index, size):
    current_index = index
    while current_index < size:
        min_child_i = min_child(array, current_index, size)
        if min_child_i < size and array[min_child_i] < array[current_index]:
            array[min_child_i], array[current_index] = array[current_index], array[min_child_i]
        current_index = min_child_i

# test case
array = [1, 3, 4, 2, 9, 7, 8, 10, 14, 16]
min_heapify(array, 1, len(array))
assert array == [1, 2, 4, 3, 9, 7, 8, 10, 14, 16]

def build_min_heap(array):
    n = len(array)
    starting_index = int(n / 2) - 1
    for current_index in range(starting_index, -1, -1):
        min_heapify(array, current_index, n)

# test case
array = [1, 3, 4, 2, 9, 7, 8, 10, 14, 16]
min_heapify(array, 1, len(array))
assert array == [1, 2, 4, 3, 9, 7, 8, 10, 14, 16]

# HW2. Heapsort: Implement heapsort that makes use of min-heap instead of max-heap. 
# This function returns a new array. The strategy is similar to max-heap, 
# but we will use a new array to store the sorted output. 
# Take note of the hints below:

# The top of the min-heap is always the smallest. 
# You can take this element and put it into the output array.
# To find the next minimum, take the last element of the heap and put it into the first element 
# of the array. Now, the tree is no longer a min-heap. 
# Use min_heapify() to restore the min-heap property. T
# This will result in a mean-heap where the first element of the array is the next minimum. 
# You can then take out the top of the min-heap and put it into the output array.
# Reduce the heap size as you go.
# Return the new output array.

import random

def gen_random_int(number, seed):
    result = []
    for i in range(number):
        result.append(i)
    random.seed(seed)
    random.shuffle(result)
    return result

def heapsort(array):
    result = []
    build_min_heap(array)
    heapsize = len(array)
    while (heapsize > 0):
        # swap the first element with the last element in the heap.
        # min_heapify()
        # pop the first element of the array and append to result
        # heapsize -= 1
        # return array
        result.append(array[0])
        heapsize -= 1
        min_heapify(array, 0, heapsize)
    return array

# test case
array = gen_random_int(10, 100)
result = heapsort(array)
print(result)
#result = heapsort(array)
#assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
