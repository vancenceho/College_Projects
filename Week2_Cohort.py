# WEEK 2 PROBLEM SET

# CS1 : Binary Heap
# Write the following functions:

# parent_of(index): returns the index of node's parent
# left_of(index): returns the index of node's left child
# right_of(index): returns the index of node's right child
# max_child(array, index, heap_size): returns the index of node's largest child. You can assume that the node has at least one child.

# Hint:
# index starts from 0.
# You can refer to the pseudocode in Binary Heap and Heapsort.
# When finding the index of the largest child, consider the following cases:
# when the node only has one child
# when the node has two children

# parent_of(index) : returns the index of node's parent
def parent_of(index):
    return int((index - 1) // 2)

# test case 
assert parent_of(1) == 0
assert parent_of(2) == 0
assert parent_of(5) == 2
assert parent_of(6) == 2

# left_of(index) : returns the index of node's left child
def left_of(index):
    return (index * 2) + 1

# test case
assert left_of(0) == 1
assert left_of(1) == 3
assert left_of(3) == 7
assert left_of(6) == 13

# right_of(index) : returns the index of node's right child
def right_of(index):
    return (index + 1) * 2

# test case
assert right_of(0) == 2
assert right_of(1) == 4
assert right_of(3) == 8
assert right_of(5) == 12

# Re-order the code for returning the index of the max child from a given node index and some particular heap size.
# Input:
# array: containing the binary heap data
# index: node index which we want to find the max child
# heap_size: the number of elements in the heap

# Note that the array size can be bigger than the heap size. In this case, ignore the rest of the element as NOT part of the heap.
# Output:
# index of the max child
codes = {
    'a': 'return right_of(index)',
    'b': 'return left_of(index)',
    'c': 'if right_of(index) >= heap_size:',
    'd': 'else:',
    'e': 'def max_child(array, index, heap_size):',
    'f': 'if array[left_of(index)] > array[right_of(index)]:'
}

# Fill in the sequence and add "sub" or "exit sub" as needed.
answers = ['e', 'sub', 'c', 'sub', 'b', 'exit sub', 'd', 'sub', 'f', 'sub', 'b', 'exit sub', 'd', 'sub', 'a']

def max_child(array, index, heap_size):
     if right_of(index) >= heap_size:
         return left_of(index)
     else:
          if array[left_of(index)] > array[right_of(index)]:
              return left_of(index)
          else:
              return right_of(index)

# test case
maxheap = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
print(max_child(maxheap, 0, len(maxheap)))
print(max_child(maxheap, 2, len(maxheap)))
print(max_child(maxheap, 3, len(maxheap)))
print(max_child(maxheap, 1, len(maxheap)))
print(max_child(maxheap, 4, len(maxheap)))

# CS2.Binary Heap : Write two functions. 
# - `max_heapify(array, index, size)`: that moves the node down so as to satisfy the heap property. 
# The first argument is the array that contains the heap. 
# The second argument is an integer index where to start the process of heapifying. 
# The third argument is the size of the heap in the array. 
# This argument will be useful in heapsort algorithm where we take out the elements in the array from the heap. 
# Hint: You should make use of `size` argument to determine the last element of the heap in the array rather than `len(array)`.

# - `build_max_heap(heap)`: that builds the max heap from any array. 
# This function should make use of `max_heapify()` in its definition.
# Hint: You can refer to the pseudocode in [Binary Heap and Heapsort](https://data-driven-world.github.io/2023/notes/category/week-2-analysing-programs) for the above functions.

# max_heapify(array, index, size) : that moves the node down so as to satisfy the heap property.
def max_heapify(array, index, size):
    # current index starting from input i.
    current_i = index
    swapped = True
    while (left_of(current_i) < size) and swapped == True:
        swapped = False
        # get the index of the largest child of the node current_i.
        max_child_i = max_child(array, current_i, size)
        if array[max_child_i] > array[current_i]:
            array[max_child_i], array[current_i] = array[current_i], array[max_child_i]
            swapped = True
        # move to the index of the largest child.
        current_i = max_child_i

# build_max_heap(heap) : that builds the max heap from any array.
def build_max_heap(array):
    n = len(array)
    # start from the middle or non-leaf node.
    starting_index = int(n / 2) - 1
    for current_index in range(starting_index, 0):
        max_heapify(array, current_index)

# CS3. Heapsort: Implement heapsort algorithm following the pseudocode in Binary Heap and Heapsort.

# heapsort(array)
def heapsort(array):
    build_max_heap(array)
    # index of the last element in the heap.
    heap_end_pos = len(array) - 1
    while (heap_end_pos > 0):
        array[0], array[heap_end_pos] = array[heap_end_pos], array[0]
        # reduce heap size.
        heap_end_pos = heap_end_pos - 1
        max_heapify(array, 0, 0)

# CS4. Measure computational time of Python's built-in sort function by filling the template below. 
# Hint: You will need the function gen_random_int() from Week 01 Problem Set.
# Use sorted(list) function of Python's list See Python's Sorting HOW TO Documentation.