# Code for insertion sort modified with the use of recursion.
# Where index == to the length of list.
def insertionSort(list, index):
    # If length of list is larger than 1,
    if index > 1:
        # Recursion for the next insertion where it will be -1 of the previous insertion.
        insertionSort(list, index - 1)

    # Value of index (e.g. 1) will be = to object called currentValue.
    currentValue = list[index]
    # Index of next descending value (e.g. 1) will be equals to its position.
    position = index
    
    # While position (e.g. 1) is more than 0 and value of position( e.g. 1-1=0) 72 is more than currentValue = 56
    # run the following code.
    while position > 0 and list[position - 1] > currentValue:
        # Value of position (e.g. 1) = 56 will be equals to position (e.g. 0). 
        list[position] = list[position - 1]
        position = position - 1

    list[position] = currentValue

list = [72, 56, 93, 8, 22, 41, 88, 23, 60]
insertionSort(list, len(list) - 1)
print(list)