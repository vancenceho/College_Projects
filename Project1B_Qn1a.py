# Code for binary search modified with the use of recursion.
# Where value = to the number that you are looking for,
#       start = to the starting position of the list and
#       end = to the ending position of the list. 
def binarySearch(list, value, start, end):
    # If the first value is less than or equal to the last value:
    # which it always will because the list must and will be sorted before the function can function properly, 
    if start <= end:
        # mid is the calculation to find the middle element in the list.
        mid = (start + end) // 2

        # Check if the middle element in the list == to value you are finding,
        # If the middle element of the list is larger than value wanted,
        if list[mid] > value:
            # the position of end would be positioned to the position where the mid position -1.
            end = mid - 1
        # Else if the middle element of the list is smaller than the value wanted,
        elif list[mid] < value:
            # the position of the start would be positioned to the position where the mid position +1.
            start = mid + 1

        # Else, when the mid element is == to the value,
        else:
            # return the position of mid.
            return mid

        # Returns the new position of start & end to repeat the function in order
        # to find the position of the value wanted.
        return binarySearch(list, value, start, end)

    # Returns None when value wanted is not in the list.
    return None


list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# Sort the list to not only meet the requirement of binary search but also,
# in order for binary search to function properly and correctly. 
list.sort()
# To display the sorted list.
print("Sorted list = ", list)
# To display the position (if any) of the value wanted in the list.
print("The position is at", str(binarySearch(list, 10, 0, len(list) - 1)) + ".")
