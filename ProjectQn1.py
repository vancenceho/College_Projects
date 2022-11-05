def sequentialSearch(list,number):
    for i in range(len(list)):
        if list[i] == number:
            print("Found it!")
            return
    print("Not found!")

list = [20, 153, 34, 14, 220, 180, 83, 160, 90, 244, 60]
sequentialSearch(list, 165)

# def binarySearch(list,value):
#     low = 0 
#     high = len(list) - 1
#     while low <= high:
#         mid = (low+high) // 2
#         if list[mid] < value:
#             low = mid + 1 
#         elif value < list[mid]:
#             high = mid - 1 
#         else:
#             return mid
#     return None

# list = [20, 153, 34, 14, 220, 180, 83, 160, 90, 244, 60]
# list.sort()
# binarySearch(list, 165)