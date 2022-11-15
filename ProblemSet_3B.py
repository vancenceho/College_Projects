# Problem 3B - Median
def median(ls):
    ls.sort()
    n = len(ls)
    if n % 2 != 0:
        median = ls[int((n - 1) / 2)]
    else:
        median = (ls[int(n / 2)] + ls[int(n / 2) - 1]) / 2 
    return median

a = median([5, 7, 3, 8, 6])
print(a)
b = median([5, 7, 3, 8, 6, 9])
print(b)

# Problem 3.11 - Middle
def middle_list(ls):
    ls = ls[1:-1]
    return ls

a = middle_list([1, 9])
print(a)
b = middle_list([1, 9, 4])
print(b)

# Problem 3.12 - Swapping of Elements
def swap_elements(ls, index1, index2):
    newls = ls.copy()
    max_pos_index = len(newls) - 1
    if index1 > max_pos_index or index2 > max_pos_index:
        return None
    else:
        temp = newls[index1]
        newls[index1] = newls[index2]
        newls[index2] = temp
        return newls

ls = [3, 6, 8, 7]
newls1 = swap_elements(ls, 2, 3)
print(newls1)   # [3, 6, 7, 8]
print(newls1 is ls) # False
newls2 = swap_elements(ls, -3, -1)
print(newls2)   # [3, 7, 8, 6]
result = swap_elements(ls, 3, 4)
print(result)   # None

# Problem 3.13 - Sum Odd Numbers
def sum_odd_numbers(ls):
    total = 0
    length = len(ls)
    for i in range(length):
        if ls[i] % 2 != 0 and ls[i] > 0:
            total = total + ls[i]
    return total

a = sum_odd_numbers([1, 2, 3])
print(a)    # 4 
b = sum_odd_numbers([43, 30, 27, -3])
print(b)    # 70

# Problem 3.14 - Hailstone
def hailstone(n):
    ls = []
    ls.append(int(n))
    while n != 1:
        if n % 2 == 0:
            n = n / 2
            ls.append(int(n))
        else:
            n = 3 * n + 1
            ls.append(int(n))
    return ls

sequence = hailstone(4)
print(sequence)     # [4, 2, 1]
sequence1 = hailstone(5)
print(sequence1)    # [5, 16, 8, 4, 2, 1]

# Problem 3.15 - Get Odd Numbers
def get_odd_numbers(ls):
    newls = []
    length = len(ls)
    for i in range(length):
        if ls[i] % 2 != 0:
            newls.append(ls[i])
    newls.sort()
    return newls

a = get_odd_numbers([3, 2, 1])
print(a)    # [1, 3]
b = get_odd_numbers([43, 30, 27, -3])
print(b)    # [-3, 27, 43]

# Problem 3.16 - Moving Average 
def moving_average(ls):
   average_days = 3
   i = 0
   averages = []

   while i < len(ls) - average_days + 1:

        # Store elements from i to i+window_size
        # in list to get the current window
        window = ls[i : i + average_days]
  
        # Calculate the average of current window
        window_average = round(sum(window) / average_days, 2)
      
        # Store the average of current
        # window in moving average list
        averages.append(window_average)
      
        # Shift window to right by one position
        i += 1
        
   return(averages)

data = [30.0, 20.0, 40.0, 50.0, 25.0, 70.0]
ma = moving_average(data)
print(ma)   # [30.0, 36.7, 38.3, 48.3]

# Problem 3.17 - Trapezodial Rule
def trapezoidal_rule(f, dx):
    if len(f) <= 1:
        return 0
    else:
        a = f[0] + f[-1]
        b = f[1:-1]
        b = sum(b)
        area = 0.5 * dx * (a + (2 * b))
    return area 

f = [3, 7, 11, 9, 3]
dx = 2
area = (trapezoidal_rule(f, dx))
print(area) # 60.0

# Problem 3.18 - Riemann Sum
def left_riemann_sum(x, y):
    area = []
    for i in range(len(x)):
        if i == 0:
            area.append(x[i] * y[i])
        else:
            area.append((x[i] - x[i - 1]) * y[i - 1])
    a = sum(area)
    return a

x = [0, 2, 3, 5, 6]
y = [1, 1.5, 1.7, 1.9, 2.0]
s = left_riemann_sum(x, y)
print(s)    # 8.8