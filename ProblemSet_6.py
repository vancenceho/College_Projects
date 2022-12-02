# Problem 6.01 - Get Average
def get_average_sublist(ls, n):
    try:
        ls[n]
        l = len(ls[n])
        total = sum(ls[n])
        average = round((total / l), 1)
        return average
    except IndexError:
        return None

a = [ [10, 12], [36, 40, 52], [10, 16, 17] ] 
print(get_average_sublist(a, 0))                                                    # Expected 11.0
print( get_average_sublist( a, 1) )                                                 # Expected 42.7
print( get_average_sublist( a, 3) )                                                 # Expected: None
print(get_average_sublist(a, -1))                                                   # Expected: 14.3
print(get_average_sublist(a, -2))                                                   # Expected: 42.7

# Problem 6.02 - Has Lists
def has_list(ls):
    for i in ls:
        if isinstance(i, list):
            return True
    return False

a = [10, 20, [30, 40] , 'apple' ]
print(has_list(a))                                                                 # Expected: True                        
b = [10, 20, 30, 40 , 'apple' ]
print(has_list(b))                                                                 # Expected: False

# Problem 6.03 - Max Lists in Lists
def max_list(inlist):
    outlist = []
    for i in inlist:
        a = max(i)
        outlist.append(a)

    return outlist

inlist = [[1 ,2 ,3] ,[4 ,5]]
print(max_list(inlist))                                                            # Expected: [3, 5]
inlist = [[3 ,4 ,5 ,2] ,[1 ,7] ,[8 ,0 , -1] ,[2]]
print(max_list(inlist))                                                            # Expected: [5, 7, 8, 2]
inlist = [[3 ,4 ,5 ,2]]
print(max_list(inlist))                                                            # Expected: [5]

# Problem 6.04 - Average in Lists
def find_average(ls):
    total_average = []
    counter = 0
    length = 0

    for i in ls:
        l = len(i)
        total = sum(i)
        average = round((total / l), 2)
        total_average.append(average)

        counter += sum(i)
        length += l

    overall_average = round((counter / length), 2)
    print(overall_average)

    return total_average, overall_average



# def find_average(ls):
#     average =[]
#     b = []
#     counter = 0

#     for i in ls:
#         total = sum(i)
#         n = len(i)
#         a = round((total / n), 2)
#         average.append(a)
#         b.append(total)
#         counter += n
         
#     total_average = sum(b) / counter

#     return average, total_average

ls = [[3,4],[5,6,9],[-1,2,8]]
ans=find_average(ls) 
print(ans)                                                                          # Expected: ([3.5, 6.67, 3.0], 4.5)
ls1 = [[3, 4, 7], [0], [-2, -3, 10]]
ans1 = find_average(ls1)
print(ans1)                                                                         # Expected: ([4.67, 0.0, 1.67], 2.71)


# Problem 6.05 - Get Zero Matrix
def get_zero_matrix(m, n):
    zeros = [ [0] * n for i in range(m)]
    return zeros
    

ls = get_zero_matrix(2, 3)
print(ls)                                                                           # Expected: [[0, 0, 0], [0, 0, 0]]
ls[1][2] = 7
print(ls)                                                                           # Expected: [[0, 0, 0], [0, 0, 7]]
ls = get_zero_matrix(1, 4)
print(ls)                                                                           # Expected: [[0,0,0,0] ]
ls = get_zero_matrix(3, 1)
print(ls)                                                                           # Expected: [[0], [0], [0]]

# Problem 6.06 - Transpose Matrix
def transpose_matrix(ls):
    rows = len(ls)
    columns = len(ls[0])
    empty = get_zero_matrix(columns, rows)

    # iterate through rows
    for i in range(len(empty)):
        # iterate through columns
        for j in range(len(ls)):
            empty[i][j] = ls[j][i]

    return empty

A = [ [1 ,  2, 3 ] ,
      [40, 50, 60] ]
print(transpose_matrix(A))                                                           # Expected: [[1, 40], [2, 50], [3, 60]]

# Problem 6.07 - Process Scores
def process_scores(f):
    ls = f.read()
    i = ls.split()

    lis = [eval(j) for j in i]
    total = sum(lis)
    n = len(lis)
    average = round((total / n), 1)

    return total, average

with open('scores.txt') as f:
    ans=process_scores(f)
    print(ans)                                                                      # Expected: (581, 38.7)

# Problem 6.08 - Read fdi
def read_fdi(f):
    ls = f.read()
    ls = ls.split()
    ls.pop(0)
    
    key = []
    val = []

    for i in ls:
        lis = i.split(",")
        lis.remove("Total")
        lis[1] = eval(lis[1])
        
        key.append(lis[0])
        val.append(lis[1])

    d = dict(zip(key, val))

    return d

with open('fdi.csv') as f:
    data = read_fdi(f)
    print(data)                                                                     # Expected: {'1998': 2995.4, '1999': 15443.7, '2000': 22174.1, '2001': 18314.6, '2002': 24611.3, '2003': 30256.9, '2004': 43943.7, '2005': 53896.3}

# Problem 6.09 - DNA Content
def gc_content(f):
    base_C = "C"
    base_G = "G"

    dna = f.read().replace("\n", "")
    
    counter_C = dna.count(base_C)
    counter_G = dna.count(base_G)
    counter = len(dna)

    total = ((counter_C + counter_G) / counter) * 100
    percentage = round(total, 1)

    return percentage

filename = "dna.txt"
with open(filename, 'r') as f:
    percentage = gc_content(f)
    print(percentage)                                                               # Expected: 52.2

# Problem 6.10a - Scalar Multiply
def scalar_multiply(ls, a):
    rows = len(ls)
    columns = len(ls[0])

    for i in range(rows):
        for j in range(columns):
            ls[i][j] = ls[i][j] * a

    return ls

A = [ [10, 20, 30 ] ,
    [40, 50, 60] ]
scalar_multiply(A, 0.2)
print(A)                                                                            # Expected: [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]

# Problem 6.10b - Scalar Multiply Copy
import copy

def scalar_multiply_new(ls, a):
    matrix_copy = copy.deepcopy(ls)

    new_matrix = scalar_multiply(matrix_copy, a)

    return new_matrix

C = [ [10, 20, 30 ] ,
      [40, 50, 60] ]
B = scalar_multiply_new(C, 0.2)
print(C)                                                                              # Expected: [[10, 20, 30], [40, 50, 60]]
print(B)                                                                              # Expected: [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]