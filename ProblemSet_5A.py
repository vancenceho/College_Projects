# Problem 5.01 - Dictionary
def create_dictionary(fruits, prices):
    d = {}
    if len(fruits) == len(prices):
        dic = list(zip(fruits, prices))
        d.update(dic)
        return d
    else:
        return None

f = ['apple', 'orange']
p = [0.43, 0.51]
dd = create_dictionary(f, p)
print(dd)                                                                           # Expected {'apple':0.43, 'orange':0.51}
f = ['apple']
p = [0.43, 0.51]
dd1 = create_dictionary(f, p)
print(dd1)                                                                          # Expected None

# Problem 5.02 - Get Value
def get_value(dd, k):
    if k in dd:
        return dd[k]
    else:
        return None

sample_dict = {1:2, 2:3}
key = 2
value = get_value(sample_dict, key)
print(value)                                                                        # Expected 3
key = 3
value = get_value(sample_dict, key)
print(value)                                                                        # Expected None

# Problem 5.03 - Extract Data
import math

def extract_data(dd, k):
    if k in dd:
        x1 = dd[k][0]
        x2 = dd[k][1]
        data = math.sqrt((x1 ** 2) + (x2 ** 2))
        data = round(data, 2)
        return data
    else:
        return None

dd = {'A': (1,2), 'B': (-3,4), 'C': (-1,2)}
key = 'A'
result = extract_data(dd, key)
print(result)                                                                       # Expected 2.24
key = 'D'
result = extract_data(dd, key)
print(result)                                                                       # Expected None

# Problem 5.04 - Get Base Counts (DNA)
def get_base_counts(dna):
    dic = {}
    dd = ["A", "C", "G", "T"]
    keys = zip(dd, [0]*len(dd))
    dic.update(keys)
    for c in (dna):
        if c in dic:
            dic[c] = dic[c] + 1
        else:
            string = "The input DNA string is invalid"
            return string
    return dic

result = get_base_counts('AACCCGT')
print(result)                                                                       # Expected {'A':2, 'C':3, 'G':1, 'T':1}
result = get_base_counts('AACCG')
print(result)                                                                       # Expected {'A':2, 'C':2, 'G':1, 'T':0}
result = get_base_counts('OOOO')
print(result)                                                                       # Expected The input DNA string is invalid

# Problem 5.05 - Evaluate Polynomial
def evaluate_polynomial(dd, x):
    total = []
    for i, j in dd.items():
        a = j * (x ** i)
        total.append(a)
    
    y = round(sum(total), 2)
    return y

dd = {3:1, 1:2, 0:1}
x = 0.5
result = evaluate_polynomial(dd, x)
print(result)                                                                       # Expected 2.12

# Problem 5.06 - Differentiation Dictionary
def diff(pp):
    dp = {}

    for k in pp:
        dp[ k - 1] = k * pp[k]

    dpp = dp.copy()

    for k in dp:
        if dp.get(k) == 0:
            del dpp[k]
    
    return dpp

p={0:-3, 3:2, 5:-1} 
q = {1:2}
r = {3:6}
result = diff(p)
result1 = diff(q)
result2 = diff(r)
print(result)                                                                       # Expected {2:6, 4:-5}
print(result1)
print(result2)

# Problem 5.07 - Read lists
def read_list(ls, reg_no, key):
    items = len(ls)
    for i in range(items):
        if ls[i]['reg'] == reg_no:
            if key in ls[i].keys():
                return ls[i][key]
            else:
                return None
        

ls = [
  {'reg': '1234A', 'make':'Caelum', 'price': 24999}, 
  {'reg': '888B', 'make':'Noctis', 'price': 12499}, 
  {'reg': '365K', 'make':'Cloud', 'price': 7499},
  {'reg': '1043W', 'price': 2499}, 
  {'reg': '7422M', 'make': 'Lucis'} ]

result = read_list(ls, '888B', 'make')
print(result)                                                                   # Expected Noctis
result = read_list(ls, '7422M', 'price')
print(result)                                                                   # Expected None
result = read_list(ls, '1234A', 'price')
print(result)                                                                   # Expected 24999

# Problem 5.08 - Get Highest Value
def get_highest_value(dd):
    max_key = max(dd, key=dd.get)
    max_value = dd.get(max_key)
    return(max_key, max_value)


data = {'a': 123, 'b': 132, 'c': 95}
results = get_highest_value(data)
print(results)                                                                  # Expected (b,132)