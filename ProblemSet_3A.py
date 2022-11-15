# Problem 3.00a - Calculate Sum Odd (n)
def calculate_sum_odd(n):
    total = 0 
    for i in range(n):
        if i % 2 != 0:
            total = total + i
    return total

print(calculate_sum_odd(10)) # output = 25
print(calculate_sum_odd(11)) # output = 25
print(calculate_sum_odd(12)) # output = 36

# Problem 3.00b - Alternating
def alternating(n):
    total = 0
    for i in range(1, n+1):
        sn = ((-1) ** (i + 1)) / i
        total = total + sn
    return total

print(alternating(2)) # output = 0.5
print(alternating(4)) # output = 0.5833333333333333
print(alternating(10)) # output = 0.6456349206349207

# Problem 3.01a - Compounding Interest
def compound_interest(principal, rate, months):
    for i in range(1, months + 1):
        principal = principal * (1 + (rate / 12))
    return round(principal, 2)

print(compound_interest(2500, 0.10, 3))
value = compound_interest(100, 0.05, 6)
print(value) # output = 102.53

# Problem 3.01b - Regular Interest
def regular_savings(deposit, rate, months):
    principal = 0
    for i in range(1, months + 1):
        savings = (deposit + principal) * (1 + (rate / 12))
        principal = savings
    return round(principal, 2)

value = regular_savings(100, 0.05, 6)
print(value) # output = 608.81

# Problem 3.02 - Sum of Series
def sum_of_series(n):
    if n < 1:
        return 0
    else:
        total = 0
        for i in range(1, n + 1):
            sn = 1 / (i ** 2)
            total = total + sn
    return total

a = sum_of_series(0)
b = sum_of_series(1)
c = sum_of_series(10)
print(a) # output = 0
print(b) # output = 1.0
print(c) # output = 1.5497677311665408

# Problem 3.03 - Prime Numbers
def is_prime(n):
    for i in range(2, n - 1):
        if n % i == 0:
            return False
    return True

print(is_prime(15))
print(is_prime(17))

# Problem 3.04 - Number of Terms required
import math

def sum_of_series(n):
    if n < 1:
        return 0
    else:
        total = 0
        for i in range(1, n + 1):
            sn = 1 / (i ** 2)
            total = total + sn
    return total

def fraction_of_pisq(s):
    s = s / ((math.pi ** 2) / 6)
    return s

def terms_required(p):
    n = 0
    frac = 0
    while frac < p:
        n += 1
        frac = fraction_of_pisq(sum_of_series(n))
    return n

solution = sum_of_series(6)
print(solution)
fraction = fraction_of_pisq(1.5)
print(fraction)
n_terms = terms_required(0.9)
print(n_terms)

# Problem 3.05a - Calculate Even Sum
def calculate_sum_even(n):
    total = 0
    i = 0
    while i < n:
        if i % 2 == 0:
            total = total + i
            i = i + 2
    return total

print(calculate_sum_even(11)) # 30
print(calculate_sum_even(12)) # 30
print(calculate_sum_even(13)) # 42
print(calculate_sum_even(19))
print(calculate_sum_even(25))

# Problem 3.05b - Alternating While 
def alternating_while(stop):
    total = 0
    n = 1
    term = 1
    while abs(term) > stop:
        total += term
        n += 1
        term = ((-1) ** (n + 1)) / n
    return total

print(alternating_while(0.249)) # 0.58333333333
print(alternating_while(0.199)) # 0.78333333333