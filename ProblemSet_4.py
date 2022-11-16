# Problem 4.00 - BMI Information
def calculate_bmi(weight, height):
    height = height / 100
    weight = float(weight)
    bmi = weight / (height ** 2)
    bmi = round(bmi, 1)
    return bmi

def describe_bmi(bmi):
    if bmi < 18.5:
        return("nutritional deficiency")
    elif bmi >= 18.5 and bmi < 23:
        return("low risk")
    elif bmi >= 23 and bmi < 27.5:
        return("moderate risk")
    else:
        return("high risk")

def bmi_information(weight, height):
    BMI = calculate_bmi(weight, height)
    category = describe_bmi(BMI)
    text = "Your BMI is {BMI} and your category is {category}.".format(BMI = BMI, category = category)
    return text

info = bmi_information(70, 167)
print(info) # Your BMI is 25.1 and your category is moderate risk. 

# Problem 4.01 - Reverse

# Slicing
def reverse1(s):
    return s[::-1]

# Looping
def reverse2(s):
    reverse_str = ""
    for i in s:
        reverse_str = i + reverse_str
    return reverse_str

result = reverse1("I choose you")
result2 = reverse2("I choose you")
print(result)     # uoy esoohc I
print(result2)    # uoy esoohc I

# Problem 4.02 - Palindrome

# Slicing 
def is_palindrome1(s):
    if s == s[::-1]:
        return True
    else:
        return False

# Looping
def is_palindrome2(s):
    reverse_str = ""
    for i in s:
        reverse_str = i + reverse_str
    if reverse_str == s:
        return True
    else:
        return False

result = is_palindrome1("civic")
print(result)
result = is_palindrome1("sonor")
print(result)

result1 = is_palindrome2("civic")
print(result1)    # True
result1 = is_palindrome2("sonor")
print(result1)    # False

# Problem 4.03 - Match Cases
def match(a, b):
    a_length = len(a)
    new_b = b[-a_length:]

    if new_b == a:
        return True
    else:
        return False

ending = "nus"
word = "parvenus"
result = match(ending, word)
print(result)

word1 = "aviatrix"
result1 = match(ending, word1)
print(result1)

ending1 = "lar"
word2 = "agricolarum"
result3 = match(ending1, word2)
print(result3)

# Problem 4.04 - Clean String
def clean_string(s):
    out = ""
    for i in s:
        if i.isalnum() or i == " ":
            out += i
    return out

print(clean_string("Hello. Are you there?!"))     # Hello Are you there
print(clean_string("If you don't take risks, you can't create a future!"))    # If you dont take risks you cant create a future

# Problem 4.05 - Check Password
def digits_in_string(s):
    digit_count = 0
    for i in s:
        if i.isnumeric():
            digit_count += 1
    return digit_count

def check_password(pwd):
    length = len(pwd)
    digits = digits_in_string(pwd)
    if length >= 8 and pwd.isalnum() == True and digits >= 2:
        return True
    else:
        return False

ans = digits_in_string('pokemon')
print(ans)    # 0
ans = digits_in_string('poke201mon')
print(ans)    # 3
ans = check_password('test') 
print(ans)    # False
ans=check_password('testtest') 
print(ans)    # False
ans=check_password('testt22') 
print(ans)    # False
ans=check_password('testte22') 
print(ans)    # True

# Problem 4.06 - Longest Prefix 
# (I DONT UNDERSTAND)
def longest_common_prefix(s1, s2):
    ls = []
    ls.append(s1)
    ls.append(s2)

    if (len(ls) == 0):
        return ""
    
    for i in range(len(ls[0])):
        a = ls[0][i]
        for j in range(len(ls)):
            if (i == len(ls[j]) or ls[j][i] != a):
                return ls[0][0:i]
    return ls[0]
    

ans=longest_common_prefix('distance','disinfection') 
print(ans)  # dis
ans=longest_common_prefix('testing','technical') 
print(ans)  # te
ans=longest_common_prefix('rosses','crosses') 
print(ans)  # empty string

# Problem 4.07 - Binary to Decimals  
# I DO NOT UNDERSTAND  
def binary_to_decimal(s):
    return int(s, 2)

print(binary_to_decimal('100'))   # 4
print(binary_to_decimal('101'))   # 5
print(binary_to_decimal('10001'))     # 17
print(binary_to_decimal('10101'))     # 21

# Problem 4.08 - Uncompressed
def uncompressed(s):
    output = ""
    num = ""
    for i in s:
        if i.isalpha():
            output += i * int(num)
            num = ""
        else:
            num += i
    return output

print(uncompressed("1x1y"))   #xy
print(uncompressed("2a5b1c"))     # aabbbbbc
print(uncompressed("1a1b2c"))     # abcc
print(uncompressed("1a9b3b1c"))   # abbbbbbbbbbbbc

# # Problem 4.09 - Get Base Counts
def get_base_count(dna):
    dna_bases = "ACGT"
    sorted_dna = sorted(dna)
    bases = []
    if all(ch in dna_bases for ch in sorted_dna) == True:
        A = sorted_dna.count("A")
        C = sorted_dna.count("C")
        G = sorted_dna.count("G")
        T = sorted_dna.count("T")
        bases.append(A)
        bases.append(C)
        bases.append(G)
        bases.append(T)
        return bases
    else:
        return bases

result1 = get_base_count("AACCCGT")
print(result1)
result2 = get_base_count("AACCG")
print(result2)
result3 = get_base_count("AAB")
print(result3)


# import collections

# def get_base_counts(dna):
#     dna_bases = "ACGT"
#     sorted_dna = sorted(dna)
#     sorted_dna = "".join(sorted_dna)
#     bases = []
#     d = collections.defaultdict(int)
#     if all(ch in dna_bases for ch in sorted_dna) == True:
#         for c in sorted_dna:
#             d[c] += 1
        
#         for c in d:
#                 bases.append(d[c])
#         return bases
    
#     else:
#         return bases

# result = get_base_counts('AACCCGT')
# print(result)   # [2, 3, 1, 1]
# result = get_base_counts('AACCG')
# print(result)   # [2, 2, 1, 0]
# result = get_base_counts('AAB')
# print(result)   #[]