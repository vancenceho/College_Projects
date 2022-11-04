# Question 1.01 - Body Mass Index (BMI)
weight = input('What is your weight in kg?')
weight = float(weight)
height = input('What is your height in m?')
height = float(height)
bmi = weight/(height*height)

print('Your BMI is', bmi)

if bmi >= 18 and bmi <= 25:
    print('congratulations! Do not forget to exercise regularly!') 
else: 
    print('good nutrition is important!')


# Question 1.02 - Quadratic Equation
x = input("Enter the value of x: ")
x = float(x)
value = (x**2) + (5 * x) - 4
print("The value of f(x) is ", value)

# Question 1.03 - Geometric Sequence
a = 2
r = 2
n = 1
result = (a * (1 - r ** n))/(1 - r)
result = float(result)
print("The sum is ", result)

# Question 1.04 - Address Format
addressee = "Singapore Post Pte Ltd"
house_number = "10"
road_name = "Eunos Road 8"
unit_number = "# 05-33"
building_name = "Singapore Post Centre"
postal_code = "408600"

s = addressee + "\n" + house_number + " " + road_name + "\n" + unit_number + " " + building_name + "\n" + "Singapore" + " " + postal_code
print(s)

# Question 1.05 - Income Tax
annual_income = 30000

if annual_income <= 20000:
    tax = 0
    print("The tax payable on an income of ",annual_income, " " + "is ", tax, "  dollars.")
else:
    tax = 0.05 * annual_income
    print("The tax payable on an income of ",annual_income," " + "is ",tax," dollars.")

# Question 1.06 - Green Bottles
for i in range(9):
    if i %2 != 0:
        print("green bottle ", i)

# Question 1.07 - General Term of Sequence
number_of_terms = 10
for n in range(number_of_terms):
     result = ((n ** 3) + (3 * n) + 5)/(n ** 2 + 1)
     print("n = ", n, ": ", result)

# Question 1.08 - Sum using a loop - Calculate sum of Sn
n = 10
total = 0  
for i in range(1, n + 1):
    total += i
    
print("The sum is ", total)

# Question 1.09 - Sum using a loop - Calculate sum of series
number_of_terms = 10
total = 0
for n in range(1, number_of_terms + 1):
    sn = (n - 1) + n 
    total += sn

print("The sum of ", number_of_terms, " is ", total)

# Question 1.10 - Swap
a = 10
b = "apple"

c = a
a = b
b = c 

print(a , b) 

# Question 1.11 - Program from a flow chart
height = 1.70
weight = 69.0
bmi = weight / (height ** 2)

condition1 = bmi >= 27.5
condition2 = bmi >= 23

print(bmi)

if( condition1 == True):
  print("High Risk")
elif( condition2 == True):
  print("Moderate Risk")
else:
    print("Low Risk")