# Problem 2.01 - Calculate BMI
def calculate_bmi(weight, height):
    height = height/100
    weight = float(weight)
    bmi = weight / (height ** 2)
    bmi = round(bmi, 1)
    return bmi

print(calculate_bmi(2.5, 50)) # output = 10.0
print(calculate_bmi(50, 150)) # output = 22.2
print(calculate_bmi(43.5, 142.3)) # output = 21.5

# Problem 2.02 - Position Velocity
# y(t) = ut - 1/2gt^2
# y'(t) = u - gt 

def position_velocity(u, t):
    g = 9.81
    position = (u * t) - (0.5 * g * (t ** 2))
    position = round(position, 3)
    velocity = u - (g * t)
    velocity = round(velocity, 3)
    return position, velocity

print(position_velocity(5.0, 0)) # output = (0.0, 5.0)
print(position_velocity(10.0, 1)) # output = (5.095, 0.19)
print(position_velocity(5.886, 0.3)) # output = (1.324, 2.943)

# Problem 2.03 - Spring 
import math

def decay(a, t):
    x = math.exp(-(a * t)) * math.cos(a * t)
    return x

print(decay(2, 0)) # output = 1.0
print(decay(2, 0.5)) # output = 0.19876611034641298

# Problem 2.04 - Decribe BMI
def describe_bmi(bmi):
    if bmi < 18.5:
        return("nutritional deficiency")
    elif bmi >= 18.5 and bmi < 23:
        return("low risk")
    elif bmi >= 23 and bmi < 27.5:
        return("moderate risk")
    else:
        return("high risk")

print(describe_bmi(18)) # output = nutritional deficiency
print(describe_bmi(20)) # output = low risk
print(describe_bmi(24)) # output = moderate risk
print(describe_bmi(27.5)) # output = high risk
print(describe_bmi(23)) # testing boundary

# Problem 2.05 - Positive/Odd Test

def is_positive_even(n):
    if n % 2 == 0 and n >= 0:
        return True
    else:
        return False

print(is_positive_even(-2)) # output = False
print(is_positive_even(2)) # output = True
print(is_positive_even(3)) # output = False

# Problem 2.06 - Grade Mark

def letter_grade(mark):
    if mark >= 90 and mark <= 100:
        return "A"
    elif mark >= 80 and mark <= 89:
        return "B"
    elif mark >= 70 and mark <= 79:
        return "C"
    elif mark >= 60 and mark <= 69:
        return "D"
    elif mark >= 0 and mark <= 59:
        return "E"
    else:
        return None

print(letter_grade(102)) # output = None
print(letter_grade(100)) # output = A
print(letter_grade(83)) # output = B
print(letter_grade(75)) # output = C
print(letter_grade(67)) # output = D
print(letter_grade(52)) # output = E
print(letter_grade(-2)) # output = None

# Problem 2.07 - Largest Area

def largest_area(s, u, v):
    if s < 0 or u < 0 or v < 0:
        return None
    elif u > s or v > s:
        return None
    else:
        area = [u * v, u * (s - v), v * (s - u), (s - u) * (s - v)]
        return area

result = largest_area(10, 3, 4)
print(result)
result = largest_area(10, 11, 4)
print(result)
result = largest_area(-10, 3, 4)
print(result)
