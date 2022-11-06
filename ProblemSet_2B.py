# Problem 2.11 - Compound Interest 
def compound_interest(amount, rate, periods, time):
    f = amount * ((1 + (rate / periods)) ** (periods * time))
    f = round(f, 3)
    return f

print(compound_interest(1, 0.03, 1, 1)) # output = 1.03
print(compound_interest(1, 0.12, 12, 1)) # output = 1.127
print(compound_interest(1, 1, 1000, 1)) # output = 2.717

# Problem 2.12 - Area Volume of Cylinder
import math

def area_vol_cylinder(radius, length):
    area = math.pi * (radius ** 2)
    volume = area * length
    area = round(area, 2)
    volume = round(volume, 2)
    return(area, volume)

print(area_vol_cylinder(1.0, 2.0)) # output = 3.14, 6.28
print(area_vol_cylinder(2.0, 2.3)) # output = 12.57, 28.9
print(area_vol_cylinder(1.5, 4)) # output = 7.07, 28.27
print(area_vol_cylinder(2.2, 5.0)) # output = 15.21, 76.03

# Problem 2.13 - Conversion of Seconds to Hours 
def seconds_to_hours(seconds):
    seconds_per_hour = 60 * 60
    seconds_per_minute = 60
    hours_in_day = 24
    seconds = seconds % (hours_in_day * seconds_per_hour)
    hour = seconds // seconds_per_hour
    seconds %= seconds_per_hour
    minutes = seconds // seconds_per_minute
    seconds %= seconds_per_minute
    return(hour, minutes, seconds)

print(seconds_to_hours(29500)) # output = 8, 11, 40
print(seconds_to_hours(7210)) # output = 2, 0, 10

# Problem 2.14 - Temperature Conversion (Fahrenheit => Celsius) 
def fahrenheit_to_celsius(f):
    c = (f - 32) * (5/9)
    if c < 0:
        return None
    else:
        c = round(c, 2)
        return c

print(fahrenheit_to_celsius(-500)) # output = None
print(fahrenheit_to_celsius(32)) # output = 0.0
print(fahrenheit_to_celsius(99)) # output = 37.22
print(fahrenheit_to_celsius(212)) # output = 100.0