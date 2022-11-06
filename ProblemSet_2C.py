# Problem 2.15 - Sequence
import math

def sequence(n):
    if n >= -3 and n < -0.5:
        return None
    else:
        sn = math.sqrt((2 * n + 1) / (n + 3))
        sn = round(sn, 4)
        return sn

print(sequence(-4.0)) # output = 2.6458
print(sequence(-2.0)) # output = None
print(sequence(0.0)) # output = 0.5774
print(sequence(2.0)) # output = 1.0

# Problem 2.16 - Checking of Values
def check_value(n1, n2, n3, x):
    if x > n1 and x > n2 and x < n3:
        return True
    else:
        return False

ans = check_value(1, 4, 8, 7)
print(ans)                      # output = T
ans = check_value(10, 4, 8, 7)
print(ans)                      # output = F
ans = check_value(1, 10, 8, 7)
print(ans)                      # output = F
ans = check_value(1, 4, 5, 7)
print(ans)                      # output = F