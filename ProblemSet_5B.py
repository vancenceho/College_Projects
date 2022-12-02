# Problem 5.11 - Increment Value
def increase_value(dd, k):
    for k in dd:
        if key == k:
            dd[key] = dd[key] + 1
    return dd

dd1 = {1:2, 2:3}
key = 2
increase_value(dd1, key)
print(dd1)                                                              # Expected: {1:2, 2:4}

dd2 = {7:10, 8:40}
key = 3
increase_value(dd2, key)
print(dd2)                                                              # Expected: {7:10, 8:40}

# Problem 5.12a - Translating Point of Vector
def translate_point(dd, key, vector):
    for k in dd:
        if key == k:
           dd[k] = tuple(map(sum, (zip(dd[k], vector))))
           return dd
           

dd = {'A': (1,2), 'B': (-3,4), 'C': (-1,2)}
key = 'A'
vector = (3, 2)
translate_point(dd, key, vector)
print(dd)                                                               # Expected: {'A': (4,4), 'B': (-3,4), 'C': (-1,2)}

dd2 = {'F': (1,2), 'G': (-3,4), 'H': (-1,2)}
key = 'D'
vector = (3, 2)
translate_point(dd2, key, vector)
print(dd2)                                                              # Expected: {'F': (1,2), 'G': (-3,4), 'H': (-1,2)}

# Problem 5.12b - Translating Point of Vector
def translate_point_new(dd, key, vector):
    dd_out = dd.copy()
    for k in dd_out:
        if key == k:
            dd_out[k] = tuple(map(sum, (zip(dd_out[k], vector))))
            return dd_out
        else:
            return dd_out

dd = {'A': (1,2), 'B': (-3,4), 'C': (-1,2)}
key = 'A'
vector = (3, 2)
d_new = translate_point_new(dd, key, vector)
print(dd)                                                               # Expected: {'A': (1, 2), 'B': (-3, 4), 'C': (-1, 2)}
print(d_new)                                                            # Expected: {'A': (4, 4), 'B': (-3, 4), 'C': (-1, 2)}
print(dd is d_new)                                                      # Expected: False

# Problem 5.13a - Replacing Values
def replace_values(ls, value1, value2):
    for i in range(len(ls)):
        if ls[i] == value1:
            ls[i] = value2
    return ls

ls1 = [3, 9, 5, 10, 2, 4, 10, 3]
value1 = 3
value2 = 12
replace_values(ls1, value1, value2)
print(ls1)                                                              # Expected: [12, 9, 5, 10, 2, 4, 10, 12]

ls2 = [4, 9, 5, 10, 2, 4, 10, 4]
value1 = 3
value2 = 12
replace_values(ls2, value1, value2)
print(ls2)                                                              # Expected: [4, 9, 5, 10, 2, 4, 10, 4]

# Problem 5.13b - Replacing Values
def replace_values_new(ls, value1, value2):
    ls_copy = ls.copy()
    for i in range(len(ls_copy)):
        if ls_copy[i] == value1:
            ls_copy[i] = value2
    return ls_copy

ls1 = [3, 9, 5, 10, 2, 4, 10, 3]
ls2 = replace_values_new(ls1, 3, 12)
print("ls1", ls1)                                                        # Expected: ls1 [3, 9, 5, 10, 2, 4, 10, 3]
print("ls2", ls2)                                                        # Expected: ls2 [12, 9, 5, 10, 2, 4, 10, 12]

ls3 = [4, 9, 5, 10, 2, 4, 10, 4]
ls4 = replace_values_new(ls3, 3, 12)
print("ls3", ls3)                                                        # Expected: ls3 [4, 9, 5, 10, 2, 4, 10, 4]
print("ls4", ls4)                                                        # Expected: ls4 [4, 9, 5, 10, 2, 4, 10, 4]

# Problem 5.14 - Reflect Triangle
def equation_of_line(point1, point2):
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    a = delta_x
    b = -(delta_y)
    c = (point1[0] * delta_y) - (point1[1] * delta_x)
    return a, b, c

print(equation_of_line( (1, 2), (-1, 2)))                                # Expected: (-2, 0, 4)

def reflect(point, eqn):
    reflect_p = ( (point[0] * ((eqn[0] ** 2) - (eqn[1] ** 2))) - ((2 * eqn[1]) * ((eqn[0] * point[1]) + eqn[2])) ) / ((eqn[0] ** 2) + (eqn[1] ** 2))
    reflect_q = ( (point[1] * ((eqn[1] ** 2) - (eqn[0] ** 2))) - ((2 * eqn[0]) * ((eqn[1] * point[0]) + eqn[2])) ) / ((eqn[0] ** 2) + (eqn[1] ** 2))
    return reflect_p, reflect_q

print(reflect((-3, 4), (-2, 0, 4)))                                      # Expected: (-3.0, 0.0)

def reflect_triangle(dd, point):
    line = []
    for i in dd:
        if point == i:
            coordinate = dd.get(point)
        else:
            line.append(dd.get(i))

    new_value = reflect(coordinate, equation_of_line(line[0], line[1]))

    dd_copy = dd.copy()

    dd_copy[point] = new_value

    return dd_copy

dd = {'A': (1,2), 'B': (-3,4), 'C': (-1,2)}
d_new = reflect_triangle(dd, 'B')
print(d_new)                                                             # Expected: {'A': (1,2), 'B': (-3.0,0.0), 'C': (-1,2)}
