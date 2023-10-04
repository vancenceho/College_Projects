# WEEK 4 - COHORT 
import math
import time
import random

# COHORT 1 
# Exercise 1 
class Dog:

    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def set_name(self, value):
        if isinstance(value, str):
            self._name = value
        else:
            self._name = "Doggy"
    
    def bark(self):
        return "{} says woof!".format(self._name)

# (1) instantiate one instance with name Fido.
my_dog = Dog("Fido")
# (2) execute the bark method and dislay the result on the screen.
print(my_dog.bark())

# Exercise 2 
class Coordinate: 

    def __init__(self, x, y):
        self._x = x
        self._y = y

    def get_x(self):
        return self._x

    def set_x(self, value):
        if isinstance(value, int):
            self._x = value
        else:
            self._x = 0

    def __str__(self):
        return "Coordinate ({},{})".format(self._x, self._y)

    def distance(self):
        return math.sqrt(self._x*self._x + self._y*self._y)

# instantiate a Coordinate object with x = 1, y = 2
my_coordinate = Coordinate(1, 2)
# display the x and y attribute on the screen 
print(my_coordinate._x, my_coordinate._y)
print(my_coordinate)
# execute the distance method
print(my_coordinate.distance())

# COHORT 2 
# Exercise 3 
class Dog_2:
    
    def __init__(self, name):
        self._name = name
        self._color = "white"

    # return a string that says "<name> wage tail"
    def wag_tail(self):
        return self._name + " wags tail"

    # write a getter for the color attribute
    def get_color(self):
        return self._color
    
    # write a setter for the color attribute
    def set_color(self, value):
        if value in ['white', 'black', 'yellow']:
            self._color = value
        else:
            self._color = 'white'

a = Dog_2('Fido')
print(a.get_color())
a.set_color('black')
print(a.get_color())

# Exercise 4
class Coordinate_2:

    def __init__(self, x, y):
        self._info = {'x': x, 'y': y}

    def get_x(self):
        return self._info['x']

a = Coordinate_2(1, 2)
print(a.get_x())

# Exercise 5 - Implementing @property & @<function>.setter
class Coordinate_3:

    def __init__(self, x, y):
        self._x = x 
        self._y = y

    # decorator for getter
    @property
    # name of getter has same name as attribute.
    def x(self):
        return self._x
    # decorator for setter
    @x.setter
    def x(self, value):
        if isinstance(value, int):
            self._x = value
        else:
            self._x = 0

a = Coordinate_3(1, 2)
# getter statement
print(a.x)

# CS1. We are going to create a simple Car Racing game. 
# First, let's create a class Car with the following properties:
# (1) - racer which stores the name of the driver. 
# This property must be non-empty string. This property should be initialized upon object instantiation.
# (2) - speed which stores the speed of the car. 
# This property can only be non-negative values and must be less than or equal to a maximum speed.
# (3) - pos which is an integer specifying the position of the car which can only be non-negative values.
# (4) - is_finished which is a computed property that returns True or False depending whether the 
# position has reached the finish line.

# Each car also has the following attributes:
# (1) - max_speed which specifies the maximum speed the car can have. 
# This attribute should be initialized upon object instantiation.
# (2) - finish which stores the finish distance the car has to go through. 
# Upon initialization, it should be set to -1.

# The class has the following methods:
# (1) - start(init_speed, finish_distance) which set the speed property to some initial value. 
# The method also set the finish distance to some value and set the pos property to 0.
# (2) - race(acceleration) which takes in an integer value for its acceleratin and modify both 
# the speed and the position of the car.

# Class definition - RacingCar
class RacingCar:

    # definition of class
    def __init__ (self, name, max_speed):
        self._racer = name
        self.max_speed = max_speed
        self.finish = -1
        # initialize other storage for properties in __init__
        self._speed = 0
        self._pos = 0

    # decorator for getter name attribute
    @property
    def racer(self):
        return self._racer
    # decorator for setter name attribute
    @racer.setter
    def racer(self, name):
        if isinstance(name, str) and len(name) > 0:
            self._racer = name
    # decorator for getter speed attribute
    @property
    def speed(self):
        # the value for the speed property is stored in self._speed
        return self._speed
    # decorator for setter speed attribute
    @speed.setter
    def speed(self, val):
        if isinstance(val, (int, float)):
            if val >= 0 and val <= self.max_speed:
                self._speed = val
        # no else ---> do nothing if condition is not met
    # decorator for getter position attribute
    @property
    def pos(self):
        return self._pos
    # decorator for setter position attribute
    @pos.setter
    def pos(self, val):
        if isinstance(val, int) and val >= 0:
            self._pos = val
        # no else ---> do nothing if condition is not met
    # decorator for getter is_finished attribute
    @property
    # computed property ---> value returned from a calculation
    def is_finished(self):
        return self._pos > 0 and self._pos > self.finish
    
    def start(self, init_speed, finish_dist):
        if isinstance(init_speed, (int, float)) and init_speed > 0:
            self._speed = init_speed
        if isintance(finish_dist, int):
            self._finish = finish_distance
            self._pos = 0
        
    def race(self, acc):
        if isinstance(acc, int) and acc >= 0:
            self._speed += acc
            self._pos += self._speed
    
    def __str__(self):
        return f"Racing Car {self.racer} at position: {self.pos}, with speed: {self.speed}."

# test case 
car = RacingCar("Hamilton", 200)
car.racer = "Verstappan"
print(car.racer)
car.speed = "Ruby"
print(car.speed)
car.pos = "Bochhi"
print(car.pos)
car.finish = 300
car.pos = 200
print(car.is_finished)
car.pos = 400
print(car.is_finished) ### <---- calls the is_finished method

# CS2. 
# Implement a RacingGame class that plays car racing using Python random module to simulate car's acceleration. 
# The class has the following attribute(s):
# (1) - car_list which is a dictionary containing all the RacingCar objects where the keys are the racer's name.
# The class has the following properties:
# (1) - winners which list the winners from the first to the last. 
#       If there is no winner, it should return None.
# Upon instantiation, it should initalize the game with some random seed. 
# This is to ensure that the behaviour can be predicted.
# It has the following methods:
# (1) - add_car(name, max_speed) which creates a new RacingCar object and add it into the car_list.
# (2) - start(finish_distance) which uses the random module to assign different initial speeds (0 to 50) to 
# (3) - each of the racing car and set the same finish distance for all cars.
# (4) - play(finish) which contains the main loop of the game that calls the RacingCar's method race() 
#       until all cars reach the finish line. It takes in an argument for the finish distance.

# Class definition - RacingGame
class RacingGame:

    def __init__(self, seed):
        self.car_list = {}
        self._winners = []
        random.seed(seed)

    # decorator for getter winner attribute
    @property
    def winner(self):
        if self._winners == []:
            return None
        else:
            return self._winners

    # self.car_list is a dictionary -> key = racer name, value = RacingCar object
    def add_car(self, name, speed):
        if name not in self.car_list:
            self.car_list[name] = RacingCar(name, speed)
        else:
            print("Racing Car with {} already exists, choose another name.".format(name))

    def start(self, finish):
        # looping through dictionary ---> name is the keys
        for name in self.car_list:
            # variable car points to a RacingCar object
            car = self.car_list[name]
            speed = random.randint(0, 50)
            # since car is a RacingCar object, I can execute its start() method
            car.start(speed, finish)

    def play(self, finish):
        self.start(finish)
        finished_car = 0
        while True:
            for racer, car in self.car_list.items():
                if not car.is_finished:
                    acc = random.randint(-10, 20)
                    car.race(acc)
                    # to check for output
                    print(car)
                    if car.is_finished:
                        self._winners.append(racer)
                        finished_car += 1
            if finished_car == len(self.car_list):
                break

# CS3. Implement the Stack abstract data type using a Class. 
# You can use list Python data type as its internal data structure. Name this list as items.

# The class should have the following interface:
# (1) - __init__() to initialize an empty list for the stack to store the items.
# (2) - push(item) which stores an item into the top of the stack.
# (3) - pop() which returns and removes the top element of the stack. 
#       The return value is optional as it may return None if there are no more elements in the stack.
# (4) - peek() which returns the top element of the stack. If the stack is empty, it returns None.

# The class should have the following properties:
# (1) - is_empty is a computed property which returns either True or False 
#        depending whether the stack is empty or not.
# (2) - size is a computed property which returns the number of items in the stack.

class Stack:

    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)
    
    def pop(self):
       return self._items.pop()

    def peek(self):
        return self._items[-1]
    
    # decorator for getter of is_empty
    @property
    def is_empty(self):
        return self.size == 0

    # decorator for getter of size
    @property
    def size(self):
        return len(self._items)

# test case
s1 = Stack()
s1.push(10)
s1.push(20)
print(s1.size)
# print(s1._Stack__items) ### Name Mangling - Python's backfoor to access the double underscore
s1.push(2)
assert not s1.is_empty
assert s1.pop() == 2
assert s1.is_empty
assert s1.pop() == None
s1.push(1)
s1.push(2)
s1.push(3)
assert not s1.is_empty
assert s1._Stack__items == [1, 2, 3]
assert s1.peek() == 3
assert s1.size == 3

# WEEK 4 - HOMEWORK 
# CS1 - Queue Class ---> FIFO Rule
class Queue:

    def __init__(self):
        self._items = []
    
    def enqueue(self, item):
        self._items = [item] + self._items ### O(n) ---> create object, then copy n elements over
    
    def dequeue(self):
        return self._items.pop() ### O(1) ---> remember pop(0) is O(n)
    
    def peek(self):
        return self._items[-1]

    @property
    def is_empty(self):
        return self.size == 0
    
    @property
    def size(self):
        return len(self._items)

# CS5 - Queue with double stack
class Stack:

    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)
    
    def pop(self):
       return self._items.pop()

    def peek(self):
        return self._items[-1]
    
    # decorator for getter of is_empty
    @property
    def is_empty(self):
        return self.size == 0

    # decorator for getter of size
    @property
    def size(self):
        return len(self._items)

class Queue_with_Stack:
    
    def __init__(self):
        # IN stack
        self.left_stack = Stack()
        # OUT stack
        self.right_stack = Stack()

    def enqueue(self, item):
        self.left_stack.push(item)
    
    # move elements from IN to OUT
    # left stack --> right stack 
    def dequeue(self):
        # move elements from left to right if right is empty
        if self.right_stack.is_empty:
            self._move_to_right()
        return self.right_stack.pop()

    def peek(self):
        if self.right_stack.is_empty:
            self._move_to_right()
        return self.right_stack.peek()

    # private method (internal)
    def _move_to_right(self):
        # assume right stack is empty
        while not self.left_stack.is_empty():
            element = self.left_stack.pop()
            self.right_stack.push(element)