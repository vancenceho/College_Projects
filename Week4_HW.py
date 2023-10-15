# WEEK 4 PROBLEM SET - HOMEWORK
import math

# HW1. Implement Queue abstract data structure using a Class. 
# You can use a list as its internal data structure. The class should have the following interface:
# (1) - _init__() to initialize an empty List for the queue to store the items.
# (2) - enqueue(item) which inserts an Integer into the queue.
# (3) - dequeue() which returns and removes the element at the head of the queue. 
#       The return value is an optional as it may return None if there are no more elements in the queue.
# (4) - peek() which returns the element at the head of the queue. If there is no element in the Queue, return None.

# The class Queue has two computed properties:
# is_empty which returns either True or False depending on whether the queue is empty or not.
# size which returns the number of items in the queue.

# Queue class definition ---> FIFO rule
class Queue:

    def __init__(self):
        self.__items = []

    def enqueue(self, value):
        if isinstance(value, int):
            self.__items.append(value)
    
    def dequeue(self):
        if len(self.__items) != 0:
            return self.__items.pop(0)
        else:
            return None
    
    def peek(self):
        if len(self.__items) != 0:
            return self.__items[0]
        else:
            return None
    
    @property
    def is_empty(self):
        return self.__items == []
    
    @property
    def size(self):
        return len(self.__items)
    
    def __str__(self) -> str:
        return str(self.__items)

# test cases
q1 = Queue()
q1.enqueue(2)
assert not q1.is_empty
assert q1.size == 1
ans = q1.dequeue()
assert ans == 2
assert q1.is_empty
q1.enqueue(1)
q1.enqueue(2)
q1.enqueue(3)
assert q1.size == 3
assert q1.peek() == 1
assert q1.dequeue() == 1
assert q1.dequeue() == 2
assert q1.dequeue() == 3
assert q1.peek() == None

# HW2. We are going to create a class that contains both RobotTurtle and Coordinate class. 
# The name of the class is TurtleWorld which is used to simulate when RobotTurtle is moving around some two dimensional space. 
# The class has the following methods:
# (1) - add_turtle(name) which is to add a new RobotTurtle into the world with the specified name.
# (2) - remove_turtle(name) which is to remove the object RobotTurtle with the specified name from the world.
# (3) - list_turtles() which is to list all the turtles in the world using their names in an ascending order.
# We give you here the class definition for the Coordinate and the RobotTurtle from the Notes.

# Coordinate class definition 
class Coordinate:

    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    
    # decorator for distance getter function
    @property 
    def distance(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def __str__(self):
        return ("({}, {})".format(self.x, self.y))
    
# RobotTurtle class definition
class RobotTurtle:

    def __init__(self, name, speed = 1):
        self.name = name
        self.speed = speed
        self._pos = Coordinate(0, 0)

    # decorator for name getter function
    @property
    def name(self):
        return self._name
    
    # decorator for name setter function
    @name.setter
    def name(self, value):
        if isinstance(value, str) and value != "":
            self._name = value
    
    # decorator for speed getter function
    @property
    def speed(self):
        return self._speed
    
    # decorator for speed setter function
    @speed.setter
    def speed(self, value):
        if isinstance(value, (int, float)) and value > 0:
            self._speed = value

    # decorator for position getter function
    @property
    def pos(self):
        return self._pos
    
    # Methods
    def move(self, direction):
        update = {'up': Coordinate(self.pos.x, self.pos.y + self.speed),
                  'down': Coordinate(self.pos.x, self.pos.y - self.speed),
                  'left': Coordinate(self.pos.x - self.speed, self.pos.y),
                  'right': Coordinate(self.pos.x + self.speed, self.pos.y)}
        self._pos = update[direction]

    def tell_name(self):
        return ("My name is {}".format(self.name))
    
# TurtleWorld class definition 
class TurtleWorld:

    def __init__(self):
        self.turtles = {}

    def add_turtle(self, name, speed):
        if name not in self.turtles:
            self.turtles[name] = RobotTurtle(name, speed)
        else:
            print("Turtle with {} already exists in the world :(".format(name))

    def remove_turtle(self, name):
        if name in self.turtles:
            del self.turtles[name]
        else:
            print("Turtle with name {} does not exist in the world.".format(name))

    def list_turtles(self):
        sorted_names = sorted(self.turtles.keys())
        return sorted_names
    
# test case
world = TurtleWorld()
world.add_turtle('t1', 1)
assert world.list_turtles() == ['t1']
world.add_turtle('t2', 2)
assert world.list_turtles() == ['t1', 't2']
world.add_turtle('abc', 3)
assert world.list_turtles() == ['abc', 't1', 't2']
world.remove_turtle('t2')
assert world.list_turtles() == ['abc', 't1']
world.remove_turtle('abc')
assert world.list_turtles() == ['t1']

# HW3. Modify the class TurtleWorld to add the following method:
# (1) - move_turtle(name, movement) which is to move the turtle with the specified name with a given input movement. 
#       The argument movement is a string containing letters: l for left, r for right, u for up, and d for down. 
#       The movement should be based on the speed. This means that if the turtle has speed of 2 and the movement argument is uulrdd, 
#       the turtle should move up four units, move left two units, move right two units and move down four units.

# TurtleWorld class definition 
class TurtleWorld:
    valid_movements = set('udlr')
    movement_map = {'u': 'up', 'd': 'down', 'l': 'left', 'r': 'right'}

    def __init__(self):
        self.turtles = {}

    def move_turtle(self, name, movement):
        for i in range(len(movement)):
            if movement[i] in self.valid_movements:
                self.turtles[name].move(self.movement_map[movement[i]])
        return self.turtles[name].pos

    def add_turtle(self, name, speed):
        if name not in self.turtles:
            self.turtles[name] = RobotTurtle(name, speed)
        else:
            print("Turtle with {} already exists in the world :(".format(name))

    def remove_turtle(self, name):
        if name in self.turtles:
            del self.turtles[name]
        else:
            print("Turtle with name {} does not exist in the world.".format(name))

    def list_turtles(self):
        sorted_names = sorted(self.turtles.keys())
        return sorted_names
    
# test case
world = TurtleWorld()
world.add_turtle('abc', 1)
world.move_turtle('abc', 'uu')
assert str(world.turtles['abc'].pos) == '(0, 2)'
world.move_turtle('abc', 'rrr')
assert str(world.turtles['abc'].pos) == '(3, 2)'
world.move_turtle('abc', 'd')
assert str(world.turtles['abc'].pos) == '(3, 1)'
world.move_turtle('abc', 'llll')
assert str(world.turtles['abc'].pos) == '(-1, 1)'
world.add_turtle('t1', 2)
world.move_turtle('t1', 'uulrdd')
assert str(world.turtles['t1'].pos) == '(0, 0)'
world.move_turtle('t1', 'ururur')
assert str(world.turtles['t1'].pos) == '(6, 6)'

# HW4. Modify the class `TurtleWorld` to include the following attribute and methods:
# - `movement_queue` which is an attribute of the type `Queue` to store the movement list.
# - `add_movement(turtle, movement)` which adds turtle movement to the queue `movement_queue` 
#    to be run later. The argument `turtle` is a string containing the turtle's name. 
#    The argument `movement` is another string for the movement. 
#    For example, value for `turtle` can be something like `'t1'` 
#    while the value for the `movement` can be something like `'uullrrdd'`.
# - `run()` which executes all the movements in the queue.

# Class definition: Modified TurtleWorld
class TurtleWorld:
    valid_movements = set('udlr')
    movement_map = {'u': 'up', 'd': 'down', 'l': 'left', 'r': 'right'}

    def __init__(self):
        self.turtles = {}
        # add a code to create a Queue for movement
        self.movement_queue = Queue()

    def add_movement(self, turtle, movement):
        # store both turtle and movement as a tuple.
        self.movement_queue.enqueue((turtle, movement))

    def run(self):
        while not self.movement_queue.is_empty:
            name, movement = self.movement_queue.dequeue()
            self.move_turtle(name, movement)

    def move_turtle(self, name, movement):
        for i in range(len(movement)):
            if movement[i] in self.valid_movements:
                self.turtles[name].move(self.movement_map[movement[i]])
        return self.turtles[name].pos

    def add_turtle(self, name, speed):
        self.turtles[name] = RobotTurtle(name, speed)

    def remove_turtle(self, name):
        del self.turtles[name]

    def list_turtles(self):
        sorted_names = sorted(self.turtles.keys())
        return sorted_names
    
# test cases 
world = TurtleWorld()
assert isinstance(world.movement_queue, Queue)

world.add_turtle('t1', 1)
world.add_turtle('t2', 2)
world.add_movement('t1', 'ur')
world.add_movement('t2', 'urz')
print(str(world.movement_queue))
print(str(world.movement_queue.size))
assert str(world.turtles['t1'].pos) == '(0, 0)'
assert str(world.turtles['t2'].pos) == '(0, 0)'
assert world.movement_queue.size == 2

world.run()
assert str(world.turtles['t1'].pos) == '(1, 1)'
assert str(world.turtles['t2'].pos) == '(2, 2)'

world.add_movement('t1', 'ur')
world.add_movement('t2', 'urz')

world.run()
assert str(world.turtles['t1'].pos) == '(2, 2)'
assert str(world.turtles['t2'].pos) == '(4, 4)'

# HW5. Implement a radix sorting machine. A radix sort for base 10 integers is a *mechanical* sorting technique that utilizes a collection of bins:
# - one main bin 
# - 10 digit-bins

# Each bin acts like a *queue* and maintains its values in the order that they arrive. The algorithm works as follows:
# - it begins by placing each number in the main bin. 
# - Then it considers each value digit by digit. 
#   The first value is removed from the main bin and placed in a digit-bin corresponding to the digit being considered. 
#   For example, if the ones digit is being considered, 534 will be placed into digit-bin 4 and 667 will placed into digit-bin 7. 
# - Once all the values are placed into their corresponding digit-bins, 
#   the values are collected from bin 0 to bin 9 and placed back in the main bin (in that order). 
# - The process continues with the tens digit, the hundreds, and so on. 
# - After the last digit is processed, the main bin will contain the values in ascending order.

# Create a class `RadixSort` that takes in a List of Integers during object instantiation. 
# The class should have the following properties:
# - `items`: is a List of Integers containing the numbers.

# It should also have the following methods:
# - `sort()`: which returns the sorted numbers from `items` as an `list` of Integers.
# - `max_digit()`: which returns the maximum number of digits of all the numbers in `items`. 
#                  For example, if the numbers are 101, 3, 1041, 
#                  this method returns 4 as the result since the maximum digit is four from 1041. 
# - `convert_to_str(items)`: which returns items as a list of Strings (instead of Integers). 
#                            This function should pad the higher digits with 0 when converting an Integer to a String. 
#                            For example if the maximum digit is 4, the following items are converted as follows. 
#                            From `[101, 3, 1041]` to `["0101", "0003", "1041"]`.
# Hint: Your implementation should make use of the generic `Queue` class, which you created, for the bins.

# Class definition: RadixSort
class RadixSort:

    def __init__(self, MyList) -> None:
        self.items = MyList

    def max_digit(self):
        largest_number_of_digits = 0
        for i in self.items:
            if len(str(i)) > largest_number_of_digits:
                largest_number_of_digits = len(str(i))
        return largest_number_of_digits
    
    def convert_to_str(self, items):
        number = self.max_digit()
        return [(number - len(str(x))) * "0" + str(x) for x in items if len(str(x)) <= number]
    
    def sort(self):
        number = self.max_digit()
        ls = self.convert_to_str(self.items)
        main_bin = Queue()
        zero_bin = Queue()
        one_bin = Queue()
        two_bin = Queue()
        three_bin = Queue()
        four_bin = Queue()
        five_bin = Queue()
        six_bin = Queue()
        seven_bin = Queue()
        eight_bin = Queue()
        nine_bin = Queue()
        digit_dict = {'0': zero_bin,
                      '1': one_bin,
                      '2': two_bin,
                      '3': three_bin,
                      '4': four_bin,
                      '5': five_bin,
                      '6': six_bin,
                      '7': seven_bin,
                      '8': eight_bin,
                      '9': nine_bin}
        for i in ls:
            # enqueue elements from list to main queue e.g. 1001, 1302, 1101
            main_bin.enqueue(i)     

        # loop through each digits place of numbers
        for i in range(number - 1, -1, -1):
            for _ in range(len(ls)):
                digit_dict[main_bin.peek()[i]].enqueue(main_bin.dequeue())
            
            for key in digit_dict:
                while not digit_dict[key].is_empty:
                    main_bin.enqueue(digit_dict[key].dequeue())

        return [int(main_bin.dequeue()) for x in range(len(ls))]
    
# test cases 
list1 = RadixSort([101, 3, 1041])
assert list1.items == [101,3,1041]
assert list1.max_digit() == 4
assert list1.convert_to_str(list1.items) == ["0101", "0003", "1041"]
ans = list1.sort()
print(ans)
assert ans == [3, 101, 1041]
list2 = RadixSort([23, 1038, 8, 423, 10, 39, 3901])
assert list2.sort() == [8, 10, 23, 39, 423, 1038, 3901]
