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
        self._items = []

    def enqueue(self, value):
        if isinstance(value, int):
            self._items.append(value)
    
    def dequeue(self):
        if len(self._items) != 0:
            return self._items.pop(0)
        else:
            return None
    
    def peek(self):
        if len(self._items) != 0:
            return self._items[0]
        else:
            return None
    
    @property
    def is_empty(self):
        if len(self._items) == 0:
            return True
        else:
            return False
    
    @property
    def size(self):
        return len(self._items)

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
        return ("{}, {}".format(self.x, self.y))
    
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
        if name in self.turtles:
            turtle = self.turtles[name]
            for move in movement:
                if move in self.valid_movements:
                    direction = self.movement_map[move]
                    for _ in range(turtle.speed):
                        turtle.move(direction)
                else:
                    print("Ignoring invalid movement: {}".format(move))

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