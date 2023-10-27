# WEEK 6 PROBLEM SET - COHORT PROBLEM

# Exercise 1
class Dog:
    def __init__(self, name, sound):
        self._name = name
        self._sound = sound 
    
    def bark(self):
        return self._name + " says " + self._sound.make_sound(self._sound) 

class Woof:
    def make_sound(self):
        return "woof woof"

a = Dog("Husky", Woof)
print(a.bark())

# Exercise 2
class Dog:
    def __init__(self, name):
        self._name = name
        self.collar_color = 'white'

    # getter function
    def get_collar_colour(self):
        return self.collar_color

    # setter function
    def set_collar_colour(self, colour):
        if isinstance(colour, str) and colour != '':
            self.collar_color = colour

    def bark(self):
        return self._name + ' says woof!'

d = Dog('Fido')
print(d.collar_color)
d.collar_color = 'black'
d.set_collar_colour = ''
print(d.get_collar_colour())
d.set_collar_colour = 'green'
print(d.get_collar_colour())

# Exercise 3 
# class Dog:
#     def __init__(self, name, colour):
#         self._name = name
#         self._collar_colour = self._acceptable(colour)

#     # decorator for getter for collar_color function
#     @property
#     def collar_color(self):
#         return self._collar_colour

#     # decorator for setter for collar_color function
#     @collar_color.setter
#     def _acceptable(self, colour):
#         if (colour in ['yellow', 'green', 'orange']):
#             return colour
#         else:
#             return 'yellow'

# a = Dog('Norman', 'yellow')
### Error occured
# print(a.__dict__)
# print(a.collar_color)

# Exercise 4
class Counter:
    def __init__(self):
        self._value = 0

    def up(self):
        self._value = self._value + 1

    def __str__(self):
        return f"Counter: {self._value}"

class CounterUpFast(Counter):

    def up_ten(self):
        self._value = self._value + 10

counter = CounterUpFast()
counter.up()
print(counter)
counter.up_ten()
print(counter)

# Exercise 5
class A:
    
    def __init__(self, value):
        print("__init__ of A")
        self._a = value

    def get_a(self):
        return self._a

    def __str__(self):
        return "A: a = " + str(self._a)

class B(A):

    def __init__(self, value):
        print("__init__ of B")
        super().__init__(value)
        self._b = value

    def get_b(self):
        return self._b

    # overwriting the __str__ method in the parent class - A
    def __str__(self):
        return "B is child class of A: a = " + str(self._a) + " b = " + str(self._b)

b = B(1)
print(b.get_a())
print(b.get_b())
print(b)

# Exercise 6 - Abstract Method
from abc import ABC, abstractmethod

class Pet(ABC):
    @abstractmethod
    def sound(self):
        pass

class Dog(Pet):

    def sound(self):
        return "woof"

class Fox(Pet):

    def sound(self):
        return "geringdingding"

class Cat(Pet):

    def sound(self):
        return "meow"

class AnimalFarm:

    def make_me_talk(self, animal):
        # remember, Pet is an abstract class.
        if isinstance(animal, Pet):
            print(animal.sound())

farm = AnimalFarm()
fox = Fox()
farm.make_me_talk(fox)

# CS1. Create a class called Fraction to represent a simple fraction. 
# The class has two properties:
# num: which represents a numerator and of the type Integer.
# den: which represents a denominator and of the type Integer. 
#      Denominator should not be a zero. If a zero is assigned, you need to replace it with a 1.

# The class should have the following method:
# __init__(num, den): to initialize the numerator and the denominator. 
#                     You should check if the denominator is zero. 
#                     If it is you should assign 1 as the denominator instead.
# __str__(): for the object instance to be convertable to String. 
#            You need to return a string in a format of num/den

class Fraction:

    def __init__(self, num, den):
        self._num = num
        self._den = self._check_den(den)

    def _check_den(self, value):
        if int(value) == 0:
            return 1
        else:
            return value

    @property
    def num(self):
        return self._num

    @num.setter
    def num(self, val):
        self._num = int(val)

    @property
    def den(self):
        return self._den

    @den.setter
    def den(self, val):
        self._den = self._check_den(val)

    def __str__(self):
        return f"{self._num}/{self._den}"

a = Fraction(2, 3)
print(a._num, a._den)
a = Fraction(2, 0)
print(a._num, a._den)
print(a.num, a.den)
a = Fraction(5, 2)
print(a)

# CS2. Extend the class Fraction to support the following operator: + and ==. 
# To do this, you need to overload the following operator:
# __add__(self, other)
# __eq__(self, other)

# You may want to write a method to simplify a fraction:
# simplify(): which simplify a fraction to its lowest terms. 
#             To simplify a fraction divide both the numerator and the denominator 
#             with the greatest common divisor of the the two. 
#             This method should return a new Fraction object.

class Fraction:

    def __init__(self, num, den):
        self._num = num
        self._den = self._check_den(den)

    def _check_den(self, value):
        if int(value) == 0:
            return 1
        else:
            return value

    @property
    def num(self):
        return self._num

    @num.setter
    def num(self, val):
        self._num = int(val)

    @property
    def den(self):
        return self._den

    @den.setter
    def den(self, val):
        self._den = self._check_den(val)

    def __str__(self):
        return f"{self._num}/{self._den}"

    def _gcd(self, a, b):
        if b == 0:
            return a
        else:
            return self._gcd(b, a % b)

    def simplify(self):
        divisor = self._gcd(self.num, self.den)
        new_numerator = int(self.num / divisor)
        new_denominator = int(self.den / divisor)
        return Fraction(new_numerator, new_denominator)
    
    def __add__(self, other):
        a = self.num
        b = self.den
        c = other.num
        d = other.den
        new_numerator = (a * d) + (b * c)
        new_denominator = (b * d)
        new_fraction = Fraction(new_numerator, new_denominator)
        result = new_fraction.simplify()
        return result

    def __eq__(self, other):
        first_fraction = self.simplify()
        second_fraction = other.simplify()
        is_num_equal = first_fraction.num == second_fraction.num
        is_den_equal = first_fraction.den == second_fraction.den
        
        return is_num_equal and is_den_equal

f1 = Fraction(1, 6)
f2 = Fraction(2, 3)
f3 = f1 + f2
print(f3)
print(f1 == f2)
f4 = Fraction(2, 4)
f5 = Fraction(2, 4)
print(f4 == f5)
f6 = Fraction(2, 4)
f7 = Fraction(1, 2)
print(f6 == f7)

# CS3. Inheritance: Create a class called MixedFraction as a subclass of Fraction. 
# A mixed fraction is a fraction that comprises of a whole number, 
# a numerator and a denominator, e.g. 1 2/3 which is the same as 5/3. 
# The class has the following way of initializing its properties:
# __init__(top, bot, whole): which takes in three Integers, the whole number, the numerator, 
#                            and the denominator, e.g. whole=1, top=2, bot=3. 
#                            The argument whole by default is 0. 
#                            You can also specify top to be greater than bot.

# The class only has two properties:
# 1) num: which is the numerator and can be greater than denominator.
# 2) den: which is the denominator and must be a non-zero number.

# The class should also have the following methods:
# 1) get_three_numbers(): which is used to calculate the whole number, numerator and the denominator 
#                         from a given numerator and denominator. The stored properties are num and 
#                         den as in Fraction class. This function returns three Integers as a tuple, 
#                         i.e. (top, bot, whole).

# The class should also override the __str__() method in this manner:
# 1) num/dem if the numerator is smaller than the denominator. For example, 2/3.
# 2) whole top/bot if the numerator is greater than the denominator. For example, 1 2/3.

# Class definition: MixedFraction(Fraction)
class MixedFraction(Fraction):

    def __init__(self, top, bot, whole=0):
        # """e.g. Mixed Fraction 1 3/4 ---> top = 3, bot = 4, whole = 1"""
        # 1 3/4 = 7/4
        num = (whole * bot) + top
        super().__init__(num, bot)

    def get_three_numbers(self):
        # """e.g. 11/8 ---> get mixed fraction = 1 3/8 ---> top = 3, bot = 8, whole = 1"""
        whole = self.num // self.den
        top = self.num % self.den
        bot = self.den
        return (top, bot, whole)

    def __str__(self):
        if self.num < self.den:
            super().__str__()
        else:
            top, bot, whole = self.get_three_numbers()
            return f"{whole} {top}/{bot}"

# test cases
mf1 = MixedFraction(5, 3)
assert mf1.num == 5 and mf1.den == 3
assert mf1.get_three_numbers() == (2, 3, 1)
mf2 = MixedFraction(2, 3, 1)
assert mf2.num == 5 and mf2.den == 3

result = mf1 + mf2
print(result.num, result.den)
assert result.num == 10 and result.den == 3

assert mf1 == mf2

# CS4. Inheritance: Create a class Deque as a subclass of Queue. 
# Use the double-stack implementation of Queue in this problem. 
# Deque has the following methods:
# 1) add_front(item): which add an item to the front of the queue.
# 2) remove_rear(): which pops out an item from the rear of the queue.
# 3) add_rear(item): which add an item from rear of the queue. This is the same as enqueue a normal queue.
# 4) remove_front(): which pops out an item from the front of the queue. 
#                    This is the same as dequeue method in a normal queue.
# 5) left_to_right(): which is a helper function to move the items from the left stack to the right stack 
#                     and keep the proper order. Use the Push and Pop operation of Stacks here.
# 6) peek_front() and peak_rear(): which peek the front or the rear of the Deque respectively. 
#                                  It should return None when the Deque is empty.

# Class definition: Stack
class Stack:

    def __init__(self) -> None:
        self.__items = []

    def push(self, item):
        self.__items.append(item)

    def pop(self):
        if self.__items != []:
            return self.__items.pop()
        
    def peek(self):
        if not self.is_empty:
            return self.__items[-1]
        return None
    
    @property
    def is_empty(self):
        return self.__items == []
    
    @property
    def size(self):
        return len(self.__items)

a = Stack()
print(a.is_empty)
a.push(1)
a.push(2)
print(a.is_empty)
print(a.size)
assert a.pop() == 2
print(a.size)
assert a.pop() == 1
print(a.size)
print(a.pop())

# Class definition: Queue -> Using Stack
class Queue:
    
    def __init__(self) -> None:
        # OUT Stack
        self.left_stack = Stack()
        # IN Stack
        self.right_stack = Stack()

    def enqueue(self, item):
        self.right_stack.push(item)

    def dequeue(self):
        # move elements from right to left 
        if self.left_stack.is_empty:
            # while the right stack has elements
            while self.right_stack.size > 0:
                # push whatever elements I have in the right stack to the left stack
                self.left_stack.push(self.right_stack.pop())
        # pop the left stack which is the first element being added in the Queue
        return self.left_stack.pop()
    
    def peek(self):
        # base case 
        if self.is_empty:
            return None
        
        if self.left_stack.is_empty:
            return self.right_stack._Stack__items[0]
        else:
            return self.left_stack.peek()
        
    @property
    def is_empty(self):
        return self.left_stack.is_empty and self.right_stack.is_empty
    
    @property
    def size(self):
        if self.left_stack.is_empty:
            return self.right_stack.size
        return self.left_stack.size

q = Queue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
print(q.is_empty)
print(q.size)
assert q.peek() == 1
print(q.dequeue())
print(q.dequeue())
assert q.peek() == 3

# Class definition: Deque
# class Deque(Queue):
    
    def add_front(self, item):
        self.left_stack.push(item)

    # same as enqueue for Queue() class
    def add_rear(self, item):
        self.enqueue(item)
    
    # same as dequeue for Queue() class
    def remove_front(self):
        return self.dequeue()
    
    def remove_rear(self):
        if self.right_stack.is_empty:
            self._left_to_right()
        return self.right_stack.pop()
    
    # same as peek for Queue() class
    def peek_front(self):
        return self.peek()
    
    def peek_rear(self):
        while self.left_stack.size > 0:
            self._left_to_right()
            return self.right_stack._Stack__items[-1]
        return None
    
    # helper function to move items from left to right stack to keep the proper order
    def _left_to_right(self):
        while not self.left_stack.is_empty:
            self.right_stack.push(self.left_stack.pop()) 

    # def _move_to_left(self):
    #     while self.right_stack.is_empty:
    #         element = self.right_stack.pop()
    #         self.left_stack.push(element)

# CS5-Prelude-1. ArrayFixedSize class (Given): 
# Write a class called ArrayFixedSize that simulate a fixed size array. 
# This class should inherint from collections.abc.Iterable base class. 
# A fixed size array is like a list which size cannot change once it is set. 
# The size and its data type is specified during object instantiation. 
# Use Numpy's array for its internal data storage. 
#  the start you can use np.empty(size) to create an uninitalized empty array. 
# The class should have the following methods:

# __getitem__(index): which returns the element at a given index using the bracket operator, i.e. array[index].
# __setitem__(index, value): which set the item at a given index with a particular value using the bracket 
#                            and assignment operators, i.e. array[index] = value.
# __iter__(): which returns the iterable object so that it can be used in a for loop and 
#             other iterable object operators.
# __str__(): which returns the string object representation of the object. 
#            It should displays as follows: [el1, el2, el3, ...].
# __len__(): which returns the number of items in the array when len() function is called upon this object
import collections.abc as c
import numpy as np

class ArrayFixedSize(c.Iterable):
    
    def __init__(self, size, dtype=int):
        self.__data = np.empty(size)
        self.__data = self.__data.astype(dtype)
        
    def __getitem__(self, index):
        return self.__data[index]
    
    def __setitem__(self, index, value):
        self.__data[index] = value      
        
    def __iter__(self):
        return iter(self.__data)
    
    def __len__(self):
        return len(self.__data)

    def __str__(self):
        out = "["
        for item in self:
            out += f"{item:}, "
        if self.__data != []:
            return out[:-2] + "]"
        else:
            return "[]"

# test case
# simulate a array with a fixed size, using python syntax list
a = ArrayFixedSize(10)
### a[10] = 1 <--- no space allocated for an 11th element.
a[0] = 1 ### calls the __setitem__ magic method
print(a[1]) ### calls the __getitem__ magic method
for element in a: ### calls the __iter__ magic method
    print(a)
print(len(a)) ### calls the __len__ magic method

# CS5-Prelude-2. 
# Implement a class called MyAbstractList which is a subclass of Python's collections.abc.Iterator class. 
# This class is meant to be an abstract class for two kinds of List data structures, 
# i.e. MyArrayList and MyLinkedList. In this exercise, you will implement MyAbstractList.

# This class has the following attribute and computed property property:
# 1) size: which gives you the size of the list in integer.
# 2) is_empty: which returns either True or False depending whether the list is empty or not.

# Implement also several other methods:
# 1) __init__(self, list_of_items): which initializes the list by adding the items in list argument 
#                                   using the append(item) method.
# 2) __getitem__(index): which returns the element at a given index using the bracket operator, 
#                        i.e. array[index]. This method should call the method _get(index) which 
#                        should be implemented by the child class.
# 3) __setitem__(index, value): which set the item at a given index with a particular value using 
#                               the bracket and assignment operators, i.e. array[index] = value. 
#                               This method should call the method _set_at(index, item) which should be 
#                               implemented in the child class.
# 4) __delitem__(index): which removes the item at a given index using the del operator, 
#                        i.e. del array[index]. This method should call the method _remove_at(index) 
#                        which should be implemented in the child class.
# 5) append(item): which adds an item at the end of the list. 
#                  This method should call _add_at(index, item) implemented in the child class which 
#                  adds the item at the specified index.
# 6) remove(item): which removes the first occurence of the item in the list. 
#                  This method should call _index_of(item) and _remove_at(index) implemented in the 
#                  child class which removes the item at the specified index.
# 7) __next__(): which returns the next element in the iterator. 
#                This method should call _get(index) in the child class which returns the item at 
#                the specified index. If there is no more element, it should raise StopIteration.

from abc import abstractmethod

class MyAbstractList(c.Iterator):
    
    def __init__(self, list_items):
        # iterate over every element and call self.append(item)
        self.size = 0
        self._idx = 0
        for item in list_items:
            self.append(item)
    
    @property
    def is_empty(self):
        return self.size == 0
    
    def append(self, item):
        # call add_at() method here
        self._add_at(self.size, item)
        
    def remove(self, item):
        # you should use remove_at() method here
        idx = self._index_of(item)
        if  idx >= 0:
            self._remove_at(idx)
            return True
        else:
            return False
        
    def __getitem__(self, index):
        return self._get(index)
    
    def __setitem__(self, index, value):
        # call set_at(index, value) method here
        self._set_at(index, value)
        
    def __delitem__(self, index):
        # call remove_at() method here
        self._remove_at(index)
    
    def __len__(self):
        return self.size
        
    def __iter__(self):
        self._idx = 0
        return self
        
    def __next__(self):
        if self._idx < self.size:
            n_item = self._get(self._idx)
            self._idx += 1
            return n_item
        else:
            raise StopIteration
    
    # the following methods should be implemented in the child class
    @abstractmethod
    def _get(self, index):
        pass

    @abstractmethod
    def _set_at(self, index, item):
        pass

    @abstractmethod
    def _add_at(self, index, item):
        pass

    @abstractmethod
    def _remove_at(self, index):
        pass

    @abstractmethod
    def _index_of(self, item):
        pass

# creating a class PythonList inheriting from MyAbstractList
# this is just for testing the MyAbstractList class

class PythonList(MyAbstractList):
    
    def __init__(self, init_data=[]):
        self.data = []
        super().__init__(init_data)
    
    def _add_at(self, index, item):
        self.data.insert(index, item)
        self.size += 1
        
    def _set_at(self, index, item):
        self.data[index] = item
        
    def _remove_at(self, index):
        self.data.pop(index)
        self.size -= 1
        
    def _get(self, index):
        if 0 <= index < self.size:
            return self.data[index]
        else:
            raise IndexError()
            
    def _index_of(self, item):
        try:
            idx = self.data.index(item)
            return idx
        except:
            return -1

# test cases
f = PythonList(list(range(10)))

# Testing init
assert f.data == list(range(10))

# Testing size property
assert f.size == 10

# Testing is_empty property
assert not f.is_empty

# Testing append method
f.append(10)
assert f.data == list(range(11))

# Testing remove method
f.remove(5)
assert f.data == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
f._add_at(5, 5)
f.append(5)
f.remove(5)
assert f.data == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 5]
assert not f.remove(11)

# Testing getitem method
assert [f[i] for i in range(len(f))] == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 5]
f[0] = -1
assert [f[i] for i in range(len(f))] == [-1, 1, 2, 3, 4, 6, 7, 8, 9, 10, 5]

# Testing delitem
del f[0]
assert f[0] == 1

# Testing __next__
assert [x for x in f] == [ 1, 2, 3, 4, 6, 7, 8, 9, 10, 5]
# Testing __iter__
assert [x for x in f] == [ 1, 2, 3, 4, 6, 7, 8, 9, 10, 5]

# CS6 
import numpy as np 

class MyArrayList(MyAbstractList):
    INITIAL_CAPACITY = 16

    def __init__(self, items, dtype=int):
        self.data = ArrayFixedSize(MyArrayList.INITIAL_CAPACITY, dtype)
        super().__init__(items)

    def _add_at(self, index, item):
        self._ensure_capacity()
        # do the following:
        # 1. copy data by shifting it to the right from index position to the end
        # 2. set item at index
        # 3. add size by 1
        self._set_at(index, item)
        self.size = self.size + 1

    def _set_at(self, index, value):
        self.data[index] = value

    def _remove_at(self, index):
        if 0 <= index < self.size:
            # do the following
            # 1. get the element at index
            # 2. copy the data by shifting it to the left from index to the end
            # 3. reduce size by 1
            # 4. return the element at index
            element = self.data[index]
            for i in range(index, self.size - 1):
                self.data[i] = self.data[i + 1]
            self.size = self.size - 1
            return element
        else:
            raise IndexError()
        
    def _get(self, index):
        if 0 <= index < self.size:
            return self.data[index]
        else:
            raise IndexError()
        
    def _index_of(self, item):
        # iterate over the data and return the index
        # if not found return -1
        for i in range(self.size):
            if item == self.data[i]:
                return i
        return -1
    
    def _ensure_capacity(self):
        if self.size >= len(self.data):
            new_data = ArrayFixedSize(self.size * 2 + 1)
            self._copy(self.data, 0, new_data, 0)
            self.data = new_data
            
    def _copy(self, source, idx_s, dest, idx_d):
        for idx in range(idx_s,len(source)):
            offset = idx - idx_s
            dest[idx_d + offset] = source[idx]
            
    def _clear(self):
        self.data = ArrayFixedSize(MyArrayList.INITIAL_CAPACITY)
        self.size = 0
        
    def __str__(self):
        out = "["
        for idx in range(self.size):
            out += f"{self.get(idx):}, "
        return out[:-2] + "]"

# test cases
a = MyArrayList([1,2,3])
assert [x for x in a] == [1,2,3]
assert a.size == 3
assert not a.is_empty
a.append(4)
assert a.size == 4
assert [x for x in a] == [1,2,3,4]
assert [a[i] for i in range(len(a))] == [1,2,3,4]
a[0] = -1
assert [x for x in a] == [-1, 2,3,4]
del a[0]
assert [x for x in a] == [2,3,4]
