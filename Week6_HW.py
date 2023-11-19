# **HW1.** Extend the class `Fraction` to implement the other operators: `- * < <= > >=`.

def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


class Fraction:

    # copy the rest of the methods here
    
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
    
    def gcd(self, a, b):
        if b == 0:
            return a
        else:
            return self.gcd(b, a % b)
    
    def simplify(self):
        divisor = self.gcd(self.num, self.den)
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
    
    def __sub__(self, other):
        a = self.num
        b = self.den
        c = other.num
        d = other.den
        new_numerator = (a * d) - (b * c)
        new_denominator = (b * d)
        new_fraction = Fraction(new_numerator, new_denominator)
        result = new_fraction.simplify()
        return result
    
    def __mul__(self, other):
        a = self.num
        b = self.den
        c = other.num
        d = other.den
        new_numerator = a * c
        new_denominator = b * d
        new_fraction = Fraction(new_numerator, new_denominator)
        result = new_fraction.simplify()
        return result
    
    # def __eq__(self, other):
    #     ###
    #     ### YOUR CODE HERE
    #     ###
    #     pass
    
    def __lt__(self, other):
        a = self.num
        b = self.den
        c = other.num
        d = other.den
        return (a * d) < (b * c)
    
    def __le__(self, other):
        a = self.num
        b = self.den
        c = other.num
        d = other.den
        return (a * d) <= (b * c)
    
    def __gt__(self, other):
        a = self.num
        b = self.den
        c = other.num
        d = other.den
        return (a * d) > (b * c)
    
    def __ge__(self, other):
        a = self.num
        b = self.den
        c = other.num
        d = other.den
        return (a * d) >= (b * c)
    
# test cases 
f1 = Fraction(3, 4)
f2 = Fraction(1, 2)
f3 = f1 - f2
assert f3 == Fraction(1, 4)
f4 = f1 * f2
assert f4 == Fraction(3, 8)
assert f2 < f1
assert f2 <= f2
assert f1 > f3
assert f3 >= f3

# Copy your solution from the Cohort problem

class MixedFraction(Fraction):
    def __init__(self, top, bot, whole=0):
        num = (whole * bot) + top
        super().__init__(num, bot)

    def get_three_numbers(self):
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
assert result.num == 10 and result.den == 3

result = mf1 * mf2
assert result.num == 25 and result.den == 9

mf3 = MixedFraction(1, 2, 1)
result = mf1 - mf3
assert result.num == 1 and result.den == 6

assert str(mf1) == "1 2/3"

# HW2. Write a class called `EvaluateFraction` that evaluates postfix notation implemented using Dequeue data structures only. 
# Postfix notation is a way of writing expressions without using parenthesis. For example, the expression `(1+2)*3` would be written as `1 2 + 3 *`. 
# The class `EvaluateFraction` has the following method:
# - `input(inp)`: which pushes the input input one at a time. For example, to create a postfix notation `1 2 + 3 *`, we can call this method repetitively, 
#    e.g. `e.input('1'); e.input('2'); e.input('+'); e.input('3'); e.input('*')`. Notice that the input is of String data type. 
# - `evaluate()`: which returns the output of the expression.
# - `get_fraction(inp)`: which takes in an input string and returns a `Fraction` object. 
# Postfix notation is evaluated using a Stack. Since `Dequeue` can be used for both Stack and Queue, we will implement using `Dequeue`. 
# The input streams from `input()` are stored in a Queue, which we will again implement using Dequeue. If the output of the Queue is a number, 
# the item is pushed into the stack. If it is an operator, we will apply the operator to the two top most item n the stacks and push the result back into the stack. 

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
    
class Queue:
    def __init__(self):
        # OUT Stack
        self.left_stack = Stack()
        # IN Stack
        self.right_stack = Stack()

    @property
    def is_empty(self):
        return self.left_stack.is_empty and self.right_stack.is_empty

    @property
    def size(self):
        if self.left_stack.is_empty:
            return self.right_stack.size
        return self.left_stack.size

    def enqueue(self, item):
        self.right_stack.push(item)

    def dequeue(self):
        if self.left_stack.is_empty:
            while self.right_stack.size > 0:
                self.left_stack.push(self.right_stack.pop())
        return self.left_stack.pop()

    def peek(self):
        if self.is_empty:
            return None
        
        if self.left_stack.is_empty:
            return self.right_stack._Stack__items[0]
        else:
            return self.left_stack.peek()
        
class Deque(Queue):
  
    def add_front(self, item):
        self.left_stack.push(item)
      
    def remove_front(self):
        if not self.is_empty:
            return self.dequeue()
        return ""
    
    def add_rear(self, item):
        self.enqueue(item)
    
    def left_to_right(self):
        while not self.left_stack.is_empty:
            self.right_stack.push(self.left_stack.pop())
    
    def remove_rear(self):
        if self.right_stack.is_empty:
            self.left_to_right()
        return self.right_stack.pop()
    
    def peek_front(self):
        return self.peek()
    
    def peek_rear(self):
        while self.left_stack.size > 0:
            self.left_to_right()
            return self.right_stack._Stack__items[-1]
        return None
    
class EvaluateFraction:

    operands = "0123456789"
    operators = "+-*/"
    
    def __init__(self):
        self.expression = Deque()
        self.stack = Deque()
    
    def input(self, item):
        self.expression.add_rear(item)
    
    # OUTPUT: assert pe.evaluate()==Fraction(7, 6)
    def evaluate(self):
        while not self.expression.is_empty:
            current = self.expression.remove_front()

            if current == "+":
                second = self.stack.remove_front()
                first = self.stack.remove_front()
                self.stack.add_front(first + second)
            elif current == "-":
                second = self.stack.remove_front()
                first = self.stack.remove_front()
                self.stack.add_front(first - second)
            elif current == "*":
                second = self.stack.remove_front()
                first = self.stack.remove_front()
                self.stack.add_front(first * second)
            elif current == "/":
                second = self.stack.remove_front()
                first = self.stack.remove_front()
                self.stack.add_front(first / second)
            else:
                self.stack.add_front(self.get_fraction(current))

        return self.stack.remove_front()
    
    def get_fraction(self, inp):
        if "/" in inp:
            num, den = inp.split("/")
            num = int(num)
            den = int(den)
            return Fraction(num, den)
        else:
            raise ValueError(f"Invalid fraction format: {inp}")
    
#     def process_operator(self, op1, op2, op):
#         fraction1 = self.get_fraction(op1)
#         fraction2 = self.get_fraction(op2)
        
#         if op == "+":
#             return fraction1 + fraction2
#         elif op == "-":
#             return fraction1 - fraction2
#         elif op == "*":
#             return fraction1 * fraction2
#         elif op == "/":
#             return fraction1 / fraction2
#         else:
#             raise ValueError("Invalid operator")

# test cases
pe = EvaluateFraction()
pe.input("1/2")
pe.input("2/3")
pe.input("+")
assert pe.evaluate()==Fraction(7, 6)

pe.input("1/2")
pe.input("2/3")
pe.input("+")
pe.input("1/6")
pe.input("-")
assert pe.evaluate()==Fraction(1, 1)

pe.input("1/2")
pe.input("2/3")
pe.input("+")
pe.input("1/6")
pe.input("-")
pe.input("3/4")
pe.input("*")
assert pe.evaluate()==Fraction(3, 4)

# HW3. Modify HW2 so that it can work with MixedFraction. Write a class called `EvaluateMixedFraction` as a subclass of `EvaluateFraction`. 
# You need to override the following methods:
# - `get_fraction(inp)`: This function should be able to handle string input for MixedFraction such as `1 1/2` or `3/2`. 
#    It should return a `MixedFraction` object.
# - `evaluate()`: This function should return `MixedFraction` object rather than `Fraction` object. 

class EvaluateMixedFraction(EvaluateFraction):
    def get_fraction(self, inp):
        if " " in inp:
            whole, frac = inp.split(" ")
            whole = int(whole)
            num, den = frac.split("/")
            num = int(num)
            den = int(den)
            return MixedFraction(num, den, whole)
        elif "/" in inp:
            num, den = inp.split("/")
            num = int(num)
            den = int(den)
            return MixedFraction(num, den, 0)
        else:
            whole = int(inp)
            return MixedFraction(0, 1, whole)
    
    def evaluate(self):
        answer = super().evaluate()
        return MixedFraction(answer.num, answer.den)
    
# test cases
pe = EvaluateMixedFraction()
pe.input("3/2")
pe.input("1 2/3")
pe.input("+")
assert pe.evaluate() == MixedFraction(1, 6, 3)

pe.input("1/2")
pe.input("2/3")
pe.input("+")
pe.input("1 1/8")
pe.input("-")
assert pe.evaluate() == MixedFraction(1, 24)

pe.input("1 1/2")
pe.input("2 2/3")
pe.input("+")
pe.input("1 1/6")
pe.input("-")
pe.input("5/4")
pe.input("*")
assert pe.evaluate() == MixedFraction( 3, 4, 3)

# HW4. *Linked List:* We are going to implement Linked List Abstract Data Type. 
#                     To do so, we will implement two classes: `Node` and `MyLinkedList`. 
#                     In this part, we will implement the class Node.

# The class `Node` has the following attribute and computed property:
# - `element`: which stores the value of the item in that node.
# - `next`: which stores the reference to the next `Node` in the list. The setter method should check if the value assigned is of type `Node`.

class Node:
    def __init__(self, e):
        self.element = e
        self.__next = None
               
    @property
    def next(self):
        return self.__next
    
    @next.setter
    def next(self, value):
        # check if value is an instance of Node object
        # you can use isinstance() function
        if isinstance(value, Node) or value is None:
            self.__next = value
        else:
            raise ValueError("Value assigned must be of type Node or None.")

# test cases
n1 = Node(1)
n2 = Node(2)
n3 = Node(3)
assert n1.element == 1 and n2.element == 2 and n3.element == 3
n1.next = n2
n2.next = n3
assert n1.next == n2 and n2.next == n3

# HW5. This is a continuation to implement a Linked List. The class `MyLinkedList` has two different properties:
# - `head`: which points to the `Node` of the first element.
# - `tail`: which points to the `Node` of the last element.

# It should also have the following methods:
# - `__init__(items)`: which create the link list object based using the arguments.
# - `_get(index)`: which returns the item at the given `index`.
# - `_add_first(item)`: which adds the `item` as the first element.
# - `_add_last(item)`: which adds the `item` as the last element.
# - `_add_at(index, item)`: which adds the `item` at the position `index`.
# - `_remove_first(item)`: which removes the `item` as the first element.
# - `_remove_last(item)`: which removes the `item` as the last element.
# - `_remove_at(index, item)`: which removes the `item` at the position `index`.
# - `_index_of(item)`: which returns the index of the given item and is called by `remove(item)` in the parent class.


import collections.abc as c
from abc import abstractmethod

class MyAbstractList(c.Iterator):

    
    def __init__(self, list_items):
        # iterate over every element and call self.add(item)
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

class MyLinkedList(MyAbstractList):

    def __init__(self, items):
        self.head = None
        self.tail = None
        super().__init__(items)

    def _get(self, index):
        # 1. traverse to the node at the index
        current = self.head
        for i in range(index):
            current = current.next
        # 2. return the element of the node
        return current.element
    
    def _add_first(self, element):
        # 1. create a new Node object using element
        new_node = Node(element)
        # 2. set the current head reference as the next reference of the new node
        new_node.next = self.head
        self.head = new_node
        # 3. increase size by 1
        self.size += 1
        # 4. if this is the last element (no tail) -> set current node as the tail
        if self.tail is None:
            self.tail = new_node

    def _add_last(self, element):
        # 1. create a new Node object using element
        new_node = Node(element)
        # 2. if there is no element as tail -> set the new node as both tail and head
        if self.head is None:
            self.head = new_node
        else:
            self.tail.next = new_node
        # 3. otherwise -> set the new node as the next reference of the tail + set the next reference of the current node as the tail's reference
        self.tail = new_node
        # 4. increase size by 1
        self.size += 1

    def _add_at(self, index, element):
        if index == 0:
            # insert at first position, call add_first() method
            self._add_first(element)
        elif index >= self.size:
            # if insert at last position, call add_last() method
            self._add_last(element)
        else:
            # if insert in between:
            # 1. start from head, traverse down the linked list to get the reference at position index - 1 using its next reference
            current = self.head
            for j in range(index - 1):
                current = current.next
            # 2. Create a new Node
            new_node = Node(element)
            # 3. set the next of the current node as the next of the new node
            new_node.next = current.next
            # 4. set the new node as the next of the current node
            current.next = new_node
            # 5. increase the size by 1
            self.size += 1

    def _set_at(self, index, element):
        current = self.head
        for _ in range(index):
            current = current.next
        current.element = element

    # def _set_at(self, index, element):
    #     if 0 <= index < self.size:
    #         current = self._get(index)
    #         current.element = element

    def _remove_first(self):
        if self.size == 0:
            # if list is empty, return the None
            return None
        else:
            # otherwise do the following:
            # 1. store the head at a temporary variable
            removed_variable = self.head.element
            # 2. set the next reference of the current head to be the head
            self.head = self.head.next
            # 3. reduce the size by 1
            self.size -= 1
            # 4. if the new head is now None, it means empty list
            if self.size == 0:
                # set the tail to None also
                self.tail = None
            # 5. return the element of the removed node
            return removed_variable
        
    def _remove_last(self):
        if self.size == 0:
            # if the list is empty, return None 
            return None
        elif self.size == 1:
            # if there is only 1 element, just remove the one node using some other method
            return self._remove_first()
        else:
            # otherwise do the following
            current = self.head
            # 1. traverse to the second last node
            for k in range(self.size - 2):
                current = current.next
            # 2. store the tail of the list to a random variable
            removed_variable = current.next.element
            # 3. set the current node as the tail
            self.tail = current
            # 4. set the next reference of the tail to be None 
            current.next = None
            # 5. reduce the size by 1
            self.size -= 1
            # 6. return the element of the removed node
            return removed_variable
        
    def _remove_at(self, index):
        if index < 0 or index >= self.size:
            return None
        elif index == 0:
            return self._remove_first()
        elif index == self.size - 1:
            return self._remove_last()
        else:
            # otherwise do the following
            current = self.head
            # 1. traverse to the node at index - 1
            for _ in range(index - 1):
                # 2. get the node at index using next reference
                current = current.next
            removed_variable = current.next.element
            # 3. set the next node of the node at index - 1
            current.next = current.next.next
            # 4. reduce the size by 1
            self.size -= 1
            # 5. return the element of the removed node
            return removed_variable
        
    def _index_of(self, item):
        # 1. initialize index to 0 and current node to be head node
        current = self.head
        index = 0
        # 2. iterate to the end of the linked list
        while current is not None:
            # 3. if the item is the same as current's node element, return the current index
            if current.element == item:
                return index
            # 4. otherwise -> increase the index and move current node to the next element
            index += 1
            current = current.next
        # if we loop to the end and have not exit, return -1 
        return -1 
    
# test cases
asean = MyLinkedList(['Singapore', 'Malaysia'])
assert asean.head.element == 'Singapore'
assert asean.tail.element == 'Malaysia'

asean.append('Indonesia')
assert asean.tail.element == 'Indonesia'
asean._add_at(0, 'Brunei')
assert asean.head.element == 'Brunei'
assert asean.size == 4
assert len(asean) == 4
assert asean.remove('Singapore')
assert len(asean) == 3
assert asean[1] == 'Malaysia'
asean._add_at(1, 'Singapore')

asean[0] = 'Cambodia'
assert asean[0] == 'Cambodia' and asean[1] == 'Singapore'
asean[2] = 'Myanmar'
assert(len(asean)) == 4 
assert [x for x in asean] == ['Cambodia', 'Singapore', 'Myanmar', 'Indonesia']


del asean[0]
assert [x for x in asean] == ['Singapore', 'Myanmar', 'Indonesia']

asean._add_at(2, 'Brunei')
assert [x for x in asean] == ['Singapore', 'Myanmar', 'Brunei', 'Indonesia']
del asean[3]
assert [x for x in asean] == ['Singapore', 'Myanmar', 'Brunei']
del asean[1]
assert [x for x in asean] == ['Singapore', 'Brunei']
del asean[1]
assert [x for x in asean] == ['Singapore']
del asean[0]
assert [x for x in asean] == []