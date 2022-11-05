# Import the random module so that we can have access to the function,
# randint, to generate a random integer needed for the question.  
import random

# Create a class name Stack
class Stack:

    # To initialize myself, making enqueue_stack & play_stack alive.
    def __init__(self):
        self.container = [] 

    # Check if the Stack size is empty by comparing if the values inside = 0.
    def isEmpty(self):
        return self.size() == 0   

    # Push the item into the Stack.
    def push(self, item): 
        self.container.append(item)

    # To check the variable that is top of the Stack.
    def peek(self):
        if self.size() > 0 :
            return self.container[-1]
        else :
            return None

    # Pop the top variable that is in the Stack. 
    def pop(self): 
         return self.container.pop()  

    # Return the size of the Stack.
    def size(self):
         return len(self.container)

# Create variables using the Stack class.
# Create two stacks named enqueue_stack & play_stack.
enqueue_stack = Stack()
play_stack = Stack()

# Push all the songs into the enqueue_stack.
enqueue_stack.push("#5 - Just The Way You Are")
enqueue_stack.push("#4 - I believe I can fly")
enqueue_stack.push("#3 - Faded")
enqueue_stack.push("#2 - Firework")
enqueue_stack.push("#1 - One Call Away")

# For loop to loop through the size of the stack.
for i in range(enqueue_stack.size()):
    # Generate a random number between 1 to 10 and assign it to the variable "x".
    x = random.randint(1, 10)
    # If x is more than 0 and less than 9, 
    if x > 0 and x < 9:
        # pop the song(s) that is in the enqueue_stack and push it into the play_stack. 
        play_stack.push(enqueue_stack.pop())
        # Print "x" which is the random number generated. 
        print("Random Number:")
        print(x)
    # Else if "x" equals to 9 or 10,    
    else:
        # print "x" which is the random number generated
        print("Random Number:")
        print(x)
        # and break the loop. 
        break
    
    # print the string.
    print("")

# print the strings.
print("")
print("Have a nice day!")
print("") 

# Once the for loop ends, IF the play_stack size does not equals to 0,
if play_stack.size() != 0:
    # print the string. 
    print("We have played the following song(s):")

# Another for loop to loop the amount of times accordingly to the size of the play_stack, 
for i in range(play_stack.size()):
    # print the song(s) that have been played, which are in the play_stack.
    print(play_stack.pop())
