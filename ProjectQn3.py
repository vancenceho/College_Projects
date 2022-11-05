# Import the random module so that we can have access to the function,
# randint, to generate a random integer needed for the question. 
import random

# Create a class name Queue.
class Queue:

    # To initialize myself, making shopper_queue alive.
    def __init__(self):
        self.container = []

    # Check if the Stack size is empty by comparing if the values inside = 0.
    def isEmpty(self):
        return self.size() == 0  

    # Push the item into the Queue.
    def enqueue(self, item):
        self.container.append(item)

    # Popping the first variable in the Queue.
    def dequeue(self):
        return self.container.pop(0)

    # Return the size of the Stack.
    def size(self):
        return len(self.container)

    # To check the variable that is last of the Queue.
    def peek(self) :
        return self.container[0]

# Create variables using Queue class.
shopper_queue = Queue()

# Intializing a variable call shopperindex for the number of shoppers.
shopperindex = 0

# Pushing all the shoppers into the shopper_queue.
shopper_queue.enqueue("Adam")
shopper_queue.enqueue("Ben")
shopper_queue.enqueue("Cassie")
shopper_queue.enqueue("Deborah")
shopper_queue.enqueue("Elvin")

# Print the strings.
print("Welcome to Pong Pong Shopping Mall Weekly Lucky Draw!")
print("")

# For loop to loop according to the shopper_queue size,
for i in range(shopper_queue.size()) : 
    # shopper index adds 1 to itself everytime it loops to have the shopper's index numbers,
    shopperindex += 1
    # print the string "Shopper #" and the shopper's index and the shoppers in the shopper_queue.
    print("Shopper #" + str(shopperindex) + ": " + shopper_queue.dequeue())

# Push all the shoppers into the shopper_queue again, since it was dequeued just now to print their names.  
shopper_queue.enqueue("Adam")
shopper_queue.enqueue("Ben")
shopper_queue.enqueue("Cassie")
shopper_queue.enqueue("Deborah")
shopper_queue.enqueue("Elvin")

# Print the strings.
print("")
print("Lucky Draw Contest starts...")
print("")

# Intializing the counters for the calculations of statistics later on in the question.
counter1 = 0
counter2 = 0
counter3 = 0

# Another for loop to loop the number of times of the size of the shopper_queue,
for i in range(shopper_queue.size()):
    # Generate a random integer between 1 and 2 and assign it to the variable "n",
    n = random.randint(1,2)
    # If n equals to 1,
    if n == 1:
        # print the string "Calling" and the shopper name,
        print("Calling " + shopper_queue.dequeue() + "...")
        # since n equals to 1, print the answer "Yes"
        print("Answer the call: Yes")
        # then generate another integer between 1 and 2 again and assign it to the variable "x",
        x = random.randint(1,2)
        # if x equals to 1,
        if x == 1:
            # counter for number of people who picked up the call and answered correctly will increase by 1,
            counter1 += 1
            # then print the following strings, saying the results of the answer and ending speech. 
            print("Answer the question correctly: Yes")
            print("Congratulations!")
        # else x equals to 2,
        else:
            # counter for number of people who picked up the call and answered wrongly will increase by 1,
            counter2 +=1
            # print the following strings of the result and ending speech.
            print("Answer the question correctly: No")
            print("Have a nice day!")

    # Else if n equals to 2,
    else:
        # Intialize a variable name "rejoin" and dequeue the shopper_queue, assigning the variable 
        # that was dequeued to "rejoin".
        rejoin = shopper_queue.dequeue()    
        # Print the shopper that was dequeued,
        print("Calling " + rejoin + "...")
        # Requeue the caller that did not pick up by:
        # pushing the shopper that was dequeued back into the shopper_queue because they did not picked up the phone.
        shopper_queue.enqueue(rejoin)
        # Counter for people who did not picked up the call will increase by 1.
        counter3 += 1
        # Print the following strings of the result.
        print("Answer the call: No")
        print("Call later")

    print("")

# Print strings, 
print("Still in queue = ")
# For loop to loop the amount of times the size of the shopper_queue, which is the people who did not
# picked up their phones.
for i in range(shopper_queue.size()):
    # Print  the shoppers who are still in the queue.
    print(shopper_queue.dequeue())

# Print the following strings,    
print("")
# Calculations for the number of shoppers who did not pick up the call,
print("Numbers of shoppers who did not pick up the call = " + str(counter3))
# Percentage of shoppers who picked up the call and answered correctly,
print("Percentage of shoppers who picked up the call and answered correctly = " + str((counter1 / 5) * 100)) 
# Percentage of shoppers who picked up the call and answered wrongly.
print("Percentage of shoppers who picked up the call and did not answered correctly = " + str((counter2 / 5) * 100)) 