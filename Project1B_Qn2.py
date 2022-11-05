# Create a class name Book
class Book:
    # To initialize a book each time a Book class is created
    def __init__ (self, author = "", title = "", nextBook = None):
         self.author = author
         self.title = title
         self.nextBook = nextBook

    # To add a book in a specific position.
    def addBookToPosition(self, book, position) :
        # If the position entered is more than 0,
        if position > 0:
            # It will add the new book to the position where the next book is occupying
            self.nextBook.addBookToPosition(book, position - 1)
        # Else when position entered is 0,
        else : 
            # The new book will be added in the next book of the previous position,
            book.nextBook = self.nextBook
            # the previous book will be moved to a position before the new book.
            self.nextBook = book
    
    # To remove a book from a specific position
    def remove(self, position) :
        # If position is > 0,
        if position > 0 :
            # the next book will remove the book before its position
            self.nextBook.remove(position - 1)
        # Else when position is 0,
        else :
            # It moves the position of the book to the position after the next
            self.nextBook = self.nextBook.nextBook

    # It displays the book in a specific way with <> by using recursion
    def displayAll(self) :
        print("<", self.author, ",", self.title + " >")
        if self.nextBook != None :
            self.nextBook.displayAll()

    # To call the sortAuthor function
    def sortAuthor(self, last=None) : 
        sortByAuthor(self,last) 

    # To call the sortTitle function
    def sortTitle(self, last=None) :
        sortByTitle(self, last)


class LinkedList:
    # To initialize the start of a linkedlist
    def __init__ (self, head=None):
        self.head = head
        if head == None :
            self.size = 0
        else :
            self.size = 1

    # To get the size of the linked list
    def getSize (self):
        return self.size

    # To add a book to the front of the list
    def addBook (self, newBook):
        # It moves the orginal first book to the next position of the new book
        newBook.nextBook = self.head
        # It assigns the new book as the first in the list
        self.head = newBook
        # Increase the size of the list by 1
        self.size+=1

    # To insert a book at a specific position
    def insertBook (self,newBook, n) :
        # Calls the addBookToPosition method and add the new book to the position 
        self.head.addBookToPosition(newBook, n - 1)
        # Increase the size of the list by 1
        self.size += 1
    
    # Function to deleteBook at a specific position by comparing
    def deleteBook(self, n):
        # If n == 0,
        if n == 0:
            # The first book would be moved to the next position following
            self.head = self.head.nextBook
        # Else when n is > 0,
        else :
            # it calls the function of remove and removes the book of the previous position
            self.head.remove(n - 1)
        # and it reduces the size of the list by `1
        self.size -= 1

    # Calls the displayAll method to print the books in the list
    def printAll (self):
        self.head.displayAll()

    # Function to sort the books by Author
    def sortAuthor(self) :
        # loop throught the amount of times of the size - 1 of the list
        for index in range(self.size - 1):
            self.head = sortByAuthor(self.head, self.head)

    #Function to sort the books by Title
    def sortTitle(self) :
        # loop through the amount of times of the size - 1 of the list
        for index in range(self.size - 1):
            self.head = sortByTitle(self.head, self.head)    


#This is out of the class so it can call both LinkedList and Book which you need
def sortByAuthor(current, last) :
    # If next book of the curret == to None, just return
    if current.nextBook == None :
        return
    
    # If current author has a hex code larger than the hex code of the next book 
    if current.author > current.nextBook.author :
        x = current
        y = x.nextBook
        z = y.nextBook

        # than set the x to the position after the next of the current book
        current.nextBook.nextBook = x
        # and set z to the next of the current
        current.nextBook = z
        # If the last book is not equals to the current
        if last != current :
            # set the next of the last which is the current to y to rest
            last.nextBook = y
        last = y
    
    # Else current author has a hex code lesser than the author of the next book
    else :
        # set last = to current
        last = current

    last.nextBook.sortAuthor(last)
    
    return last

#This is out of the class so it can call both LinkedList and Book which you need
def sortByTitle(current, last) :
    # If next book of the curret == to None, just return
    if current.nextBook == None :
        return
    
    # If current title has a hex code larger than the hex code of the next book 
    if current.title > current.nextBook.title :
        x = current
        y = x.nextBook
        z = y.nextBook

        # than set the x to the position after the next of the current book
        current.nextBook.nextBook = x
        # and set z to the next of the current
        current.nextBook = z
        # If the last book is not equals to the current
        if last != current :
            # set the next of the last which is the current to y to rest
            last.nextBook = y
        last = y
    
    # Else current title has a hex code lesser than the title of the next book
    else :
        # set last = to current
        last = current

    last.nextBook.sortTitle(last)
    
    return last

booklist = LinkedList()
book1 = Book("Lewis Carroll", "Alice's Adventures in Wonderland")
book2 = Book("Stephenie Meyer", "Breaking Dawn")
book3 = Book("Suzanne Collins", "Catching Fire")
book4 = Book("Veronica Roth", "Divergent")
book5 = Book("Stephenie Meyer", "Eclipse")
book6 = Book("Dan Brown", "Inferno")

booklist.addBook(book1)
booklist.addBook(book2)
booklist.addBook(book3)
booklist.addBook(book4)
booklist.addBook(book5)

print("The following shows the result of part a. of the project for the function AddBookToFront: ")
print("")
booklist.printAll()

print("")
print("The following shows the result of part b. of the project for the function AddBookAtPosition(n): ")
print("")
booklist.insertBook(book6, 3)
booklist.printAll()

print("")
print("The following shows the result of part c. of the project for the function RemoveBookAtPosition(n): ")
print("")
booklist.deleteBook(3)
booklist.printAll()

print("")
print("The following shows the result of part d. of the project for the function DisplayBook: ")
print("")
booklist.printAll()

print("")
print("The following shows the result of part e. of the project for the function SortByAuthorName: ")
print("")
booklist.sortAuthor()
booklist.printAll()

print("")
print("The following shows the result of the additional feature of the project for the function SortByBookTitle: ")
print("")
booklist.sortTitle()
booklist.printAll()

