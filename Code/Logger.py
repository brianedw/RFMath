#!/usr/bin/env python
# coding: utf-8

# # Logger

# The logger will generate a detailed printed history of the evaluation function or series of functions.  The main advantage of it over using simply print statements is that it provides automatic indenting as one increases the depth of the context.  This dramatically increases the readabilty of the output for complicated.
# 
# It provides special indented printing for numpy arrays.

# In[ ]:


import numpy as np


# In[ ]:


class Logger:

    def __init__(self, indentStep, printQ):
        self.indentStep = indentStep
        self.indentLevel = 0
        self.indentTxt = " ."+" "*(indentStep-2)
        self.printQ = printQ

    def open(self, name, reset=False):
        """
        Increases indent level at function level.
        Applied at the beginning of a function definition
        """
        if reset:
            self.indentLevel = 0
        if self.printQ:
            print(self.indentTxt*self.indentLevel, "==", name, "==")
        self.indentLevel += 1

    def openContext(self, message):
        """
        Increases indent level with message.
        Applied just prior to a local context such as "if" or "for"
        """
        if self.printQ:
            print(self.indentTxt*self.indentLevel, message)
        self.indentLevel += 1

    def print(self, *msg):
        """
        Generic message.  Unpacks the list similar to native 'print("Hello", "World")'
        """
        if self.printQ:
            print(self.indentTxt*self.indentLevel, *msg)

    def printVar(self, name, var):
        """
        Prints "name: var"
        """
        if self.printQ:
            print(self.indentTxt*self.indentLevel+" " + name + ":", var)

    def printVarX(self, name, scope):
        """
        Prints "name: eval(name)"
        """
        if self.printQ:
            print(self.indentTxt*self.indentLevel+" "+ name + ":", eval(name, scope))

    def printNPArray(self, name, array, *options):
        """
        Prints
        name:
          [[1,2]
           [3,4]]
        """
        padding = self.indentTxt*self.indentLevel+"   "
        txt = padding + np.array2string(array, *options)
        txt2 = txt.replace("\n", "\n"+padding)
        if self.printQ:
            print(self.indentTxt*self.indentLevel+" "+ name + ":")
            print(txt2)

    def close(self):
        """
        Closes function level indent.
        """
        self.indentLevel -= 1
        if self.printQ:
            print(self.indentTxt*self.indentLevel, "====")
    
    def closeContext(self):
        """
        Closes context level indent.
        """
        self.indentLevel -= 1
        if self.printQ:
            print(self.indentTxt*self.indentLevel, "----")


# In[ ]:


def foo():
    log.open("foo", reset=True)
    log.print("I'm in foo.")
    a = 2
    b = 4
    log.printVar("a", a)
    log.printVarX("a", locals())
    log.print("The variable 'a' is", a, "and 'b' is", b)
    bar()
    log.openContext("in loop")
    for i in range(2):
        j = i**2
        log.printVar("j", j)
    log.closeContext()
    log.close()

def bar():
    log.open("bar")
    log.print("in bar")
    a = np.array([[1,2],[3,4]])
    log.printNPArray("a", a)
    log.close()


# In[ ]:


# Lines below are commented out so as to not print during import.  Uncomment for testing
# log = Logger(4, True)
# foo()


# In[ ]:


log = Logger(4, False)
foo()

