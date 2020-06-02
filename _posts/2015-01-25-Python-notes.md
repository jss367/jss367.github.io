---
layout: post
title: "Python Cheatsheet"
description: "A basic cheatsheet for programming in Python"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/python.jpg"
tags: [Python, Cheat Sheet]
---

This notebook is a collection of Python notes, tricks, and tips. It is set up to act as a quick reference for basic Python programming.

<b>Table of contents</b>
* TOC
{:toc}

# Basic Math

### Most basic


```python
print(5+5)
print(4-3)
print(2*2)
print(7/3)
print(7//3)
```

    10
    1
    4
    2.3333333333333335
    2
    

Note that the division sign, "/", is used differently in Python 2 and 3. In Python 2, 5/2 is 2. In Python 3, 5 / 2 is 2.5, and 5 // 2 is 2.

### Exponentiation


```python
print(2**3)
```

    8
    

### Modulo


```python
print(7%3)
```

    1
    

# Strings

## Concatenating


```python
a='ab'
c='cd'
print("The easiest way is this: " + a + c)
print("You can also multiply a string by an int to make it repeat: {}".format(3*a))
try:
    a-'a'
except TypeError:
    print('But you cannot do the same with subtraction or division')
```

    The easiest way is this: abcd
    You can also multiply a string by an int to make it repeat: ababab
    But you cannot do the same with subtraction or division
    

## Appending


```python
s=[]
s.append(a)
s.append(c)
print(s)
st = ''.join(s)
print(st)
```

    ['ab', 'cd']
    abcd
    

## Removing and changing letters


```python
#String are immutable, so you will have to create a new string each time
old_str = 'Python sucks'
new_str = old_str.replace('su', 'ro')
print(new_str)
```

    Python rocks
    

## Finding a string inside a string


```python
phrase = 'Now for ourself and for this time of meeting'
if 'meeting' in phrase:
     print('found "meeting"')
```

    found "meeting"
    


```python
# Or find the location of the beginning of the string
phrase.find('for')
```




    4




```python
# Index does the same thing when a string is present
phrase.index('for')
```




    4




```python
# When the string isn't there, they differ
phrase.find('fort')
```




    -1




```python
phrase.index('fort')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-59-d241ba1fe19d> in <module>()
    ----> 1 phrase.index('fort')
    

    ValueError: substring not found



```python
# Find and index the first instance. You can also search for the last instance
phrase.rfind('for')
```




    20



## Splitting and joining strings


```python
sen = 'This is my sentence'
# Split into words
words = sen.split(' ')
print(words)
```

    ['This', 'is', 'my', 'sentence']
    


```python
# Join does the opposite of split
another = ['This', 'is', 'another', 'sentence']
' '.join(another)
```




    'This is another sentence'



## Unique words


```python
# The "set" function does this
# Note that we'll have to convert to all lower case beforehand so it doesn't think "this" and "This" are different words
lower = [w.lower() for w in words]
unique_words = set(lower)
print(unique_words)
print("You have used {} unique words.".format(len(unique_words)))
```

    {'is', 'this', 'sentence', 'my'}
    You have used 4 unique words.
    

## Extract longer words from a sentence


```python
# Extract the words longer than three letters
[w for w in words if len(w) > 3]
```




    ['This', 'sentence']



## Capitalization


```python
# Find capitalized words
print([w for w in words if w.istitle()]) # istitle looks to see if first and only first character is capitalized
```

    ['This']
    


```python
# Note that isupper checks if ALL the letters are uppercase
print('This'.isupper())
print('This'.istitle())
print('UPPER'.isupper())
```

    False
    True
    True
    


```python
print('Great'.lower())
print('great'.title())
print('Great'.upper())
```

    great
    Great
    GREAT
    

## More string methods


```python
# Words that end with "y"
[w for w in words if w.endswith('y')]
```




    ['my']




```python
a = 'too many spaces at the end      '
a.strip()
```




    'too many spaces at the end'



# Lists


```python
# Lists are defined using square brackets
a = [1,2,3,4]
type(a)
```




    list




```python
# Lists can consistent of different types, but this isn't common
a = ['This', 'is', 4, 1+4, 's'*3]
print(a)
```

    ['This', 'is', 4, 5, 'sss']
    

## Indexing


```python
# The first item has an index of 0
print(a[0])
# And the last has an index of one less than the number of element
print(a[len(a)-1])
```

    This
    sss
    

## Slicing

List slicing is done by [start : end]
The first value is included, but the last one isn't, so it's of this form: [inclusive : exclusive]


```python
# list slicing: [start : end] is [inclusive : exclusive], so the last value is the slice is not included
a[1:3]

```




    ['is', 4]




```python
# Lists also include steps: [start : end :step]
a[0:len(a):2]
```




    ['This', 4, 'sss']




```python
# If you leave them blank, the slicing values will be implied as follows
print(a[0:len(a):1])
print(a[::])
```

    ['This', 'is', 4, 5, 'sss']
    ['This', 'is', 4, 5, 'sss']
    


```python
# So you can just enter the part you want to be different than the defaults
print(a[::2])
print(a[:3])
print(a[:4:2])
```

    ['This', 4, 'sss']
    ['This', 'is', 4]
    ['This', 4]
    


```python
# The end value can be more than the length of the list
a[2:10]
```




    [4, 5, 'sss']



## Other list stuff


```python
# Lists are mutable
a[2] = 'hello'
print(a)
```

    ['This', 'is', 'hello', 5, 'sss']
    


```python
# You can also have lists of lists
list_of_lists= [[1,2,3,4,5], ['a','b','c'], a]
print(list_of_lists)
```

    [[1, 2, 3, 4, 5], ['a', 'b', 'c'], ['This', 'is', 'hello', 5, 'sss']]
    


```python

```


      File "<ipython-input-35-ccd4c4cd69ff>", line 2
        list of lists= [['list1', 3], ['list2', 4]]
              ^
    SyntaxError: invalid syntax
    


# Tuples


```python
# Tuples are defined using parentheses
tu = (1, 2, 3)
```


```python
#tuple are immutable, so the following will give you an error:
tu[2] = 5
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-108-6f2f318298ea> in <module>()
          1 #tuple are immutable, so the following will give you an error:
    ----> 2 tu[2] = 5
    

    TypeError: 'tuple' object does not support item assignment



```python
# Use tuples to flip values
a=1
b=2
(b,a) = (a,b)
print("a is {a} and b is {b}".format(a=a,b=b))
```

    a is 2 and b is 1
    

## Finding a value in a tuple


```python
tu.index(3)
```




    2



# Dictionaries

Dictionaries are implementations of hash tables. They are very quick for looking up, inserting, updating, and deleting values


```python
# Dictionaries are defined with curly brackets
# Instead of using indexes, uses key-value pairs
entries = {'Alice' : True, 'Bob' : 0, 'Charlie' : 12}
print(entries['Charlie'])
```

    12
    

## Adding values


```python
# First, check if it is already there
print('New' in entries)
```

    False
    


```python
entries['New'] = 'yes'
```

## Exploring dictionaries


```python
print(entries.keys())
print(entries.values())
```

    dict_keys(['Alice', 'Bob', 'Charlie', 'New'])
    dict_values([True, 0, 12, 'yes'])
    

# Printing

`print` is a function in Python 3, so the text to be printed must have parenthesis around it. In Python 2, this was not that case


```python
print("Hello world!")
```

    Hello world!
    


```python
# In Python 2, you can do this
print "Hello world!"
# But it doesn't work in Python 3
```


      File "<ipython-input-137-e74dbf0a8ff8>", line 2
        print "Hello world!"
                           ^
    SyntaxError: Missing parentheses in call to 'print'
    



```python
#printing multiple lines
lines = """Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar."""
print(lines)
```

    Friends, Romans, countrymen, lend me your ears;
    I come to bury Caesar, not to praise him.
    The evil that men do lives after them;
    The good is oft interred with their bones;
    So let it be with Caesar.
    


```python
# If you use a comma in the print statement you get an extra space
a = 'Et tu, Brute!'
b = 'Then fall, Caesar.'
print(a + b)
print(a, b)
```

    Et tu, Brute!Then fall, Caesar.
    Et tu, Brute! Then fall, Caesar.
    


```python
# You can print with single or double quotes. This makes it easier to print a single or double quote
print('Antony said "This was the most unkindest cut of all."')
print("For when the noble Caesar saw him stab,\n Ingratitude, more strong than traitors' arms")
```

    Antony said "This was the most unkindest cut of all."
    For when the noble Caesar saw him stab,
     Ingratitude, more strong than traitors' arms
    

## Using the format printing method


```python
print('{:.2f}'.format(3.1415))
print('{:.2%}'.format(.333))
print("{perc:.2%}% of {num_verbs} are weak".format(perc=.501, num_verbs=7))
```

    3.14
    33.30%
    50.10%% of 7 are weak
    

# For loops


```python
for i in range(10):
    print(i**2)
```

    0
    1
    4
    9
    16
    25
    36
    49
    64
    81
    


```python
# Also works with text
for word in ['Beware', 'the', 'ides', 'of', 'March']:
    print(word)

```

    Beware
    the
    ides
    of
    March
    




    '\nPython Expression\tComment\nfor item in s\titerate over the items of s\nfor item in sorted(s)\titerate over the items of s in order\nfor item in set(s)\titerate over unique elements of s\nfor item in reversed(s)\titerate over elements of s in reverse\nfor item in set(s).difference(t)\titerate over elements of s not in t'



# List comprehensions

List comprehensions are an alternative to for loops. Every list comprehension can be rewritten as a for loop, but not every for loop can be written as a list comprehension.


```python
list_of_numbers = [2,54,8,4,26,3,8,3,7]
```


```python
small_numbers = []
for number in list_of_numbers:
    if number < 5:
        small_numbers.append(2*number)
print(small_numbers)
```

    [4, 8, 6, 6]
    


```python
# This can be rewritten as a list comprehension
small_numbers = [2*number for number in list_of_numbers if number < 5]
print(small_numbers)
```

    [4, 8, 6, 6]
    

[Here](http://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/) is an easy to visualize explanation.

# Error handling

There are [a bunch of python error messages](https://www.tutorialspoint.com/python/standard_exceptions.htm), which are known as standard exceptions. Here are some of the most common

### Index Error


```python
a = [1,2,3,4]
print(a[5]) # There is no element #5, so you get an error
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-152-3328b7d9ec57> in <module>()
          1 a = [1,2,3,4]
    ----> 2 print(a[5]) # There is no element #5, so you get an error
    

    IndexError: list index out of range


### Name error


```python
my_variable = 4
print(my_veriable)
# I have introduced a typo, so call the variable "my_veriable" returns an error
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-155-30993cfdcc4a> in <module>()
          1 my_variable = 4
    ----> 2 print(my_veriable)
          3 # I have introduced a typo, so call the variable "my_veriable" returns an error
    

    NameError: name 'my_veriable' is not defined


### Type error


```python
# Trying to use a type in a way it cannot be
print(a[2]) # works fine
print(a['two']) # returns an error
```

    3
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-159-45c59a70da68> in <module>()
          1 # Trying to use a type in a way it cannot be
          2 print(a[2]) # works fine
    ----> 3 print(a['two']) # returns an error
    

    TypeError: list indices must be integers or slices, not str


### Syntax error


```python
print("Syntax errors often result from missing parentheses"
```


      File "<ipython-input-156-edd9a7d47009>", line 1
        print("Syntax errors often result from missing parentheses"
                                                                  ^
    SyntaxError: unexpected EOF while parsing
    


## Try, except statements


```python
#Can also be more specific:

print("If you provide two integers, I will divide the first by the second")
try:
    a = int(input('Give me a number: '))
    b = int(input('Give me another: '))
    print(a/b)
except ValueError:
    print("That's not an int")
except ZeroDivisionError:
    print("Can't divide by zero")
except:
    print("I don't even know what you did wrong")
```

    If you provide two integers, I will divide one by the other
    Give me a number: 2
    Give me another: 0
    Can't divide by zero
    


```python
# You can also use a finally statement to do something even after an error has been raised
try:
    a = int(input('Give me a number: '))
    b = int(input('Give me another: '))
    print(a/b)
except ValueError:
    print("That's not an int")
finally:
    print("Whether there's an exception or not, this runs. Good for closing a file.")
```

    Give me a number: r
    That's not an int
    Whether there's an exception or not, this runs. Good for closing a file.
    

You can also raise exceptions directly


```python
try:
    1/0
except:
    print("I've just picked up a fault in the AE35 unit. It's going to go 100% failure in 72 hours.")
    raise
```

    I've just picked up a fault in the AE35 unit. It's going to go 100% failure in 72 hours.
    


    ---------------------------------------------------------------------------

    ZeroDivisionError                         Traceback (most recent call last)

    <ipython-input-172-af3e15a5729f> in <module>()
          1 try:
    ----> 2     1/0
          3 except:
          4     print("I've just picked up a fault in the AE35 unit. It's going to go 100% failure in 72 hours.")
          5     raise
    

    ZeroDivisionError: division by zero


## Assert


```python
def div_by_two(x):
    assert (x%2 == 0), "Number must be even"#Assert that x is even; this makes the program stop immediately if it is not
    return x / 2
print(div_by_two(3))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-3-00ddaa3cd990> in <module>()
          2     assert (x%2 == 0), "Number must be even"#Assert that x is even; this makes the program stop immediately if it is not
          3     return x / 2
    ----> 4 print(div_by_two(3))
    

    <ipython-input-3-00ddaa3cd990> in div_by_two(x)
          1 def div_by_two(x):
    ----> 2     assert (x%2 == 0), "Number must be even"#Assert that x is even; this makes the program stop immediately if it is not
          3     return x / 2
          4 print(div_by_two(3))
    

    AssertionError: Number must be even


## Aliasing

Python uses references. This can create aliasing problems


```python
# Here's an example that isn't a problem
old_list = [1,2,3,4]
new_list = old_list
new_list[2]=7
print(old_list)
```

    [1, 2, 7, 4]
    

`old_list` has changed without being changed directly. This is because when I did `new_list = old_list`, it created a reference from the value of `old_list` to a new variable, `new_list`. But it did not make a second copy of the value, so they are pointing to the same value. If that value is changed both variables will see the change.

This could be the desired result, but sometimes it isn't. In those cases, you can make a copy of the value instead of just getting a reference to the old value. Do this by setting `new_list` equal to `old_list[:]` or `list(old_list)`


```python
old_list = [1,2,3,4]
new_list = old_list[:]
new_list[2]=7
print(old_list)
```

    [1, 2, 3, 4]
    


```python
old_list = [1,2,3,4]
new_list = list(old_list)
new_list[2]=7
print(old_list)
```

    [1, 2, 3, 4]
    

To see this in more detail, you can look at the id of the variable


```python
old_list = [1,2,3,4]
new_list = old_list
print(id(old_list))
print(id(new_list))
new_list = old_list[:]
# By using a copy, new_list gets a different id")
print(id(new_list))
```

    328784200
    328784200
    328893896
    
# Testing

For testing, I highly recommend [pytest](https://docs.pytest.org/en/latest/). One issue I had with it when I was getting started was that if it mocked the inputs I couldn't run the test as a file (like to debug in VSCode). It turns out this is all you need:

```python
if __name__ == "__main__":
    pytest.main([__file__])
```

Or, if you just one to test a function or two, you can do

```python
if __name__ == "__main__":
    pytest.main([test_my_func()])
```

# Zip


```python
a = range(5)
b = range(5,10)
c = zip(a,b)
c = list(c)
print(c)
```

    [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    


```python
# Or, you can unzip
ca, cb = zip(*c)
```


```python
print(ca)
print(cb)
```

    (0, 1, 2, 3, 4)
    (5, 6, 7, 8, 9)
    

# Learning your environment 

## Where am I?


```python
import os
os.getcwd() #get current working directory
```

## What version of Python am I using?


```python
import sys
sys.version
```

## Python modules

### Where are the site packages held?


```python
import site
site.getsitepackages()
```




    ['C:\\Users\\HMISYS\\Anaconda3',
     'C:\\Users\\HMISYS\\Anaconda3\\lib\\site-packages']



### Where is a particular package?


```python
import nltk
print(nltk.__file__)
import matplotlib.pyplot as plt
print(plt.__file__)
#This returns the location of the compiled .pyc file for the module
```

    C:\Users\HMISYS\Anaconda3\lib\site-packages\nltk\__init__.py
    C:\Users\HMISYS\Anaconda3\lib\site-packages\matplotlib\pyplot.py
    
