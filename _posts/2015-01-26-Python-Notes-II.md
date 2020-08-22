---
layout: post
title: "Python Cheatsheet II"
description: "A more advanced cheatsheet for programming in Python"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/black_headed_python.jpg"
tags: [Python, Cheatsheet]
---

These tips and tricks are a little bit more advanced than the ones in the previous notebook. I try to update the post every once in a while with the latest version of Python, so it should be roughly up to date.

<b>Table of contents</b>
* TOC
{:toc}

# Learning your environment 

## Where am I?


```python
import os
os.getcwd() #get current working directory
```




    'C:\\Users\\Julius\\Google Drive\\JupyterNotebooks\\Blog'



## What verison of Python am I using?


```python
import sys
sys.version
```




    '3.7.7 (default, May  6 2020, 11:45:54) [MSC v.1916 64 bit (AMD64)]'



## Where is my Python interpreter located?


```python
sys.executable
```




    'C:\\Users\\Julius\\anaconda3\\envs\\tf\\python.exe'



## What conda environment am I in?


```python
!conda env list
```

    # conda environments:
    #
    base                     C:\Users\Julius\anaconda3
    pt                       C:\Users\Julius\anaconda3\envs\pt
    pyt                      C:\Users\Julius\anaconda3\envs\pyt
    solaris                  C:\Users\Julius\anaconda3\envs\solaris
    tf                    *  C:\Users\Julius\anaconda3\envs\tf
    tf2                      C:\Users\Julius\anaconda3\envs\tf2
    
    

## Python modules

### Where are the site packages held?


```python
import site
site.getsitepackages()
```




    ['C:\\Users\\Julius\\anaconda3\\envs\\tf',
     'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages']



### Where is a particular package?


```python
import tensorflow
print(tensorflow.__file__)
```

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\tensorflow\__init__.py
    


```python
import matplotlib.pyplot as plt
print(plt.__file__)
```

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\matplotlib\pyplot.py
    

# Aliasing

Python uses references. This can create aliasing problems.


```python
# Here's an example that isn't a problem
old_list = [1,2,3,4]
new_list = old_list
new_list[2]=7
print(old_list)
```

    [1, 2, 7, 4]
    

`old_list` has changed without being changed directly. This is because when I did `new_list = old_list`, it created a reference from the value of `old_list` to a new variable, `new_list`. But it did not make a second copy of the value, so they are pointing to the same value. If that value is changed both variables will see the change.

This can be a desired result, but sometimes it isn't. In those cases you can make a copy of the value instead of just getting a reference to the old value. Do this by setting `new_list` equal to `old_list[:]` or `list(old_list)`


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
    

To see this in more detail, you can look at the id of the variable.


```python
old_list = [1,2,3,4]
new_list = old_list
print(id(old_list))
print(id(new_list))
new_list = old_list[:]
# By using a copy, new_list gets a different id")
print(id(new_list))
```

    2186639106504
    2186639106504
    2186639105672
    

# Functional programming

### Zip


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
    

You can use this to quickly sort two lists while keeping them in sync.


```python
preds = [0.1, 0.95, 0.11, 0.35, 0.75, 0.8]
y_true = [0, 1, 0, 0, 1, 0]
```


```python
sorted_preds, sorted_y_true = zip(*sorted(zip(preds, y_true), reverse=True))
```


```python
print(sorted_preds)
print(sorted_y_true)
```

    (0.95, 0.8, 0.75, 0.35, 0.11, 0.1)
    (1, 0, 1, 0, 0, 0)
    

### Filter

Filter is good for, as it sounds, filtering. 


```python
list(filter(lambda x: x > 5, range(10)))
```




    [6, 7, 8, 9]



Note that you could also do this with a list comprehension.


```python
[x for x in range(10) if x > 5]
```




    [6, 7, 8, 9]



### Map

Map can actually changes the values.


```python
list(map(lambda x : str(x), range(10)))
```




    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



And the list comprehensions way.


```python
[str(x) for x in range(10)]
```




    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



`range`, `map`, `zip`, and `filter` are all iterables in Python 3.X, so to see the actual values you can use the `list` command.


```python
list(map(abs, [-5, -2, 1]))
```




    [5, 2, 1]



Or you can use `lambda`, the anonymous function.


```python
list(map(lambda x: x**2, range(4)))
```




    [0, 1, 4, 9]



You can also use multiple arguments with `lamb


```python
list(map(lambda x, y: x + y, range(4), [10, 20, 30, 40]))
```




    [10, 21, 32, 43]



### Reduce


```python
from functools import reduce
```


```python
reduce(lambda x,y: x+y, [23, 34, 12, 23])
```




    92



# Error handling

There are [a bunch of python error messages](https://www.tutorialspoint.com/python/standard_exceptions.htm), which are known as standard exceptions. Here are some of the most common.

### IndexError


```python
a = [1,2,3,4]
print(a[5]) # There is no element #5, so you get an error
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-28-c5031d474716> in <module>
          1 a = [1,2,3,4]
    ----> 2 print(a[5]) # There is no element #5, so you get an error
    

    IndexError: list index out of range


### NameError


```python
my_variable = 4
# I have introduced a typo, so the call to variable 'my_veriable' returns an error
print(my_veriable)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-37-6c6800b68d90> in <module>
          1 my_variable = 4
          2 # I have introduced a typo, so the call to variable 'my_veriable' returns an error
    ----> 3 print(my_veriable)
    

    NameError: name 'my_veriable' is not defined


### TypeError


```python
# Trying to use a type in a way it cannot be
print(a[2]) # works fine
print(a['two']) # returns an error
```

    3
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-38-01695fff8160> in <module>
          1 # Trying to use a type in a way it cannot be
          2 print(a[2]) # works fine
    ----> 3 print(a['two']) # returns an error
    

    TypeError: list indices must be integers or slices, not str


### SyntaxError


```python
print("Syntax often result from missing parentheses"
```


      File "<ipython-input-39-c2f0579e9fa6>", line 1
        print("Syntax often result from missing parentheses"
                                                            ^
    SyntaxError: unexpected EOF while parsing
    


## Creating Your Own Exceptions

It's a good idea to create your own exceptions. They don't actually need to do anything a lot of the time (other than inherit from `Exception`). Their name alone is valuable.


```python
class NiException(Exception):
    pass

if 'ni' in 'knights who say ni':
    raise NiException
```


    ---------------------------------------------------------------------------

    NiException                               Traceback (most recent call last)

    <ipython-input-40-d76c5f072f95> in <module>
          3 
          4 if 'ni' in 'knights who say ni':
    ----> 5     raise NiException
    

    NiException: 


## Try, except statements


```python
#Can also be more specific:

print("If you provide two integers, I will devide the first by the second")
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

    If you provide two integers, I will devide the first by the second
    Give me a number: 00
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

    Give me a number: 9
    Give me another: 5
    1.8
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

    <ipython-input-44-b11c5b22cc8c> in <module>
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

    <ipython-input-45-c78c5e458a78> in <module>
          2     assert (x%2 == 0), "Number must be even"#Assert that x is even; this makes the program stop immediately if it is not
          3     return x / 2
    ----> 4 print(div_by_two(3))
    

    <ipython-input-45-c78c5e458a78> in div_by_two(x)
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

This can be a desired result, but sometimes it isn't. In those cases you can make a copy of the value instead of just getting a reference to the old value. Do this by setting `new_list` equal to `old_list[:]` or `list(old_list)`


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

    2186641155016
    2186641155016
    2186641154568
    

# Testing

For testing, I highly recommend [pytest](https://docs.pytest.org/en/latest/). One issue I had with it when I was getting started was that if it mocked the inputs I couldn't run the test as a file (like to debug in VSCode). It turns out this is all you need.


```python
import pytest
if __name__ == "__main__":
    pytest.main([__file__])
```

Or, if you just one to test a function or two, you can do


```python
if __name__ == "__main__":
    pytest.main([test_my_func()])
```

# Enums


```python
from enum import Enum
```

Enums are a simple way of aliasing values.


```python
class Animals(Enum):
    cat = 1
    dog = 2
    fish = 3
```


```python
Animals.cat
```




    <Animals.cat: 1>




```python
for animal in Animals:
    print(animal)
    print(animal.value)
```

    Animals.cat
    1
    Animals.dog
    2
    Animals.fish
    3
    

## IntEnums

IntEnums are like Enums except that you can also do integer comparison with them.


```python
from enum import IntEnum
```


```python
class Birds(IntEnum):
    cardinal = 1
    blue_jay = 2
```

### Enums vs IntEnums


```python
# can int compare IntEnums
print(Animals.dog == 2)
print(Birds.blue_jay == 2)
```

    False
    True
    


```python
print(Birds.blue_jay < Birds.cardinal + 3)
try:
    print(Animals.dog < Animals.cat + 3)
except TypeError:
    print("Can't do interger comparison with standard Enums")
```

    True
    Can't do interger comparison with standard Enums
    

# Pathlib

Pathlib does much of the same work as `os.path` but it contains a lot more. I strongly recommend that people checkout Pathlib.


```python
from pathlib import Path
```


```python
path = Path(r'E:\Data\Raw\PennFudanPed')
```


```python
all_objs = path.glob('**/*')
```


```python
all_objs
```




    <generator object Path.glob at 0x000001B136675EC8>




```python
files = [f for f in all_objs if f.is_file()]
```


```python
files[:5]
```




    [WindowsPath('E:/Data/Raw/PennFudanPed/added-object-list.txt'),
     WindowsPath('E:/Data/Raw/PennFudanPed/readme.txt'),
     WindowsPath('E:/Data/Raw/PennFudanPed/Annotation/FudanPed00001.txt'),
     WindowsPath('E:/Data/Raw/PennFudanPed/Annotation/FudanPed00002.txt'),
     WindowsPath('E:/Data/Raw/PennFudanPed/Annotation/FudanPed00003.txt')]




```python
path.stem
```




    'PennFudanPed'




```python
path.name
```




    'PennFudanPed'




```python
file_path = Path(files[0])
file_path
```




    WindowsPath('E:/Data/Raw/PennFudanPed/added-object-list.txt')




```python
file_path.stem
```




    'added-object-list'




```python
file_path.name
```




    'added-object-list.txt'



Pathlib also has a great one-liner for reading in text files. No more context managers.


```python
text = Path(r'E:/Data/Raw/PennFudanPed/readme.txt').read_text()
```


```python
text[:500]
```




    '1. Directory structure:\n\nPNGImages:   All the database images in PNG format.\n\nPedMasks :   Mask for each image, also in PNG format. Pixels are labeled 0 for background, or > 0 corresponding\nto a particular pedestrian ID.\n\nAnnotation:  Annotation information for each image.  Each file is in the following format (take FudanPed00001.txt as an example):\n\n# Compatible with PASCAL Annotation Version 1.00\nImage filename : "PennFudanPed/PNGImages/FudanPed00001.png"\nImage size (X x Y x C) : 559 x 536 x 3'



# Scope

Scope is very important in Python. Different objects perform differently when they are modified in a function.


```python
a = [1,2,3]
```


```python
def cl(s):
    s[1] = 5
```


```python
cl(a)
```


```python
a
```




    [1, 5, 3]




```python
def cv(a):
    a = 5
```


```python
q = 2
```


```python
cv(q)
```


```python
q
```




    2



# Python Disassembler

The Python disassembler is a great tool if you want to see how a Python statement would be done in bytecode. For example, let's say you're wondering which of the two commands is better to use:

`x is not None`

`not x is None`

We can use `dis` to figure out what the difference is.


```python
from dis import dis
```


```python
def func1(x):
    return x is not None
```


```python
def func2(x):
    return not x is None
```


```python
dis(func1)
```

      2           0 LOAD_FAST                0 (x)
                  2 LOAD_CONST               0 (None)
                  4 COMPARE_OP               9 (is not)
                  6 RETURN_VALUE
    


```python
dis(func2)
```

      2           0 LOAD_FAST                0 (x)
                  2 LOAD_CONST               0 (None)
                  4 COMPARE_OP               9 (is not)
                  6 RETURN_VALUE
    

It turns out there is no difference (although I think the first is easier to read and is [recommended by PEP-8](https://www.python.org/dev/peps/pep-0008/#programming-recommendations)).
