---
layout: post
title: "Python Cheat Sheet II"
description: "A more advanced cheatsheet for programming in Python"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/black_headed_python.jpg"
tags: [Python, Cheat Sheet]
---

These tips and tricks are a little bit more advanced than the ones in the previous notebook. I try to update the post every once in a while with the latest version of Python, so it should be roughly up to date.

<b>Table of Contents</b>
* TOC
{:toc}

# Learning your environment 

## Where am I?


```python
import os
os.getcwd() #get current working directory
```




    'C:\\Users\\Julius\\Google Drive\\JupyterNotebooks\\Blog'



## What version of Python am I using?


```python
import sys
sys.version
```




    '3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]'



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
    base                  *  C:\Users\Julius\anaconda3
    pt                       C:\Users\Julius\anaconda3\envs\pt
    tf                       C:\Users\Julius\anaconda3\envs\tf
    tf-gpu                   C:\Users\Julius\anaconda3\envs\tf-gpu
    
    

## Where will Python look for modules?


```python
import sys
```


```python
print(sys.path)
```

    ['C:\\Users\\Julius\\Google Drive\\JupyterNotebooks\\Blog', 'C:\\Users\\Julius\\Documents\\GitHub', 'C:\\Users\\Julius\\Documents\\GitHub\\cv\\src\\py', 'C:\\Users\\Julius\\Documents\\GitHub\\fastai-pythonic', 'C:\\Users\\Julius\\Documents\\GitHub\\facv\\src', 'C:\\Users\\Julius\\Documents\\GitHub\\fastai2', 'C:\\Users\\Julius\\Documents\\GitHub\\ObjectDetection', 'C:\\Users\\Julius\\Documents\\GitHub\\fastcore', 'C:\\Users\\Julius\\Documents\\GitHub\\cv_dataclass\\src', 'C:\\Users\\Julius\\Google Drive\\JupyterNotebooks\\Blog', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\python38.zip', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\DLLs', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib', 'C:\\Users\\Julius\\anaconda3\\envs\\tf', '', 'C:\\Users\\Julius\\AppData\\Roaming\\Python\\Python38\\site-packages', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\win32', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\win32\\lib', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\Pythonwin', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\IPython\\extensions', 'C:\\Users\\Julius\\.ipython']
    

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
# By using a copy, new_list gets a different id
print(id(new_list))
```

    3179513172992
    3179513172992
    3179513163712
    

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

Map can actually change the values.


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



You can also use multiple arguments with `lambda`s


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

There is a [hierarchy of Python Exceptions](https://docs.python.org/3/library/exceptions.html#exception-hierarchy). It is best practice to use the most specific one that applies to your case, and if none do, to raise a custom Exception. Here are some of the most common Exceptions.

### IndexError


```python
a = [1,2,3,4]
try:
    print(a[5])
except IndexError as e:
    print("There is no element #5, so you get an IndexError: ", e)
```

    There is no element #5, so you get an IndexError:  list index out of range
    

### NameError


```python
my_variable = 4
try:
    print(my_veriable)
except NameError:
    print("I have introduced a typo, so the call to variable 'my_veriable' returns an error")
```

    I have introduced a typo, so the call to variable 'my_veriable' returns an error
    

### TypeError


```python
# Trying to use a type in a way it cannot be
print(a[2]) # works fine
try:
    print(a['two']) # returns an error
except TypeError as err:
    print(err)
```

    3
    list indices must be integers or slices, not str
    

### SyntaxError

Syntax errors can be a little different. That's because the syntax is wrong, which prevents the `try`/`except` block from being set up.


```python
try:
    print("Syntax often result from missing parentheses"
except SyntaxError:
    print("This is never printed")
```


      File "C:\Users\Julius\AppData\Local\Temp/ipykernel_29776/3723339239.py", line 3
        except SyntaxError:
        ^
    SyntaxError: invalid syntax
    


There is a way around this using `eval`. Let's say you're trying to use a try/except in the following:


```python
try:
    [2 * x for x in [1,2,3] if x > 1 else 0]
except SyntaxError:
    print("This is never printed")
```


      File "C:\Users\Julius\AppData\Local\Temp/ipykernel_29776/3576225009.py", line 2
        [2 * x for x in [1,2,3] if x > 1 else 0]
                                         ^
    SyntaxError: invalid syntax
    


If you simply wrap it in an eval statement, it still doesn't work.


```python
try:
    eval([2 * x for x in [1,2,3] if x > 1 else 0])
except SyntaxError:
    print("This is never printed")
```


      File "C:\Users\Julius\AppData\Local\Temp/ipykernel_29776/241822697.py", line 2
        eval([2 * x for x in [1,2,3] if x > 1 else 0])
                                              ^
    SyntaxError: invalid syntax
    


But if it's a string, it will catch the `SyntaxError`.


```python
try:
    eval("[2 * x for x in [1,2,3] if x > 1 else 0]")
except SyntaxError:
    print("But this is printed")
```

    But this is printed
    

And if there wasn't an error, it would still evaluate it.


```python
try:
    a = eval("[2 * x for x in [1,2,3] if x > 1]")
except SyntaxError:
    print("But this is printed")
print(a)
```

    [4, 6]
    

## Creating Your Own Exceptions

It's a good idea to create your own exceptions. They don't actually need to do anything a lot of the time (other than inherit from `Exception`). Their name alone is valuable.


```python
class NiException(Exception):
    pass

try:
    if 'ni' in 'knights who say ni':
        raise NiException
except NiException:
    print("We raised and caught our custom exception")
```

    We raised and caught our custom exception
    

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

    If you provide two integers, I will divide the first by the second
    Give me a number: 2
    Give me another: 0
    Can't divide by zero
    

You can also use a finally statement to do something even after an error has been raised


```python
try:
    a = int(input('Give me a number: '))
    b = int(input('Give me another: '))
    print(a/b)
except ValueError:
    print("That's not an int")
finally:
    print("Whether there's an exception or not, this runs. Good for closing a file.")
```

    Give me a number: 1
    Give me another: 1.5
    That's not an int
    Whether there's an exception or not, this runs. Good for closing a file.
    

You can also raise exceptions directly


```python
try:
    1/0
except ZeroDivisionError:
    print("I've just picked up a fault in the AE35 unit. It's going to go 100% failure in 72 hours.")
    raise NiException
```

    I've just picked up a fault in the AE35 unit. It's going to go 100% failure in 72 hours.
    


    ---------------------------------------------------------------------------

    ZeroDivisionError                         Traceback (most recent call last)

    <ipython-input-45-413944f2b329> in <module>
          1 try:
    ----> 2     1/0
          3 except ZeroDivisionError:
    

    ZeroDivisionError: division by zero

    
    During handling of the above exception, another exception occurred:
    

    NiException                               Traceback (most recent call last)

    <ipython-input-45-413944f2b329> in <module>
          3 except ZeroDivisionError:
          4     print("I've just picked up a fault in the AE35 unit. It's going to go 100% failure in 72 hours.")
    ----> 5     raise NiException
    

    NiException: 


Note that you can't use this on some `SyntaxError`s as the compiler has to parse everything to set up the `try`/`except` blocks, and a `SyntaxError` can prevent that from happening. Here's an example:


```python
try:
    [2 * x for x in [1,2,3] if x > 1 else 0]
except SyntaxError:
    print("This doesn't get printed")
```


      File "<ipython-input-46-f08e10e9850e>", line 2
        [2 * x for x in [1,2,3] if x > 1 else 0]
                                         ^
    SyntaxError: invalid syntax
    


If you want to do this, you can wrap the statement in an `eval` statement, so the compiler has time to set up the `try`/`except` block.


```python
try:
    eval("[2 * x for x in [1,2,3] if x > 1 else 0]")
except SyntaxError:
    print("This DOES get printed")
```

    This DOES get printed
    

## Assert


```python
def div_by_two(x):
    assert (x%2 == 0), "Number must be even"#Assert that x is even; this makes the program stop immediately if it is not
    return x / 2
try:
    print(div_by_two(3))
except AssertionError as err:
    print('Error:', err)
```

    Error: Number must be even
    

# Testing

For testing, I highly recommend [pytest](https://docs.pytest.org/en/latest/). One issue I had with it when I was getting started was that if it mocked the inputs I couldn't run the test as a file (like to debug in VSCode). It turns out this is all you need.


```python
import pytest
if __name__ == "__main__":
    pytest.main([__file__])
```

Or, if you just want to test a function or two, you can do


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
    print("Can't do integer comparison with standard Enums")
```

    True
    Can't do integer comparison with standard Enums
    

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



# Type Annotations

I'm really into type annotations in Python, but I have to say, sometimes they make things ugly. Compare the following, with and without type annotations.


```python
import pandas as pd
def func(inp):
    df = pd.read_csv(inp)
    return df
```


```python
import pandas as pd
from pathlib import Path
from typing import AnyStr, IO, Union
def f(inp: Union[str, Path, IO[AnyStr]]) -> pd.DataFrame:
    df = pd.read_csv(inp)
    return df
```

You can see how [pandas tried to solve the problem here](https://github.com/pandas-dev/pandas/blob/1ce1c3c1ef9894bf1ba79805f37514291f52a9da/pandas/_typing.py#L48:1), but it's still messy.

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

# Inspection


```python
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width
        
    def get_area(self):
        return self.length * self.width
```


```python
r = Rectangle(3,5)
```


```python
r.get_area()
```




    15




```python
import inspect
```


```python
print(inspect.getsource(Rectangle.__init__))
```

        def __init__(self, length, width):
            self.length = length
            self.width = width
    
    


```python
print(inspect.getsource(Rectangle.get_area))
```

        def get_area(self):
            return self.length * self.width
    
    

# Namespace Mangling


```python
class MyClass:
    
    def __dunder_example(self):
        print("Dunder Example")
```

if you want to access `MyClass.__dunder_example`, you'll need to use `self._MyClass__dunder_example`. This is especially useful when debugging.


```python
mc = MyClass()
```


```python
try:
    mc.__dunder_example()
except AttributeError:
    print("Because of namespace mangling, you can't access this")
```

    Because of namespace mangling, you can't access this
    


```python
mc._MyClass__dunder_example()
```

    Dunder Example
    
