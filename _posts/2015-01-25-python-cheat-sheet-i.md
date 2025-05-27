---
layout: post
title: "Python Cheat Sheet I"
description: "A basic cheatsheet for programming in Python"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/python.jpg"
tags: [Python, Cheat Sheet]
---

This notebook is a collection of Python notes, tricks, and tips. It is set up to act as a quick reference for basic Python programming. I try to update the post every once in a while with the latest version of Python, so it should be roughly up to date.

<b>Table of Contents</b>
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
# Strings are immutable, so you will have to create a new string each time
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
try:
    phrase.index('fort')
except ValueError as e:
    print(e)
```

    substring not found
    


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

    {'is', 'sentence', 'my', 'this'}
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
# Lists can consist of different types, but this isn't common
a = ['This', 'is', 4, 1+4, 's'*3]
print(a)
```

    ['This', 'is', 4, 5, 'sss']
    

## Indexing


```python
# The first item has an index of 0
print(a[0])
# And the last has an index of one less than the number of elements
print(a[len(a)-1])
```

    This
    sss
    

## Slicing

List slicing is done by [start : end]

The first value is included, but the last one isn't, so it looks like [inclusive : exclusive].


```python
# list slicing: [start : end] is [inclusive : exclusive], so the last value in the slice is not included
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
    

# Tuples


```python
# Tuples are defined using parentheses
tu = (1, 2, 3)
```


```python
#tuple are immutable, so the following will give you an error:
try:
    tu[2] = 5
except TypeError as e:
    print(e)
```

    'tuple' object does not support item assignment
    


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


```python
print("Hello world!")
```

    Hello world!
    


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
print("{perc:.2%} of {num_verbs} are weak".format(perc=.501, num_verbs=7))
```

    3.14
    33.30%
    50.10% of 7 are weak
    

## f-strings

You can also use f-strings, which are an even faster way of using the functionality of `format`.


```python
perc=.501
num_verbs=7
print(f"{perc:.2%} of {num_verbs} are weak")
```

    50.10% of 7 are weak
    

# For-loops


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
    

Note that you can end a for-loop with an `else` statement. This might seem a little strange, but you can think of "else" acting as a "then" in this situation. If the loop successfully completes, the `else` statement will run. But if there's a `break` in the loop, it doesn't.


```python
for i in [0, 3, 5]:
    print(i*i)
else:
    print("Print if loop completes")

for i in [0, 3, 5]:
    print(i*i)
    if i > 2:
        break
else:
    print("Won't print if there's a break")
```

    0
    9
    25
    Print if loop completes
    0
    9
    

# List Comprehensions

List comprehensions are an alternative to for-loops. Every list comprehension can be rewritten as a for-loop, but not every for-loop can be written as a list comprehension.


```python
numbers = [1,2,3,4,5]
```


```python
small_numbers = []
for number in numbers:
    if number < 5:
        small_numbers.append(2*number)
print(small_numbers)
```

    [2, 4, 6, 8]
    


```python
# This can be rewritten as a list comprehension
small_numbers = [2*number for number in numbers if number < 5]
print(small_numbers)
```

    [2, 4, 6, 8]
    

Check out this [excellent visualization of the relationship between for-loop and list comprehensions](http://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/) to see how they relate.

## Conditional List Comprehensions

When there's only an `if`, you can do this:


```python
[2*number for number in numbers if number < 5]
```




    [2, 4, 6, 8]



But if you want to add an `else` statement, you can't just add it after the `if` statement like so:


```python
try:
    [2*number for number in numbers if number < 5 else 0]
except SyntaxError as err:
    print(err)
```


      File "C:\Users\Julius\AppData\Local\Temp/ipykernel_15528/3025308397.py", line 2
        [2*number for number in numbers if number < 5 else 0]
                                                      ^
    SyntaxError: invalid syntax
    


Instead, you have to move the `if` and `else` statements near the beginning, like so:


```python
[2*number if number < 5 else number**2 for number in numbers]
```




    [2, 4, 6, 8, 25]



I find this a bit odd because if you have the `if` statement there WITHOUT the `else`, it raises an error:


```python
[2*number if number < 5 for number in numbers]
```


      File "C:\Users\Julius\AppData\Local\Temp/ipykernel_15528/2907721880.py", line 1
        [2*number if number < 5 for number in numbers]
                                ^
    SyntaxError: invalid syntax

