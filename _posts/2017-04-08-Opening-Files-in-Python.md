---
layout: post
title: "Opening Files in Python"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/python_resting.jpg"
tags: [Python]
---

Python offers several different methods for opening text files, but are all options equally good? What's the best method for the generic case of opening a text file?

The most basic is the following:


```python
# Open the file
f = open('file.txt', 'r')
```

The `r` means that the file is being opened as read-only. The other options are `w` for write-only, `a` for append-only, and `r+` for both reading and writing. If you use `r+`, you could accidentally write over your file, so it is best to use `r` unless you specifically want to edit the file.

From there, you can read the file with:


```python
# Read the file
f.read()
```




    'Hello World'



It's important to always close files after you're done reading them, so the entire code should look like:


```python
# Open the file
f = open('file.txt', 'r')

# Read the file
print(f.read())

# Do other things with the file

#Close the file
f.close()
```

    Hello World
    

The problem with this is that if something happens in your code which interrupts the program, it will abort before it closes your file. What you can do to avoid this is to use a `try`/`finally` statement. This way you `try` a section, and if there's a bug it will end the `try` section but still complete the `finally` section, so you can be guaranteed that the file will be closed properly.


```python
# Open the file
f = open('file.txt', 'r')

try:
    # Read the file
    print(f.read())

    # Do other things with the file

finally:
    #Close the file
    f.close()
```

    Hello World
    

However, there's an easier way to do this, and that's using the `with` statement. **This is a simple and safe way to open a file.**


```python
# Use a with statement
with open('file.txt') as f:
    # Read the file
    print(f.read())

    # Do other things with the file
```

    Hello World
    

Whenever the `with` statement ends, the file will be properly closed.

`pathlib` also offers another way to open and read files. It's simple, safe, and can be done with a one-liner. It's my go-to way of reading files.


```python
from pathlib import Path
```


```python
Path('file.txt').read_text()
```




    'Hello World'


