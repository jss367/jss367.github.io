---
layout: post
title: "Pathlib Cheat Sheet"
description: "A post showing basic operations with Pathlib"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/windmills.jpg"
tags: [Python, Cheatsheet]
---

Pathlib does much of the same work as `os.path` but it contains a lot more. I strongly recommend that people checkout [Pathlib](https://docs.python.org/3/library/pathlib.html).


```python
from pathlib import Path
```


```python
path = Path(r'E:\Data\Raw\PennFudanPed')
```

## Getting a list of files from a path


```python
all_objs = path.glob('**/*')
```

Pathlib creates a generator for the files.


```python
all_objs
```




    <generator object Path.glob at 0x00000253CC2FD2E0>



If you want to put them all in a list, you can do that like so:


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



## Working with Pathlib objects

Path objects are also really easy to work with to extract useful information.


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



## Reading text files

Pathlib also has a great one-liner for reading in text files. No more context managers.


```python
text = Path(r'E:/Data/Raw/PennFudanPed/readme.txt').read_text()
```


```python
text[:500]
```




    '1. Directory structure:\n\nPNGImages:   All the database images in PNG format.\n\nPedMasks :   Mask for each image, also in PNG format. Pixels are labeled 0 for background, or > 0 corresponding\nto a particular pedestrian ID.\n\nAnnotation:  Annotation information for each image.  Each file is in the following format (take FudanPed00001.txt as an example):\n\n# Compatible with PASCAL Annotation Version 1.00\nImage filename : "PennFudanPed/PNGImages/FudanPed00001.png"\nImage size (X x Y x C) : 559 x 536 x 3'


