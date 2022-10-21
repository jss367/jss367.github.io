---
layout: post
title: "Pathlib Cheat Sheet"
description: "A post showing basic operations with Pathlib"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/windmills.jpg"
tags: [Python, Cheat Sheet]
---

[Pathlib](https://docs.python.org/3/library/pathlib.html) is a built-in Python library that is similar to `os.path` but contains a lot more. This post walks through some of the basics with pathlib.

<b>Table of Contents</b>
* TOC
{:toc}


```python
import sys
from pathlib import Path
```

Since Python is built in, the version of pathlib you have will be based on the version of Python. I try to keep this notebook up-to-date with the latest version. Here's the latest version it was run on:


```python
sys.version
```




    '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]'




```python
path = Path(r'E:\Data\Raw\PennFudanPed')
```

## Getting a list of files from a path

You can do two sets of asterisks like so to get subdirectories.


```python
all_objs = path.glob('**/*')
```

Pathlib creates a generator for the files.


```python
all_objs
```




    <generator object Path.glob at 0x000002283654A120>



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



Here's a way to list what's in a directory.


```python
list(path.iterdir())
```




    [WindowsPath('E:/Data/Raw/PennFudanPed/added-object-list.txt'),
     WindowsPath('E:/Data/Raw/PennFudanPed/Annotation'),
     WindowsPath('E:/Data/Raw/PennFudanPed/PedMasks'),
     WindowsPath('E:/Data/Raw/PennFudanPed/PNGImages'),
     WindowsPath('E:/Data/Raw/PennFudanPed/readme.txt')]



> Note: This is what [FastAI uses for path.ls()](https://fastcore.fast.ai/xtras.html#Path.ls).

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


You can even get the file extension:


```python
file_path.suffix
```




    '.txt'



It's also easy to make new directories.


```python
path.mkdir(parents=True, exist_ok=True)
```

## Reading text files

Pathlib also has a great one-liner for reading in text files. No more context managers.


```python
text = Path(r'E:/Data/Raw/PennFudanPed/readme.txt').read_text()
```


```python
text[:500]
```




    '1. Directory structure:\n\nPNGImages:   All the database images in PNG format.\n\nPedMasks :   Mask for each image, also in PNG format. Pixels are labeled 0 for background, or > 0 corresponding\nto a particular pedestrian ID.\n\nAnnotation:  Annotation information for each image.  Each file is in the following format (take FudanPed00001.txt as an example):\n\n# Compatible with PASCAL Annotation Version 1.00\nImage filename : "PennFudanPed/PNGImages/FudanPed00001.png"\nImage size (X x Y x C) : 559 x 536 x 3'


