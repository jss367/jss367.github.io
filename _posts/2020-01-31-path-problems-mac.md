---
layout: post
title: "Path Problems"
description: "A guide to some of the path problems you may face on Macs"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/dark_path.jpg"
tags: [Mac, Python]
---

Path problems are some of the most common and annoying problems machine learning engineers face, especially when frequently switching between operating systems. There are so many different issues ways to have path problems that no post could cover them all, but in this post, I'll try to provide some background on possible issues and how to resolve them.

<b>Table of Contents</b>
* TOC
{:toc}

The first thing you need to realize are that there are multiple different things that are at times called a "path". The two main ones that Python programmers will run into are the system path and the Python path. 

# System Path

The first step is being able to find out what's on your path. You can do this by looking at the `$PATH` environmental variable. You can do this from the command line, but the exact command depends on which operation system you're using.

## Viewing Your Path

If you're using either a Mac or Linux, including Windows Subsystem for Linux, it's as simple as:

`echo $PATH`

The result might be a little hard to read, so if you want a more readable version you can use:

`echo "${PATH//:/$'\n'}"`


## Adding to Your Path

If you want to temporarily add to your path, type the following in the terminal:

`export PATH=$PATH:/path/to/directory`

If you want to permanently add to your path, simply take that command and place it in your `.bashrc` file (or wherever you keep your environmental variables - [I recommend `.profile`](https://jss367.github.io/shell-and-environment-setup.html))

# Python Path

If you can't import a module you wrote, it's probably because that location is missing from your path. But it's your *Python* path, not your system path, that it's missing from.

```
import my_module
ModuleNotFoundError: No module named 'my_module'
```

## Viewing your Python Path

#### Unix

`echo $PYTHONPATH`


## Adding to your PYTHONPATH

I use the User variables for this



Right-click anywhere in the editor window and select Run Python File in Terminal (which saves the file automatically):

- doesn't use selected interpreter





On a brand new Mac your path is:

```
julius@Juliuss-MacBook-Pro ~ % echo $PATH
/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
```


Note:
If you change the environment variables you'll need to restart VSCode or Jupyter Notebooks. Once you restart you'll see the addition in `sys.path`


## Finding your Python Interpreter

From within Python, you can do:

``` python
import sys
sys.executable
```


## Other

If you want to know more about how the Python `sys.path` is initialized, there's a [detailed StackOverflow answer](https://stackoverflow.com/questions/897792/where-is-pythons-sys-path-initialized-from).
