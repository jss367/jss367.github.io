---
layout: post
title: "Debugging in Python"
description: "This post contains information on different debugging options in Python."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/wasp.jpg"
tags: [Python, Cheat Sheet]
---

My favorite place to debug a Python application is in a full IDE like VSCode or PyCharm. However, sometimes those options aren't available, so it's good to know the alternatives. This post shows some basic functionality of some Python debuggers.

<b>Table of Contents</b>
* TOC
{:toc}

## breakpoint()

Since Python 3.7, the easiest way to drop into a debugger is with the built-in `breakpoint()` function. Just add it anywhere in your code:

```python
def my_function(x):
    breakpoint()
    return x + 1
```

By default, it drops you into `pdb`. If you have `ipdb` installed, you can switch to it by setting the `PYTHONBREAKPOINT` environment variable:

```
PYTHONBREAKPOINT=ipdb.set_trace python my_script.py
```

You can also disable all breakpoints at once with `PYTHONBREAKPOINT=0`.

## pdb

`pdb` is the default Python debugger and for this reason alone it's good to be familiar with. I often use it when I am tunneling into somewhere.

### Main commands

The main commands to know are:

* `n` - proceed to next step
* `s` - step into
* `u` - move up stack
* `d` - move down stack
* `l` - show surrounding lines
* `ll` - "long list", show all of the code

### List all the attributes of an object

`p dir(a)`

### See the traceback

```
import traceback
traceback.print_stack()
```

## ipdb

You can use the same commands as in `pdb` but there's some extra functionality on top.

### Context
*Automatically* see the context above and below your current line. I think this feature alone makes it preferable to pdb

`import ipdb; ipdb.set_trace(context=11)`

## Jupyter notebooks

If you have an error, you can recover the state with `%debug`.

## Command line

You can also set the debugger to capture the traceback if you have an error in a command, right from the command line:

`python -m pdb algo_trainer.py`
