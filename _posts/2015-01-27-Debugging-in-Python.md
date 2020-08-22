---
layout: post
title: "Debugging in Python"
description: "This post contains information on different debugging options in Python."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/wasp.jpg"
tags: [Python, Cheatsheet]
---

# Debugging in Python

VSCode and PyCharm have the strongest debugging capabilities. Sometimes those options aren't available to you, so here are some alternatives.

## pdb

`pdb` is the default Python debugger and for this reason alone it's good to be familiar with. I often use it when I am tunneling into somewhere.

### Main commands

The main commandes to know are:

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
*Automatically* see the context above and below the your current line. I think this feature alone makes it preferable to pdb

`ipdb.set_trace(context=11)`

## Jupyter notebooks

If you have an error, you can recover the state with `%debug`.

## Command line

You can also set the debugger to capture the traceback if you have an error in a command, right from the command line:

`python -m pdb algo_trainer.py`