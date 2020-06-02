---
layout: post
title: "Debugging Python"
tags: [Python]
---

Debugging

VSCode and PyCharm have the strongest debugging capabilities.


## pdb

it's the default

### List all the attributes of an object

`p dir(a)`

### See the traceback

```
import traceback
traceback.print_stack()
```

## ipdb

## Context
*Automatically* see the context above and below the your current line. I think this feature alone makes it preferable to pdb

`ipdb.set_trace(context=11)`

## Jupyter notebooks

Now, if you have an error, you can recover the state with `%debug`


## Command line

`python -m pdb algo_trainer.py`