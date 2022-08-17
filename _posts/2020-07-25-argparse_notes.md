---
layout: post
title: "Argparse Notes"
description: "This post contains notes about how to work with argparse"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/kea.jpg"
tags: [Jupyter Notebooks, Python]
---

## What Arguments Were Passed

If you ever want to see what arguments were passed into a module, you can do the following:
```python
import sys
print(sys.argv)
```

Note that the first argument will be the path to the module, not an actual argument.

## Required vs Optional Arguments

Arguments that start with  `-` or `--` are optional arguments, while those that don't are positional and therefore required (much like positional arguments in Python functions). Even though, arguments starting with  `-` or `--` are generally optional, `argparse` still lets you mark them as required. This is considered bad design in most casee but isn't prevented.

For example, you can do:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--foo', required=True)
```


This is how it's put in the documentation [documentation](https://docs.python.org/3/library/argparse.html#required):
> Required options are generally considered bad form because users expect options to be optional, and thus they should be avoided when possible.


