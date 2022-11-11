---
layout: post
title: "Argparse Notes"
description: "This post contains notes about how to work with argparse"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/kea.jpg"
tags: [Jupyter Notebooks, Python]
---

This post contains some of my notes on the `argparse` package.

## What Arguments Were Passed

If you ever want to see what arguments were passed into a module, you can do the following:
```python
import sys
print(sys.argv)
```

Note that the first argument will be the path to the module, not an actual argument.

## Required vs Optional Arguments

Arguments that start with  `-` or `--` are optional arguments, while those that don't are positional and therefore required (much like positional arguments in Python functions). Even though arguments starting with  `-` or `--` are generally optional, `argparse` still lets you mark them as required. This is considered bad design in most cases but isn't prevented.

For example, you can do:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--foo', required=True)
```

This is how it's put in the [documentation](https://docs.python.org/3/library/argparse.html#required):
> Required options are generally considered bad form because users expect options to be optional, and thus they should be avoided when possible.


## Defaults for booleans

If you want an argument to default to true:

```python
parser.add_argument('--foo', action='store_false')
```

This means `foo` will be `True` unless `--foo` is added to the command, in which case it would become `False`.

If you want an argument to default to false:

```python
parser.add_argument('--foo', action='store_true')
```

