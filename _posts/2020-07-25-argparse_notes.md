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

This might be a little bit counter intuitive. Let's think about a case where you want something to do true. Let's say you want to include the training set in a train.

The common way to do this would be:
```python
parser.add_argument('--exclude_train', action='store_true')
```
This means "if I add this tag, store true; otherwise, store false". But that's kind of like a double negative - leaving out the command to exclude the training set. Instead, let's say you want to affirm it. You would do that like so:

parser.add_argument('--include_train', action='store_true')
parser.add_argument('--include_val', action='store_true')
parser.add_argument('--include_test', action='store_true')

This is probably better in this case. This means if I didn't specify that set, don't include it.


