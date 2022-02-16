---
layout: post
title: "Argparse Notes"
description: "This post contains notes about how to work with argparse"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/kea.jpg"
tags: [Jupyter Notebooks, Python]
---

## Required Arguments

Arguments that don't begin with "-" or "--" are positional and therefore required (much like arguments in Python functions). Arguments that start with  "-" or "--" are optional arguments, although `argparse` lets you mark this as required. This is considered bad design in most casee but isn't prevented.

For example, you can do:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--foo', required=True)
```


This is how it's put in the documentation [documentation](https://docs.python.org/3/library/argparse.html#required):
> Required options are generally considered bad form because users expect options to be optional, and thus they should be avoided when possible.


