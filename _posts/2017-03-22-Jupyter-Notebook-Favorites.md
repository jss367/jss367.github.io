---
layout: post
title: My Favorite Things about Jupyter Notebooks
description: "A notebook that shows some my of favorites things to do with Jupyter Notebooks"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/jupiter.jpg"
tags: [Jupyter Notebooks]
---

This demonstrates some of my favorites tweaks for Jupyter Notebooks.<!--more--> 

<b>Table of contents</b>
* TOC
{:toc}

# Jupyter Notebook extensions

One of the many great things about Jupyter Notebooks is the ability to add extensions.

## Installation

The easiest way to add Jupyter Notebook extensions is through nbextensions. You can install it using `conda-forge`:
```python
conda install -c conda-forge jupyter_contrib_nbextensions
```
You can also use `pip`:
```
pip install jupyter_contrib_nbextensions
```

Then:
* Enter: `jupyter contrib nbextension install --user`
* Open up Jupyter Notebook and you will have a tab called Nbextensions
* Go to the Nbextensions tab and enable it from there

# Running pip from Jupyter Notebooks

You can run external commands by putting an exclamation mark in front of the command. This means that you can make pip install directly from Jupyter Notebooks:


```python
!pip install numpy
```

    Requirement already satisfied: numpy in c:\users\hmisys\anaconda3\lib\site-packages
    

# Jupyter Notebook extensions

## [2to3 converter](http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/code_prettify/README_2to3.html)

Converts Python 2 to Python 3 in a single click

http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/code_prettify/README_2to3.html


## Autopep8

Formats your code based on the PEP8 guide in a single click.

Install with `pip install autopep8`, then enable it.

## spellchecker

Does what you think it does.

# Printing LaTeX

Jupyter Notebooks use MathJax to render LaTeX in Markdown. To add LaTeX, simply surround your statement with `$$`:

$$c = \sqrt{a^2 + b^2}$$

Some things that work in Jupyter Notebooks don't work with the renderer I use for this blog, so I can't show everything. But in Jupyter Notebooks you can get LaTeX to work with a single `$` and use a double `$$` to center it.

'''If you want to center your formula, surround it with `$$`
$$
\begin{align}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\   \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\
\nabla \cdot \vec{\mathbf{B}} & = 0
\end{align}
$$'''

# nbconvert



Convert Jupyter Notebooks to various formats, including HTML, LaTeX, PDF, and Markdown

`jupyter nbconvert --to html mynotebook.ipynb`

or 

`jupyter nbconvert --to markdown mynotebook.ipynb`

See how I use it to [prepare Jupyter Notebooks for my blog](https://jss367.github.io/jupyter-notebooks-in-blog.html).

# Unix commands

You can also use some Unix commands in Jupyter Notebooks, such as `ls` or `pwd`


```python
pwd
```




    'C:\\Users\\Julius\\Google Drive\\JupyterNotebooks\\Blog'



# Viewing Source Code 

Just put two `??` behind something to view the source code. You won't be able to see it in this post, 


```python
import tensorflow as tf
```


```python
tf.einsum??
```

It show you the docstring:


Signature: tf.einsum(equation, *inputs, **kwargs)
Source:   
@tf_export('einsum', 'linalg.einsum')
def einsum(equation, *inputs, **kwargs):
  """Tensor contraction over specified indices and outer product.

  This function returns a tensor whose elements are defined by `equation`,
  which is written in a shorthand form inspired by the Einstein summation
  convention.  As an example, consider multiplying two matrices
  A and B to form a matrix C.  The elements of C are given by:

  ```
    C[i,k] = sum_j A[i,j] * B[j,k]
  ```

  The corresponding `equation` is:

  ```
    ij,jk->ik
  ```
 
 ...


And then the code


```python
if fwd_compat.forward_compatible(2019, 10, 18):
    return _einsum_v2(equation, *inputs, **kwargs)
else:
    return _einsum_v1(equation, *inputs, **kwargs)
```

If you just wanted to see the docstring, you could do this:


```python
def func_with_docstring():
    """
    This is a useful docstring
    """
```


```python
help(func_with_docstring)
```

    Help on function func_with_docstring in module __main__:
    
    func_with_docstring()
        This is a useful docstring
    
    

# Shift + Tab

Shift + Tab is a great keyboard shortcut that every Jupyter Notebook user should know. You simply start to call a function and when you're inside the parentheses and can't remember the right inputs, hit Shift + Tab and it will show you the documentation. If you hit Tab multiple times while holding Shift it cycles through various displays of the documentation. Try it out!

# Command Palatte

Jupyter Notebooks has a command palatte that you can access with Ctrl + Shift + P (Windows/Linux) / Cmd + Shift + P (Mac) (just like VSCode). From there you can search for whatever feature you're looking for.

One hot key that I like but sometimes forget is how to split a cell where my cursor is. So I just open up the command palatte and type in "split" and I see that it is Ctrl + Shift + - (windows/Linux) Cmd + Shift + - (Mac).


