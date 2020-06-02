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
```
conda install -c conda-forge jupyter_contrib_nbextensions
```
You can also use `pip`:
```
pip install jupyter_contrib_nbextensions
```

Then:
* Enter: `jupyter contrib nbextension install --user`

Then open up Jupyter Notebook and you will have a tab called Nbextensions
* Go to the Nbextensions tab and enable it from there


## [2to3 converter](http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/code_prettify/README_2to3.html)

Converts Python 2 to Python 3 in a single click


## Autopep8

Formats your code based on the PEP8 guide in a single click

You'll need to install it and enable it in the notebook

Install with `pip install autopep8`, then enable it


# Running pip from Jupyter Notebooks

You can run external commands by putting an exclamation mark in front of the command. This means that you can make pip install directly from Jupyter Notebooks:


```python
!pip install numpy
```

    Requirement already satisfied: numpy in c:\users\hmisys\anaconda3\lib\site-packages
    

# Printing LaTeX

Jupyter Notebooks use MathJax to render LaTeX in Markdown. To add LaTeX, simply surround your statement with `$`:

For example $c = \sqrt{a^2 + b^2}$ prints an equation

If you want to center your formula, surround it with `$$`
$$
\begin{align}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\   \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\
\nabla \cdot \vec{\mathbf{B}} & = 0
\end{align}
$$

It may render differently in different browsers. In some cases, like Chrome, it may give you an error that says "This page is trying to load scripts from unauthenticated sources." You will have to allow unauthenticated sources to see the equations.

# nbconvert

Convert Jupyter Notebooks to various formats, including HTML, LaTeX, PDF, and Markdown

`jupyter nbconvert --to html mynotebook.ipynb`

`jupyter nbconvert --to markdown mynotebook.ipynb`

See how I use it here in [this post](https://jss367.github.io/jupyter-notebooks-in-blog.html).

<script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>