---
layout: post
title: "Sigmoid Functions for Mathematical Modeling"
description: "This post shows how to use sigmoid functions to model data."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/tiger_snake.jpg"
tags: [Python]
---

I use sigmoids all the time for fitting data. They are smooth and differentiable, as well as being easy to tie to boundaries. They natural exhibit the property of gradual then sudden increase without exploding. In this post, I provide some tips for how to adapt them to different problem cases.


```python
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
```

Here's the basic sigmoid function:


```python
def sigmoid(x: float) -> float:
    """
    Compute the sigmoid function for the input value x.
    For any output between negative infinity and positive infinity, it returns a response between 0 and 1
    """
    return 1 / (1 + np.exp(-x))
```

Let's see what it does.


```python
print(sigmoid(1))
print(sigmoid(0))
print(sigmoid(10))
print(sigmoid(-99))
```

    0.7310585786300049
    0.5
    0.9999546021312976
    1.0112214926104486e-43
    

Now let's make a function to plot functions so we can visualize them.


```python
def plot_function(func: Callable, start: float = -10, end: float = 10, step: float = 0.1, **kwargs):
    """
    Plot the given function within the specified range and step.

    Args:
        func: A function to plot.
        start: Start value of the x-axis range.
        end: End value of the x-axis range.
        step: Step size for x-axis values. Default is 0.1.
    """
    x_values = np.arange(start, end, step)
    y_values = func(x_values, **kwargs)

    plt.plot(x_values, y_values)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plot of the function")
    plt.grid(True)
    plt.show()
```


```python
plot_function(sigmoid)
```


    
![png](2023-03-18-sigmoid-functions-for-mathematical-modeling_files/2023-03-18-sigmoid-functions-for-mathematical-modeling_9_0.png)
    


Let's say we want to use to it model something. The y-bounds at 0 and 1 aren't necessarily what we want. Nor is the inflection point at x=0 or the amount of stretch. To allow us to tweak these, let's write a new sigmoid function that gives us parameters to play with.


```python
def sigmoid(x, x_shift=0, y_shift=0, x_scale=1, y_scale=1):
    """
    Parameterized sigmoid function
    """
    x_transformed = (x - x_shift) / x_scale
    sigmoid_value = 1 / (1 + np.exp(-x_transformed))
    y_transformed = y_scale * sigmoid_value + y_shift
    return y_transformed
```

We can see that the base case is the same.


```python
plot_function(sigmoid)
```


    
![png](2023-03-18-sigmoid-functions-for-mathematical-modeling_files/2023-03-18-sigmoid-functions-for-mathematical-modeling_13_0.png)
    


But now we can also move it around. Let's slide it to the right.


```python
plot_function(sigmoid, x_shift=4)
```


    
![png](2023-03-18-sigmoid-functions-for-mathematical-modeling_files/2023-03-18-sigmoid-functions-for-mathematical-modeling_15_0.png)
    


Now drop it down.


```python
plot_function(sigmoid, x_shift=4, y_shift=-5)
```


    
![png](2023-03-18-sigmoid-functions-for-mathematical-modeling_files/2023-03-18-sigmoid-functions-for-mathematical-modeling_17_0.png)
    


Now stretch it in the y-axis. Note the change in the y-axis labels below.


```python
plot_function(sigmoid, x_shift=4, y_shift=-5, y_scale=10)
```


    
![png](2023-03-18-sigmoid-functions-for-mathematical-modeling_files/2023-03-18-sigmoid-functions-for-mathematical-modeling_19_0.png)
    


Depending on your use-case, you may want to specify certain conditions. For example, say you wanted to specify the min and max of the function. There's no explicit parameter for that, so we'll have to figure out how to express that given the parameters we have. The two that we care about for this case are `y_shift` and `y_scale`. The `x_shift` and `x_scale` parameters could be anything in this case because we haven't specified them. We could add additional constraints for them, but in this example I'll simply leave them alone. That leaves use with two unknowns, `y_shift` and `y_scale` and two conditions, which we can solve for.

We know two points:
1. x approaches infinity and y approaches the desired max
2. x approaches negative infinity and y approaches the desired min

We'll use $$ \sigma $$ to represent the sigmoid function.

Our starting formula is what we wrote in the sigmoid function:

$$ \sigma(x) = \frac{y_\text{scale}}{1 + e^{-x_\text{scale}(x - x_\text{shift})}} + y_\text{shift} $$


Now let's plug in the following:

$$ x = \infty $$

$$ y = max_\text{desired} $$

Here's what we get:

$$ \sigma(\infty) = \frac{y_\text{scale}}{1 + e^{-\infty}} + y_\text{shift} = \frac{y_\text{scale}}{1 + 0} + y_\text{shift} = y_\text{scale} + y_\text{shift} $$

Therefore:

$$ y_\text{scale} + y_\text{shift} = max_\text{desired} $$

At negative inifinity, we've got:

$$ \sigma(-\infty) = \frac{y_\text{scale}}{1 + e^{\infty}} + y_\text{shift} = \frac{y_\text{scale}}{\infty} + y_\text{shift} = y_\text{shift} $$

Therefore:

$$ y_\text{shift} = min_\text{desired} $$

plugging this into the above equation, we have:

$$ y_\text{scale} + min_\text{desired} = max_\text{desired} $$

Ending with:

$$ y_\text{shift} = min_\text{desired} $$

$$ y_\text{scale} = max_\text{desired} - min_\text{desired} $$

Let's give it a try.


```python
desired_max = 100
desired_min = 85
```


```python
y_shift = desired_min
y_scale = desired_max - desired_min
```


```python
plot_function(sigmoid, y_shift=y_shift, y_scale=y_scale)
```


    
![png](2023-03-18-sigmoid-functions-for-mathematical-modeling_files/2023-03-18-sigmoid-functions-for-mathematical-modeling_44_0.png)
    


Another thing you might do is fit an equation with an inflection point and a desired max. Again, we have two equations and two unknowns.

Let's start with our sigmoid equation again.

$$ \sigma(x) = \frac{y_\text{scale}}{1 + e^{-x_\text{scale}(x - x_\text{shift})}} + y_\text{shift} $$


We'll start with the following:

$$ x = \infty $$

$$ y = max_\text{desired} $$

We already know the answer:

$$ y_\text{scale} + y_\text{shift} = max_\text{desired} $$

And therefore:

$$ y_\text{shift} = max_\text{desired} - y_\text{scale} $$

At the inflection point, we know that the inflection point in x in just `x_shift`, so we can say that $$ x=x_\text{inflection}=x_\text{shift} $$ and $$ y = y_\text{inflection} $$ (our desired point). Plugging that in, we get:

$$ \sigma(x_\text{inflection}) = \frac{y_\text{scale}}{1 + e^{0}} + y_\text{shift} = \frac{y_\text{scale}}{2} + y_\text{shift} = y_\text{inflection} $$


Plugging in $$ y_\text{shift} = max_\text{desired} - y_\text{scale} $$, we get:

$$ \frac{y_\text{scale}}{2} + max_\text{desired} - y_\text{scale} = y_\text{inflection} $$


Ending with:

$$ y_\text{scale} = 2 * (max_\text{desired} - y_\text{inflection}) $$
$$ y_\text{shift} = max_\text{desired} - y_\text{scale} $$

$$ y_\text{scale} = 2 * (max_\text{desired} - y_\text{inflection}) $$

$$ y_\text{shift} = max_\text{desired} - y_\text{scale} $$


```python
x_inflection = 10
y_inflection = -12
desired_max = 0
```


```python
x_shift = x_inflection
y_scale = 2 * (desired_max - y_inflection)
y_shift = desired_max - y_scale
```


```python
plot_function(sigmoid, -10, 20, x_shift=x_shift, y_shift=y_shift, y_scale=y_scale)
```


    
![png](2023-03-18-sigmoid-functions-for-mathematical-modeling_files/2023-03-18-sigmoid-functions-for-mathematical-modeling_65_0.png)
    


Let's do another.


```python
x_inflection = 0.5
y_inflection = 1
desired_max = 2
```


```python
x_shift = x_inflection
y_scale = 2 * (desired_max - y_inflection)
y_shift = desired_max - y_scale
```


```python
plot_function(sigmoid, x_shift=x_shift, y_shift=y_shift, y_scale=y_scale)
```


    
![png](2023-03-18-sigmoid-functions-for-mathematical-modeling_files/2023-03-18-sigmoid-functions-for-mathematical-modeling_69_0.png)
    


Last one:


```python
x_inflection = 0
y_inflection = 0
desired_max = 1
```


```python
x_shift = x_inflection
y_scale = 2 * (desired_max - y_inflection)
y_shift = desired_max - y_scale
```


```python
plot_function(sigmoid, x_shift=x_shift, y_shift=y_shift, y_scale=y_scale)
```


    
![png](2023-03-18-sigmoid-functions-for-mathematical-modeling_files/2023-03-18-sigmoid-functions-for-mathematical-modeling_73_0.png)
    

