---
layout: post
title: "Distributions"
description: "This post shows distributions and how to plot them."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/cradle_mountain.jpg"
tags: [Data Visualization, Python, Statistics]
---

Distributions are super important. In this post I'll talk about some common distributions, how to plot them, and what they can be used for.

<b>Table of Contents</b>
* TOC
{:toc}

Distributions can be categorized as discrete or continuous based on whether the variable is discrete or continuous.


```python
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import binom, norm, pareto, poisson
from tweedie import tweedie # used for tweedie distribution; download with `pip install tweedie`
```


```python
# set seed
np.random.seed(0)
```

# Continuous Distributions

## Continuious Uniform Distribution

The most basic distribution is the uniform distribution. In this case every value between the limits is equally likely to be called. These can be continuous or discrete. Continuous uniform distributions are also called rectangular distributions.

#### Plot


```python
def plot_hist(bins):
    num_bins = len(bins) - 1
    plt.xlabel('Variable')
    plt.ylabel('Count')
    plt.title(f"Continuous Uniform Distribution with {num_bins} Bins")
    plt.show()
```


```python
num_samples = 10000
```


```python
x = np.random.uniform(0, 1, num_samples)
```


```python
num_bins = 20
count, bins, ignored = plt.hist(x, bins=num_bins, color='g', alpha=0.9) 
plot_hist(bins)
```


    
![png](2022-09-17-distributions_files/2022-09-17-distributions_13_0.png)
    


Note that the number of bins used to plot it can make it look more or less spiky. Here's the same data plotted with 200 bins.


```python
num_bins = 200
count, bins, ignored = plt.hist(x, bins=num_bins, color='g', alpha=0.9) 
plot_hist(bins)
```


    
![png](2022-09-17-distributions_files/2022-09-17-distributions_15_0.png)
    


#### Uses

Whether things are truly continuously distributed or not can be a bit of a question of definition. If time and space are quantized, then nothing in the physical world is truly a continuous uniform distribution. But these things are close enough. Some uniform distributions are:
* Random number generators (in the ideal case)
* The location a dart lands on a dartboard
* Asteroid impacts (roughly - it would only be continuous given a steady state of the universe)

## Gaussian (or Normal) Distribution

Gaussian distributions are all around us. These are the famous bell curves. I've found that physicists are more likely to say "Gaussian" and mathematicians are more likely to say "normal". It seems that "normal" is more popular, although old habits die hard, so I still say "Gaussian".

#### Equation

$$
f(x) = \frac{e^{-\frac{x^2}{2}}}{\sqrt{2\pi} } 
$$

Or, with more parameters:

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi} } e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
$$

Many distributions look roughly Gaussian but are not truly Gaussian. A paper on "[The unicorn, the normal curve, and other improbable creatures](https://psycnet.apa.org/record/1989-14214-001)" looked at 440 real datasets and found that none of them were perfectly normally distributed.

#### Plot


```python
fig = plt.figure()

mean = 0
standard_deviation = 1
x = np.random.normal(loc=mean, scale=standard_deviation, size=num_samples)

num_bins = 50
n, bins, patches = plt.hist(x, num_bins, density=True, color='g', alpha=0.9)

# add a line
y = norm.pdf(bins, loc=mean, scale=standard_deviation)
plt.plot(bins, y, 'r--')

plt.xlabel('Variable')
plt.ylabel('Probability')

plt.title("Gaussian Distribution")
plt.show()
```


    
![png](2022-09-17-distributions_files/2022-09-17-distributions_26_0.png)
    


You can also do this with `seaborn` and `displot`.


```python
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, num_samples)
```


```python
ax = sns.displot(s, color='skyblue')
ax.set(xlabel='Variable', ylabel='Count');
```


    
![png](2022-09-17-distributions_files/2022-09-17-distributions_29_0.png)
    


#### Uses

* Height is a famous example of a Gaussian distribution

## Tweedie Distribution

#### Plot


```python
num_params = 20
```


```python
# Generate random exogenous variables, num_samples x (num_params - 1)
exog = np.random.rand(num_samples, num_params - 1)

# Add a column of ones to the exogenous variables, num_samples x num_params
exog = np.hstack((np.ones((num_samples, 1)), exog))

# Generate random coefficients for the exogenous variables, num_params x 1
beta = np.concatenate(([500], np.random.randint(-100, 100, num_params - 1))) / 100

# Compute the linear predictor, num_samples x 1
eta = np.dot(exog, beta)

# Compute the mean of the Tweedie distribution, num_samples x 1
mu = np.exp(eta)

# Generate random samples from the Tweedie distribution, num_samples x 1
x = tweedie(mu=mu, p=1.5, phi=20).rvs(num_samples)
```


```python
num_bins = 50
n, bins, patches = plt.hist(x, num_bins, density=True, color='g', alpha=0.9, rwidth=0.9)
plt.xlabel('Variable')
plt.ylabel('Probability')

plt.title("Tweedie Distribution")
plt.show()
```


    
![png](2022-09-17-distributions_files/2022-09-17-distributions_36_0.png)
    


## Pareto Distribution

The distribution between the famous "80-20 rule", the Pareto distribution is based on a power law. The Pareto distribution is seen all the time, from social issues to scientific ones.

#### Plot


```python
# Distribution parameters
a = 2.0 # shape parameter
b = 1.0 # scale parameter

# Generate random samples from the distribution
pareto_samples = pareto.rvs(a, scale=b, size=1000)

# Plot the histogram of the samples
plt.hist(pareto_samples, bins=50, density=True, color='g', alpha=0.9, rwidth=0.9)

# Plot the probability density function (PDF)
x = np.linspace(pareto.ppf(0.01, a, scale=b), pareto.ppf(0.99, a, scale=b), 100)
plt.plot(x, pareto.pdf(x, a, scale=b), 'r-', lw=2, alpha=0.6, label='pareto pdf')

plt.xlabel('Variable')
plt.ylabel('Probability density')
plt.title('Pareto Distribution')
plt.legend()
plt.show()
```


    
![png](2022-09-17-distributions_files/2022-09-17-distributions_40_0.png)
    


#### Uses

* 80-20 rule

# Discrete Distributions

But lots of distributions aren't continuous, they deal with discrete predictions. For example, you can't flip 7.5 heads after 15 tries.

## Discrete Uniform Distribution

Lots of things are discrete uniform distribution.

#### Plot


```python
x = np.random.randint(0, 10, num_samples)
```


```python
count, bins, ignored = plt.hist(x, color='g', alpha=0.9, rwidth=0.9) 
num_bins = len(bins) - 1
plt.xlabel('Variable')
plt.ylabel('Count')
plt.title(f"Discrete Uniform Distribution")
plt.show()
```


    
![png](2022-09-17-distributions_files/2022-09-17-distributions_49_0.png)
    


#### Uses

* The results of rolling a die
* The card number from a card randomly drawn from a deck
* Winning lottery numbers
* Birthdays aren't perfectly uniform because they tend to bunch up for a variety of reasons, but they are close to uniform

## Binomial Distribution

Binomial distribution are based on discrete results from discrete events, like flipping a coin `n` times. From Wikipedia, a "binomial distribution with parameters n and p is the discrete probability distribution of the number of successes in a sequence of n independent experiments, each asking a yes–no question, and each with its own Boolean-valued outcome"

#### Plot

Let's say with run something with 100 trials, each of which has a 10% change of happening. To run that experiment, we can do:


```python
num_trials = 100
prob_success = 0.1
```


```python
x = np.random.binomial(n=100, p=0.1)
print(f"In this experiment, we got heads {x} times")
```

    In this experiment, we got heads 11 times
    

But we can run this whole experiment over and over again and see what we get.


```python
fig = plt.figure()

# Generate data
x = np.random.binomial(n=num_trials, p=prob_success, size=num_samples)
# you could also do this with scipy like so:
# x = binom.rvs(n=num_trials, p=prob_success, size=num_samples)

num_bins = 25
# Define bin edges explicitly to avoid gaps
bin_edges = np.arange(min(x), max(x) + 2)

# Plot the histogram
plt.hist(x, bins=bin_edges, density=True, color='g', alpha=0.9, rwidth=0.9)

plt.xlabel('Variable')
plt.ylabel('Probability')

plt.title("Normal Distribution Histogram (Bin size {})".format(num_bins))
plt.show()

```


    
![png](2022-09-17-distributions_files/2022-09-17-distributions_59_0.png)
    


#### Uses

* Flipping a coin
* Shooting free throws
* Lots of things that seem to be Gaussian but are discrete

## Poisson Distribution

Many reallife things follow a Poisson distribution. Poisson distributions are like binomial distributions in that they count discrete events. But it's different in that there is no fixed number of experiments to run. Imagine a soccer game. So it's 
discrete occurances (yes/no for a goal being scored) along a continuous distribution (time).

At any moment, a goal could be scored. Thus it is continuous. There's no specific number of "goal attempts" in a Poisson distribution.

If you wanted to find the number of shots on goal and see how many of them were goals, then it would become binomial. But if you look at it over time it's Poisson. It's like the limit of a binomial distribution where you have infinite attempts and each has an infintessimal change of being positive.

Note that Poisson and binomial distributions can't be negative.

#### Equation

Mathematically, this looks like: $$ P(x = k) = e^{-\lambda}\frac{\lambda^k}{k!} $$

#### Plot


```python
data_poisson = poisson.rvs(mu=3, size=num_samples)
```


```python
ax = sns.displot(data_poisson, color='skyblue')
ax.set(xlabel='Variable', ylabel='Count');
```


    
![png](2022-09-17-distributions_files/2022-09-17-distributions_68_0.png)
    


#### Uses

* Decay events from radioactive sources
* Meteorites hitting the Earth
* Emails sent in a day

#### Example

Poisson distributions are very useful in predicting rare events. If once-in-a-lifetime floods occur once every hundred years, what is the chance that we have five within a hundred year period?

In this case $$ \lambda = 1 $$ and $$ k = 5 $$


```python
math.exp(-1)*(1**5/math.factorial(5))
```




    0.0030656620097620196


