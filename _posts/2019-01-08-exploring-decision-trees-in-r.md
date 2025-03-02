---
layout: post
title: Exploring Decision Trees in R
description: "An exploration of decision trees in R based on chapter 8 of An Introduction to Statistical Learning with Applications in R."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/tree_roots.jpg"
tags: [R, Machine Learning, Decision Trees]
---

This post aims to explore decision trees for the NOVA Deep Learning Meetup. It is based on chapter 8 of *An Introduction to Statistical Learning with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani. It also discusses methods to improve decision tree performance, such as bagging, random forest, and boosting. There are two posts with the same material, one in R and [one in Python](https://jss367.github.io/exploring-decision-trees-in-python.html).

<b>Table of Contents</b>
* TOC
{:toc}

# Decision Trees for Regression

Decision trees are models for decision making that rely on repeatedly splitting the data based on various parameters. That might sound complicated, but the concept should be familiar to most people. Imagine that you are deciding what to do today. First, you ask yourself if it is a weekday. If it is, you'll go to work, but if it isn't, you'll have to decide what else to do with your free time, so you have to ask another question. Is it raining? If it is, you'll read a book, but if it isn't, you'll go to the park. You've already made a decision tree.

![Decision Tree]({{site.baseurl}}/assets/img/decision_tree.png "Decision Tree")

## Overview

There are good reasons that decision trees are popular in statistical learning. For one thing, they can perform either classification or regression tasks. But perhaps the most important aspect of decision trees is that they are interpretable. They combine the power of machine learning with a result that can be easily visualized. This is both a powerful and somewhat rare capability for something created with a machine learning algorithm.

To see how they work, let's explore the Hitters database which contains statistics and salary information for professional baseball players. The dataset has a lot of information in it, but we're just going to focus on three things: Hits, Years, and Salary. In particular, we're going to see if we can predict the Salary of a player from their Hits and Years.


```R
library(ISLR)
library(dplyr)
library(ggplot2)
library(rcompanion)
library(tree)
library(rpart)
library(formattable)
```

    
    Attaching package: 'dplyr'
    
    The following objects are masked from 'package:stats':
    
        filter, lag
    
    The following objects are masked from 'package:base':
    
        intersect, setdiff, setequal, union
    
    





```R
attach(Hitters)
```

Let's take a quick look at the data.


```R
glimpse(Hitters)
```

    Observations: 322
    Variables: 20
    $ AtBat     <int> 293, 315, 479, 496, 321, 594, 185, 298, 323, 401, 574, 20...
    $ Hits      <int> 66, 81, 130, 141, 87, 169, 37, 73, 81, 92, 159, 53, 113, ...
    $ HmRun     <int> 1, 7, 18, 20, 10, 4, 1, 0, 6, 17, 21, 4, 13, 0, 7, 3, 20,...
    $ Runs      <int> 30, 24, 66, 65, 39, 74, 23, 24, 26, 49, 107, 31, 48, 30, ...
    $ RBI       <int> 29, 38, 72, 78, 42, 51, 8, 24, 32, 66, 75, 26, 61, 11, 27...
    $ Walks     <int> 14, 39, 76, 37, 30, 35, 21, 7, 8, 65, 59, 27, 47, 22, 30,...
    $ Years     <int> 1, 14, 3, 11, 2, 11, 2, 3, 2, 13, 10, 9, 4, 6, 13, 3, 15,...
    $ CAtBat    <int> 293, 3449, 1624, 5628, 396, 4408, 214, 509, 341, 5206, 46...
    $ CHits     <int> 66, 835, 457, 1575, 101, 1133, 42, 108, 86, 1332, 1300, 4...
    $ CHmRun    <int> 1, 69, 63, 225, 12, 19, 1, 0, 6, 253, 90, 15, 41, 4, 36, ...
    $ CRuns     <int> 30, 321, 224, 828, 48, 501, 30, 41, 32, 784, 702, 192, 20...
    $ CRBI      <int> 29, 414, 266, 838, 46, 336, 9, 37, 34, 890, 504, 186, 204...
    $ CWalks    <int> 14, 375, 263, 354, 33, 194, 24, 12, 8, 866, 488, 161, 203...
    $ League    <fct> A, N, A, N, N, A, N, A, N, A, A, N, N, A, N, A, N, A, A, ...
    $ Division  <fct> E, W, W, E, E, W, E, W, W, E, E, W, E, E, E, W, W, W, W, ...
    $ PutOuts   <int> 446, 632, 880, 200, 805, 282, 76, 121, 143, 0, 238, 304, ...
    $ Assists   <int> 33, 43, 82, 11, 40, 421, 127, 283, 290, 0, 445, 45, 11, 1...
    $ Errors    <int> 20, 10, 14, 3, 4, 25, 7, 9, 19, 0, 22, 11, 7, 6, 8, 0, 10...
    $ Salary    <dbl> NA, 475.000, 480.000, 500.000, 91.500, 750.000, 70.000, 1...
    $ NewLeague <fct> A, N, A, N, N, A, A, A, N, A, A, N, N, A, N, A, N, A, A, ...
    

First we'll clean up the data and then make a plot of it. We're just going to focus on how Years and Hits affect Salary.


```R
clean.hitters <- na.omit(Hitters)
```


```R
salary.plot <- ggplot(data = clean.hitters, aes(x = Years, y = Hits, color = Salary)) + geom_point()
# Make a color gradient
salary.plot + scale_color_gradientn(colours = rainbow(5, start = 0, end = .8))
```




![png]({{site.baseurl}}/assets/img/{{site.baseurl}}/assets/img/2019-01-08-Exploring-Decision-Trees-in-R_files/2019-01-08-Exploring-Decision-Trees-in-R_16_1.png)


Just from this we can see some outliers with really high salaries. This will cause a right-skewed distribution. One nice thing about decision trees is that we don't have to do any feature processing on the predictors. We do, however, have to take the target variable distribution into account. We don't want a few high salaries to have too much weight in the squared error minimization. So to make the distribution more even, we may have to take a log transformation of the salary data. Let's look at the distribution.

## Data Transformation


```R
plotNormalHistogram(clean.hitters$Salary)
```


![png]({{site.baseurl}}/assets/img/{{site.baseurl}}/assets/img/2019-01-08-Exploring-Decision-Trees-in-R_files/2019-01-08-Exploring-Decision-Trees-in-R_19_0.png)


It's definitely right-skewed. Let's look at a log transform to see if that's better.


```R
plotNormalHistogram(log(clean.hitters$Salary))
```


![png]({{site.baseurl}}/assets/img/{{site.baseurl}}/assets/img/2019-01-08-Exploring-Decision-Trees-in-R_files/2019-01-08-Exploring-Decision-Trees-in-R_21_0.png)


This we can work with. Let's add this to the dataframe.


```R
log.salary <- log(clean.hitters$Salary)
log.hitters <- data.frame(clean.hitters, log.salary)
```

## Building the Decision Tree

OK, let's try to predict `log.salary` based on everything in `log.hitters` (we'll exclude Salary for obvious reasons).

Here's the part where the machine learning comes in. Instead of creating our own rules, we're going to feed all the data into a decision tree algorithm and let the algorithm build the tree for us by finding the optimal decision points.

OK, let's build the tree. We'll use the rpart library in R and sklearn in Python.


```R
tree <- rpart(log.salary ~Hits + Years,
    data = log.hitters, control = rpart.control(minbucket = 5))
```


```R
summary(tree)
```

    Call:
    rpart(formula = log.salary ~ Hits + Years, data = log.hitters, 
        control = rpart.control(minbucket = 5))
      n= 263 
    
              CP nsplit rel error    xerror       xstd
    1 0.44457445      0 1.0000000 1.0050559 0.06544393
    2 0.11454550      1 0.5554255 0.5637088 0.05906918
    3 0.04446021      2 0.4408800 0.4601615 0.05756653
    4 0.01831268      3 0.3964198 0.4201148 0.05884280
    5 0.01690198      4 0.3781072 0.4302385 0.06369715
    6 0.01675238      5 0.3612052 0.4242531 0.06352470
    7 0.01107214      6 0.3444528 0.4330694 0.06377200
    8 0.01000000      7 0.3333807 0.4363175 0.06490774
    
    Variable importance
    Years  Hits 
       74    26 
    
    Node number 1: 263 observations,    complexity param=0.4445745
      mean=5.927222, MSE=0.7876568 
      left son=2 (90 obs) right son=3 (173 obs)
      Primary splits:
          Years < 4.5   to the left,  improve=0.4445745, (0 missing)
          Hits  < 117.5 to the left,  improve=0.2229369, (0 missing)
      Surrogate splits:
          Hits < 29.5  to the left,  agree=0.669, adj=0.033, (0 split)
    
    Node number 2: 90 observations,    complexity param=0.04446021
      mean=5.10679, MSE=0.4705907 
      left son=4 (62 obs) right son=5 (28 obs)
      Primary splits:
          Years < 3.5   to the left,  improve=0.2174595, (0 missing)
          Hits  < 112.5 to the left,  improve=0.1823794, (0 missing)
      Surrogate splits:
          Hits < 154.5 to the left,  agree=0.722, adj=0.107, (0 split)
    
    Node number 3: 173 observations,    complexity param=0.1145455
      mean=6.354036, MSE=0.4202619 
      left son=6 (90 obs) right son=7 (83 obs)
      Primary splits:
          Hits  < 117.5 to the left,  improve=0.32636580, (0 missing)
          Years < 6.5   to the left,  improve=0.03890075, (0 missing)
      Surrogate splits:
          Years < 11.5  to the right, agree=0.543, adj=0.048, (0 split)
    
    Node number 4: 62 observations,    complexity param=0.01831268
      mean=4.891812, MSE=0.3711076 
      left son=8 (43 obs) right son=9 (19 obs)
      Primary splits:
          Hits  < 114   to the left,  improve=0.16487440, (0 missing)
          Years < 1.5   to the left,  improve=0.03525102, (0 missing)
    
    Node number 5: 28 observations
      mean=5.582812, MSE=0.3619427 
    
    Node number 6: 90 observations,    complexity param=0.01690198
      mean=5.99838, MSE=0.3121523 
      left son=12 (26 obs) right son=13 (64 obs)
      Primary splits:
          Years < 6.5   to the left,  improve=0.1246296, (0 missing)
          Hits  < 72.5  to the left,  improve=0.0804416, (0 missing)
      Surrogate splits:
          Hits < 112.5 to the right, agree=0.733, adj=0.077, (0 split)
    
    Node number 7: 83 observations
      mean=6.739687, MSE=0.2516033 
    
    Node number 8: 43 observations,    complexity param=0.01675238
      mean=4.727386, MSE=0.3987367 
      left son=16 (38 obs) right son=17 (5 obs)
      Primary splits:
          Hits  < 40.5  to the right, improve=0.20240190, (0 missing)
          Years < 1.5   to the left,  improve=0.01335827, (0 missing)
    
    Node number 9: 19 observations
      mean=5.263932, MSE=0.1089185 
    
    Node number 12: 26 observations
      mean=5.688925, MSE=0.2783727 
    
    Node number 13: 64 observations,    complexity param=0.01107214
      mean=6.124096, MSE=0.2711673 
      left son=26 (12 obs) right son=27 (52 obs)
      Primary splits:
          Hits  < 50.5  to the left,  improve=0.13216210, (0 missing)
          Years < 7.5   to the right, improve=0.02132578, (0 missing)
    
    Node number 16: 38 observations
      mean=4.624337, MSE=0.08631659 
    
    Node number 17: 5 observations
      mean=5.510558, MSE=2.079066 
    
    Node number 26: 12 observations
      mean=5.730017, MSE=0.2241199 
    
    Node number 27: 52 observations
      mean=6.215037, MSE=0.2379161 
    
    

Now let's take a look at it.


```R
plot(tree, main = "Decision Tree for Hitters Dataset")
text(tree)
```


![png]({{site.baseurl}}/assets/img/{{site.baseurl}}/assets/img/2019-01-08-Exploring-Decision-Trees-in-R_files/2019-01-08-Exploring-Decision-Trees-in-R_31_0.png)


We've made our first decision tree. Let's go over some terminology. At the top it says `Years < 4.5`. This is the first decision point in the tree and is called the *root node*. From there, we see a horizontal line that we could take either to the left or to the right. These are called *left-hand* and *right-hand branches*. If we go down the right-hand branch by following the line to the right and then down, we see `Hits<117.5` and another decision point. This is called an *internal node*. At the bottom, we have a few different numbers. Because no more decision points emanate from them, these are called *terminal nodes* or a *leaves*. A decision tree will always have one more terminal node than split point. As you can see, the "tree" is actually upside down (perhaps statistician don't spend enough time outside?).

OK, so now how do we use it?

Let's go back to the root node where it says `Years < 4.5`. From here the tree splits into two paths because we've reached a decision point. If the statement is true for a given instance, as in the player has fewer than 4.5 Years, we go to the left. There, we see 5.107 and we've reached the end of the line. So for players with fewer than 4.5 Years, the model predicts 5.107. Remember that we're predicting the natural logarithm of the salary in thousands of dollars. So let's calculate what the actual number would be to see if it makes sense.


```R
currency(1000 * exp(5.107), digits = 0L)
```


$165,174


That looks right. But where did that number come from? When using decision trees for regression, the prediction, 5.107 in this case, comes from taking the mean of all the instances in that category.

Let's explore what happens for players with more than 4.5 Years. Now we go down the right branch of the and we reach `Hits<117.5`. Another decision point! Here, if the number of Hits is fewer than 117.5, we go down the left side of the tree to 5.998. If it's greater than 117.5, we go down the right side to 6.74. Let's see how much this is.


```R
currency(1000 * exp(6.74), digits = 0L)
```


$845,561


## Regions in Decision Trees

Another way to think about decision trees is that they are algorithms that split the input space into different regions. This isn't particularly easy to see in the tree, so we'll plot the predictor space.


```R
plot(Years, Hits, pch = 16, col = "orange")
abline(v = 4.5, col = "dark green", lwd = 3)
segments(4.5, 117.5, 25, 117.5, col = " dark green", lwd = 3)
text(1.5, 120, "R1")
text(12, 50, "R2")
text(12, 175, "R3")
```


![png]({{site.baseurl}}/assets/img/{{site.baseurl}}/assets/img/2019-01-08-Exploring-Decision-Trees-in-R_files/2019-01-08-Exploring-Decision-Trees-in-R_41_0.png)


You can see that the predictor space is broken into three regions.

$$ R1 ={X | Years< 4.5} $$

$$ R2 ={X | Years>=4.5, Hits<117.5} $$

$$ R3 ={X | Years>=4.5, Hits>=117.5} $$

We can see the same tree features here that we saw above. R1, R2, and R3 are the three leaves or terminal nodes. Each decision point results in a line, so you can count the number of lines to get the number of decision points, which are also the number of internal nodes plus the root node.

We make make predictions just as easily with this plot. To make a prediction, we see what region the input falls into, then assign the value to the mean of the entities in that region. So if someone has 10 years and last year had 50 hits, we would put them in R2. The mean logSalary of all our training samples in that region was 5.998, so we'll predict that value.

## Interpreting the Tree

One of the good things about decision trees is that most people already have an intuition about them, so interpretation comes easily. Imagine, for example, that you're playing 20 Questions. You know the key of the game is to ask good questions up front that help to split the possible solutions into (ideally) two equal sizes. The exact same applies here.

So, why did the tree start with Years? Just like in decision trees for 20 Questions, they start with the most important factor first. This is valuable information; in this case, it tells us that Years is the most important factor in determining logSalary (out of Years and Hits). Then, the next factor is Hits (admittedly these are the only two factors we're looking at). And Hits has a bigger effect on the Salary of players who have played longer than 4.5 years versus those who haven't. If we had done a fourth split, it could very well split the less experienced players by Hits as well.

As you might imagine, this model wouldn't perform particularly well in the real world. On the other hand, it's got some predictive power and is easy to interpret. That's an important tradeoff.

## Decision Tree Algorithms

Let's talk a little about how this works mathematically.

#### Building the Tree

The goal of decision tree algorithms is to find $$ J $$ regions $$ R_1, ..., R_J $$ that minimize the RSS given by

$$ \sum_{j=1}^{J}{\sum_{i\in R_j}}({y_i-\hat{y}_{R_j}})^2 $$

Note that the regions that minimize that equation don't have to be rectangular, but we'll impose an addition constraint that they are so that we can can easily interpret the decision points. This means that each question can only be about a single variable. The alternative would be to allow questions like "Is feature 1 times feature 2 greater than 25?" This is not allowed in decision tree implementations. So we'll assume that each region is a box.

Even so, it would be computational expensive (or even infeasible) to consider every possible partition of the feature space into $$ J $$ boxes. So instead we use a *top-down*, *greedy* approach known as *recursive binary splitting*. "Top-down" means that it begins at the top of the tree and works its way down. "Greedy" means that it maximizes the value of the split at that particular step, without considering future splits. This is done to make the calculations feasible and may not actually result in the optimal decision tree. It's possible that a less optimal decision early on allows for much better decisions later, resulting in an overall better tree.

The algorithm continues to create more split points until a stopping condition is met. The stopping conditions are:

1. All the samples have been correctly classified (the leaf nodes are pure)
2. The maximum node depth (set by user) is reached
3. Splitting nodes no longer leads to a sufficient information gain

The three most common algorithms are ID3, C4.5, and CART. They are all similar in some ways but have tradeoffs. ID3 was the first of these to be invented. It had significant limitations, such as it could only handle categorical data, couldn't handle missing values, and is subject to overfitting. C4.5 is an improved version of ID3. It can handle numeric data as well as missing values and has methods to reduce overfitting. CART (Classification And Regression Trees) is perhaps the most popular decision tree algorithm. It handles outliers better than the other approaches. They all use different criteria to determine a split, which we'll get into later.

## Pruning

So, how big could we make a decision tree? In theory, there is no limit. We could keep building up a decision tree, making it bigger and bigger until every training example is predicted correctly. But this would grossly overfit the data. It turns out that even if we limit the size of the tree by, say, limiting the minimum bucket size, we could still be subject to overfitting. Another way to stop the tree would be to stop when each split decreases the RSS by some threshold. But this has downsides. An unimportant split early on could lead to a better split in the later stages. So what do we do?

The best way to handle overfitting is not to build a small tree, but to build a large tree and prune it. The method used to do this is called "Cost complexity pruning" or "weakest link pruning". Because there are more powerful methods than pruning a single tree, we'll just touch on pruning without going into detail on it.

The key to pruning is to add a penalizing factor to the size of the tree (number of terminal nodes in it) just like we do with lasso regression.

For each value of $$ \alpha $$ there exists a subtree $$ T \subset T_0 $$ where we minimize:

$$ \sum_{m=1}^{\vert T\vert}{\sum_{x_i\in R_m}{(y_i-\hat{y}_{R_m})^2+\alpha\vert T\vert}} $$

where

$$ \vert T\vert $$ - number of terminal nodes in tree $$ T $$

$$ \alpha $$ - the penalty parameter; we can tune this to control the trade-off between the subtree's complexity and its fit to the training data

$$ \sum_{x_i\in R_m}{(y_i-\hat{y}_{R_m})^2} $$ - sum of squares in region around region m

$$ \sum_{m=1}^{\vert T\vert} $$ - add the above sum up for all the terminal nodes

We use cross-validation to find the best alpha, know as $$ \hat{\alpha} $$. Then we return to the full dataset and obtain the subtree corresponding to the best alpha $$ (\hat{\alpha}) $$.

Different decision tree algorithms handle pruning differently. ID3 doesn't do any pruning. CART uses cost-complexity pruning. C4.5 uses error-based pruning.

#### Summary

* Use recursive binary splitting to grow a large tree
* Use a stopping criterion, such as stopping when each terminal node has fewer than some minimum number of observations, to stop growing the tree; it's OK to build a large tree that overfits the data
* Apply pruning to the large tree in order to obtain a sequence of best subtrees, as a function of alpha
* use k-fold cross-validation to find alpha; pick that alpha that minimizes the average error
* Choose the subtree that corresponds to that value of alpha


```R
smp_size <- floor(0.75 * nrow(log.hitters))

set.seed(0)
train_ind <- sample(seq_len(nrow(log.hitters)), size = smp_size)

train <- log.hitters[train_ind, ]
test <- log.hitters[-train_ind, ]
```


```R
fit2 <- rpart(log.salary ~. - Salary,
    data = log.hitters)
```


```R
plot(fit2, main = "Decision Tree for Hitters Dataset")
text(fit2)
```


![png]({{site.baseurl}}/assets/img/{{site.baseurl}}/assets/img/2019-01-08-Exploring-Decision-Trees-in-R_files/2019-01-08-Exploring-Decision-Trees-in-R_70_0.png)


Note that the length of the arms in the graphic is related to the size of the split.

![Cross validation]({{site.baseurl}}/assets/img/mse_tree_size.jpg "Tree size")
Image from ISLR.

Looks like a tree size of three is best. Size of tree is actually the number of terminal nodes.

# Decision Trees for Classification

Now let's talk a little bit about using decision tree for classification, which are sometimes called *classification trees*. Most of the concepts mentioned above apply equally well to classification trees, so let's focus on the differences.

The first difference is that we'll be classifying inputs into a category instead of predicting a value for them. This works as you might expect. Instead of taking the mean of all the instances in a region, we'll have those instances all vote, and use majority wins to determine what to classify new samples in that region.


Because we're doing classification, we can't use the same loss function as before. The most obvious substitute would be to find the fraction of training observations in a particular region don't belong to the majority class, then subtract that to get the error. This is called the *classification error rate*.

This can be written mathematically as:

$$ E=1-max_k(\hat{p}_{mk}) $$

Here $$ \hat{p}_{mk} $$ represents the proportion of training observations in the $$ m $$th region that are from the $$ k $$th class.

Although this may be the most straightforward method of identifying error, it isn't always the most effective. Two other measures of impurity are common used: The Gini index and cross-entropy. Here are the formulas for them:

Gini index:

$$ G=\sum_{k=1}^{K}{\hat{p}_{mk}}(1-\hat{p}_{mk}) $$

Cross-enropy:

$$ D=-\sum_{k=1}^{K}{\hat{p}_{mk}}\log\hat{p}_{mk}\frac{1}{2\log(2)} $$

Note that we've added a scaling factor to the cross-entropy function to make it easier to compare with the other methods.

The Gini index (also called Gini coefficient) and cross-entropy are common used measures of impurity. Gini index is the default criterion for scikit-learn.

Let's look at an example. Say there are ten instances in a region, three of which are labeled as True and seven of which as labeled as False.

To calculate the classification error rate, we can see that the majority class, $$ k $$, is False. Thus, $$ \hat{p}_{mk} $$, the proportion of observations in this region that are False, is 0.7. So the final classification error rate would be $$ E = 1 - 0.7 = 0.3 $$.

The Gini index in the above scenario would be $$ 0.7*0.3 + 0.3 * 0.7 = 0.21 + 0.21 = 0.42 $$.

The cross-entropy is harder to calculate by hand so we'll plug it into the computer.


```R
- ( (0.7 * log(0.7)) + (0.3 * log(0.3))) / (2 * log(2))
```


0.440645449615346


Note that the (scaled) cross-entropy is the greatest, then the Gini index, then the classification error rate. If we plot the formulas, we'll see that this is what we should expect.


```R
classification_error_rate <- function(p)
    return (1 - pmax(p, 1 - p))

gini_index <- function(p)
    return (2 * p * (1 - p))

cross_entropy <- function(p)
    return ( (p * log( (1 - p) / p) - log(1 - p)) / (2 * log(2)))
```


```R
p <- seq(0, 1, 0.001)
plot(p, classification_error_rate(p), pch = 16, col = "red", ylab = "Impurity Index")
points(p, gini_index(p), pch = 16, col = "blue")
points(p, cross_entropy(p), pch = 16, col = "green")
legend(0.7, 0.5, legend = c("Class. Error Rate", "Gini Index", "Cross-entropy"),
       col = c("red", "blue", "green"), lty = 1, cex = .94)
```


![png]({{site.baseurl}}/assets/img/{{site.baseurl}}/assets/img/2019-01-08-Exploring-Decision-Trees-in-R_files/2019-01-08-Exploring-Decision-Trees-in-R_93_0.png)


## Heart Disease Example

Let's look at an example using heart disease data.


```R
heart <- read.table("data/Heart.csv", header = TRUE, sep = ",")
```


```R
glimpse(heart)
```

    Observations: 303
    Variables: 15
    $ X         <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17...
    $ Age       <int> 63, 67, 67, 37, 41, 56, 62, 57, 63, 53, 57, 56, 56, 44, 5...
    $ Sex       <int> 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, ...
    $ ChestPain <fct> typical, asymptomatic, asymptomatic, nonanginal, nontypic...
    $ RestBP    <int> 145, 160, 120, 130, 130, 120, 140, 120, 130, 140, 140, 14...
    $ Chol      <int> 233, 286, 229, 250, 204, 236, 268, 354, 254, 203, 192, 29...
    $ Fbs       <int> 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, ...
    $ RestECG   <int> 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, ...
    $ MaxHR     <int> 150, 108, 129, 187, 172, 178, 160, 163, 147, 155, 148, 15...
    $ ExAng     <int> 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...
    $ Oldpeak   <dbl> 2.3, 1.5, 2.6, 3.5, 1.4, 0.8, 3.6, 0.6, 1.4, 3.1, 0.4, 1....
    $ Slope     <int> 3, 2, 2, 3, 1, 1, 3, 1, 2, 3, 2, 2, 2, 1, 1, 1, 3, 1, 1, ...
    $ Ca        <int> 0, 3, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...
    $ Thal      <fct> fixed, normal, reversable, normal, normal, normal, normal...
    $ AHD       <fct> No, Yes, Yes, No, No, No, Yes, No, Yes, Yes, No, No, Yes,...
    


```R
clean.heart <- na.omit(heart)
```


```R
glimpse(clean.heart)
```

    Observations: 297
    Variables: 15
    $ X         <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17...
    $ Age       <int> 63, 67, 67, 37, 41, 56, 62, 57, 63, 53, 57, 56, 56, 44, 5...
    $ Sex       <int> 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, ...
    $ ChestPain <fct> typical, asymptomatic, asymptomatic, nonanginal, nontypic...
    $ RestBP    <int> 145, 160, 120, 130, 130, 120, 140, 120, 130, 140, 140, 14...
    $ Chol      <int> 233, 286, 229, 250, 204, 236, 268, 354, 254, 203, 192, 29...
    $ Fbs       <int> 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, ...
    $ RestECG   <int> 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, ...
    $ MaxHR     <int> 150, 108, 129, 187, 172, 178, 160, 163, 147, 155, 148, 15...
    $ ExAng     <int> 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...
    $ Oldpeak   <dbl> 2.3, 1.5, 2.6, 3.5, 1.4, 0.8, 3.6, 0.6, 1.4, 3.1, 0.4, 1....
    $ Slope     <int> 3, 2, 2, 3, 1, 1, 3, 1, 2, 3, 2, 2, 2, 1, 1, 1, 3, 1, 1, ...
    $ Ca        <int> 0, 3, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...
    $ Thal      <fct> fixed, normal, reversable, normal, normal, normal, normal...
    $ AHD       <fct> No, Yes, Yes, No, No, No, Yes, No, Yes, Yes, No, No, Yes,...
    

Lot of data to work with. Let's try to predict AHD.


```R
heart.tree <- rpart(AHD ~., data = clean.heart)
```


```R
plot(heart.tree)
text(heart.tree)
```


![png]({{site.baseurl}}/assets/img/{{site.baseurl}}/assets/img/2019-01-08-Exploring-Decision-Trees-in-R_files/2019-01-08-Exploring-Decision-Trees-in-R_102_0.png)


# Ensembling for Better Models

Decision trees are often not as successful as some of the other methods. So we'll also talk about bagging, random forests, and boosting. Each of these is a method to produce and combine multiple trees to that gives a single prediction.

Although decision trees can be powerful machine learning models, they often don't match up against other models, such as support vector machines. But there's a way to make up for this by combining the output of several algorithms into a single classifier. This is known as *ensembling*.

Any machine learning model can be combined with others in an ensemble, so these techniques are not specific to decision trees. However, decision trees have many properties that make them especially good for ensemble learning.

The most popular techniques for ensemble learning are bagging, random forest, and boosting. They can all be used with decision trees to build more powerful models.

## Bagging

Heavily trained decision trees can often suffer from high variance. This means that if we build a decision tree of off two halves of the same dataset, they could have significant differences. This might sound like a negative, and it is, but we can also use it to our benefit. The high variance between the models allows us to combine decision trees to make better predictions. If the variance were low, we wouldn't gain much from combining models trained from different data sets. Overall, this can significantly improve the accuracy of your model. Let's talk about a few ways we can do that.

The first is called *bagging*, which comes from bootstrap aggregation. Its job is to reduce the variance of a model.

Bagging relies on an important concept in statistics. Recall that if you have $$ N $$ observations, each of which has a variance of $$ \sigma^2 $$, then the variance of the mean will be $$ \sigma^2/N $$. We'll we can use the same principle here. In bagging, we take our single training set and generate B different bootstrapped sets by taking a random sample with replacement. Then we make B decision trees each making their own prediction. Then we average those predictions into a final answer.

Mathematically, it looks like this:

$$ \hat{f}_{avg}(x)=\frac{1}{B}\sum_{b=1}^B{\hat{f}^b(x)} $$

When bagging with regression, you take the average of all the resulting predictions. For classification, simply take a majority vote.

#### Overfitting

One of the great aspects of bagging is that creating more trees does not result in overfitting. Not only that, but you can input large, overfit trees that haven't been pruned. The variance in the different trees will balance each other out. You can use hundreds or even thousands of trees to average. The way to determine the number of trees is to watch when your accuracy stops decreasing.

#### Out of bag error estimation

How do we estimate the error? There's actually an easy way to do this and you don't need to use cross validation. It's called *out-of-bag error estimation*. Imagine the following situation.

If there are N training examples in total and you choose N examples for each bagged dataset, you will almost certainly have duplicates of some (because you're replacing them each time) and have completely missed others. In fact, on average you will have $$ 1-\frac{1}{e} \approx 0.632 $$ of the example in the original dataset appear in the bagged one. That leaves another 36.8% of the data in the original training set that your model has never seen. These are called *out-of-bag* (OOB) instances because they aren't included in the bagged set. And because your model hasn't seen them, then they can be used as a test set to get an accuracy score.

We'll take the OOB values and generate predictions with them. Then we'll use the predictions to calculate the error (mean squared error for regression and classification error for classification). This gives us a good way to measure the error.

#### Summary

Bagging is a great option for combining models with high variance, and decision trees are a great use-case. Even more, bagging doesn't lead to overfitting. However, there's a significant downside as well. One of the greatest benefits of decision trees is their interpretability. That interpretability is lost when bagging is used. There's no longer any way to display the model or easily show how a decision was made.

However, you can still find the most important predictors, either in terms of RSS for regression trees or Gini index for classification trees.

## Random Forests

Another powerful technique is known as *random forests*. Random forests offers an improvement over bagged trees because it allows us to decorrelate the trees. We start the same way as in bagging - build a number of decision trees based on bootstrapped training samples. But each time there's a split, we only allow a random subset, $$ m $$, of the $$ p $$ predictors to be chosen as split candidates. So if we're looking at the baseball dataset again, for a particular split you might only to able to split based on Hits and Runs, but not Home Runs or RBIs. For a dataset with $$ p $$ different features, $$ m $$ is typically chosen such as $$ m \approx \sqrt{p} $$.

So this means we work with less than half of the features. This means we often *don't* use the strongest predictor as our first split. $$ (p - m) / p $$ of the time the strongest predictor won't even be available. This makes the trees more varied. 

Note that bagging is actually just a subset of random forests. Specifically, it is a random forest with $$ m=p $$.

The more correlated the different features are the lower the value of m we should use.

Again, more trees won't cause more overfitting. The key to determining the number of trees is to see when the error rate settles down.

## Boosting

Now let's talk about a technique known as *boosting*. Boosting is another way we can improve decision tree performance. Like the other techniques, it can be applied to many machine learning models, but is most commonly used with decision trees. We're going to focus on boosting regression trees but the concept is the same for classification trees.

This process does not involve bootstrapping - we'll train the decision trees on the entire dataset. But then we'll take the output from that and input it into a second tree. Thus the output of one model will be fed into the next. We fit a decision tree to the residuals from the previous model. Then use update the residuals and do it again.

These trees can be quite small, maybe with just a few nodes in them. By adding these small trees, we gradually improve the overall model in the areas where it needs the most help.

There are three parameters of primary interest when boosting:

$$ B $$ - the number of trees. Unlike in bagging and random forests, boosted model *can* overfit. This means you will need to limit $$ B $$. Use cross-validation to find the optimal number for $$ B $$.

$$ \lambda $$ - the shrinkage parameter. This is like the learning rate in other algorithms. It determines the rate at which the model learns.

$$ d $$ - the number of splits in the tree. This controls the complexity of each tree. Often, numbers as low a 1 work well for boosting.

The final boosted model looks like:

$$ \hat{f}(x)=\sum_{b=1}^B{\lambda\hat{f}^b(x)} $$

I like to think of boosting as working in series while bagging works in parallel.
