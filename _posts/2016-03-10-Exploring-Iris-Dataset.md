---
layout: post
title: "Exploring the Iris Dataset"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/iris.jpg"
tags: [Python, Matplotlib, Seaborn, Data Exploration, Data Visualization]
---

In this notebook, we'll demonstrate some data exploration techniques using the famous iris dataset. In [the second notebook](https://jss367.github.io/Visualize-shallow-learning.html), we'll use this data set to visualize a bunch of machine learning algorithms.

<b>Table of contents</b>
* TOC
{:toc}

![Iris]({{site.baseurl}}/assets/img/iris_square.jpg "Iris")


```python
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="dark")
import matplotlib.pyplot as plt
```

# Load the Data

Load the data using seaborn. The dataset is also available from Scikit-learn and Keras, but it loads as a pandas DataFrame from seaborn, saving a step.


```python
df = sns.load_dataset("iris")
```

# Explore

Let's look at what features are in the data set.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



Then check how many of each species is recorded.


```python
df['species'].value_counts()
```




    setosa        50
    versicolor    50
    virginica     50
    Name: species, dtype: int64



And let's see what types of values are in the dataset and do some basic statistics on the set.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   sepal_length  150 non-null    float64
     1   sepal_width   150 non-null    float64
     2   petal_length  150 non-null    float64
     3   petal_width   150 non-null    float64
     4   species       150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.057333</td>
      <td>3.758000</td>
      <td>1.199333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.435866</td>
      <td>1.765298</td>
      <td>0.762238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>



Fortunately, the data set is really clean so we can jump right into visualization.

# Visualize

Let's see how the different categories compare with each other.


```python
hue_order = df['species'].unique()[::-1]
palette = sns.color_palette('bright')
sns.pairplot(df, hue="species", hue_order=hue_order, palette=palette, markers=["o", "s", "D"], diag_kind='kde');
```


![png]({{site.baseurl}}/assets/img/2016-03-10-Exploring-Iris-Dataset_files/2016-03-10-Exploring-Iris-Dataset_19_0.png)


Nothing looks noticeably wrong with the data, and there aren't any outliers that would confound a model.

Petal length and petal width appear to be good variables to distinguish the species, especially sestosa. Let's take a closer look at those.


```python
sns.FacetGrid(df, hue='species', hue_order=hue_order, palette=palette, height=8) \
    .map(plt.scatter, 'petal_length','petal_width') \
    .add_legend();
```


![png]({{site.baseurl}}/assets/img/2016-03-10-Exploring-Iris-Dataset_files/2016-03-10-Exploring-Iris-Dataset_21_0.png)


OK, it will be very easy to extract the setosa from the others.

Let's see what the best way to separate versicolor from virginica is. We'll a new dataframe with just the two we're focusing on.


```python
# Exclude setosa
vvdf = df[df['species'] != 'setosa']
```


```python
sns.pairplot(vvdf, hue="species", hue_order=hue_order, palette=palette, diag_kind='kde');
```


![png]({{site.baseurl}}/assets/img/2016-03-10-Exploring-Iris-Dataset_files/2016-03-10-Exploring-Iris-Dataset_24_0.png)


 OK, these are not as easy to separate. We make have to do the best that we can. In [Part II](https://jss367.github.io/Visualize-shallow-learning.html), we'll look at how we can use machine learning models to analyze the data.
