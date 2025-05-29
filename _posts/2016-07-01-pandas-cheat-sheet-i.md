---
layout: post
title: "Pandas Cheat Sheet I"
description: "A helpful cheatsheet of common tasks using the Python library pandas"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/koala.jpg"
tags: [Pandas, Python, Cheat Sheet]
---

This post demonstrates some basic techniques for the Python library [pandas](https://pandas.pydata.org/).

<b>Table of Contents</b>
* TOC
{:toc}

# Version

I rerun this code every once in a while to ensure it's up-to-date. Here's the latest version it was tested on:


```python
import pandas as pd
from pathlib import Path
print(pd.__version__)
```

    1.4.4
    

# Importing Data

## DataFrames from CSV Files

In pandas, DataFrames are the primary structure for dealing with data. They provide indexed rows and columns of a dataset, much like a spreadsheet. There are many ways to get data into DataFrames. Perhaps the most common way of getting data into DataFrames is by importing CSV files.


```python
shakespeare_path = Path("data") / "shakespeare.csv"
```


```python
df = pd.read_csv(shakespeare_path)
```


```python
df
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
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
    </tr>
  </tbody>
</table>
</div>



Although importing from a CSV file is perhaps the most common way of getting data into a DataFrame, there are many alternatives.

There are a lot of options that you can do with `read_csv`. One in particular that I like is the ability to limit the number of rows you read, which allows for you to experiment with smaller DataFrames to make debugging easier.


```python
df = pd.read_csv(shakespeare_path, nrows=2)
```


```python
df
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
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



## DataFrames from Strings

Another way you can create a DataFrame is by using StringIO, which allows you to read a string as if it were a CSV file. It even allows for missing data.


```python
from io import StringIO

csv_data = """A,B,C,D
1.0, 2.0, 3.0, 4.0
5.0, 6.0,, 8.0
0.0, 11.0, 12.0,"""
df = pd.read_csv(StringIO(csv_data))
```


```python
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## DataFrames from .data Files

Machine learning datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) often have datasets with the .data extension. These can also be read using `pd.read_csv`.


```python
splice_path = "C:/Users/Julius/Google Drive/JupyterNotebooks/data/splice.data"
df = pd.read_csv(splice_path, names=["Class", "Instance", "Sequence"])
```


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
      <th>Class</th>
      <th>Instance</th>
      <th>Sequence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EI</td>
      <td>ATRINS-DONOR-521</td>
      <td>CCAGCTGCATCACAGGAGGCCAGCGAGCAGG...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EI</td>
      <td>ATRINS-DONOR-905</td>
      <td>AGACCCGCCGGGAGGCGGAGGACCTGCAGGG...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EI</td>
      <td>BABAPOE-DONOR-30</td>
      <td>GAGGTGAAGGACGTCCTTCCCCAGGAGCCGG...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EI</td>
      <td>BABAPOE-DONOR-867</td>
      <td>GGGCTGCGTTGCTGGTCACATTCCTGGCAGGT...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EI</td>
      <td>BABAPOE-DONOR-2817</td>
      <td>GCTCAGCCCCCAGGTCACCCAGGAACTGACGTG...</td>
    </tr>
  </tbody>
</table>
</div>



Note that you can also read files directly from the Internet without downloading them first.


```python
df = pd.read_csv("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data")
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
      <th>row.names</th>
      <th>sbp</th>
      <th>tobacco</th>
      <th>ldl</th>
      <th>adiposity</th>
      <th>famhist</th>
      <th>typea</th>
      <th>obesity</th>
      <th>alcohol</th>
      <th>age</th>
      <th>chd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>160</td>
      <td>12.00</td>
      <td>5.73</td>
      <td>23.11</td>
      <td>Present</td>
      <td>49</td>
      <td>25.30</td>
      <td>97.20</td>
      <td>52</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>144</td>
      <td>0.01</td>
      <td>4.41</td>
      <td>28.61</td>
      <td>Absent</td>
      <td>55</td>
      <td>28.87</td>
      <td>2.06</td>
      <td>63</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>118</td>
      <td>0.08</td>
      <td>3.48</td>
      <td>32.28</td>
      <td>Present</td>
      <td>52</td>
      <td>29.14</td>
      <td>3.81</td>
      <td>46</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>170</td>
      <td>7.50</td>
      <td>6.41</td>
      <td>38.03</td>
      <td>Present</td>
      <td>51</td>
      <td>31.99</td>
      <td>24.26</td>
      <td>58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>134</td>
      <td>13.60</td>
      <td>3.50</td>
      <td>27.78</td>
      <td>Present</td>
      <td>60</td>
      <td>25.99</td>
      <td>57.34</td>
      <td>49</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## DataFrames from Lists

You can also create a DataFrame from a list. For a single list, you can put it directly in a DataFrame.


```python
names = [
    "Titus Andronicus",
    "The Comedy of Errors",
    "Richard II",
    "Romeo and Juliet",
    "A Midsummer Night’s Dream",
    "King John",
    "Julius Caesar",
    "Othello",
    "Macbeth",
]
pd.DataFrame(names, columns=["Name"])
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
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
    </tr>
  </tbody>
</table>
</div>



But if you're combining multiple lists, you need to zip them first.


```python
years = [1592, 1594, 1595, 1595, 1595, 1596, 1599, 1604, 1606]
categories = ["Tragedy", "Comedy", "History", "Tragedy", "Comedy", "History", "Tragedy", "Tragedy", "Tragedy"]
```


```python
df = pd.DataFrame(list(zip(names, years, categories)), columns=["Name", "Year", "Category"])
df
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
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
    </tr>
  </tbody>
</table>
</div>



## DataFrames from Dicts

A more common way to do it is through dictionaries


```python
my_dict = {"Name": names, "Year": years, "Category": categories}
```


```python
my_dict
```




    {'Name': ['Titus Andronicus',
      'The Comedy of Errors',
      'Richard II',
      'Romeo and Juliet',
      'A Midsummer Night’s Dream',
      'King John',
      'Julius Caesar',
      'Othello',
      'Macbeth'],
     'Year': [1592, 1594, 1595, 1595, 1595, 1596, 1599, 1604, 1606],
     'Category': ['Tragedy',
      'Comedy',
      'History',
      'Tragedy',
      'Comedy',
      'History',
      'Tragedy',
      'Tragedy',
      'Tragedy']}




```python
df = pd.DataFrame(my_dict)
df
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
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
    </tr>
  </tbody>
</table>
</div>




If you have a simple dictionary like this: `d = {'a': 1, 'b': 2}`, you can't just put it in a `pd.DataFrame`, because it'll give you a `ValueError` for not passing an index. Instead, you can do this: `pd.DataFrame(list(d.items()))`. You could also pass column names so it looks more like this: `pd.DataFrame(list(d.items()), columns=['Key', 'Value'])`.

You might not want the dictionary keys, in which can you can do this: `pd.DataFrame(list(building_dict.values()))`



## Saving a DataFrame

There are many ways to save a DataFrame, including in a pickle, msgpack, CSV, and HDF5Store. They all follow similar syntax.


```python
df.to_csv("shakespeare.csv")
```

There's something I find a little weird about saving a loading DataFrames. It's that if you save it as above and then load it, you'll get an extra column.


```python
df = pd.read_csv("shakespeare.csv")
```


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
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



There are two ways to avoid this. The first I recommend if your row names are just 0-4 (or any number) as they are above. In that case, when you save it, you want to save it like so:


```python
df.to_csv("shakespeare.csv", index=False)
```


```python
df = pd.read_csv("shakespeare.csv")
```


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
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



The second way is better in the case when you do have index names that you want to save, but it will still work in this case. What you want to do is save your DataFrame like you did at first:


```python
df.to_csv("shakespeare.csv")
```

Then when you open it, pass `index_col=0` like so:


```python
df = pd.read_csv("shakespeare.csv", index_col=0)
```


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
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



# Exploring Data

## Displaying Parts of the DataFrame

OK, now that we have the data into a DataFrame, let's explore it. To get a quick preview of the data, you can use `head()`.


```python
df = pd.read_csv(shakespeare_path)
df.head(3)
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
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
    </tr>
  </tbody>
</table>
</div>



To see a preview of the end, use `tail()`. If you want to see some columns at random, use `sample()`


```python
df.sample(5)
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
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
    </tr>
  </tbody>
</table>
</div>



Or you can slice the DataFrame


```python
df[5:]
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
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[0:5:2]
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
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



You can also select by column. A single column of a Pandas DataFrame is known as a Pandas Series.


```python
type(df["Name"])
```




    pandas.core.series.Series



You can also make selections by inequalities, such as finding all the points where a column has a value greater than a specific amount.


```python
df[df["Year"] > 1600]
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
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
    </tr>
  </tbody>
</table>
</div>



You can also select a specific cell based on its label.


```python
df.loc[1, "Name"]
```




    'The Comedy of Errors'



## Built-in Descriptors

There are a few built-in descriptors that are good to know.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9 entries, 0 to 8
    Data columns (total 3 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   Name      9 non-null      object
     1   Year      9 non-null      int64 
     2   Category  9 non-null      object
    dtypes: int64(1), object(2)
    memory usage: 344.0+ bytes
    


```python
df.shape
```




    (9, 3)



For quantitative data, there are a few more worth knowing.

### Quantitative Data

You can also find the data type of each column


```python
df.dtypes
```




    Name        object
    Year         int64
    Category    object
    dtype: object



You can do even more with quantitative data. You can find the mean of each quantitative column in the dataset (e.g. ints and floats)


```python
df["Year"].mean()
```




    1597.3333333333333



Or you can use describe to find even more.


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
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1597.333333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.743416</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1592.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1595.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1595.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1606.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Sorting

Sorting is very simple with DataFrames.


```python
df.sort_values("Name", ascending=True)
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
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
    </tr>
  </tbody>
</table>
</div>



## Searching for Text Within a DataFrame


```python
df["Name"].str.contains("Julius")
```




    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    7    False
    8    False
    Name: Name, dtype: bool



If you want to find both uppper and lower case examples, you can pass `(?i)` to the regex parser to tell it to ignore cases.


```python
df["Name"].str.contains("(?i)julius")
```




    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    7    False
    8    False
    Name: Name, dtype: bool



## Splitting by Strings

If you want to split on the first element, you have to reference `str` again.


```python
df["Name"].str.split(" ").str[0]
```




    0      Titus
    1        The
    2    Richard
    3      Romeo
    4          A
    5       King
    6     Julius
    7    Othello
    8    Macbeth
    Name: Name, dtype: object



## Grouping and Counting by Label


```python
df["Category"].groupby(df["Category"]).count()
```




    Category
    Comedy     2
    History    2
    Tragedy    5
    Name: Category, dtype: int64



# Cleaning Data

Bad values can come in many forms, including missing values, NaN, NA, ?, etc. Let's go over how to find them


```python
df = pd.read_csv(StringIO(csv_data))
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Now we can clean that dataset. One of the first things to look for is empty values


```python
df.isnull()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



This works for tiny datasets, but if it is larger we won't be able to see them all, so we can just sum up all the True values


```python
df.isnull().sum()
```




    A    0
    B    0
    C    1
    D    1
    dtype: int64



## Adding

To add a new column, simply declare it and make sure it has the correct number of rows.


```python
df["E"] = df["A"] * df["B"] + 1
```


```python
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



You can also add columns using conditional logic


```python
import numpy as np

df["F"] = np.where(df["B"] > df["E"], 1, 0)
```


```python
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Dropping

There are many ways to drop missing data. The basic `dropna` drops every row that has a missing value.


```python
df.dropna()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Note that this doesn't drop the data from the original DataFrame, it just returns a new one without the dropped values.

Alternatively, you could drop all the columns that have a missing value


```python
df.dropna(axis=1)
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
      <th>A</th>
      <th>B</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



But you may want more precision than that. One way to do this is by dropping specific rows or columns by name. To drop columns, use `axis=1`; to drop rows, use `axis=0`, or leave it out, because 0 is the default value


```python
df.drop([0,2])
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(["D", "C"], axis=1)
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
      <th>A</th>
      <th>B</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Selecting specific columns is commonly used when preparing a dataset for machine learning. For example, you might select the features and labels like so


```python
features = df.drop("A", axis=1)
labels = df["A"]
```

You can also selective decide which rows to drop. You can decide to only drop rows where all columns are NaN.


```python
df.dropna(how="all")
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Or drop all rows that have fewer than 4 non-NaN values


```python
df.dropna(thresh=4)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Or just drop rows where NaN appears in a specific column


```python
df.dropna(subset=["D"])
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Finding Missing Values

You will need to clean up missing values before doing any machine learning on the data. Scikit-learn will give you an error if you try to run an algorithm on a dataset with missing values.

pandas has two commands to find missing values in a DataFrame, `isna` and `isnull`. These are 100% identical. [The documentation literally says `isnull = isna`.](https://github.com/pandas-dev/pandas/blob/0409521665bd436a10aea7e06336066bf07ff057/pandas/core/dtypes/missing.py#L109)


```python
df.isnull()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Note that pandas will only call a certain type of data as missing. It's possible that the user used a certain value to denote missing data but pandas doesn't see it as missing. Below, there are many types of values that could have been intended to denote missing data, but pandas only sees the `None` value as missing.


```python
a = [False, 0, "na", "NaN"]
b = ["None", None, "nan", "NA"]
```


```python
df_missing = pd.DataFrame(list(zip(a, b)), columns=(["a", "b"]))
```


```python
df_missing
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
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>na</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NA</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_missing.isnull()
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
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



But when you import from a CSV, only the missing values will show up as missing.


```python
bad_data = """A,B,C,D
None, none, 3.0, 4.0
Missing, 6.0,, 'None'
0.0, NaN, False,"""
```


```python
df2 = pd.read_csv(StringIO(bad_data))
df2
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>none</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Missing</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>'None'</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.isnull()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## Interpolating

Sometimes deleting data isn't the right approach. In these cases, you can interpolate the data. SKLearn has a good library for that.


```python
from sklearn.impute import SimpleImputer
```


```python
imp = SimpleImputer(strategy="mean")
```

 It uses the same fit and transform approach as other SKLearn tools.


```python
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
imp.fit(df)
imp.transform(df)
```




    array([[ 1. ,  2. ,  3. ,  4. ,  3. ,  0. ],
           [ 5. ,  6. ,  7.5,  8. , 31. ,  0. ],
           [ 0. , 11. , 12. ,  6. ,  1. ,  1. ]])



# Data Types

Data are usually split into three types: **continuous**, **ordinal**, and **categorical**. With continuous data the values associated with the data have specific meaning. For example, the price of a good is a continuous value. Common statistical analyses are usually appropriate for continuous values. For example, you can take the average of two continuous values and the result is a meaningful description of the data.

For ordinal data, the values are ordered, but they don't necessarily have meaning. T-shirt size is a good example of this, where a small is less than a medium, but it's not less by a specific value. If you converted "small" into a specific measurement (e.g. convert it to inches) it would become a continuous value.

The last type is categorical, and that's where the differences aren't associated with any ranking or order. For example, shirtstyle is a categorical variable. We might have "polo" and "T-shirt", but neither is greater than the other. We could (and will) assign numbers to these categories for machine learning, but we need to remember that those numbers aren't associated with values as they would be in continuous or ordinal data.


```python
df = pd.DataFrame([["green", "M", 20, "polo"], ["red", "L", 15, "T-shirt"], ["red", "S", 15, "polo"]])
df.columns = ["color", "size", "price", "type"]
df
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
      <th>color</th>
      <th>size</th>
      <th>price</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>green</td>
      <td>M</td>
      <td>20</td>
      <td>polo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>L</td>
      <td>15</td>
      <td>T-shirt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red</td>
      <td>S</td>
      <td>15</td>
      <td>polo</td>
    </tr>
  </tbody>
</table>
</div>



Because size can be ranked, it is ordinal. We can map it to numbers to make it easier for machine learning.


```python
size_mapping = {"S": 1, "M": 2, "L": 3}
df["size"] = df["size"].map(size_mapping)  # we can do a reverse mapping if we want to undo this in the end
df
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
      <th>color</th>
      <th>size</th>
      <th>price</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>green</td>
      <td>2</td>
      <td>20</td>
      <td>polo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>3</td>
      <td>15</td>
      <td>T-shirt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red</td>
      <td>1</td>
      <td>15</td>
      <td>polo</td>
    </tr>
  </tbody>
</table>
</div>



Even though the type is categorical, we can still map it to numbers


```python
import numpy as np

class_mapping = {label: idx for idx, label in enumerate(np.unique(df["type"]))}
df["type"] = df["type"].map(class_mapping)
df
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
      <th>color</th>
      <th>size</th>
      <th>price</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>green</td>
      <td>2</td>
      <td>20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>3</td>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red</td>
      <td>1</td>
      <td>15</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



After we perform whatever analysis we want to on the data, we could then invert that mapping.


```python
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df["type"] = df["type"].map(inv_class_mapping)
df
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
      <th>color</th>
      <th>size</th>
      <th>price</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>green</td>
      <td>2</td>
      <td>20</td>
      <td>polo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>3</td>
      <td>15</td>
      <td>T-shirt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red</td>
      <td>1</td>
      <td>15</td>
      <td>polo</td>
    </tr>
  </tbody>
</table>
</div>



Another way to do this is the LabelEncoder class in scikit-learn


```python
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df["type"].values)  # fit_transform is a shortcut for calling fit and transform separately
print(y)
class_le.inverse_transform(y)
```

    [1 0 1]
    




    array(['polo', 'T-shirt', 'polo'], dtype=object)



# Preparing Data for Machine Learning

It's a good idea to do one-hot encoding of categorical variables before using them for machine learning. It can also be a good idea for ordinal variables as well, although that's not always the case. A good rule of thumb is if the mean of two values isn't a meaningful value, that category should be one-hot encoded.

The downside of treating ordinal data as categorical is that we throw away information about the relative order. The downside of treating it as continuous data is that we introduce a notion of distance. For example, if we set "small" as "1", "medium" as "2", and "large" as "3", we're telling the model that a large is 3 times a small. This isn't a meaningful thing to say, so it can reduce model performance.

## Shuffling Data

pandas makes it very easy to shuffle data.


```python
df.sample(frac=1).reset_index(drop=True)
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
      <th>color</th>
      <th>size</th>
      <th>price</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red</td>
      <td>3</td>
      <td>15</td>
      <td>T-shirt</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>1</td>
      <td>15</td>
      <td>polo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>green</td>
      <td>2</td>
      <td>20</td>
      <td>polo</td>
    </tr>
  </tbody>
</table>
</div>



## Separating Features from Labels

`pandas` also has a convenient way to extract the labels from a DataFrame, and that's by using the `pop` method. It will remove the specified column from the DataFrame and put it into a Series of its own.


```python
labels = df.pop("price")
```

You can see that it's no longer in the DataFrame:


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
      <th>color</th>
      <th>size</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>green</td>
      <td>2</td>
      <td>polo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>3</td>
      <td>T-shirt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red</td>
      <td>1</td>
      <td>polo</td>
    </tr>
  </tbody>
</table>
</div>




```python
labels
```




    0    20
    1    15
    2    15
    Name: price, dtype: int64



## One-hot Encoding

pandas can also do one-hot encoding.


```python
pd.get_dummies(df[["color", "size"]])
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
      <th>size</th>
      <th>color_green</th>
      <th>color_red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


# Other Tricks

## Don't Truncate

pandas will by default truncate text that is over a certain limit. When working with text data, sometimes you don't want this. Here's how to stop it.


```python
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
```

`pd.set_option` is a nice tool. But if for some reason you don't want to use that, you can also set it directly.


```python
pd.options.display.max_columns = None
pd.options.display.max_rows = None
```

## Transposing

You can switch the rows and columns of a DataFrame by transposing it by adding `.T` to the end.


```python
df = pd.DataFrame(list(zip(names, years, categories)), columns=["Name", "Year", "Category"])
```


```python
df
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
      <th>Name</th>
      <th>Year</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.T
```


```python
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Name</th>
      <td>Titus Andronicus</td>
      <td>The Comedy of Errors</td>
      <td>Richard II</td>
      <td>Romeo and Juliet</td>
      <td>A Midsummer Night’s Dream</td>
      <td>King John</td>
      <td>Julius Caesar</td>
      <td>Othello</td>
      <td>Macbeth</td>
    </tr>
    <tr>
      <th>Year</th>
      <td>1592</td>
      <td>1594</td>
      <td>1595</td>
      <td>1595</td>
      <td>1595</td>
      <td>1596</td>
      <td>1599</td>
      <td>1604</td>
      <td>1606</td>
    </tr>
    <tr>
      <th>Category</th>
      <td>Tragedy</td>
      <td>Comedy</td>
      <td>History</td>
      <td>Tragedy</td>
      <td>Comedy</td>
      <td>History</td>
      <td>Tragedy</td>
      <td>Tragedy</td>
      <td>Tragedy</td>
    </tr>
  </tbody>
</table>
</div>


