---
layout: post
title: "Pandas Cheat Sheet II"
description: "A helpful cheatsheet of common tasks using the Python library pandas"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/koala2.jpg"
tags: [Pandas, Python, Cheat Sheet]
---

This notebook demonstrates some basic techniques for the Python library [pandas](https://pandas.pydata.org/). This is part II of the [Pandas Cheat Sheet](https://jss367.github.io/pandas-cheat-sheet-i.html).

<b>Table of Contents</b>
* TOC
{:toc}

# Version

I rerun this code every once in a while to ensure it's up-to-date. Here's the latest version it was tested on:


```python
import pandas as pd
print(pd.__version__)
```

    1.4.4
    

# Setup

We'll use the same data as last time.


```python
shakespeare_path = "C:/Users/Julius/Google Drive/JupyterNotebooks/data/shakespeare.csv"
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



## Applying a Function

There are many ways to apply functions to pandas DataFrames. One of the most flexible is to use `.apply`.

It works well with anonymous `lambda` functions.


```python
df["Next Year"] = df.apply(lambda x: x["Year"] + 1, axis=1)
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
      <th>Next Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
      <td>1593</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
      <td>1595</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
      <td>1596</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
      <td>1596</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
      <td>1596</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
      <td>1597</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
      <td>1600</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
      <td>1607</td>
    </tr>
  </tbody>
</table>
</div>




```python
def make_year_even(row):
    if row["Year"] % 2 == 0:
        return row["Year"]
    return row["Year"] + 1
```

Test it by calling it on a single row.


```python
make_year_even(df.iloc[0])
```




    1592




```python
df["Even Year"] = df.apply(make_year_even, axis=1)
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
      <th>Next Year</th>
      <th>Even Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
      <td>1593</td>
      <td>1592</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
      <td>1595</td>
      <td>1594</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
      <td>1596</td>
      <td>1596</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
      <td>1596</td>
      <td>1596</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
      <td>1596</td>
      <td>1596</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
      <td>1597</td>
      <td>1596</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
      <td>1600</td>
      <td>1600</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
      <td>1605</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
      <td>1607</td>
      <td>1606</td>
    </tr>
  </tbody>
</table>
</div>



You can also use functions that require arguments.


```python
def add_number(row, number=1):
    return row["Year"] + number
```


```python
add_number(df.iloc[0], 10)
```




    1602



Then pass the arguments as a list.


```python
df["Next Decade"] = df.apply(add_number, axis=1, args=[10])
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
      <th>Next Year</th>
      <th>Even Year</th>
      <th>Next Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
      <td>1593</td>
      <td>1592</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
      <td>1595</td>
      <td>1594</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
      <td>1597</td>
      <td>1596</td>
      <td>1606</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
      <td>1600</td>
      <td>1600</td>
      <td>1609</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
      <td>1605</td>
      <td>1604</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
      <td>1607</td>
      <td>1606</td>
      <td>1616</td>
    </tr>
  </tbody>
</table>
</div>



Note that if you try to pass the arguments in a tuple you'll get an error.


```python
try:
    df["Next Decade"] = df.apply(add_number, axis=1, args=(10))
except TypeError as err:
    print(err)
```

    add_number() argument after * must be an iterable, not int
    

# Editing DataFrames

## using .at and .iat

`.at` and `.iat` aren't as commonly used, but they are the recommended way to edit pandas DataFrames.


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
      <th>Next Year</th>
      <th>Even Year</th>
      <th>Next Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
      <td>1593</td>
      <td>1592</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
      <td>1595</td>
      <td>1594</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
      <td>History</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
      <td>1597</td>
      <td>1596</td>
      <td>1606</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
      <td>1600</td>
      <td>1600</td>
      <td>1609</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
      <td>1605</td>
      <td>1604</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
      <td>1607</td>
      <td>1606</td>
      <td>1616</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.at[2, "Name"]
```




    'Richard II'




```python
df.at[2, "Name"] = "Richard the Second"
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
      <th>Next Year</th>
      <th>Even Year</th>
      <th>Next Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
      <td>1593</td>
      <td>1592</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Comedy</td>
      <td>1595</td>
      <td>1594</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard the Second</td>
      <td>1595</td>
      <td>History</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
      <td>1597</td>
      <td>1596</td>
      <td>1606</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
      <td>1600</td>
      <td>1600</td>
      <td>1609</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
      <td>1605</td>
      <td>1604</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
      <td>1607</td>
      <td>1606</td>
      <td>1616</td>
    </tr>
  </tbody>
</table>
</div>



You can also query by row and column number. Note the index doesn't count, so column 2 is "Category".


```python
df.iat[1, 2]
```




    'Comedy'




```python
df.iat[1, 2] = "Tragicomedy"
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
      <th>Next Year</th>
      <th>Even Year</th>
      <th>Next Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
      <td>1593</td>
      <td>1592</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Tragicomedy</td>
      <td>1595</td>
      <td>1594</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard the Second</td>
      <td>1595</td>
      <td>History</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
      <td>1597</td>
      <td>1596</td>
      <td>1606</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
      <td>1600</td>
      <td>1600</td>
      <td>1609</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
      <td>1605</td>
      <td>1604</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
      <td>1607</td>
      <td>1606</td>
      <td>1616</td>
    </tr>
  </tbody>
</table>
</div>



## .loc vs .iloc

.loc searches for labels by name (aka label-based indexing), .iloc gets rows by index number (aka positional indexing)

For a DataFrame where the index matches the row number (i.e. it starts at 0 and doesn't skip any values), there is no difference between .loc and .iloc


```python
df.loc[1]
```




    Name           The Comedy of Errors
    Year                           1594
    Category                Tragicomedy
    Next Year                      1595
    Even Year                      1594
    Next Decade                    1604
    Name: 1, dtype: object




```python
df.iloc[1]
```




    Name           The Comedy of Errors
    Year                           1594
    Category                Tragicomedy
    Next Year                      1595
    Even Year                      1594
    Next Decade                    1604
    Name: 1, dtype: object



But once the index and row numbers become different, you have to be careful about which one to use.


```python
new_df = df.drop(1)
new_df
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
      <th>Next Year</th>
      <th>Even Year</th>
      <th>Next Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
      <td>1593</td>
      <td>1592</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard the Second</td>
      <td>1595</td>
      <td>History</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
      <td>1597</td>
      <td>1596</td>
      <td>1606</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
      <td>1600</td>
      <td>1600</td>
      <td>1609</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
      <td>1605</td>
      <td>1604</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
      <td>1607</td>
      <td>1606</td>
      <td>1616</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_df.iloc[1]
```




    Name           Richard the Second
    Year                         1595
    Category                  History
    Next Year                    1596
    Even Year                    1596
    Next Decade                  1605
    Name: 2, dtype: object



There is still an index of value 1, but there is no longer a label. So if we try `.loc[1]`, we'll get an error.


```python
try:
    new_df.loc[1]
except KeyError:
    print("There is nothing with index 1")
```

    There is nothing with index 1
    

If you want to update your indices you can do so with `reset_index`.


```python
new_df.reset_index(drop=True)
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
      <th>Next Year</th>
      <th>Even Year</th>
      <th>Next Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
      <td>1593</td>
      <td>1592</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Richard the Second</td>
      <td>1595</td>
      <td>History</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
      <td>1597</td>
      <td>1596</td>
      <td>1606</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
      <td>1600</td>
      <td>1600</td>
      <td>1609</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
      <td>1605</td>
      <td>1604</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
      <td>1607</td>
      <td>1606</td>
      <td>1616</td>
    </tr>
  </tbody>
</table>
</div>



When you use `.iloc`, you can treat the results just like a `numpy` matrix. This means that trailing colons are always optional (just like `numpy`).

# Changing Specific Rows

Note that there are a lot of wrong ways to do this in pandas.


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
      <th>Next Year</th>
      <th>Even Year</th>
      <th>Next Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
      <td>1593</td>
      <td>1592</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Tragicomedy</td>
      <td>1595</td>
      <td>1594</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard the Second</td>
      <td>1595</td>
      <td>History</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
      <td>1597</td>
      <td>1596</td>
      <td>1606</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
      <td>1600</td>
      <td>1600</td>
      <td>1609</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
      <td>1605</td>
      <td>1604</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
      <td>1607</td>
      <td>1606</td>
      <td>1616</td>
    </tr>
  </tbody>
</table>
</div>



Here you might think you're setting the value, but you're not.


```python
df.loc[df.index[-4:]]["Category"] = "Drama"
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
      <th>Next Year</th>
      <th>Even Year</th>
      <th>Next Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
      <td>1593</td>
      <td>1592</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Tragicomedy</td>
      <td>1595</td>
      <td>1594</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard the Second</td>
      <td>1595</td>
      <td>History</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>History</td>
      <td>1597</td>
      <td>1596</td>
      <td>1606</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Tragedy</td>
      <td>1600</td>
      <td>1600</td>
      <td>1609</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Tragedy</td>
      <td>1605</td>
      <td>1604</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Tragedy</td>
      <td>1607</td>
      <td>1606</td>
      <td>1616</td>
    </tr>
  </tbody>
</table>
</div>



This is how you have to do it.


```python
df.loc[df.index[-4:], "Category"] = "Drama"
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
      <th>Next Year</th>
      <th>Even Year</th>
      <th>Next Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
      <td>Tragedy</td>
      <td>1593</td>
      <td>1592</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
      <td>Tragicomedy</td>
      <td>1595</td>
      <td>1594</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard the Second</td>
      <td>1595</td>
      <td>History</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
      <td>Tragedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
      <td>Comedy</td>
      <td>1596</td>
      <td>1596</td>
      <td>1605</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
      <td>Drama</td>
      <td>1597</td>
      <td>1596</td>
      <td>1606</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
      <td>Drama</td>
      <td>1600</td>
      <td>1600</td>
      <td>1609</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
      <td>Drama</td>
      <td>1605</td>
      <td>1604</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
      <td>Drama</td>
      <td>1607</td>
      <td>1606</td>
      <td>1616</td>
    </tr>
  </tbody>
</table>
</div>


