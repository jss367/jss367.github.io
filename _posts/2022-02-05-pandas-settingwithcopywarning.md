---
layout: post
title: "Pandas SettingWithCopyWarning"
description: "How to deal with a common warning in Pandas"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/bear_with_fish.jpg"
tags: [Pandas, Python]
---

There's a common warning in `pandas` about a `SettingWithCopyWarning`. While the error message covers some of the possible reasons for the error, it doesn't cover them all. In this post, I'll show another source of the error and how to fix it.


```python
import os
import pandas as pd
```


```python
os.getenv('DATA')
```




    'I:\\Data'




```python
shakespeare_path = os.path.join(os.getenv('DATA'), 'shakespeare.csv')
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



Let's say you want to make a subset of the data by copying a couple columns.


```python
sdf = df[['Name', 'Year']]
```


```python
sdf
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Titus Andronicus</td>
      <td>1592</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Comedy of Errors</td>
      <td>1594</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richard II</td>
      <td>1595</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Romeo and Juliet</td>
      <td>1595</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Midsummer Night’s Dream</td>
      <td>1595</td>
    </tr>
    <tr>
      <th>5</th>
      <td>King John</td>
      <td>1596</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Julius Caesar</td>
      <td>1599</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Othello</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Macbeth</td>
      <td>1606</td>
    </tr>
  </tbody>
</table>
</div>



Then you want to continue cleaning it up.


```python
sdf.loc[:, 'Year'] = pd.to_datetime(sdf['Year'])
```

    C:\Users\Julius\anaconda3\lib\site-packages\pandas\core\indexing.py:1773: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_single_column(ilocs[0], value, pi)
    

The error message doesn't make any sense. You're already doing what they suggest trying. What you need to do to avoid this is to make a complete copy of the DataFrame using `.copy()`.


```python
sdf2 = df[['Name', 'Year']].copy()
```


```python
sdf2.loc[:, 'Year'] = pd.to_datetime(sdf2['Year'])
```

No more error message!
