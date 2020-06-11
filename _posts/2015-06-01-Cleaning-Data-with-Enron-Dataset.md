---
layout: post
title: "Data Cleaning Tutorial with the Enron Dataset"
description: "Tools, techniques, and tips to clean data in Python using the Enron dataset"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/sun_tas.jpg"
tags: [Python, Data Cleaning, Pandas]
---

In this notebook, I'm going to look at the basics of cleaning data with Python. I will be using a dataset of people involved in the Enron scandal. I first saw this dataset in the Intro to Machine Learning class at [Udacity](https://www.udacity.com/).

<b>Table of contents</b>
* TOC
{:toc}


```python
# Basic imports that we'll use
import pandas as pd
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os.path import join
```

# Loading the data

The first step is to load the data. It's saved as a pickle file, which is a filetype created by the pickle module - a Python module to efficiently serialize and de-serialize data. Serializing is a process of converting a Python data object, like a list or dictionary, into a stream of characters.


```python
path = 'datasets/Enron/'
file = 'final_project_dataset.pkl'
with open(join(path, file), 'rb') as f:
    enron_data = pickle.load(f)
```

# Exploring the data

Now let's look at the data. First, we'll see what type it is.


```python
print("The dataset is a", type(enron_data))
```

    The dataset is a <class 'dict'>
    

OK, we have a dictionary. That means we'll have to find the values by calling the associated keys. Let's see what different keys we have in the dataset.


```python
print("There are {} entities in the dataset.\n".format(len(enron_data)))
print(enron_data.keys())
```

    There are 146 entities in the dataset.
    
    dict_keys(['METTS MARK', 'BAXTER JOHN C', 'ELLIOTT STEVEN', 'CORDES WILLIAM R', 'HANNON KEVIN P', 'MORDAUNT KRISTINA M', 'MEYER ROCKFORD G', 'MCMAHON JEFFREY', 'HAEDICKE MARK E', 'PIPER GREGORY F', 'HUMPHREY GENE E', 'NOLES JAMES L', 'BLACHMAN JEREMY M', 'SUNDE MARTIN', 'GIBBS DANA R', 'LOWRY CHARLES P', 'COLWELL WESLEY', 'MULLER MARK S', 'JACKSON CHARLENE R', 'WESTFAHL RICHARD K', 'WALTERS GARETH W', 'WALLS JR ROBERT H', 'KITCHEN LOUISE', 'CHAN RONNIE', 'BELFER ROBERT', 'SHANKMAN JEFFREY A', 'WODRASKA JOHN', 'BERGSIEKER RICHARD P', 'URQUHART JOHN A', 'BIBI PHILIPPE A', 'RIEKER PAULA H', 'WHALEY DAVID A', 'BECK SALLY W', 'HAUG DAVID L', 'ECHOLS JOHN B', 'MENDELSOHN JOHN', 'HICKERSON GARY J', 'CLINE KENNETH W', 'LEWIS RICHARD', 'HAYES ROBERT E', 'KOPPER MICHAEL J', 'LEFF DANIEL P', 'LAVORATO JOHN J', 'BERBERIAN DAVID', 'DETMERING TIMOTHY J', 'WAKEHAM JOHN', 'POWERS WILLIAM', 'GOLD JOSEPH', 'BANNANTINE JAMES M', 'DUNCAN JOHN H', 'SHAPIRO RICHARD S', 'SHERRIFF JOHN R', 'SHELBY REX', 'LEMAISTRE CHARLES', 'DEFFNER JOSEPH M', 'KISHKILL JOSEPH G', 'WHALLEY LAWRENCE G', 'MCCONNELL MICHAEL S', 'PIRO JIM', 'DELAINEY DAVID W', 'SULLIVAN-SHAKLOVITZ COLLEEN', 'WROBEL BRUCE', 'LINDHOLM TOD A', 'MEYER JEROME J', 'LAY KENNETH L', 'BUTTS ROBERT H', 'OLSON CINDY K', 'MCDONALD REBECCA', 'CUMBERLAND MICHAEL S', 'GAHN ROBERT S', 'BADUM JAMES P', 'HERMANN ROBERT J', 'FALLON JAMES B', 'GATHMANN WILLIAM D', 'HORTON STANLEY C', 'BOWEN JR RAYMOND M', 'GILLIS JOHN', 'FITZGERALD JAY L', 'MORAN MICHAEL P', 'REDMOND BRIAN L', 'BAZELIDES PHILIP J', 'BELDEN TIMOTHY N', 'DIMICHELE RICHARD G', 'DURAN WILLIAM D', 'THORN TERENCE H', 'FASTOW ANDREW S', 'FOY JOE', 'CALGER CHRISTOPHER F', 'RICE KENNETH D', 'KAMINSKI WINCENTY J', 'LOCKHART EUGENE E', 'COX DAVID', 'OVERDYKE JR JERE C', 'PEREIRA PAULO V. FERRAZ', 'STABLER FRANK', 'SKILLING JEFFREY K', 'BLAKE JR. NORMAN P', 'SHERRICK JEFFREY B', 'PRENTICE JAMES', 'GRAY RODNEY', 'THE TRAVEL AGENCY IN THE PARK', 'UMANOFF ADAM S', 'KEAN STEVEN J', 'TOTAL', 'FOWLER PEGGY', 'WASAFF GEORGE', 'WHITE JR THOMAS E', 'CHRISTODOULOU DIOMEDES', 'ALLEN PHILLIP K', 'SHARP VICTORIA T', 'JAEDICKE ROBERT', 'WINOKUR JR. HERBERT S', 'BROWN MICHAEL', 'MCCLELLAN GEORGE', 'HUGHES JAMES A', 'REYNOLDS LAWRENCE', 'PICKERING MARK R', 'BHATNAGAR SANJAY', 'CARTER REBECCA C', 'BUCHANAN HAROLD G', 'YEAP SOON', 'MURRAY JULIA H', 'GARLAND C KEVIN', 'DODSON KEITH', 'YEAGER F SCOTT', 'HIRKO JOSEPH', 'DIETRICH JANET R', 'DERRICK JR. JAMES V', 'FREVERT MARK A', 'PAI LOU L', 'HAYSLETT RODERICK J', 'BAY FRANKLIN R', 'MCCARTY DANNY J', 'FUGH JOHN L', 'SCRIMSHAW MATTHEW', 'KOENIG MARK E', 'SAVAGE FRANK', 'IZZO LAWRENCE L', 'TILNEY ELIZABETH A', 'MARTIN AMANDA K', 'BUY RICHARD B', 'GRAMM WENDY L', 'CAUSEY RICHARD A', 'TAYLOR MITCHELL S', 'DONAHUE JR JEFFREY M', 'GLISAN JR BEN F'])
    

The keys are different people who worked at Enron. I see several names familiar from the Enron scandal, including Kenneth Lay, Jeffery Skilling, Andrew Fastow, and Cliff Baxter (who's listed as John C Baxter). There's also an entity named "The Travel Agency in the Park". From footnote `j` in the [original document](http://news.findlaw.com/hdocs/docs/enron/enron61702insiderpay.pdf), this business was co-owned by the sister of Enron's former Chairman. It may be of interest to investigators, but it will mess up the machine learning algorithms as it's not an employee, so I will remove it.

Now let's see what information we have about each person. We'll start with Ken Lay.


```python
enron_data['LAY KENNETH L']
```




    {'salary': 1072321,
     'to_messages': 4273,
     'deferral_payments': 202911,
     'total_payments': 103559793,
     'loan_advances': 81525000,
     'bonus': 7000000,
     'email_address': 'kenneth.lay@enron.com',
     'restricted_stock_deferred': 'NaN',
     'deferred_income': -300000,
     'total_stock_value': 49110078,
     'expenses': 99832,
     'from_poi_to_this_person': 123,
     'exercised_stock_options': 34348384,
     'from_messages': 36,
     'other': 10359729,
     'from_this_person_to_poi': 16,
     'poi': True,
     'long_term_incentive': 3600000,
     'shared_receipt_with_poi': 2411,
     'restricted_stock': 14761694,
     'director_fees': 'NaN'}



There are checksums built into the dataset, like total_payments and total_stock_value. We should be able to calculate these from the other values to double check the values.


We can also query for a specific value, like this.


```python
print("Jeff Skilling's total payments were ${:,.0f}.".format(enron_data['SKILLING JEFFREY K']['total_payments']))
```

    Jeff Skilling's total payments were $8,682,716.
    

Before we go any further, we'll put the data into a pandas DataFrame to make it easier to work with.


```python
# The keys of the dictionary are the people, so we'll want them to be the rows of the dataframe
df = pd.DataFrame.from_dict(enron_data, orient='index')
pd.set_option('display.max_columns', len(df.columns))
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
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>email_address</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>expenses</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALLEN PHILLIP K</th>
      <td>201955</td>
      <td>2902</td>
      <td>2869717</td>
      <td>4484442</td>
      <td>NaN</td>
      <td>4175000</td>
      <td>phillip.allen@enron.com</td>
      <td>-126027</td>
      <td>-3081055</td>
      <td>1729541</td>
      <td>13868</td>
      <td>47</td>
      <td>1729541</td>
      <td>2195</td>
      <td>152</td>
      <td>65</td>
      <td>False</td>
      <td>304805</td>
      <td>1407</td>
      <td>126027</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>178980</td>
      <td>182466</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>257817</td>
      <td>3486</td>
      <td>NaN</td>
      <td>257817</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BANNANTINE JAMES M</th>
      <td>477</td>
      <td>566</td>
      <td>NaN</td>
      <td>916197</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>james.bannantine@enron.com</td>
      <td>-560222</td>
      <td>-5104</td>
      <td>5243487</td>
      <td>56301</td>
      <td>39</td>
      <td>4046157</td>
      <td>29</td>
      <td>864523</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>465</td>
      <td>1757552</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>267102</td>
      <td>NaN</td>
      <td>1295738</td>
      <td>5634343</td>
      <td>NaN</td>
      <td>1200000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1386055</td>
      <td>10623258</td>
      <td>11200</td>
      <td>NaN</td>
      <td>6680544</td>
      <td>NaN</td>
      <td>2660303</td>
      <td>NaN</td>
      <td>False</td>
      <td>1586055</td>
      <td>NaN</td>
      <td>3942714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>239671</td>
      <td>NaN</td>
      <td>260455</td>
      <td>827696</td>
      <td>NaN</td>
      <td>400000</td>
      <td>frank.bay@enron.com</td>
      <td>-82782</td>
      <td>-201641</td>
      <td>63014</td>
      <td>129142</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>145796</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We can already see missing values, and it looks like they're entered as NaN, which Python will see as a string and not recognized as a null value. We can do a quick replacement on those.


```python
df = df.replace('NaN', np.nan)
```

Pandas makes summarizing the data simple. First, we'll configure pandas to print in standard notation with no decimal places.


```python
pd.options.display.float_format = '{:10,.0f}'.format
```


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
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>expenses</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>95</td>
      <td>86</td>
      <td>39</td>
      <td>125</td>
      <td>4</td>
      <td>82</td>
      <td>18</td>
      <td>49</td>
      <td>126</td>
      <td>95</td>
      <td>86</td>
      <td>102</td>
      <td>86</td>
      <td>93</td>
      <td>86</td>
      <td>66</td>
      <td>86</td>
      <td>110</td>
      <td>17</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>562,194</td>
      <td>2,074</td>
      <td>1,642,674</td>
      <td>5,081,526</td>
      <td>41,962,500</td>
      <td>2,374,235</td>
      <td>166,411</td>
      <td>-1,140,475</td>
      <td>6,773,957</td>
      <td>108,729</td>
      <td>65</td>
      <td>5,987,054</td>
      <td>609</td>
      <td>919,065</td>
      <td>41</td>
      <td>1,470,361</td>
      <td>1,176</td>
      <td>2,321,741</td>
      <td>166,805</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2,716,369</td>
      <td>2,583</td>
      <td>5,161,930</td>
      <td>29,061,716</td>
      <td>47,083,209</td>
      <td>10,713,328</td>
      <td>4,201,494</td>
      <td>4,025,406</td>
      <td>38,957,773</td>
      <td>533,535</td>
      <td>87</td>
      <td>31,062,007</td>
      <td>1,841</td>
      <td>4,589,253</td>
      <td>100</td>
      <td>5,942,759</td>
      <td>1,178</td>
      <td>12,518,278</td>
      <td>319,891</td>
    </tr>
    <tr>
      <th>min</th>
      <td>477</td>
      <td>57</td>
      <td>-102,500</td>
      <td>148</td>
      <td>400,000</td>
      <td>70,000</td>
      <td>-7,576,788</td>
      <td>-27,992,891</td>
      <td>-44,093</td>
      <td>148</td>
      <td>0</td>
      <td>3,285</td>
      <td>12</td>
      <td>2</td>
      <td>0</td>
      <td>69,223</td>
      <td>2</td>
      <td>-2,604,490</td>
      <td>3,285</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>211,816</td>
      <td>541</td>
      <td>81,573</td>
      <td>394,475</td>
      <td>1,600,000</td>
      <td>431,250</td>
      <td>-389,622</td>
      <td>-694,862</td>
      <td>494,510</td>
      <td>22,614</td>
      <td>10</td>
      <td>527,886</td>
      <td>23</td>
      <td>1,215</td>
      <td>1</td>
      <td>281,250</td>
      <td>250</td>
      <td>254,018</td>
      <td>98,784</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>259,996</td>
      <td>1,211</td>
      <td>227,449</td>
      <td>1,101,393</td>
      <td>41,762,500</td>
      <td>769,375</td>
      <td>-146,975</td>
      <td>-159,792</td>
      <td>1,102,872</td>
      <td>46,950</td>
      <td>35</td>
      <td>1,310,814</td>
      <td>41</td>
      <td>52,382</td>
      <td>8</td>
      <td>442,035</td>
      <td>740</td>
      <td>451,740</td>
      <td>108,579</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>312,117</td>
      <td>2,635</td>
      <td>1,002,672</td>
      <td>2,093,263</td>
      <td>82,125,000</td>
      <td>1,200,000</td>
      <td>-75,010</td>
      <td>-38,346</td>
      <td>2,949,847</td>
      <td>79,952</td>
      <td>72</td>
      <td>2,547,724</td>
      <td>146</td>
      <td>362,096</td>
      <td>25</td>
      <td>938,672</td>
      <td>1,888</td>
      <td>1,002,370</td>
      <td>113,784</td>
    </tr>
    <tr>
      <th>max</th>
      <td>26,704,229</td>
      <td>15,149</td>
      <td>32,083,396</td>
      <td>309,886,585</td>
      <td>83,925,000</td>
      <td>97,343,619</td>
      <td>15,456,290</td>
      <td>-833</td>
      <td>434,509,511</td>
      <td>5,235,198</td>
      <td>528</td>
      <td>311,764,000</td>
      <td>14,368</td>
      <td>42,667,589</td>
      <td>609</td>
      <td>48,521,928</td>
      <td>5,521</td>
      <td>130,322,299</td>
      <td>1,398,517</td>
    </tr>
  </tbody>
</table>
</div>



The highest salary was \$26 million, and the highest value for total payments was \$309 million. The value for total payments seems too high, even for Enron, so we'll have to look into that.

# Cleaning the data

It looks like we have lots of integers and strings and a single column of booleans. The booleans column "poi" indicates whether the person is a "Person Of Interest". This is the column we'll be trying to predict using the other data. Every person in the DataFrame is marked as either a poi or not. Unfortunately, this is not the case with the other columns. From the first five rows, we can tell that the other columns have lots of missing data. Let's look at how bad it is.


```python
# Print the number of missing values
num_missing_values = df.isnull().sum()
print(num_missing_values)
```

    salary                        51
    to_messages                   60
    deferral_payments            107
    total_payments                21
    loan_advances                142
    bonus                         64
    email_address                 35
    restricted_stock_deferred    128
    deferred_income               97
    total_stock_value             20
    expenses                      51
    from_poi_to_this_person       60
    exercised_stock_options       44
    from_messages                 60
    other                         53
    from_this_person_to_poi       60
    poi                            0
    long_term_incentive           80
    shared_receipt_with_poi       60
    restricted_stock              36
    director_fees                129
    dtype: int64
    

That's a lot, and it isn't easy to analyze a dataset with that many missing values. Let's graph it to see what we've got. Remember there are 146 different people in this dataset.


```python
fig, ax = plt.subplots(figsize=(16, 10))
x = np.arange(0, len(num_missing_values))
matplotlib.rcParams.update({'font.size': 18})
plt.xticks(x, (df.columns), rotation='vertical')

# create the bars
bars = plt.bar(x, num_missing_values, align='center', linewidth=0)

ax.set_ylabel('Number of missing values')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# direct label each bar with Y axis values
for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())), 
                 ha='center', color='w', fontsize=16)
    
plt.show()
```


![png]({{site.baseurl}}/assets/img/2015-06-01-Cleaning-Data-with-Enron-Dataset_files/2015-06-01-Cleaning-Data-with-Enron-Dataset_29_0.png)


The most common missing values are loan_advances and director_fees. I imagine these are likely to be zero for most employees. Based on the columns that have the most missing values, the complete lack of zeros in the dataset, and the way it's presented in the spreadsheet, I think we can say that all NaN values should actually be zero. Let's make that change.


```python
df = df.fillna(0)
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
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>email_address</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALLEN PHILLIP K</th>
      <td>201,955</td>
      <td>2,902</td>
      <td>2,869,717</td>
      <td>4,484,442</td>
      <td>0</td>
      <td>4,175,000</td>
      <td>phillip.allen@enron.com</td>
      <td>-126,027</td>
      <td>-3,081,055</td>
      <td>1,729,541</td>
      <td>...</td>
      <td>47</td>
      <td>1,729,541</td>
      <td>2,195</td>
      <td>152</td>
      <td>65</td>
      <td>False</td>
      <td>304,805</td>
      <td>1,407</td>
      <td>126,027</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>0</td>
      <td>0</td>
      <td>178,980</td>
      <td>182,466</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>257,817</td>
      <td>...</td>
      <td>0</td>
      <td>257,817</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BANNANTINE JAMES M</th>
      <td>477</td>
      <td>566</td>
      <td>0</td>
      <td>916,197</td>
      <td>0</td>
      <td>0</td>
      <td>james.bannantine@enron.com</td>
      <td>-560,222</td>
      <td>-5,104</td>
      <td>5,243,487</td>
      <td>...</td>
      <td>39</td>
      <td>4,046,157</td>
      <td>29</td>
      <td>864,523</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>465</td>
      <td>1,757,552</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>267,102</td>
      <td>0</td>
      <td>1,295,738</td>
      <td>5,634,343</td>
      <td>0</td>
      <td>1,200,000</td>
      <td>0</td>
      <td>0</td>
      <td>-1,386,055</td>
      <td>10,623,258</td>
      <td>...</td>
      <td>0</td>
      <td>6,680,544</td>
      <td>0</td>
      <td>2,660,303</td>
      <td>0</td>
      <td>False</td>
      <td>1,586,055</td>
      <td>0</td>
      <td>3,942,714</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>239,671</td>
      <td>0</td>
      <td>260,455</td>
      <td>827,696</td>
      <td>0</td>
      <td>400,000</td>
      <td>frank.bay@enron.com</td>
      <td>-82,782</td>
      <td>-201,641</td>
      <td>63,014</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>69</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>145,796</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Let's look at the checksum values to make sure everything checks out. By "checksums", I'm referring to values that are supposed to be the sum of other values in the table. They can be used to find errors in the data quickly. In this case, the total_payments field is supposed to be the sum of all the payment categories: salary, bonus, long_term_incentive, deferred_income, deferral_payments, loan_advances, other, expenses, and director_fees.

The original spreadsheet divides up the information into payments and stock value. We'll do that now and add a separate category for the email.


```python
payment_categories = ['salary', 'bonus', 'long_term_incentive', 'deferred_income',
                      'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments']
stock_value_categories = ['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']
```

Now let's sum together all the payment categories (except total_payments) and compare it to the total payments. It should be the same value. We'll print out any rows that aren't the same.


```python
# Look at the instances where the total we calculate is not equal to the total listed on the spreadsheet
df[df[payment_categories[:-1]].sum(axis='columns') != df['total_payments']][payment_categories]
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
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELFER ROBERT</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-102,500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3,285</td>
      <td>102,500</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>137,864</td>
      <td>0</td>
      <td>137,864</td>
      <td>15,456,290</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df[stock_value_categories[:-1]].sum(axis='columns') != df['total_stock_value']][stock_value_categories]
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
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELFER ROBERT</th>
      <td>3,285</td>
      <td>0</td>
      <td>44,093</td>
      <td>-44,093</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>2,604,490</td>
      <td>-2,604,490</td>
      <td>15,456,290</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



On the original spreadsheet, 102,500 is listed in Robert Belfer's deferred income section, not in deferral payments. And 3,285 should be the expenses column, and director fees should be 102,500, and total payments should be 3,285. It looks like everything shifted one column to the right. We'll have to move it one to the left to fix it. The opposite happened to Sanjay Bhatnagar, so we'll have to move his columns to the right to fix them. Let's do that now.


```python
df.loc['BELFER ROBERT']
```




    salary                               0
    to_messages                          0
    deferral_payments             -102,500
    total_payments                 102,500
    loan_advances                        0
    bonus                                0
    email_address                        0
    restricted_stock_deferred       44,093
    deferred_income                      0
    total_stock_value              -44,093
    expenses                             0
    from_poi_to_this_person              0
    exercised_stock_options          3,285
    from_messages                        0
    other                                0
    from_this_person_to_poi              0
    poi                              False
    long_term_incentive                  0
    shared_receipt_with_poi              0
    restricted_stock                     0
    director_fees                    3,285
    Name: BELFER ROBERT, dtype: object



Unfortunately, the order of the columns in the actual spreadsheet is different than the one in this dataset, so I can't use `pop` to push them all over one. I'll have to manually fix every incorrect value.


```python
df.loc[('BELFER ROBERT','deferral_payments')] = 0
df.loc[('BELFER ROBERT','total_payments')] = 3285
df.loc[('BELFER ROBERT','restricted_stock_deferred')] = -44093
df.loc[('BELFER ROBERT','deferred_income')] = -102500
df.loc[('BELFER ROBERT','total_stock_value')] = 0
df.loc[('BELFER ROBERT','expenses')] = 3285
df.loc[('BELFER ROBERT','exercised_stock_options')] = 0
df.loc[('BELFER ROBERT','restricted_stock')] = 44093
df.loc[('BELFER ROBERT','director_fees')] = 102500
```


```python
df.loc['BHATNAGAR SANJAY']
```




    salary                                                0
    to_messages                                         523
    deferral_payments                                     0
    total_payments                               15,456,290
    loan_advances                                         0
    bonus                                                 0
    email_address                sanjay.bhatnagar@enron.com
    restricted_stock_deferred                    15,456,290
    deferred_income                                       0
    total_stock_value                                     0
    expenses                                              0
    from_poi_to_this_person                               0
    exercised_stock_options                       2,604,490
    from_messages                                        29
    other                                           137,864
    from_this_person_to_poi                               1
    poi                                               False
    long_term_incentive                                   0
    shared_receipt_with_poi                             463
    restricted_stock                             -2,604,490
    director_fees                                   137,864
    Name: BHATNAGAR SANJAY, dtype: object




```python
df.loc[('BHATNAGAR SANJAY','total_payments')] = 137864
df.loc[('BHATNAGAR SANJAY','restricted_stock_deferred')] = -2604490
df.loc[('BHATNAGAR SANJAY','total_stock_value')] = 15456290
df.loc[('BHATNAGAR SANJAY','expenses')] = 137864
df.loc[('BHATNAGAR SANJAY','exercised_stock_options')] = 15456290
df.loc[('BHATNAGAR SANJAY','other')] = 0
df.loc[('BHATNAGAR SANJAY','restricted_stock')] = 2604490
df.loc[('BHATNAGAR SANJAY','director_fees')] = 0
```

Let's check to make sure that fixes our problems.


```python
print(df[df[payment_categories[:-1]].sum(axis='columns') != df['total_payments']][payment_categories])
print(df[df[stock_value_categories[:-1]].sum(axis='columns') != df['total_stock_value']][stock_value_categories])
```

    Empty DataFrame
    Columns: [salary, bonus, long_term_incentive, deferred_income, deferral_payments, loan_advances, other, expenses, director_fees, total_payments]
    Index: []
    Empty DataFrame
    Columns: [exercised_stock_options, restricted_stock, restricted_stock_deferred, total_stock_value]
    Index: []
    

Looks good!

## Look for anomalies in the data

Now we're going to use a couple of functions provided by the Udacity class. They help to get data out of the dictionaries and into a more usable form. Here they are:


```python
def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """


    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = list(dictionary.keys())

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)


def targetFeatureSplit( data ):
    """ 
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features
```

One of the best ways to detect anomalies is to graph the data. Anomalies often stick out in these graphs. Let's take a look at how salary correlates with bonus. I suspect it will be positive and fairly strong.


```python
### read in data dictionary, convert to numpy array

features = ["salary", "bonus"]
data = featureFormat(enron_data, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
```


![png]({{site.baseurl}}/assets/img/2015-06-01-Cleaning-Data-with-Enron-Dataset_files/2015-06-01-Cleaning-Data-with-Enron-Dataset_51_0.png)


OK, someone's bonus and salary are way higher than everyone else's. That looks suspicious so let's take a look at it.


```python
df[df['salary'] > 10000000]
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
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>email_address</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TOTAL</th>
      <td>26,704,229</td>
      <td>0</td>
      <td>32,083,396</td>
      <td>309,886,585</td>
      <td>83,925,000</td>
      <td>97,343,619</td>
      <td>0</td>
      <td>-7,576,788</td>
      <td>-27,992,891</td>
      <td>434,509,511</td>
      <td>...</td>
      <td>0</td>
      <td>311,764,000</td>
      <td>0</td>
      <td>42,667,589</td>
      <td>0</td>
      <td>False</td>
      <td>48,521,928</td>
      <td>0</td>
      <td>130,322,299</td>
      <td>1,398,517</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>



Ah, there's an "employee" named "TOTAL" in the spreadsheet. Having a row that is the total of our other rows will mess up our statistics, so we'll remove it. We'll also remove The Travel Agency in the Park that we noticed earlier.


```python
entries_to_delete = ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL']
for entry in entries_to_delete:
    if entry in df.index:
        df = df.drop(entry)
```

Now let's look again.


```python
### read in data dictionary, convert to numpy array

features = ["salary", "bonus"]
#data = df["salary", "bonus"]

salary = df['salary'].values / 1000
bonus = df['bonus'].values / 1000
plt.scatter(salary, bonus)

plt.xlabel("salary (thousands of dollars)")
plt.ylabel("bonus (thousands of dollars)")
plt.show()
```


![png]({{site.baseurl}}/assets/img/2015-06-01-Cleaning-Data-with-Enron-Dataset_files/2015-06-01-Cleaning-Data-with-Enron-Dataset_57_0.png)


That looks better.

Now that we've cleaned up the data let's save it as a CSV so we can pick it up and do some analysis on it another time.


```python
clean_file = 'clean_df.csv'
df.to_csv(path+clean_file, index_label='name')
```
