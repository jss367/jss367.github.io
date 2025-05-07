---
layout: post
title: "Working with US Census Bureau Data"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/seal.jpg"
tags: [Datasets]
---

I found that the US Census API is difficult to work with and even LLMs don't provide working code for it. So I thought it might be helpful to share some techniques that did work. In this post, I'm going to focus on both raw API calls and the Python wrapper.


```python
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from census import Census
```

You need to get an API key. Fortunately, this part is really easy—all you need to do is sign up on their website.


```python
API_KEY = os.getenv('CENSUS_API_KEY')
```

There are different ways to access the census data, including through their website, through direct API calls, and through a Python wrapper.

## Direct API Calls

We're going to look at the American Community Survey (ACS) Select Population Profiles (SPP) data.

You need to provide `fields` and `iteration codes`. You can find which population is associated with which iteration code here: [https://www2.census.gov/programs-surveys/decennial/datasets/summary-file-2/attachment.pdf](https://www2.census.gov/programs-surveys/decennial/datasets/summary-file-2/attachment.pdf)

For example, it tells you that `013` is the iteration code for Asian Indians.


```python
YEAR     = 2022  # latest year that I could find that had everything I was looking for
DATASET  = f"https://api.census.gov/data/{YEAR}/acs/acs1/spp"
FIELDS   = "NAME,S0201_214E"                             # median household income
POP_CODE = "013"                                         # <-- Asian Indian alone
URL      = (f"{DATASET}?get={FIELDS}"
            f"&for=us:1&POPGROUP={POP_CODE}&key={API_KEY}")

resp = requests.get(URL, timeout=30)
rows = resp.json()
```


```python
print(resp.status_code)
print(resp.text[:500])
```

    200
    [["NAME","S0201_214E","POPGROUP","us"],
    ["United States","152341","013","1"]]



```python
df = pd.DataFrame(rows[1:], columns=rows[0])
df["S0201_214E"] = pd.to_numeric(df["S0201_214E"])
print("Median HH income (Asian-Indian-American, 2022):",
      f"${int(df.at[0,'S0201_214E']):,}")
```

    Median HH income (Asian-Indian-American, 2022): $152,341


## Getting More Data

OK, so we can get a single data point from a query, but it's inefficient to do that for lots of data. Let's grab data for multiple groups in a single request.


```python
YEAR     = 2022
DATASET  = f"https://api.census.gov/data/{YEAR}/acs/acs1/spp"
FIELDS   = "NAME,S0201_214E,POPGROUP"                    # median household income + population group
URL      = (f"{DATASET}?get={FIELDS}"
            f"&for=us:1&key={API_KEY}")
resp = requests.get(URL, timeout=30)
rows = resp.json()

# Convert to pandas DataFrame
import pandas as pd
df = pd.DataFrame(rows[1:], columns=rows[0])
df['S0201_214E'] = pd.to_numeric(df['S0201_214E'], errors='coerce')

# df['POPGROUP_DESC'] = df['POPGROUP'].map(popgroup_dict)
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
      <th>NAME</th>
      <th>S0201_214E</th>
      <th>POPGROUP</th>
      <th>us</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>United States</td>
      <td>74755</td>
      <td>001</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>United States</td>
      <td>79933</td>
      <td>002</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>United States</td>
      <td>78636</td>
      <td>003</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>51374</td>
      <td>004</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>United States</td>
      <td>52238</td>
      <td>005</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The codes are not that helpful directly and need to be converted using the link above. You can find the full dictionary here:  
[code_to_population.py (Gist)](https://gist.github.com/jss367/44e041c913f87a11b2830e01e295c241). We'll use `curl` to download it:


```python
!curl -o census_popgroup_dict.py https://gist.githubusercontent.com/jss367/44e041c913f87a11b2830e01e295c241/raw/c54c8ffaf838791c1a1c42fc03d493bbb3fe3b84/gistfile1.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  6932  100  6932    0     0  31496      0 --:--:-- --:--:-- --:--:-- 31509



```python
from census_popgroup_dict import code_to_population
```


```python
df['POPGROUP_DESC'] = df['POPGROUP'].map(code_to_population)
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
      <th>NAME</th>
      <th>S0201_214E</th>
      <th>POPGROUP</th>
      <th>us</th>
      <th>POPGROUP_DESC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>United States</td>
      <td>74755</td>
      <td>001</td>
      <td>1</td>
      <td>Total Population</td>
    </tr>
    <tr>
      <th>1</th>
      <td>United States</td>
      <td>79933</td>
      <td>002</td>
      <td>1</td>
      <td>White alone</td>
    </tr>
    <tr>
      <th>2</th>
      <td>United States</td>
      <td>78636</td>
      <td>003</td>
      <td>1</td>
      <td>White alone or in combination with one or more...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>51374</td>
      <td>004</td>
      <td>1</td>
      <td>Black or African American alone</td>
    </tr>
    <tr>
      <th>4</th>
      <td>United States</td>
      <td>52238</td>
      <td>005</td>
      <td>1</td>
      <td>Black or African American alone or in combinat...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
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
      <th>NAME</th>
      <th>S0201_214E</th>
      <th>POPGROUP</th>
      <th>us</th>
      <th>POPGROUP_DESC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>342</th>
      <td>United States</td>
      <td>126414</td>
      <td>931</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>343</th>
      <td>United States</td>
      <td>135643</td>
      <td>932</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>344</th>
      <td>United States</td>
      <td>55352</td>
      <td>946</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>345</th>
      <td>United States</td>
      <td>80191</td>
      <td>9Z8</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>346</th>
      <td>United States</td>
      <td>78411</td>
      <td>9Z9</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Unfortunately, many are missing and I'm not sure what the issue is at the moment.


```python
df.dropna(inplace=True)
```


```python
df.sample(10)
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
      <th>NAME</th>
      <th>S0201_214E</th>
      <th>POPGROUP</th>
      <th>us</th>
      <th>POPGROUP_DESC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138</th>
      <td>United States</td>
      <td>51978</td>
      <td>454</td>
      <td>1</td>
      <td>Black or African American alone or in combinat...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>United States</td>
      <td>61778</td>
      <td>009</td>
      <td>1</td>
      <td>American Indian and Alaska Native alone or in ...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>United States</td>
      <td>91261</td>
      <td>023</td>
      <td>1</td>
      <td>Korean alone (440-441)</td>
    </tr>
    <tr>
      <th>42</th>
      <td>United States</td>
      <td>74089</td>
      <td>052</td>
      <td>1</td>
      <td>Native Hawaiian alone (500-503)</td>
    </tr>
    <tr>
      <th>69</th>
      <td>United States</td>
      <td>44671</td>
      <td>110</td>
      <td>1</td>
      <td>Black or African American; American Indian and...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>United States</td>
      <td>52238</td>
      <td>005</td>
      <td>1</td>
      <td>Black or African American alone or in combinat...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>United States</td>
      <td>78636</td>
      <td>003</td>
      <td>1</td>
      <td>White alone or in combination with one or more...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>United States</td>
      <td>80288</td>
      <td>014</td>
      <td>1</td>
      <td>Bangladeshi alone (402)</td>
    </tr>
    <tr>
      <th>124</th>
      <td>United States</td>
      <td>93798</td>
      <td>414</td>
      <td>1</td>
      <td>Argentinean (231)</td>
    </tr>
    <tr>
      <th>125</th>
      <td>United States</td>
      <td>87192</td>
      <td>415</td>
      <td>1</td>
      <td>Bolivian (232)</td>
    </tr>
  </tbody>
</table>
</div>



We can see the Asian Indian data again.


```python
df[df['POPGROUP'] == '013']
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
      <th>NAME</th>
      <th>S0201_214E</th>
      <th>POPGROUP</th>
      <th>us</th>
      <th>POPGROUP_DESC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>United States</td>
      <td>152341</td>
      <td>013</td>
      <td>1</td>
      <td>Asian Indian alone (400-401)</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## Using the Census Wrapper

There is also a census wrapper you can use that's available for download at [https://pypi.org/project/census/](https://pypi.org/project/census/). Let's use it to get some income data.

Here, we'll use a different table. You can see the different tables here: https://api.census.gov/data/2021/acs/acs5/groups/

In the census data, you'll also see IDs like `B19013A_001E`. Let's take a moment to understand what it means. It's in the following format:

`[TABLE_ID][SUBGROUP]_[LINE][SUFFIX]`

So, in this case:
* `B19013A` is the **table id**. This specific table is titled "Median Household Income in the Past 12 Months (in 2021 Inflation-Adjusted Dollars)". 
* `001` refers to the **line number** within that table, corresponding to a specific row (e.g., "Median household income").
* `E` stands for **Estimate** — as opposed to M, which would be the Margin of Error for that estimate.

Let's look at the `B19013` table. You'll note that not all subgroups are available here. If you want to dig deeper into, say, Asian subgroups, you need to look at a different table.

```
{
  "name": "B19013A",
  "description": "MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2021 INFLATION-ADJUSTED DOLLARS) (WHITE ALONE HOUSEHOLDER)",
  "variables": "http://api.census.gov/data/2021/acs/acs5/groups/B19013A.json",
  "universe ": "Households with a householder who is White alone"
}
```

Now we call the census wrapper.


```python
c = Census(API_KEY)
```


```python
race_data = c.acs5.get(
    ('NAME', 
     'B19013_001E',  # Total population
     'B19013A_001E', # White alone
     'B19013B_001E', # Black alone
     'B19013C_001E', # American Indian/Alaska Native alone
     'B19013D_001E', # Asian alone
     'B19013E_001E', # Native Hawaiian/Pacific Islander alone
     'B19013F_001E', # Some other race alone
     'B19013G_001E', # Two or more races
     'B19013H_001E', # White alone, not Hispanic
     'B19013I_001E', # Hispanic/Latino origin (any race)
    ),
    {'for': 'us:*'}
)
```


```python
race_data
```




    [{'NAME': 'United States',
      'B19013_001E': 78538.0,
      'B19013A_001E': 83784.0,
      'B19013B_001E': 53444.0,
      'B19013C_001E': 59393.0,
      'B19013D_001E': 113106.0,
      'B19013E_001E': 78640.0,
      'B19013F_001E': 65558.0,
      'B19013G_001E': 73412.0,
      'B19013H_001E': 84745.0,
      'B19013I_001E': 68890.0,
      'us': '1'}]




```python
df = pd.DataFrame(race_data)
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
      <th>NAME</th>
      <th>B19013_001E</th>
      <th>B19013A_001E</th>
      <th>B19013B_001E</th>
      <th>B19013C_001E</th>
      <th>B19013D_001E</th>
      <th>B19013E_001E</th>
      <th>B19013F_001E</th>
      <th>B19013G_001E</th>
      <th>B19013H_001E</th>
      <th>B19013I_001E</th>
      <th>us</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>United States</td>
      <td>78538.0</td>
      <td>83784.0</td>
      <td>53444.0</td>
      <td>59393.0</td>
      <td>113106.0</td>
      <td>78640.0</td>
      <td>65558.0</td>
      <td>73412.0</td>
      <td>84745.0</td>
      <td>68890.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



It's... a little ugly. So we can rename the columns.


```python
df = df.rename(columns={
    'B19013_001E': 'Median_Income_Total',
    'B19013A_001E': 'Median_Income_White_Alone',
    'B19013B_001E': 'Median_Income_Black_Alone',
    'B19013C_001E': 'Median_Income_AmIndian_Alone',
    'B19013D_001E': 'Median_Income_Asian_Alone',
    'B19013E_001E': 'Median_Income_Hawaiian_Alone',
    'B19013F_001E': 'Median_Income_Other_Alone',
    'B19013G_001E': 'Median_Income_TwoOrMore',
    'B19013H_001E': 'Median_Income_White_NonHispanic',
    'B19013I_001E': 'Median_Income_Hispanic',
})
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
      <th>NAME</th>
      <th>Median_Income_Total</th>
      <th>Median_Income_White_Alone</th>
      <th>Median_Income_Black_Alone</th>
      <th>Median_Income_AmIndian_Alone</th>
      <th>Median_Income_Asian_Alone</th>
      <th>Median_Income_Hawaiian_Alone</th>
      <th>Median_Income_Other_Alone</th>
      <th>Median_Income_TwoOrMore</th>
      <th>Median_Income_White_NonHispanic</th>
      <th>Median_Income_Hispanic</th>
      <th>us</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>United States</td>
      <td>78538.0</td>
      <td>83784.0</td>
      <td>53444.0</td>
      <td>59393.0</td>
      <td>113106.0</td>
      <td>78640.0</td>
      <td>65558.0</td>
      <td>73412.0</td>
      <td>84745.0</td>
      <td>68890.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


## Using the Website

I don't find the website particularly easy to use, either. You can see some of the same information though. Here, for example is the ACS data on Asian Indians:
* https://data.census.gov/table?t=013:Income+and+Poverty&g=010XX00US.

Here's the same for total population:
* https://data.census.gov/table?t=001:Income+and+Poverty&g=010XX00US

You can see in the URL how the iteration codes work. You can either change that value directly or use the filters on the left sidebar.

## Errors in the Data

I was surprised to find lots of errors in the data. Here are [a couple of examples](https://x.com/JuliusSimonelli/status/1918827800695292341). Beware, I guess!

## Note on Dates

You might have noticed that I used 2022 in the example above. That's because that's the 2023 (and beyond) data doesn't seem to be there.


```python
YEAR     = 2023 
DATASET  = f"https://api.census.gov/data/{YEAR}/acs/acs1/spp"
FIELDS   = "NAME,S0201_214E"
POP_CODE = "013"
URL      = (f"{DATASET}?get={FIELDS}"
            f"&for=us:1&POPGROUP={POP_CODE}&key={API_KEY}")

empty_resp = requests.get(URL, timeout=30)
```


```python
print(empty_resp.status_code)
print(empty_resp.text[:500])
```

    204
    


You can see that I got a 204 back, indicating that there was no content returned.
