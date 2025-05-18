---
layout: post
title: "Working with US Census Bureau Data"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/seal.jpg"
tags: [Datasets]
---

I found that the US Census API is difficult to work with and even LLMs don't provide working code for it. So I thought it might be helpful to share some techniques that did work. In this post, I'm going to focus on both raw API calls and the Python wrapper.

<b>Table of Contents</b>
* TOC
{:toc}

## API key


```python
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from census import Census
```

You need to get an API key. Fortunately, this part is really easy—all you need to do is sign up on [their website](https://api.census.gov/data/key_signup.html).


```python
API_KEY = os.getenv('CENSUS_API_KEY')
```

There are different ways to access the census data, including through their website, through direct API calls, and through a Python wrapper.

## Understanding Tables

Here, we'll use a different table. You can see the different tables here: https://api.census.gov/data/2021/acs/acs5/groups/

In the census data, you'll also see IDs like `B19013A_001E`. Let's take a moment to understand what it means. It's in the following format:

`[TABLE_ID][SUBGROUP]_[LINE][SUFFIX]`

So, in this case:
* `B19013A` is the **table id**. This specific table is titled "Median Household Income in the Past 12 Months (in 2021 Inflation-Adjusted Dollars)". 
* `001` refers to the **line number** within that table, corresponding to a specific row (e.g., "Median household income").
* `E` stands for **Estimate** — as opposed to M, which would be the Margin of Error for that estimate.

One table with a lot of data is S0201. S0201 refers to the Selected Population Profile (SPP) table series. This is used for detailed demographic, social, economic, and housing data by race, Hispanic origin, tribal group, or ancestry.

## Direct API Calls

We're going to look at the American Community Survey (ACS) Select Population Profiles (SPP) data.

You need to provide `fields` and `iteration codes`. You can find which population is associated with which iteration code here: [https://www2.census.gov/programs-surveys/decennial/datasets/summary-file-2/attachment.pdf](https://www2.census.gov/programs-surveys/decennial/datasets/summary-file-2/attachment.pdf)

For fields, we're using `S0201_214E`. S0201 is the table and _214 is the line number within the S0201 table, which corresponds to a specific data item. This is how we can get median household income.

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


### Verifying Results

It's good to have a way to verify the data as well. For example, you can verify some of the results simply by Googling the number and making sure that's what other people got. By Googling $152,341 you can see other newsites that use the same value and describe it as Indian annual median household [income]((https://economictimes.indiatimes.com/opinion/et-editorial/indian-high-earners-of-the-world-unite/articleshow/109112116.cms?from=mdr)).

## Getting More Data

OK, so we can get a single data point from a query, but it's inefficient to do that for lots of data. Let's grab data for multiple groups in a single request.

Here we also need a field. We're going to use `S0201_214E`. You can see on the [SPP variables table](https://api.census.gov/data/2022/acs/acs1/spp/variables.xml) that `S0201_214E` corresponds to "Median household income (dollars)".


```python
YEAR     = 2022
DATASET  = f"https://api.census.gov/data/{YEAR}/acs/acs1/spp"
FIELDS   = "NAME,S0201_214E,POPGROUP"  # median household income + population group
URL      = (f"{DATASET}?get={FIELDS}"
            f"&for=us:1&key={API_KEY}")
resp = requests.get(URL, timeout=30)
rows = resp.json()

# Convert to pandas DataFrame
income_df = pd.DataFrame(rows[1:], columns=rows[0])
income_df['S0201_214E'] = pd.to_numeric(income_df['S0201_214E'], errors='coerce')
```


```python
len(income_df)
```




    347




```python
income_df.head()
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



The codes are not that helpful directly and need to be converted using the link above. You can find the full dictionary here: [code_to_population.py (Gist)](https://gist.github.com/jss367/44e041c913f87a11b2830e01e295c241). We'll use `curl` to download it:


```python
!curl -o census_popgroup_dict.py https://gist.githubusercontent.com/jss367/44e041c913f87a11b2830e01e295c241/raw/c54c8ffaf838791c1a1c42fc03d493bbb3fe3b84/gistfile1.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  6932  100  6932    0     0  28768      0 --:--:-- --:--:-- --:--:-- 28883



```python
from census_popgroup_dict import code_to_population
```


```python
income_df['POPGROUP_DESC'] = income_df['POPGROUP'].map(code_to_population)
```


```python
income_df.head()
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
income_df.tail()
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
income_df.dropna(inplace=True)
```


```python
len(income_df)
```




    86




```python
income_df.sample(10)
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
      <th>129</th>
      <td>United States</td>
      <td>77024</td>
      <td>420</td>
      <td>1</td>
      <td>Peruvian (237)</td>
    </tr>
    <tr>
      <th>146</th>
      <td>United States</td>
      <td>78234</td>
      <td>462</td>
      <td>1</td>
      <td>Some Other Race alone or in combination with o...</td>
    </tr>
    <tr>
      <th>147</th>
      <td>United States</td>
      <td>72601</td>
      <td>463</td>
      <td>1</td>
      <td>Two or More Races, not Hispanic or Latino</td>
    </tr>
    <tr>
      <th>0</th>
      <td>United States</td>
      <td>74755</td>
      <td>001</td>
      <td>1</td>
      <td>Total Population</td>
    </tr>
    <tr>
      <th>132</th>
      <td>United States</td>
      <td>82993</td>
      <td>423</td>
      <td>1</td>
      <td>Spaniard (200-209)</td>
    </tr>
    <tr>
      <th>74</th>
      <td>United States</td>
      <td>85527</td>
      <td>117</td>
      <td>1</td>
      <td>Asian; Native Hawaiian and Other Pacific Islander</td>
    </tr>
    <tr>
      <th>145</th>
      <td>United States</td>
      <td>75631</td>
      <td>461</td>
      <td>1</td>
      <td>Some Other Race alone, not Hispanic or Latino</td>
    </tr>
    <tr>
      <th>113</th>
      <td>United States</td>
      <td>66241</td>
      <td>403</td>
      <td>1</td>
      <td>Cuban (270-274)</td>
    </tr>
    <tr>
      <th>46</th>
      <td>United States</td>
      <td>76421</td>
      <td>060</td>
      <td>1</td>
      <td>Native Hawaiian and Other Pacific Islander alo...</td>
    </tr>
    <tr>
      <th>66</th>
      <td>United States</td>
      <td>95428</td>
      <td>107</td>
      <td>1</td>
      <td>White; Asian</td>
    </tr>
  </tbody>
</table>
</div>



We can see the Asian Indian data again.


```python
income_df[income_df['POPGROUP'] == '013']
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



## Getting Population Groups


```python
params = {
    "get": "POPGROUP,POPGROUP_LABEL",  # Request codes and labels
    "for": "us:1",                     # National level
    "key": API_KEY
}
```


```python
year = 2023
```


```python
base_url = f"https://api.census.gov/data/{year}/acs/acs1/spp"
```


```python
# Make the request
response = requests.get(base_url, params=params)

# Check if request was successful
response.raise_for_status()

# Parse JSON response
data = response.json()

# Create DataFrame from the response (skip header row)
popgroups_df = pd.DataFrame(data[1:], columns=data[0])

# Convert to appropriate data types
popgroups_df = popgroups_df.convert_dtypes()
```


```python
len(popgroups_df)
```




    5545




```python
popgroups_df.sample(10)
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
      <th>POPGROUP</th>
      <th>POPGROUP_LABEL</th>
      <th>us</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1066</th>
      <td>1462</td>
      <td>Native Village of Buckland alone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>21H</td>
      <td>Tlingit alone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1233</th>
      <td>2124</td>
      <td>Skull Valley Band of Goshute Indians of Utah a...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1262</th>
      <td>2193</td>
      <td>Upper Chinook alone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1523</th>
      <td>3885</td>
      <td>Rotuman alone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3451</th>
      <td>2822</td>
      <td>Village of Solomon alone or in any combination</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4873</th>
      <td>2907</td>
      <td>Cherokee Alabama alone or in any combination</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1566</th>
      <td>563</td>
      <td>African</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1080</th>
      <td>095</td>
      <td>Mariana Islander alone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3867</th>
      <td>2590</td>
      <td>Central Council of the Tlingit and Haida India...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Using the Census Wrapper

There is also a census wrapper you can use that's available for download at [https://pypi.org/project/census/](https://pypi.org/project/census/). Let's use it to get some income data.

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


```python
df = pd.DataFrame(race_data)
df
```

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

## Using the Website

I don't find the website particularly easy to use, either. You can see some of the same information though. Here is the page for the American Community Survey, contains a lot of their data:
* [https://data.census.gov/table?q=American+Community+Survey&t=-A0](https://data.census.gov/table?q=American+Community+Survey&t=-A0)

Here is the ACS data on Asian Indians:
* [https://data.census.gov/table?t=013:Income+and+Poverty&g=010XX00US](https://data.census.gov/table?t=013:Income+and+Poverty&g=010XX00US).

Here's the same for total population:
* [https://data.census.gov/table?t=001:Income+and+Poverty&g=010XX00US](https://data.census.gov/table?t=001:Income+and+Poverty&g=010XX00US)

You can see in the URL how the iteration codes work. You can either change that value directly or use the filters on the left sidebar.

## Population Counts Mixed with Dollar Amounts

You will also find population counts mixed in with dollar amounts, which can be confusing. In the image below, the per capita income is in dollars, the  "With earnings for full-time, year-round workers" (Male and Female) is in number of people, and the "Mean earnings (dollars) for full-time, year-round workers" (Male and Female) is in dollars again.

<img width="819" alt="image" src="https://github.com/user-attachments/assets/b47c9c60-dae3-4f97-aa2c-7086c2194d14" />

## Note on Dates

You might have noticed that I used 2022 in the example above. That's because that's the 2023 (and beyond) data doesn't seem to be there for every table. Sometimes they are available though so I think you just have to check.


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

You can see that I got a 204 back, indicating that there was no content returned.
