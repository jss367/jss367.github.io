---
layout: post
title: "Visualizing US Census Bureau Data"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/bobcat.jpg"
tags: [Data Visualization, Datasets]
---

In this post I'm going to build off the last post on [Working with US Census Bureau Data](https://jss367.github.io/working-with-us-census-bureau-data.html) and discuss how to visualize it. That post walked through working with the Census Bureau's API, so in this post I'll skip those details.

<b>Table of Contents</b>
* TOC
{:toc}



```python
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
```


```python
API_KEY = os.getenv('CENSUS_API_KEY')
```

## Get a Population DataFrame

We start by getting a dataframe of population data, like we showed in the last post. In this post, I'm going to be using 2023 data because it's available for the tables I'm looking at.


```python
year=2023
```


```python
base_url = f"https://api.census.gov/data/{year}/acs/acs1/spp"
```


```python
params = {
    "get": "POPGROUP,POPGROUP_LABEL",  # Request codes and labels
    "for": "us:1",                     # National level
    "key": API_KEY                     # Your API key
}
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
popgroups_df
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
      <th>0</th>
      <td>2675</td>
      <td>Native Village of Ekuk alone or in any combina...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2680</td>
      <td>Native Village of Fort Yukon alone or in any c...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2828</td>
      <td>Yakutat Tlingit Tribe alone or in any combination</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2779</td>
      <td>Salamatof Tribe alone or in any combination</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2799</td>
      <td>Tsimshian alone or in any combination</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5540</th>
      <td>2311</td>
      <td>French Canadian/French American Indian alone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5541</th>
      <td>2318</td>
      <td>Heiltsuk Band alone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5542</th>
      <td>232</td>
      <td>United Houma Nation tribal grouping alone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5543</th>
      <td>2320</td>
      <td>Hiawatha First Nation alone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5544</th>
      <td>2328</td>
      <td>Kahkewistahaw First Nation alone</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5545 rows × 3 columns</p>
</div>



## Get Selected Population Profile

Now let's get the Selection Population Profile table.


```python
def get_spp_table(year: int = 2023,
                  variables: str | None = None,
                  geography: str = "us:1",
                  api_key: str | None = None,
                  timeout: int = 20) -> pd.DataFrame:
    """
    Download the full ACS‑1‑year Selected Population Profile (S0201) table
    for all population groups at the chosen geography.
    """
    if api_key is None:
        api_key = os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("Census API key not provided (argument or CENSUS_API_KEY).")

    # Default to median‑household‑income column if caller didn’t ask for more
    if variables is None:
        variables = "NAME,S0201_214E"

    base_url = f"https://api.census.gov/data/{year}/acs/acs1/spp"

    # Wildcard for *all* population‑group strings — allowed for string predicates :contentReference[oaicite:0]{index=0}
    params = {
        "get": variables,
        "for": geography,
        "POPGROUP": "*",          # same as POPGROUP:* in query string
        "key": api_key
    }

    try:
        resp = requests.get(base_url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Census API request failed: {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Unable to decode JSON: {e}") from e

    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError(f"Unexpected response format: {data}")

    # First row is the header
    df = pd.DataFrame(data[1:], columns=data[0])

    # POPGROUP is returned automatically because it’s a default‑display variable :contentReference[oaicite:1]{index=1}
    # Convert any numeric columns that arrive as text
    numeric_cols = [c for c in df.columns if c.endswith(("E", "EA"))]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df
```


```python
df = get_spp_table(year=2023)
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
      <th>S0201_214E</th>
      <th>POPGROUP</th>
      <th>us</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>77719</td>
      <td>001</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>82531</td>
      <td>002</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>81643</td>
      <td>003</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>53927</td>
      <td>004</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>55195</td>
      <td>005</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>363</th>
      <td>NaN</td>
      <td>80358</td>
      <td>920</td>
      <td>1</td>
    </tr>
    <tr>
      <th>364</th>
      <td>NaN</td>
      <td>81340</td>
      <td>921</td>
      <td>1</td>
    </tr>
    <tr>
      <th>365</th>
      <td>NaN</td>
      <td>102394</td>
      <td>930</td>
      <td>1</td>
    </tr>
    <tr>
      <th>366</th>
      <td>NaN</td>
      <td>141182</td>
      <td>931</td>
      <td>1</td>
    </tr>
    <tr>
      <th>367</th>
      <td>NaN</td>
      <td>149067</td>
      <td>932</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>368 rows × 4 columns</p>
</div>



## Merge Them

Merge them.


```python
mdf = pd.merge(df, popgroups_df, on='POPGROUP')
```


```python
mdf
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
      <th>us_x</th>
      <th>POPGROUP_LABEL</th>
      <th>us_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>77719</td>
      <td>001</td>
      <td>1</td>
      <td>Total population</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>82531</td>
      <td>002</td>
      <td>1</td>
      <td>White alone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>81643</td>
      <td>003</td>
      <td>1</td>
      <td>White alone or in combination with one or more...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>53927</td>
      <td>004</td>
      <td>1</td>
      <td>Black or African American alone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>55195</td>
      <td>005</td>
      <td>1</td>
      <td>Black or African American alone or in combinat...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>363</th>
      <td>NaN</td>
      <td>80358</td>
      <td>920</td>
      <td>1</td>
      <td>Ukraine</td>
      <td>1</td>
    </tr>
    <tr>
      <th>364</th>
      <td>NaN</td>
      <td>81340</td>
      <td>921</td>
      <td>1</td>
      <td>USSR</td>
      <td>1</td>
    </tr>
    <tr>
      <th>365</th>
      <td>NaN</td>
      <td>102394</td>
      <td>930</td>
      <td>1</td>
      <td>Oceania</td>
      <td>1</td>
    </tr>
    <tr>
      <th>366</th>
      <td>NaN</td>
      <td>141182</td>
      <td>931</td>
      <td>1</td>
      <td>Australia and New Zealand Subregion</td>
      <td>1</td>
    </tr>
    <tr>
      <th>367</th>
      <td>NaN</td>
      <td>149067</td>
      <td>932</td>
      <td>1</td>
      <td>Australia</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>368 rows × 6 columns</p>
</div>




```python
df = mdf.drop(['NAME', 'us_x', 'us_y'], axis=1)
```

## Plotting


```python
# Define label groups
ethnicities = [
    'Total population', 'White alone', 'Black or African American alone',
    'Hispanic or Latino (of any race)', 'American Indian and Alaska Native alone',
    'Two or More Races'
]
asia = ['Taiwanese alone', 'Asian Indian alone', 'Pakistani alone', 'Chinese alone', 'Filipino alone']
europe = ['English', 'Spaniard']
americas = ['Brazilian', 'Mexican']
africa = ['Nigerian', 'Egyptian', 'Congolese', 'Somali']
me = ['Iranian', 'Iraqi', 'Palestinian']

labels = ethnicities + asia + europe + americas + africa + me

# Filter for just those groups
selected = df[df['POPGROUP_LABEL'].isin(labels)].copy()
```


```python
# Region‐type map for coloring
region_map = {lbl: 'Ethnicity'      for lbl in ethnicities}
region_map.update({lbl: 'Asia'       for lbl in asia})
region_map.update({lbl: 'Europe'     for lbl in europe})
region_map.update({lbl: 'Americas'   for lbl in americas})
region_map.update({lbl: 'Africa'     for lbl in africa})
region_map.update({lbl: 'Middle East'for lbl in me})
selected['GroupType'] = selected['POPGROUP_LABEL'].map(region_map)

# Build DisplayLabel (“… ancestry” for each ancestry group)
display_map = {}
for lbl in ethnicities:
    display_map[lbl] = lbl
for lbl in asia + europe + americas + africa + me:
    base = lbl[:-6] if lbl.endswith(' alone') else lbl
    display_map[lbl] = f"{base} ancestry"
selected['DisplayLabel'] = selected['POPGROUP_LABEL'].map(display_map)
```


```python
# Sort descending by income
selected_sorted = selected.sort_values('S0201_214E', ascending=False)
```


```python
# Choose a colormap (tab10 has at least 6 distinct colors)
palette = plt.get_cmap('tab10').colors

region_order = [
    'Asia',
    'Middle East',
    'Europe',
    'Africa',
    'Americas',
    'Ethnicity', 
]

# Build a stable color map
color_map = {
    region: palette[i]
    for i, region in enumerate(region_order)
}
```


```python
# Plot
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

fig, ax = plt.subplots(figsize=(20, 16))

bars = ax.bar(
    selected_sorted['DisplayLabel'],
    selected_sorted['S0201_214E'],
    color=[color_map[gt] for gt in selected_sorted['GroupType']]
)

# X‐axis labels
# ax.set_xticklabels(selected_sorted['DisplayLabel'], rotation=90, ha='right')

positions = range(len(selected_sorted))
ax.set_xticks(positions)
ax.set_xticklabels(
    selected_sorted['DisplayLabel'],
    rotation=90,
    ha='right'
)

# Axis labels & title
ax.set_xlabel("Population group (self-reported U.S. ancestry)")
ax.set_ylabel("Median household income (USD)")
ax.set_title("Median Income by Selected Population Groups", pad=20)
fig.suptitle(
    "Self-reported ancestry of U.S. households only",
    fontsize="small", y=0.93, color="gray"
)

# Annotate bars
ax.bar_label(
    bars,
    labels=[f"${v:,}" for v in selected_sorted['S0201_214E']],
    padding=3,
    rotation=90,
    fontsize='small'
)

# Grid lines
ax.grid(axis='y', linestyle='--', alpha=0.5)

# U.S. median reference line
total_val = selected_sorted.loc[
    selected_sorted['POPGROUP_LABEL']=='Total population',
    'S0201_214E'
].iloc[0]
ax.axhline(total_val, linestyle='--', linewidth=1.2, alpha=0.7, color='gray')

# Label the reference line at right edge
n = len(selected_sorted)
ax.text(
    n - 0.5, total_val,
    f"U.S. median: ${total_val:,}",
    va='bottom', ha='right',
    fontsize='small', color='gray'
)

# Legend
unique_gt = selected_sorted['GroupType'].unique()
handles = [Patch(color=color_map[gt], label=gt) for gt in unique_gt]
ax.legend(
    handles=handles,
    title="Region / race category",
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()
```


    
![png]({{site.baseurl}}/assets/img/2025-05-10-visualizing-us-census-bureau-data_files/2025-05-10-visualizing-us-census-bureau-data_27_0.png)
    



```python
# Save
plt.savefig(
    'median_income_by_group_vertical.png',
    dpi=300,
    bbox_inches='tight'
)
```


    <Figure size 640x480 with 0 Axes>


## Larger Graph

Let's look at more groups. To do so, we'll need to flip the graph to allow for more room.


```python
# Define label groups
ethnicities = [
    'Total population', 'White alone', 'Black or African American alone',
    'Hispanic or Latino (of any race)', 'American Indian and Alaska Native alone',
    'Native Hawaiian and Other Pacific Islander alone', 'Some Other Race alone',
    'Two or More Races'
]

asia = [
    'Asian alone', 'Taiwanese alone', 'Asian Indian alone', 'Pakistani alone',
    'Chinese alone', 'Japanese alone', 'Korean alone', 'Vietnamese alone',
    'Filipino alone', 'Bangladeshi alone', 'Indonesian alone', 'Hmong alone',
    'Cambodian alone', 'Thai alone', 'Laotian alone',
    # +10 new
    'Sri Lankan alone', 'Nepalese alone', 'Bhutanese alone', 'Mongolian alone',
    'Tibetan alone', 'Kazakh alone', 'Uzbek alone', 'Kyrgyz alone',
    'Afghan alone', 'Malaysian alone'
]

europe = [
    'French (except Basque)', 'German', 'English', 'Spaniard', 'Italian',
    'Dutch', 'Swedish', 'Norwegian', 'Greek', 'Polish', 'Romanian',
    'Hungarian', 'Belgian',
    # +12 new
    'Russian', 'Ukrainian', 'Portuguese', 'Swiss', 'Austrian', 'Czech',
    'Bulgarian', 'Belarusian', 'Finnish', 'Irish', 'Scottish', 'Welsh'
]

americas = [
    'Brazilian', 'Mexican', 'Puerto Rican', 'Cuban', 'Dominican', 'Haitian',
    'Jamaican', 'Trinidadian and Tobagonian', 'Colombian', 'Venezuelan',
    'Peruvian', 'Chilean', 'Uruguayan', 'Bolivian', 'Ecuadorian',
    'Costa Rican', 'Guatemalan', 'Honduran', 'Salvadoran', 'Panamanian'
]

africa = [
    'Nigerian', 'Egyptian', 'Congolese', 'Nigerien', 'Malian', 'Botswana',
    'Kenya', 'Ethiopian', 'Ghanaian', 'Somali', 'Cabo Verdean',
    'South African', 'Moroccan', 'Algerian', 'Tunisian', 'Senegalese',
    'Ugandan', 'Sudanese', 'Rwandan', 'Cameroonian', 'Gabonese',
    'Zambian', 'Zimbabwean'
]

me = [
    'Iranian', 'Iraqi', 'Syrian', 'Lebanese', 'Jordanian', 'Turkish',
    'Palestinian',
    'Israeli', 'Saudi Arabian', 'Emirati', 'Qatari', 'Kuwaiti', 'Omani'
]

labels = ethnicities + asia + europe + americas + africa + me
```


```python
# Filter to only selected labels
selected = df[df['POPGROUP_LABEL'].isin(labels)].copy()

# Map to group types for coloring
region_map = {lbl: 'Ethnicity' for lbl in ethnicities}
region_map.update({lbl: 'Asia'       for lbl in asia})
region_map.update({lbl: 'Europe'     for lbl in europe})
region_map.update({lbl: 'Americas'   for lbl in americas})
region_map.update({lbl: 'Africa'     for lbl in africa})
region_map.update({lbl: 'Middle East'for lbl in me})
selected['GroupType'] = selected['POPGROUP_LABEL'].map(region_map)

# Build a display‐name map that adds “ancestry” to ancestry groups
display_map = {}

# Leave the broad race/ethnicity categories unchanged
for lbl in ethnicities:
    display_map[lbl] = lbl

# For everything else, strip any trailing " alone" then add " American"
for lbl in asia + europe + americas + africa + me:
    base = lbl[:-6] if lbl.endswith(' alone') else lbl
    display_map[lbl] = f"{base} ancestry"

# Apply it
selected['DisplayLabel'] = selected['POPGROUP_LABEL'].map(display_map)

```

Now let's plot it.


```python
# Sort ascending so the largest ends up at the top of the chart
selected_sorted = selected.sort_values('S0201_214E', ascending=True)

# Create the figure—taller than it is wide to fit long category names
plt.figure(figsize=(12, 20))

# Draw horizontal bars
bars = plt.barh(
    selected_sorted['DisplayLabel'],
    selected_sorted['S0201_214E'],
    color=[color_map[gt] for gt in selected_sorted['GroupType']]
)

# Labels & title
plt.xlabel('Median Household Income (S0201_214E)')
plt.title('Median Household Income by Selected Population Groups (2023)')

# Annotate each bar with its value
plt.bar_label(
    bars,
    labels=[f"${v:,}" for v in selected_sorted['S0201_214E']],
    padding=5,
    fontsize='small'
)

# Light grid for the x-axis
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Reference line at the “Total population” median
total_val = selected_sorted.loc[
    selected_sorted['POPGROUP_LABEL']=='Total population',
    'S0201_214E'
].iloc[0]
plt.axvline(total_val, linestyle='--', color='gray', alpha=0.7)
plt.text(
    total_val, 
    -1,  # slightly above the top bar
    f"U.S. median: ${total_val:,}",
    va='bottom',
    ha='right',
    color='gray',
    fontsize='small'
)

# Legend
handles = [Patch(color=color_map[gt], label=gt) for gt in unique_gt]
plt.legend(
    handles=handles,
    title='Group Type',
    bbox_to_anchor=(1.05,1),
    loc='upper left'
)

# Plot

plt.tight_layout()

plt.savefig(
    'median_income_by_group_horizontal.png',  # path & name
    dpi=300,                       # resolution
    bbox_inches='tight'           # trim extra whitespace
)

plt.show()

```


    
![png]({{site.baseurl}}/assets/img/2025-05-10-visualizing-us-census-bureau-data_files/2025-05-10-visualizing-us-census-bureau-data_34_0.png)
    



```python

```


```python

```

## Visualizing Percentage Differences


```python
YEAR     = 2023                                          # latest SPP with this slice
DATASET  = f"https://api.census.gov/data/{YEAR}/acs/acs1/spp"
FIELDS   = "NAME,S0201_214E,POPGROUP"                    # median household income + population group
URL      = (f"{DATASET}?get={FIELDS}"
            f"&for=us:1&key={API_KEY}")
resp = requests.get(URL, timeout=30)
rows = resp.json()

# Convert to pandas DataFrame
df = pd.DataFrame(rows[1:], columns=rows[0])
df['S0201_214E'] = pd.to_numeric(df['S0201_214E'], errors='coerce')
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
      <th>S0201_214E</th>
      <th>POPGROUP</th>
      <th>us</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>United States</td>
      <td>77719</td>
      <td>001</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>United States</td>
      <td>82531</td>
      <td>002</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>United States</td>
      <td>81643</td>
      <td>003</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>53927</td>
      <td>004</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>United States</td>
      <td>55195</td>
      <td>005</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>363</th>
      <td>United States</td>
      <td>81340</td>
      <td>921</td>
      <td>1</td>
    </tr>
    <tr>
      <th>364</th>
      <td>United States</td>
      <td>102394</td>
      <td>930</td>
      <td>1</td>
    </tr>
    <tr>
      <th>365</th>
      <td>United States</td>
      <td>141182</td>
      <td>931</td>
      <td>1</td>
    </tr>
    <tr>
      <th>366</th>
      <td>United States</td>
      <td>149067</td>
      <td>932</td>
      <td>1</td>
    </tr>
    <tr>
      <th>367</th>
      <td>United States</td>
      <td>50883</td>
      <td>946</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>368 rows × 4 columns</p>
</div>




```python
!curl -o census_popgroup_dict.py https://gist.githubusercontent.com/jss367/44e041c913f87a11b2830e01e295c241/raw/14423b1e2ffaad75afd641c81e7435065d2c43d6/gistfile1.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  6427  100  6427    0     0  22379      0 --:--:-- --:--:-- --:--:-- 22393



```python
from census_popgroup_dict import code_to_population
```


```python
df['population'] = df['POPGROUP'].apply(lambda x: code_to_population.get(str(x), "Unknown code"))
```


```python
# Clean up
df = df[df['population'] != 'Unknown code']
df = df.drop_duplicates()
```


```python
# Sort the dataframe by median income in descending order
df_sorted = df.sort_values('S0201_214E', ascending=False)

# Calculate income relative to total population
total_pop_income = df[df['POPGROUP'] == '001']['S0201_214E'].iloc[0]
df['income_ratio'] = df['S0201_214E'] / total_pop_income
df['percent_of_total'] = df['income_ratio'] * 100 - 100

# Create a simplified dataframe for better visualization
# Extract race/ethnicity categories (top level identifiers)
main_categories = ['001', '002', '003', '004', '006', '010', '013', '016', '019', '022', '023', '026', '029', '043', '046', '060', '110', '112', '113', '116', '118', '120', '125']
df_main = df[df['POPGROUP'].isin(main_categories)].copy()
```


```python
df_main
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
      <th>population</th>
      <th>income_ratio</th>
      <th>percent_of_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>United States</td>
      <td>77719</td>
      <td>001</td>
      <td>1</td>
      <td>Total Population</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>United States</td>
      <td>82531</td>
      <td>002</td>
      <td>1</td>
      <td>White alone</td>
      <td>1.061915</td>
      <td>6.191536</td>
    </tr>
    <tr>
      <th>2</th>
      <td>United States</td>
      <td>81643</td>
      <td>003</td>
      <td>1</td>
      <td>White alone or in combination with one or more...</td>
      <td>1.050490</td>
      <td>5.048958</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>53927</td>
      <td>004</td>
      <td>1</td>
      <td>Black or African American alone</td>
      <td>0.693872</td>
      <td>-30.612849</td>
    </tr>
    <tr>
      <th>5</th>
      <td>United States</td>
      <td>61061</td>
      <td>006</td>
      <td>1</td>
      <td>American Indian and Alaska Native alone (300, ...</td>
      <td>0.785664</td>
      <td>-21.433626</td>
    </tr>
    <tr>
      <th>8</th>
      <td>United States</td>
      <td>105393</td>
      <td>016</td>
      <td>1</td>
      <td>Chinese alone</td>
      <td>1.356078</td>
      <td>35.607766</td>
    </tr>
    <tr>
      <th>13</th>
      <td>United States</td>
      <td>79639</td>
      <td>060</td>
      <td>1</td>
      <td>Native Hawaiian and Other Pacific Islander alo...</td>
      <td>1.024704</td>
      <td>2.470438</td>
    </tr>
    <tr>
      <th>27</th>
      <td>United States</td>
      <td>50484</td>
      <td>110</td>
      <td>1</td>
      <td>Black or African American; American Indian and...</td>
      <td>0.649571</td>
      <td>-35.042911</td>
    </tr>
    <tr>
      <th>29</th>
      <td>United States</td>
      <td>58541</td>
      <td>113</td>
      <td>1</td>
      <td>Black or African American; Some Other Race</td>
      <td>0.753239</td>
      <td>-24.676077</td>
    </tr>
    <tr>
      <th>30</th>
      <td>United States</td>
      <td>72490</td>
      <td>116</td>
      <td>1</td>
      <td>American Indian and Alaska Native; Some Other ...</td>
      <td>0.932719</td>
      <td>-6.728085</td>
    </tr>
    <tr>
      <th>32</th>
      <td>United States</td>
      <td>76363</td>
      <td>118</td>
      <td>1</td>
      <td>Asian; Some Other Race</td>
      <td>0.982553</td>
      <td>-1.744747</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Income ratio compared to total population (as percentage difference)
plt.figure(figsize=(14, 10))
df_main_sorted = df_main.sort_values('percent_of_total')
bars = plt.barh(df_main_sorted['population'], df_main_sorted['percent_of_total'])

# Color bars based on whether they're above or below average
for i, bar in enumerate(bars):
    if df_main_sorted.iloc[i]['percent_of_total'] >= 0:
        bar.set_color('green')
    else:
        bar.set_color('red')

plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
plt.xlabel('Percentage Difference from U.S. Total Population Median Income (%)')
plt.ylabel('Population Group')
plt.title('Income Gap: How Much Each Group Earns Relative to Total Population (2023)')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add percentage values as labels
for i, v in enumerate(df_main_sorted['percent_of_total']):
    text_color = 'black'
    plt.text(v + (2 if v >= 0 else -2), i, f'{v:.1f}%', va='center', ha='left' if v >= 0 else 'right', color=text_color)

plt.tight_layout()
plt.savefig('income_gap_percentage.png')
plt.show()

```


    
![png]({{site.baseurl}}/assets/img/2025-05-10-visualizing-us-census-bureau-data_files/2025-05-10-visualizing-us-census-bureau-data_46_0.png)
    



```python

```
