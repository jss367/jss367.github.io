---
layout: post
title: "Visualizing a Machine Learning Algorithm"
description: "Visualizing a decision trees algorithm using the mushroom dataset"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/boobook.jpg"
tags: [Decision Trees, Python, Pandas, SKLearn, Data Visualization, Machine Learning]
---

What goes on inside the black box of a machine learning algorithm? While it may be impossible for a human to understand precisely why a large neural network produced the results it did, some algorithms are far more transparent. Decision trees are just such an example of a machine learning algorithm whose results can be understood by people.

To explore the power of decision trees, I'll use them to attempt to classify mushrooms into either poisonous or edible based on their look, smell, and habitat. The Audubon Society Field Guide to North American Mushrooms (1981) recorded these attributes for each of 8124 different mushrooms. As well as the toxicity, they came up with 22 other ways to characterize the mushrooms, such as cap shape, odor, and gill size. Credit for publishing [the dataset](https://archive.ics.uci.edu/ml/datasets/mushroom) goes to: Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.


```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn import tree
import pydotplus
import collections
```


```python
# There is a bug in sklearn which makes it give an error the way we reverse the transform
# There is a pull request to fix this bug in the next version, but for now have to suppress warnings
import warnings
warnings.filterwarnings('ignore')
```

The attributes are all described on [the dataset's website](https://archive.ics.uci.edu/ml/datasets/mushroom), but they aren't included with the dataset. We'll have to tell pandas to skip the header and we'll add the column names manually afterward.


```python
df = pd.read_csv('agaricus-lepiota.data', header=-1)
```

Here are the attributes provided. The first is the toxicity, which is what we will be trying to predict from the other attributes.


```python
columns = ['toxicity', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 
           'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
           'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 
          'spore-print-color', 'population', 'habitat']
df.columns = columns
```

## Data exploration

Now we'll take a quick look at the data.


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
      <th>toxicity</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>t</td>
      <td>l</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>w</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



For toxicity, we have 'p' for poisonous and 'e' for edible. The other attributes are:
1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s 
2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s 
3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y 
4. bruises?: bruises=t,no=f 
5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s 
6. gill-attachment: attached=a,descending=d,free=f,notched=n 
7. gill-spacing: close=c,crowded=w,distant=d 
8. gill-size: broad=b,narrow=n 
9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y 
10. stalk-shape: enlarging=e,tapering=t 
11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? 
12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s 
13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s 
14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
16. veil-type: partial=p,universal=u 
17. veil-color: brown=n,orange=o,white=w,yellow=y 
18. ring-number: none=n,one=o,two=t 
19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z 
20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y 
21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y 
22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d


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
      <th>toxicity</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>...</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>6</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>...</td>
      <td>4</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>9</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>top</th>
      <td>e</td>
      <td>x</td>
      <td>y</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>b</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>w</td>
      <td>v</td>
      <td>d</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4208</td>
      <td>3656</td>
      <td>3244</td>
      <td>2284</td>
      <td>4748</td>
      <td>3528</td>
      <td>7914</td>
      <td>6812</td>
      <td>5612</td>
      <td>1728</td>
      <td>...</td>
      <td>4936</td>
      <td>4464</td>
      <td>4384</td>
      <td>8124</td>
      <td>7924</td>
      <td>7488</td>
      <td>3968</td>
      <td>2388</td>
      <td>4040</td>
      <td>3148</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 23 columns</p>
</div>



In total, we have 8124 samples and 22 different attributes (not counting toxicity). That should be enough to train a decision tree.


We also have a good class distribution. The most common toxicity category is 'e' for edible, but that's only 4208 samples. That means that there are 3196 poisonous samples in the dataset. That's close enough to 50-50, so we've got plenty of both categories.


## Data cleaning

The first thing to do is to check for missing data.


```python
df.isnull().sum()
```




    toxicity                    0
    cap-shape                   0
    cap-surface                 0
    cap-color                   0
    bruises                     0
    odor                        0
    gill-attachment             0
    gill-spacing                0
    gill-size                   0
    gill-color                  0
    stalk-shape                 0
    stalk-root                  0
    stalk-surface-above-ring    0
    stalk-surface-below-ring    0
    stalk-color-above-ring      0
    stalk-color-below-ring      0
    veil-type                   0
    veil-color                  0
    ring-number                 0
    ring-type                   0
    spore-print-color           0
    population                  0
    habitat                     0
    dtype: int64



It looks like there are no missing values. Beautiful! That should make things more manageable. We should also go through the data and make sure all the values are what we're expecting.


```python
for x in range(len(df.columns)):
    print(df.columns[x] + ":")
    print(df[df.columns[x]].unique())
    print("")
```

    toxicity:
    ['p' 'e']
    
    cap-shape:
    ['x' 'b' 's' 'f' 'k' 'c']
    
    cap-surface:
    ['s' 'y' 'f' 'g']
    
    cap-color:
    ['n' 'y' 'w' 'g' 'e' 'p' 'b' 'u' 'c' 'r']
    
    bruises:
    ['t' 'f']
    
    odor:
    ['p' 'a' 'l' 'n' 'f' 'c' 'y' 's' 'm']
    
    gill-attachment:
    ['f' 'a']
    
    gill-spacing:
    ['c' 'w']
    
    gill-size:
    ['n' 'b']
    
    gill-color:
    ['k' 'n' 'g' 'p' 'w' 'h' 'u' 'e' 'b' 'r' 'y' 'o']
    
    stalk-shape:
    ['e' 't']
    
    stalk-root:
    ['e' 'c' 'b' 'r' '?']
    
    stalk-surface-above-ring:
    ['s' 'f' 'k' 'y']
    
    stalk-surface-below-ring:
    ['s' 'f' 'y' 'k']
    
    stalk-color-above-ring:
    ['w' 'g' 'p' 'n' 'b' 'e' 'o' 'c' 'y']
    
    stalk-color-below-ring:
    ['w' 'p' 'g' 'b' 'n' 'e' 'y' 'o' 'c']
    
    veil-type:
    ['p']
    
    veil-color:
    ['w' 'n' 'o' 'y']
    
    ring-number:
    ['o' 't' 'n']
    
    ring-type:
    ['p' 'e' 'l' 'f' 'n']
    
    spore-print-color:
    ['k' 'n' 'u' 'h' 'w' 'r' 'o' 'y' 'b']
    
    population:
    ['s' 'n' 'a' 'v' 'y' 'c']
    
    habitat:
    ['u' 'g' 'm' 'd' 'p' 'w' 'l']
    
    

They all look good except for stalk root. For some reason, there's a question mark category. Let's see how many question marks there are. Hopefully, it's just one or two, and we can remove those.


```python
len(df[df['stalk-root']=='?'])
```




    2480



OK, it's a lot. The website says that those values are missing. Because so many samples are affected, I think the best thing to do would be to not include stalk root in the decision tree. Let's remove it from the dataset.


```python
df = df.drop(['stalk-root'], axis=1)
```

We'll also remove the label from the list of columns.


```python
del columns[columns.index('stalk-root')]
```

## Preparing the data

As we can see, the data are all categorical. We'll have to convert them to numbers before we use them. Let's go ahead and do that.


```python
d = defaultdict(LabelEncoder)

# Encoding the variable
numbered_df = df.apply(lambda x: d[x.name].fit_transform(x))
```


```python
numbered_df.head()
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
      <th>toxicity</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>...</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



If we need to, we can always reverse the labeling like this:


```python
# Inverse the encoded
numbered_df.apply(lambda x: d[x.name].inverse_transform(x)).head()
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
      <th>toxicity</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>t</td>
      <td>l</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>w</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



Before we feed the data through an algorithm, we'll have to remove the toxicity because that's the variables that the algorithms is trying to predict.


```python
labels = numbered_df['toxicity']
samples = numbered_df.drop('toxicity', axis=1)
```

Now we want to build a model and test it. But to get a fair test of how it would work in the real world, we can't use any of the testing data for training the model. So we'll withhold a random 20% of the dataset and train the model with the other 80%.


```python
X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size = 0.2, random_state = 0)
```


```python
print("Number of samples to train the model: " + str(len(X_train)))
print("Number of samples to test the model: " + str(len(X_test)))
```

    Number of samples to train the model: 6499
    Number of samples to test the model: 1625
    

## Building the model

Now we'll build the actual decision tree and train it on the data.


```python
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')



Let's see how well it did with the data we gave it.

## Testing the model


```python
print("The model is able to categorize {:.0%} of the training set.".format(accuracy_score(y_train, clf.predict(X_train))))
```

    The model is able to categorize 100% of the training set.
    

Wow. We were able to build a model that accurately categorized every mushroom into either poisonous or edible. That doesn't mean it can predict 100% of mushroom accurately. To determine how many it can, we'll have to see how well it does on the mushroom it has never seen before.


```python
print("The model correctly predicted {:.0%} of the test set.".format(accuracy_score(y_test, clf.predict(X_test))))
```

    The model correctly predicted 100% of the test set.
    

Even more impressive, we were able to correctly predict all 1625 samples that the model hadn't seen before. That's a pretty good model. Now, let's try to visualize it.

## Visualizing the model

We'll use [Graphviz](https://www.graphviz.org/) and [PyDotPlus](http://pydotplus.readthedocs.io/) to visualize the model.


```python
dot_data = tree.export_graphviz(clf,
                                feature_names=columns[1:],
                                out_file=None,
                                filled=True,
                                rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
nodes = graph.get_node_list()
colors = []
for node in nodes:
    if node.get_label():
        values = [int(ii) for ii in node.get_label().split('value = [')[1].split(']')[0].split(',')]
        values = [int(255 * v / sum(values)) for v in values]
        color = '#{:02x}{:02x}00'.format(values[1], values[0])
        colors.append(color)
        node.set_fillcolor(color)

graph.write_png('tree.png')
```




    True



Here is the resulting decision tree. It's a bit complex, but as it correctly determined the toxicity of every mushroom in the test set, I'm not complaining about it. 

![Decision tree]({{site.baseurl}}/assets/img/tree.png "Decision tree")

We can learn a lot from this tree. Of the 6499 samples, 3356 of them are edible, and 3143 of them are poisonous. The color of the box corresponds to the distribution of edible versus poisonous. The greener the box, the higher the proportion of those are edible. The redder the box, the higher the proportion are poisonous.

The first thing the decision tree decided to look at was gill color. It first asks the question of whether the gill color is in a category of 3.5 or less. These categories are zero-indexed, so this equates to the first four categories. Here are the categories:


```python
d['gill-color'].classes_
```




    array(['b', 'e', 'g', 'h', 'k', 'n', 'o', 'p', 'r', 'u', 'w', 'y'],
          dtype=object)



The first four correspond to buff, red, gray, and chocolate. Out of the 6499 samples that we used to train the decision tree, 2644 had gills of one of these colors, and the other 3855 did not.

Let's say it isn't one of those colors, so we go down the "False" path of the tree. Now we look at spore print color and ask if the category is less than 1.5, which would be categories 0 and 1.


```python
d['spore-print-color'].classes_
```




    array(['b', 'h', 'k', 'n', 'o', 'r', 'u', 'w', 'y'], dtype=object)



 Those correspond to black and chocolate. Let's say it is one of those types. So we go down the tree and look at the odor. This is the final deciding factor. Here are the categories for that attribute


```python
d['odor'].classes_
```




    array(['a', 'c', 'f', 'l', 'm', 'n', 'p', 's', 'y'], dtype=object)



From the graph, we can see that if the odor is in a category less than or equal to 3.5, which would be the first four categories, it is poisonous. Those categories are almond, creosote, foul, and anise. If it's not one of those, the mushroom is edible.

## Random forests

In a more complex scenario where we weren't able to predict with 100% accuracy from a single tree, we would build a "random forest" by combining many different decision trees. Then the trees can be weighed and combined in various ways to make an overall algorithm. The algorithm becomes harder to visualize at that point, but you can still learn a lot about the data from this technique.

We already know that we can accurately predict every mushroom in the dataset with a single decision tree, but let's put some constraints on it. Before, we could have as many forks as we wanted, allowing us to build a very complicated decision tree. Now, let's limit the number of forks to 4 and see how well we can do.


```python
forest_clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
forest_clf.fit(X_train, y_train)
print("The random forest model correctly predicted {:.0%} of the test set.".format(accuracy_score(y_test, forest_clf.predict(X_test))))
```

    The random forest model correctly predicted 99% of the test set.
    

Because of the way random forests bootstrap different subsets of the data, you can't have an "average" of the random forest that's a single decision tree. That means there's no way to visualize a random forest. To get a sense of what the model is doing, we can see what it considers the most important features, and see how what percentage of the predictive power they account for.


```python
feat_importance_dict = dict(zip(columns[1:], forest_clf.feature_importances_))
most_important = sorted(feat_importance_dict, key=feat_importance_dict.get, reverse=True)[:5]
for item in most_important:
    print("{item}: {value:.2%}".format(item=item, value=feat_importance_dict[item]))
```

    odor: 17.14%
    gill-size: 15.81%
    gill-color: 14.05%
    spore-print-color: 11.51%
    ring-type: 6.67%
    

So it turns out that odor is most important, followed by gill size and gill color.

## Conclusion

Decision trees are incredible algorithms and have the wonderful attribute that they can be understood by people. Even more, they can be combined together to form powerful random forest classifiers. Random forests are often considered one of the most important machine learning algorithms and are also one of the most powerful. In fact, a [study](http://jmlr.csail.mit.edu/papers/v15/delgado14a.html) in 2014 tested 179 algorithms and determined that the random forest was the best overall algorithm.