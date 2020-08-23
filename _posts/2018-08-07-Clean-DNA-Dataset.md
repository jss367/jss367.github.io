---
layout: post
title: "DNA Splice Junctions I: Cleaning and Preparing the Data"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/falcon.jpg"
tags: [Python, Data Cleaning, Pandas, Biology]
---

Splice junctions are locations on sequences of DNA or RNA where superflous sections are removed when proteins are created. After the splice, a section, known as the intron, is removed and the remaining sections, known as the exons, are joined together. Being able to identify these sequences of DNA is useful but time-consuming. This begs the question: Can spliced sections of DNA be determined with machine learning? 

In the next two posts we'll try to do exactly that. To do this, we're going to use the [UCI Splice-junction Gene Sequence dataset](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29). It consist of sequences of DNA that contain either the part of the DNA retained after splicing, the part that was spliced out, or neither. Our problem is to distinguish between these cases.

In this post, I'm going to focus on cleaning and preparing the data set. In the next I'll walk through logistic regression to show how it works.

<b>Table of contents</b>
* TOC
{:toc}

![splice](/assets/img/splice.jpg "Picture of RNA splice")
Image from Wikipedia


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pandas_profiling import ProfileReport
```

## Getting the data

This dataset was prepared by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), so we can download it from their page and read it using pandas.


```python
df = pd.read_csv(r"datasets/DNA/splice.data",
    names=["Class", "Instance", "Sequence"])
```

## Examining the data


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
      <th>Class</th>
      <th>Instance</th>
      <th>Sequence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3190</td>
      <td>3190</td>
      <td>3190</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>3</td>
      <td>3178</td>
      <td>3092</td>
    </tr>
    <tr>
      <th>top</th>
      <td>N</td>
      <td>HUMMYLCA-DONOR-644</td>
      <td>GCCGTGGTTTTTTTGCTTCACCACCCTGAGGTGCG...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1655</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Looks like we have only three different classes. Almost all the instances and sequences are unique though. Let's look at the classes.

We can also use an amazing tool called [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling) to learn a lot about the dataset.


```python
profile = ProfileReport(df, title="Pandas Profiling Report")
```


```python
#profile.to_widgets()
```

### Class

Let's see what our three classes are and how many we have of each type.


```python
Y = df['Class']
```


```python
Y.groupby(Y).count()
```




    Class
    EI     767
    IE     768
    N     1655
    Name: Class, dtype: int64



The labels are currently strings of either 'IE', 'EI', or 'N'. They're well-balanced between EI and IE, although most of the samples are neither. To use these labels to train an algorithm, we'll need to encode them as integers.


```python
le = LabelEncoder()
le.fit(Y)
# record the label encoder mapping
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
y = le.transform(Y)
```

Here's the encoding we've done:


```python
le_name_mapping
```




    {'EI': 0, 'IE': 1, 'N': 2}




```python
y.shape
```




    (3190,)



Now our labels are an (N,) shape array of 0, 1, and 2. That's all we need to do with them. Let's dig into the instance names.

### Instance

Let's look at some of them and see if we can find a pattern. Let's look at the first 20 then another random sample of 20.


```python
df['Instance'][:20]
```




    0             ATRINS-DONOR-521
    1             ATRINS-DONOR-905
    2             BABAPOE-DONOR-30
    3            BABAPOE-DONOR-867
    4           BABAPOE-DONOR-2817
    5           CHPIGECA-DONOR-378
    6           CHPIGECA-DONOR-903
    7          CHPIGECA-DONOR-1313
    8          GCRHBBA1-DONOR-1260
    9          GCRHBBA1-DONOR-1590
    10          GCRHBBA6-DONOR-461
    11          GCRHBBA6-DONOR-795
    12         GIBHBGGL-DONOR-2278
    13         GIBHBGGL-DONOR-2624
    14         GIBHBGGL-DONOR-7198
    15         GIBHBGGL-DONOR-7544
    16         HUMA1ATP-DONOR-1972
    17         HUMA1ATP-DONOR-7932
    18         HUMA1ATP-DONOR-9653
    19        HUMA1ATP-DONOR-11057
    Name: Instance, dtype: object




```python
df['Instance'].sample(20, random_state=0)
```




    982         HUMCP21OHC-ACCEPTOR-1015
    702                HUMTNFA-DONOR-952
    461              HUMLYL1B-DONOR-2722
    480             HUMMHB27B-DONOR-1478
    298                HUMFOS-DONOR-2005
    1495          HUMTUBAG-ACCEPTOR-2034
    3055                 HUMTHR-NEG-1501
    3163                 HUMZFY-NEG-2341
    543              HUMMRP8A-DONOR-1504
    1615                 HUMADH2E2-NEG-1
    1748                   HUMASA-NEG-61
    1180           HUMIL1B-ACCEPTOR-1506
    1503          HUMUBILP-ACCEPTOR-1488
    1652                 HUMALDH03-NEG-1
    1875               HUMCGPRA-NEG-1981
    847          HUMALBGC-ACCEPTOR-13672
    661               HUMSODA-DONOR-3967
    1760             HUMATP1A2-NEG-11221
    2078               HUMFGRINT-NEG-421
    33              HUMACCYBB-DONOR-2438
    Name: Instance, dtype: object



It looks like we have a sequence of letters that often repeat in nearby rows; followed by one of the three words DONOR, ACCEPTOR, and NEG; followed by a number. We could parse the different parts out and use them individually as features. The dashes make this very easy with pandas' built-in regular expression matching.


```python
df['Instance_Prefix'] = df['Instance'].str.extract("(.*)-(\w*)-(\d*)")[0]
df['Instance_Donor'] = df['Instance'].str.extract("(.*)-(\w*)-(\d*)")[1]
df['Instance_Number'] = df['Instance'].str.extract("(.*)-(\w*)-(\d*)")[2]
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
      <th>Instance_Prefix</th>
      <th>Instance_Donor</th>
      <th>Instance_Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EI</td>
      <td>ATRINS-DONOR-521</td>
      <td>CCAGCTGCATCACAGGAGGCCAGCGAGCAGG...</td>
      <td>ATRINS</td>
      <td>DONOR</td>
      <td>521</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EI</td>
      <td>ATRINS-DONOR-905</td>
      <td>AGACCCGCCGGGAGGCGGAGGACCTGCAGGG...</td>
      <td>ATRINS</td>
      <td>DONOR</td>
      <td>905</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EI</td>
      <td>BABAPOE-DONOR-30</td>
      <td>GAGGTGAAGGACGTCCTTCCCCAGGAGCCGG...</td>
      <td>BABAPOE</td>
      <td>DONOR</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EI</td>
      <td>BABAPOE-DONOR-867</td>
      <td>GGGCTGCGTTGCTGGTCACATTCCTGGCAGGT...</td>
      <td>BABAPOE</td>
      <td>DONOR</td>
      <td>867</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EI</td>
      <td>BABAPOE-DONOR-2817</td>
      <td>GCTCAGCCCCCAGGTCACCCAGGAACTGACGTG...</td>
      <td>BABAPOE</td>
      <td>DONOR</td>
      <td>2817</td>
    </tr>
  </tbody>
</table>
</div>



#### Prefix

Let's see how many unique prefixes we have


```python
len(df['Instance_Prefix'].unique())
```




    1614



That's a lot. Sometimes with categorical data like this, we would use one-hot encoding, where we make each instance its own column and give the value of 1 if it's that type and 0 otherwise. But that would result in a large sparse matrix. That's not necessarily a problem, but we'll put it aside for now and maybe use it later.

Now let's look at the donor part.

#### Donor


```python
df['Instance_Donor'].unique()
```




    array(['DONOR', 'ACCEPTOR', 'NEG'], dtype=object)



OK, just the three we saw before. That's good.

Let's encode these in our dataset using one-hot encoding. We could use SKLearn's labelEncoder and then One Hot, but pandas has something called `get_dummies` built-in that works with strings.


```python
donor_one_hot_df = pd.get_dummies(df['Instance_Donor'])
donor_one_hot_df.sample(10, random_state=0)
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
      <th>ACCEPTOR</th>
      <th>DONOR</th>
      <th>NEG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>982</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>702</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>461</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>480</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>298</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1495</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3055</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3163</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>543</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1615</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Now we'll convert it to a numpy array.


```python
X_donor = np.asarray(donor_one_hot_df)
X_donor.shape
```




    (3190, 3)



#### Number

I don't think the `Instance_Number` is going to help us, so I'm going to ignore it for the moment. Now let's look at the sequence.

### Sequence

#### Finding unique values

I would assume the Sequence consists of only A, C, T, and G, but we'll double check just to be sure.


```python
df['Sequence'][0]
```




    '               CCAGCTGCATCACAGGAGGCCAGCGAGCAGGTCTGTTCCAAGGGCCTTCGAGCCAGTCTG'



The first example looks good except for the extra white space. We'll use `.strip()` to remove it. Let's go through the rest.


```python
def find_letters(row):
    return set(row.strip())
```

Let's turn the pandas series `df['Sequence']` into a pandas series of the set of each row. This will remove duplicate values.


```python
set_series = df['Sequence'].apply(find_letters)
```

Find the union of the sets to show all the unique values.


```python
set.union(*set_series)
```




    {'A', 'C', 'D', 'G', 'N', 'R', 'S', 'T'}



OK, we have a lot more letters than I expected. Let's see how common they are and how they're used.

#### Exploring unexpected values


```python
set_series[set_series.str.contains('D', regex=False)]
```




    1247    {T, G, A, D, C}
    2578    {T, G, A, D, C}
    Name: Sequence, dtype: object



Only two instances, that's good. Let's look at one of them.


```python
df.loc[1247].Sequence.strip()
```




    'DGACGGGGCTGACCGCGGGGGCGGGTCCAGGGTCTCACACCCTCCAGAATATGTATGGCT'



After consulting the original documentation again, it looks like these letters are used when there's uncertainty in the actual letter:

- D: A or G or T 
- N: A or G or C or T 
- S: C or G 
- R: A or G

Let's look at the other letters as well.


```python
set_series[set_series.str.contains('N', regex=False)]
```




    107     {T, G, N, C, A}
    239     {T, G, N, A, C}
    365     {T, G, N, C, A}
    366     {T, G, N, A, C}
    485     {T, G, N, C, A}
    1804    {T, G, N, A, C}
    2069    {T, G, N, A, C}
    2135    {T, G, N, A, C}
    2636    {T, G, N, A, C}
    2637    {T, G, N, A, C}
    2910    {T, G, N, A, C}
    Name: Sequence, dtype: object




```python
df.loc[107].Sequence.strip()
```




    'CACACAGGGCACCCCCTCANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN'




```python
set_series[set_series.str.contains('R', regex=False)]
```




    1440    {T, G, R, A, C}
    Name: Sequence, dtype: object




```python
df.loc[1440].Sequence.strip()
```




    'ATACCCCTTTTCACTTTCCCCACCTCTTAGGGTARTCAGTACTGGCGCTTTGAGGATGGT'




```python
set_series[set_series.str.contains('S', regex=False)]
```




    1441    {T, S, G, C, A}
    Name: Sequence, dtype: object




```python
df.loc[1441].Sequence.strip()
```




    'CCCTCCTAATGCCCACCATCCCGTCCTCAGGGAAASAGTACTGGGAGTACCAGTTCCAGC'



OK, there aren't too many rows with these missing values. We could remove them, but if there's enough information in the rest of the sequence to distinguish the class, we would be throwing away useful data. If there's not, the instance won't have much effect on the model. So I'm going to keep them in.

#### Checking the size

The dataset claims that every row has 60 characters (30 before and 30 after the possible splice. Let's check to make sure that's true.


```python
df['Sequence'].str.strip().map(len).unique()
```




    array([60], dtype=int64)



OK, looks good.

#### Cleaning and Transforming

Now we have the DNA sequences represented by a large string of letters. To use the sequence in machine learning, we're going to need to remove the whitespace and separate each letter into a distinct place on a list. We'll also need to convert the letters to integers. We'll do that now.


```python
letter_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'D': 4, 'N': 5, 'R': 6, 'S': 7}
```


```python
def convert_base_to_num(bases):
    return [letter_mapping[letter] for letter in bases.strip()]
```


```python
df['Sequence_list'] = df['Sequence'].apply(convert_base_to_num)
```


```python
df['Sequence_list'].head()
```




    0    [1, 1, 0, 2, 1, 3, 2, 1, 0, 3, 1, 0, 1, 0, 2, ...
    1    [0, 2, 0, 1, 1, 1, 2, 1, 1, 2, 2, 2, 0, 2, 2, ...
    2    [2, 0, 2, 2, 3, 2, 0, 0, 2, 2, 0, 1, 2, 3, 1, ...
    3    [2, 2, 2, 1, 3, 2, 1, 2, 3, 3, 2, 1, 3, 2, 2, ...
    4    [2, 1, 3, 1, 0, 2, 1, 1, 1, 1, 1, 0, 2, 2, 3, ...
    Name: Sequence_list, dtype: object



Now we've converted the letters to integers. We could continue with this and train the model, but that has downsides. Even though the values are integers, the data are still categorical because each number represents a category, not a quantitative or ordinal value. For example, A is 0, C is 1, and G is 2, but this doesn't mean that A is closer to or more similar to C than it is to G. But our algorithm won't know that and could be misled, thinking the data are quantitative when they're not.

To avoid that, we'll use one-hot encoding.

We have to be careful with data types. We currently have a bunch of integers inside a list within a pandas series. We want to split those lists so that each individual integer is in its own column. To do that, we'll convert the pandas series into an ndarray.


```python
X_sequence = np.array(df['Sequence_list'].values.tolist())
```

One-hot encoding our matrix will change the shape of it. Let's check the shape now so we can compare.


```python
X_sequence.shape
```




    (3190, 60)




```python
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_sequence)
X_one_hot = X_encoded.toarray()
```


```python
X_one_hot.shape
```




    (3190, 287)



This array is ready to go. Let's look at the labels next.

## Putting it all back together

I was originally going to combine the `X_donor` array with `X_one_hot` as the final X-values in the dataset. But after testing some models on it, it looks like the `X_donor` data too accurately predicts the label. Using just that information I can predict 100% of the labels, so the sequence data becomes meaningless. To me, that means that the Instance data shouldn't actually be used to predict the labels. So our final dataset will just be the sequence data.


```python
#X = np.append(X_donor, X_sequence, axis=1)
X = X_one_hot
```

Now we save it off into a csv file to examine in the [next notebook](https://jss367.github.io/DNA-Logistic-Regression.html).


```python
np.save('dna_cleaner.npy', X)
```


```python
np.save('dna_clean_labels', y)
```
