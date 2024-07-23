---
layout: post
title: "FastAI Tabular Data Tutorial"
description: "This tutorial describes how to work with the FastAI library for tabular data"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/signs.jpg"
tags: [FastAI, Python, Tabular Data]
---

This post is a tutorial on working with tabular data using FastAI. One of FastAI's biggest contributions in working with tabular data is the ease with which embeddings can be used for categorical variables. I have found that using embeddings for categorical variables results in significantly better models than the alternatives (e.g. one-hot encoding). I have found that the combination of embeddings and neural networks reach very high performance with tabular data.


```python
from fastai.tabular.all import *
from pyxtend import struct
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

We'll use the [UCI Adult Data Set](https://archive.ics.uci.edu/ml/datasets/Adult) where the task is to predict whether a person makes over 50k a year. FastAI makes downloading the dataset easy.


```python
path = untar_data(URLs.ADULT_SAMPLE)
```

Once it's downloaded we can load it into a DataFrame.


```python
df = pd.read_csv(path/'adult.csv')
```

Many times machine learning practitioners are dealing with datasets that have already been split into train and test sets. In this case we have all of the data, but I am going to split the data into a train and test split to simulate a pre-defined split.

## Part I


```python
train_df, test_df = train_test_split(df, random_state=42)
```

Let's take a look at the data.


```python
train_df.head(10)
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>42</td>
      <td>Private</td>
      <td>70055</td>
      <td>11th</td>
      <td>7.0</td>
      <td>Married-civ-spouse</td>
      <td>NaN</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>12181</th>
      <td>25</td>
      <td>Private</td>
      <td>253267</td>
      <td>Some-college</td>
      <td>10.0</td>
      <td>Married-civ-spouse</td>
      <td>Adm-clerical</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>1902</td>
      <td>36</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>18114</th>
      <td>53</td>
      <td>Self-emp-not-inc</td>
      <td>145419</td>
      <td>1st-4th</td>
      <td>2.0</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>7688</td>
      <td>0</td>
      <td>67</td>
      <td>Italy</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4278</th>
      <td>37</td>
      <td>State-gov</td>
      <td>354929</td>
      <td>Assoc-acdm</td>
      <td>12.0</td>
      <td>Divorced</td>
      <td>Protective-serv</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>12050</th>
      <td>25</td>
      <td>Private</td>
      <td>404616</td>
      <td>Masters</td>
      <td>14.0</td>
      <td>Married-civ-spouse</td>
      <td>Farming-fishing</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>99</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>14371</th>
      <td>20</td>
      <td>Private</td>
      <td>303565</td>
      <td>Some-college</td>
      <td>10.0</td>
      <td>Never-married</td>
      <td>Handlers-cleaners</td>
      <td>Own-child</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Germany</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>32541</th>
      <td>24</td>
      <td>Private</td>
      <td>241857</td>
      <td>Some-college</td>
      <td>10.0</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3362</th>
      <td>48</td>
      <td>Private</td>
      <td>398843</td>
      <td>Some-college</td>
      <td>10.0</td>
      <td>Separated</td>
      <td>Sales</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>19009</th>
      <td>46</td>
      <td>Private</td>
      <td>109227</td>
      <td>Some-college</td>
      <td>10.0</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>70</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>16041</th>
      <td>26</td>
      <td>Private</td>
      <td>171114</td>
      <td>Bachelors</td>
      <td>13.0</td>
      <td>Never-married</td>
      <td>Exec-managerial</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.describe()
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
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>24420.000000</td>
      <td>2.442000e+04</td>
      <td>24057.000000</td>
      <td>24420.000000</td>
      <td>24420.000000</td>
      <td>24420.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.578911</td>
      <td>1.895367e+05</td>
      <td>10.058361</td>
      <td>1066.490254</td>
      <td>86.502457</td>
      <td>40.393366</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.696620</td>
      <td>1.043135e+05</td>
      <td>2.580948</td>
      <td>7243.366967</td>
      <td>400.848415</td>
      <td>12.380526</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>1.228500e+04</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>1.183052e+05</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>1.784825e+05</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>2.366420e+05</td>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>1.455435e+06</td>
      <td>16.000000</td>
      <td>99999.000000</td>
      <td>4356.000000</td>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 24420 entries, 29 to 23654
    Data columns (total 15 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   age             24420 non-null  int64  
     1   workclass       24420 non-null  object 
     2   fnlwgt          24420 non-null  int64  
     3   education       24420 non-null  object 
     4   education-num   24057 non-null  float64
     5   marital-status  24420 non-null  object 
     6   occupation      24031 non-null  object 
     7   relationship    24420 non-null  object 
     8   race            24420 non-null  object 
     9   sex             24420 non-null  object 
     10  capital-gain    24420 non-null  int64  
     11  capital-loss    24420 non-null  int64  
     12  hours-per-week  24420 non-null  int64  
     13  native-country  24420 non-null  object 
     14  salary          24420 non-null  object 
    dtypes: float64(1), int64(5), object(9)
    memory usage: 3.0+ MB
    


```python
train_df['salary'].value_counts()
```




    <50k     18537
    >=50k     5883
    Name: salary, dtype: int64



The first thing to note is that there is missing data. We'll have to deal with that; fortunately, FastAI has tools that make this easy. Also, it looks like we have both continuous and categorical data. We'll split those apart so we can put the categorical data through embeddings. Also, the data is highly imbalanced. We could correct for this but I'll skip over that for now. The imbalance isn't so bad that it would completely stop the network from learning.

Note that the variable we're trying to predict, salary, is in the DataFrame. That's fine, we'll just need to tell `cont_cat_split` what the dependent variable is so it isn't included in the training variables.


```python
dep_var = 'salary'
```


```python
continuous_vars, categorical_vars = cont_cat_split(train_df, dep_var=dep_var)
```

The `cont_cat_split` function usually works well, but I always double check the results to see that they make sense.


```python
train_df[continuous_vars].nunique()
```




    age                  72
    fnlwgt            17545
    education-num        16
    capital-gain        116
    capital-loss         90
    hours-per-week       93
    dtype: int64




```python
train_df[categorical_vars].nunique()
```




    workclass          9
    education         16
    marital-status     7
    occupation        15
    relationship       6
    race               5
    sex                2
    native-country    41
    dtype: int64



Let's think about the data. One thing that sticks out to me is that `native-country` has 41 different unique values in the train set. This means there's a good chance there will be a new `native-country` in the test set (or after we deploy it!). This will be a problem if we use embeddings. There are ways to deal with unknown categories and embeddings but it's easiest to simply remove it.


```python
categorical_vars.remove('native-country')
```


```python
categorical_vars
```




    ['workclass',
     'education',
     'marital-status',
     'occupation',
     'relationship',
     'race',
     'sex']



Now we need to decide what preprocessing we need to do. We noted there is missing data, so we'll need to use `FillMissing` to clean that up. Also, we should always `Normalize` the data. Finally, we'll use `Categorify` to transform the categorical variables to be similar to `pd.Categorical`.


```python
preprocessing = [Categorify, FillMissing, Normalize]
```

We've already split our data because we're simulating that it's already been split for us. But we will still need to pass a splitter to `TabularPandas`, so we'll make one that puts everything in the train set and nothing in the validation set.


```python
def no_split(obj):
    """
    Put everything in the train set
    """
    return list(range(len(obj))), []
```


```python
splits = no_split(range_of(train_df))
```


```python
struct(splits)
```




    {tuple: [{list: [int, int, int, '...24420 total']}, {list: []}]}



There are a lot of things that don't work as well in FastAI if you don't have a validation set, like `get_preds` and the output from training, so I'm going to add it here. This is simple to do.


```python
full_df = pd.concat([train_df, test_df])
```


```python
val_indices = list(range(len(train_df),len(train_df) + len(test_df)))
```


```python
ind_splitter = IndexSplitter(val_indices)
```


```python
splits = ind_splitter(full_df) 
```

Now we need to create a `TabularPandas` for our data. A `TabularPandas` is wrapper for a pandas DataFrame where the continuous, categorical, and dependent variables are known. FastAI uses lots of inheritance, and the inheritances aren't always intuitive to me, so it's good to look at the method resolution order to get a sense of what the class is supposed to do. You can do so like this:


```python
TabularPandas.__mro__
```




    (fastai.tabular.core.TabularPandas,
     fastai.tabular.core.Tabular,
     fastcore.foundation.CollBase,
     fastcore.basics.GetAttr,
     fastai.data.core.FilteredBase,
     object)


If we just wanted to pass the train set, we would use `train_df` and `no_split(range_of(train_df))`. But we're going to pass the validation set as well, so we'll use `full_df` and `ind_splitter(full_df)`.

```python
df_wrapper = TabularPandas(full_df, procs=preprocessing, cat_names=categorical_vars, cont_names=continuous_vars,
                   y_names=dep_var, splits=splits)
```

Let's look at some examples to make sure they look right. All the data should be ready for deep learning.

If we wanted to get the data in the familiar `X_train, y_train, X_test, y_test` format a scikit-learn model, all we have to do is this:


```python
X_train, y_train = df_wrapper.train.xs, df_wrapper.train.ys.values.ravel()
X_test, y_test = df_wrapper.valid.xs, df_wrapper.valid.ys.values.ravel()
```

Now the data are in a DataFrame fully ready to be used in a scikit-learn or xgboost model. We can explore the data to see this.


```python
X_train.head()
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
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>0.249781</td>
      <td>-1.145433</td>
      <td>-1.193588</td>
      <td>-0.147240</td>
      <td>-0.215803</td>
      <td>0.372095</td>
    </tr>
    <tr>
      <th>12181</th>
      <td>5</td>
      <td>16</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>-0.991426</td>
      <td>0.610962</td>
      <td>-0.022445</td>
      <td>-0.147240</td>
      <td>4.529230</td>
      <td>-0.354868</td>
    </tr>
    <tr>
      <th>18114</th>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>1.052916</td>
      <td>-0.422942</td>
      <td>-3.145492</td>
      <td>0.914167</td>
      <td>-0.215803</td>
      <td>2.149115</td>
    </tr>
    <tr>
      <th>4278</th>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>12</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>-0.115280</td>
      <td>1.585564</td>
      <td>0.758317</td>
      <td>-0.147240</td>
      <td>-0.215803</td>
      <td>-0.193321</td>
    </tr>
    <tr>
      <th>12050</th>
      <td>5</td>
      <td>13</td>
      <td>3</td>
      <td>6</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>-0.991426</td>
      <td>2.061897</td>
      <td>1.539079</td>
      <td>-0.147240</td>
      <td>-0.215803</td>
      <td>4.733873</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the continuous variables are all normalized. This looks good! 


```python
y_train[:5]
```




    array([0, 1, 1, 0, 1], dtype=int8)



## Continuing with FastAI

If we wanted to use the data on a FastAI model, we'd need to create `DataLoaders`.


```python
batch_size = 128
dls = df_wrapper.dataloaders(bs=batch_size)
```

Let's look at our data to make sure it looks right.


```python
batch = next(iter(dls.train))
```

We are expecting three objects in each batch: the categorical variables, the continuous variables, and the labels. Let's take a look.


```python
len(batch)
```




    3




```python
cat_vars, cont_vars, labels = batch
```


```python
cat_vars[:5]
```




    tensor([[ 5, 12,  5,  9,  3,  3,  1,  1],
            [ 2, 10,  3, 11,  1,  5,  2,  1],
            [ 5, 12,  5, 14,  2,  5,  2,  1],
            [ 5,  2,  7,  9,  5,  5,  1,  1],
            [ 5, 12,  3,  5,  6,  5,  1,  1]])




```python
cont_vars[:5]
```




    tensor([[-0.5534,  0.0047, -0.4128, -0.1472, -0.2158, -0.0318],
            [ 0.3228,  1.7249,  1.1487, -0.1472, -0.2158, -0.0318],
            [-0.1883,  1.5283, -0.4128, -0.1472, -0.2158,  0.7760],
            [ 0.1768,  1.4803, -1.1936, -0.1472, -0.2158, -0.0318],
            [-0.0423, -0.0218, -0.4128, -0.1472, -0.2158,  1.1798]])




```python
labels[:5]
```




    tensor([[0],
            [1],
            [0],
            [0],
            [0]], dtype=torch.int8)



Looks good!

Now we make a learner. This data isn't very complex so we'll use a relatively small model for it.


```python
learn = tabular_learner(dls, layers=[20,10])
```

Let's fit the model.


```python
learn.fit(4, 1e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.331239</td>
      <td>0.322867</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.323588</td>
      <td>0.318893</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.320338</td>
      <td>0.325158</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.324844</td>
      <td>0.321952</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


If we didn't pass a validation set we wouldn't have gotten any `valid_loss`.

Now we can save the model.


```python
save_path = Path(os.environ['MODELS']) / 'adult_dataset'
os.makedirs(save_path, exist_ok=True)
```


```python
learn.save(save_path / 'baseline_neural_network')
```




    Path('I:/Models/adult_dataset/baseline_neural_network.pth')



## Part II

To fully simulate this being a separate test, I'm going to reload the model from the weights. Note that we would have to create a `learn` object before we load the weights. In this case we'll use the same `learn` as before.


```python
learn.load(save_path / 'baseline_neural_network')
```




    <fastai.tabular.learner.TabularLearner at 0x1e79f48c730>



Let's look at the model and make sure it loaded correctly.


```python
learn.summary()
```








    TabularModel (Input shape: 128 x 8)
    ============================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ============================================================================
                         128 x 6             
    Embedding                                 60         True      
    ____________________________________________________________________________
                         128 x 8             
    Embedding                                 136        True      
    ____________________________________________________________________________
                         128 x 5             
    Embedding                                 40         True      
    ____________________________________________________________________________
                         128 x 8             
    Embedding                                 128        True      
    ____________________________________________________________________________
                         128 x 5             
    Embedding                                 35         True      
    ____________________________________________________________________________
                         128 x 4             
    Embedding                                 24         True      
    ____________________________________________________________________________
                         128 x 3             
    Embedding                                 9          True      
    Embedding                                 9          True      
    Dropout                                                        
    BatchNorm1d                               12         True      
    ____________________________________________________________________________
                         128 x 20            
    Linear                                    960        True      
    ReLU                                                           
    BatchNorm1d                               40         True      
    ____________________________________________________________________________
                         128 x 10            
    Linear                                    200        True      
    ReLU                                                           
    BatchNorm1d                               20         True      
    ____________________________________________________________________________
                         128 x 2             
    Linear                                    22         True      
    ____________________________________________________________________________
    
    Total params: 1,695
    Total trainable params: 1,695
    Total non-trainable params: 0
    
    Optimizer used: <function Adam at 0x000001E7A3A0D670>
    Loss function: FlattenedLoss of CrossEntropyLoss()
    
    Model unfrozen
    
    Callbacks:
      - TrainEvalCallback
      - Recorder
      - ProgressCallback



Looks good. Let's look at the test data.


```python
test_df.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14160</th>
      <td>30</td>
      <td>Private</td>
      <td>81282</td>
      <td>HS-grad</td>
      <td>9.0</td>
      <td>Never-married</td>
      <td>Other-service</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>27048</th>
      <td>38</td>
      <td>Federal-gov</td>
      <td>172571</td>
      <td>Some-college</td>
      <td>10.0</td>
      <td>Divorced</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>28868</th>
      <td>40</td>
      <td>Private</td>
      <td>223548</td>
      <td>HS-grad</td>
      <td>9.0</td>
      <td>Married-civ-spouse</td>
      <td>Adm-clerical</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Mexico</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>5667</th>
      <td>28</td>
      <td>Local-gov</td>
      <td>191177</td>
      <td>Masters</td>
      <td>14.0</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>7827</th>
      <td>31</td>
      <td>Private</td>
      <td>210562</td>
      <td>HS-grad</td>
      <td>9.0</td>
      <td>Married-civ-spouse</td>
      <td>Transport-moving</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>65</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>
</div>



Because the data is imbalanced we'll have to adjust our baseline. A completely "dumb" classifier that only guesses the most common class will be right more than 50% of the time. Let's see what that percentage is.


```python
test_df['salary'].value_counts()
```




    <50k     6183
    >=50k    1958
    Name: salary, dtype: int64




```python
test_df['salary'].value_counts()[0] / np.sum(test_df['salary'].value_counts())
```




    0.7594890062645867



OK, so 75% is our baseline that we have to beat.

The data looks like we expected. Now we follow a similar process as what we did before.


```python
test_splits = no_split(range_of(test_df))
```


```python
test_df_wrapper = TabularPandas(test_df, preprocessing, categorical_vars, continuous_vars, splits=test_splits, y_names=dep_var)
```

Now we can turn that into a `DataLoaders` object.

> Note: If your test set size isn't divisible by your batch size you'll need to `drop_last`. If I don't I get an error, although I've only noticed this happening with the test set.


```python
test_dls = test_df_wrapper.dataloaders(batch_size, drop_last=False)
```

Now we've got everything in place to make predictions.


```python
preds, ground_truth = learn.get_preds(dl=test_dls.train)
```





Let's see what they look like.


```python
preds[:5]
```




    tensor([[0.9943, 0.0057],
            [0.9559, 0.0441],
            [0.6239, 0.3761],
            [0.4550, 0.5450],
            [0.7262, 0.2738]])




```python
ground_truth[:5]
```




    tensor([[0],
            [1],
            [0],
            [1],
            [0]], dtype=torch.int8)



Depending on your last layer, converting the prediction into an actual prediction will be different. In this case have a probability associated with each value, so to get the final prediction we need to take an argmax. Had you just had one value in the last layer, you could extract the label prediction with `np.rint(preds)`.

You can test this by seeing that each prediction sums to 1.


```python
preds.sum(1)
```




    tensor([1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000])




```python
torch.argmax(preds, dim=1)
```




    tensor([0, 0, 0,  ..., 1, 0, 1])



Let's see what our final accuracy is on the test set.


```python
accuracy_score(ground_truth, torch.argmax(preds, dim=1))
```




    0.851001105515293


