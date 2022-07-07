---
layout: post
title: "Test Set Metrics on Unbalanced Datasets: Part II"
description: "This post shows how to get accurate and precise metrics from a test set when you have an unbalanced dataset."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/bald_eagle2.jpg"
tags: [Deep Learning, Machine Learning, Unbalanced Data, Python]
---

This is Part II of two posts demonstrating how to get test metrics from a highly unbalanced dataset. In [Part I](https://jss367.github.io/test-set-metrics-on-unbalanced-datasets-part-i.html), I showed how you could theoretically estimate the precision, recall, and f1 score on highly unbalanced data. In this part, I'll do the same thing with a real dataset. To do so, I'll use the [Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult).


```python
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll.base import scope
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
```

Let's download the dataset using sklearn's `fetch_openml`.


```python
dataset = fetch_openml("adult", version=2, target_column=None, as_frame=True)
df = dataset.data
```

Let's look at what we've got.


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
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25.0</td>
      <td>Private</td>
      <td>226802.0</td>
      <td>11th</td>
      <td>7.0</td>
      <td>Never-married</td>
      <td>Machine-op-inspct</td>
      <td>Own-child</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>Private</td>
      <td>89814.0</td>
      <td>HS-grad</td>
      <td>9.0</td>
      <td>Married-civ-spouse</td>
      <td>Farming-fishing</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28.0</td>
      <td>Local-gov</td>
      <td>336951.0</td>
      <td>Assoc-acdm</td>
      <td>12.0</td>
      <td>Married-civ-spouse</td>
      <td>Protective-serv</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44.0</td>
      <td>Private</td>
      <td>160323.0</td>
      <td>Some-college</td>
      <td>10.0</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>7688.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18.0</td>
      <td>NaN</td>
      <td>103497.0</td>
      <td>Some-college</td>
      <td>10.0</td>
      <td>Never-married</td>
      <td>NaN</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['class'].value_counts()
```




    <=50K    37155
    >50K     11687
    Name: class, dtype: int64



Just under a quarter of the data are of people with salaries above 50k. That's unbalanced but for this we want to look at a much more extreme example. We'll make it so that people in the >50K class are really rare.

Caveat: If I was really trying to build an algorithm to classify this data, I would go over the data in great detail. I can already see that education is listed as 11th in one case and HS-grad in another, so it needs to be cleaned up. But in this case I'm going to ignore all of that and jump right to encoding the data so it can be used in a model. It would also be good practice to split off test data before doing label encoding, but we'll skip that as well.


```python
df = df.apply(LabelEncoder().fit_transform)
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
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>3</td>
      <td>19329</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>3</td>
      <td>4212</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>49</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>1</td>
      <td>25340</td>
      <td>7</td>
      <td>11</td>
      <td>2</td>
      <td>10</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27</td>
      <td>3</td>
      <td>11201</td>
      <td>15</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>98</td>
      <td>0</td>
      <td>39</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>8</td>
      <td>5411</td>
      <td>15</td>
      <td>9</td>
      <td>4</td>
      <td>14</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29</td>
      <td>38</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now let's split up the data by class so we can make the unbalance more extreme.


```python
rich_df = df[df['class'].astype(bool)]
poor_df = df[~df['class'].astype(bool)]
```

We'll split off some of the rich examples so we can add them in later.


```python
split_rich_df = rich_df.sample(50)
```


```python
split_rich_df.head()
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
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21325</th>
      <td>38</td>
      <td>3</td>
      <td>669</td>
      <td>12</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>55</td>
      <td>39</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17501</th>
      <td>28</td>
      <td>5</td>
      <td>20941</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3097</th>
      <td>19</td>
      <td>3</td>
      <td>7924</td>
      <td>9</td>
      <td>12</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>44</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24504</th>
      <td>31</td>
      <td>3</td>
      <td>21000</td>
      <td>9</td>
      <td>12</td>
      <td>2</td>
      <td>11</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>70</td>
      <td>0</td>
      <td>49</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41022</th>
      <td>8</td>
      <td>5</td>
      <td>21708</td>
      <td>15</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>52</td>
      <td>0</td>
      <td>49</td>
      <td>38</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We'll remove those from the dataset so we don't have to worry about data leakage.


```python
rich_df = rich_df.drop(split_rich_df.index)
```


```python
len(poor_df)
```




    37155



Now let's create our dataset. We'll create a really high imbalance, like 1,000 : 1.


```python
num_pos = 30
num_neg = 30000
```


```python
representative_rich_df = rich_df.sample(num_pos, random_state=42)
representative_poor_df = poor_df.sample(num_neg, random_state=42)
```


```python
rich_df = rich_df.drop(representative_rich_df.index)
poor_df = poor_df.drop(representative_poor_df.index)
```


```python
df = pd.concat([representative_rich_df, representative_poor_df])
```

`df` contains all the positives and negatives.


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
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15459</th>
      <td>45</td>
      <td>3</td>
      <td>1914</td>
      <td>9</td>
      <td>12</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>81</td>
      <td>0</td>
      <td>39</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18869</th>
      <td>16</td>
      <td>3</td>
      <td>24958</td>
      <td>15</td>
      <td>9</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>49</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13384</th>
      <td>43</td>
      <td>0</td>
      <td>2282</td>
      <td>9</td>
      <td>12</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1937</th>
      <td>40</td>
      <td>3</td>
      <td>16039</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3875</th>
      <td>7</td>
      <td>3</td>
      <td>16177</td>
      <td>8</td>
      <td>10</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>38</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



This looks adequate for feeding into a model.


```python
df_train, df_test = train_test_split(df, random_state=42)
```


```python
df_test.head()
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
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42029</th>
      <td>4</td>
      <td>8</td>
      <td>3469</td>
      <td>15</td>
      <td>9</td>
      <td>4</td>
      <td>14</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27469</th>
      <td>33</td>
      <td>3</td>
      <td>8739</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39161</th>
      <td>26</td>
      <td>3</td>
      <td>26001</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13781</th>
      <td>7</td>
      <td>5</td>
      <td>5851</td>
      <td>15</td>
      <td>9</td>
      <td>2</td>
      <td>11</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19096</th>
      <td>33</td>
      <td>3</td>
      <td>20253</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>13</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>25</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Let's record the number of positives and negatives from our test set to reuse later.


```python
num_pos_test = np.sum(df_test['class'] == 1)
num_neg_test = np.sum(df_test['class'] == 0)
```

Let's add the rest of the data to the train set. It won't be completely balanced but it will still be close enough.


```python
df_train_balanced = pd.concat([df_train, rich_df])
```

Now let's shuffle the data.


```python
df_train_balanced = df_train_balanced.sample(frac=1, random_state=42)
```


```python
y_train = df_train_balanced.pop('class')
y_test = df_test.pop('class')
```


```python
X_train = df_train_balanced
X_test = df_test
```

To train a model I'm just going to copy code from the [High Performance Models on Tabular Data](https://jss367.github.io/high-performance-models-on-tabular-data.html) post.


```python
def train_xgb(params):
    """
    xgb needs eval_metric or it produces lots of warnings
    """
    clf=XGBClassifier(**params)
    clf.fit(X_train, y_train, eval_metric='logloss')
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)

    return {'loss': -accuracy, 'status': STATUS_OK}
```


```python
num_trials = 50
```


```python
xgb_space={'max_depth': scope.int(hp.quniform("max_depth", 3, 18, 1)),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }
```


```python
trials = Trials()

fmin(fn = train_xgb,
    space = xgb_space,
    algo = tpe.suggest,
    max_evals = num_trials,
    trials = trials,
    rstate=np.random.default_rng(42))
```

      0%|                                                                           | 0/50 [00:00<?, ?trial/s, best loss=?]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

      2%|▉                                                | 1/50 [00:02<01:39,  2.03s/trial, best loss: -0.895045285029302]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

      4%|█▉                                               | 2/50 [00:02<01:01,  1.28s/trial, best loss: -0.895045285029302]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

      6%|██▉                                              | 3/50 [00:04<01:15,  1.60s/trial, best loss: -0.895045285029302]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

      8%|███▉                                             | 4/50 [00:06<01:07,  1.47s/trial, best loss: -0.895045285029302]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     10%|████▉                                            | 5/50 [00:08<01:22,  1.83s/trial, best loss: -0.895045285029302]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     12%|█████▉                                           | 6/50 [00:11<01:38,  2.23s/trial, best loss: -0.895045285029302]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     14%|██████▊                                          | 7/50 [00:12<01:19,  1.85s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     16%|███████▊                                         | 8/50 [00:13<01:05,  1.55s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     18%|████████▊                                        | 9/50 [00:16<01:17,  1.88s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     20%|█████████▌                                      | 10/50 [00:18<01:27,  2.18s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     22%|██████████▌                                     | 11/50 [00:22<01:40,  2.59s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     24%|███████████▌                                    | 12/50 [00:23<01:19,  2.10s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     26%|████████████▍                                   | 13/50 [00:24<01:05,  1.77s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     28%|█████████████▍                                  | 14/50 [00:26<01:12,  2.01s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     30%|██████████████▍                                 | 15/50 [00:27<00:58,  1.67s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     32%|███████████████▎                                | 16/50 [00:28<00:49,  1.45s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     34%|████████████████▎                               | 17/50 [00:29<00:42,  1.29s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     36%|█████████████████▎                              | 18/50 [00:32<00:52,  1.65s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     38%|██████████████████▏                             | 19/50 [00:33<00:44,  1.44s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     40%|███████████████████▏                            | 20/50 [00:34<00:37,  1.25s/trial, best loss: -0.898108684070325]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     42%|███████████████████▋                           | 21/50 [00:37<00:58,  2.01s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     44%|████████████████████▋                          | 22/50 [00:38<00:49,  1.76s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     46%|█████████████████████▌                         | 23/50 [00:42<01:02,  2.32s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     48%|██████████████████████▌                        | 24/50 [00:43<00:51,  1.98s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     50%|███████████████████████▌                       | 25/50 [00:44<00:42,  1.71s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     52%|████████████████████████▍                      | 26/50 [00:45<00:36,  1.51s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     54%|█████████████████████████▍                     | 27/50 [00:49<00:52,  2.27s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     56%|██████████████████████████▎                    | 28/50 [00:53<00:56,  2.57s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     58%|███████████████████████████▎                   | 29/50 [00:57<01:02,  2.96s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     60%|████████████████████████████▏                  | 30/50 [01:01<01:05,  3.25s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     62%|█████████████████████████████▏                 | 31/50 [01:03<00:59,  3.11s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     64%|██████████████████████████████                 | 32/50 [01:04<00:45,  2.52s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     66%|███████████████████████████████                | 33/50 [01:05<00:33,  1.98s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     68%|███████████████████████████████▉               | 34/50 [01:08<00:35,  2.20s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     70%|████████████████████████████████▉              | 35/50 [01:09<00:27,  1.82s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     72%|█████████████████████████████████▊             | 36/50 [01:10<00:22,  1.58s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     74%|██████████████████████████████████▊            | 37/50 [01:12<00:23,  1.81s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     76%|███████████████████████████████████▋           | 38/50 [01:13<00:18,  1.53s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     78%|████████████████████████████████████▋          | 39/50 [01:14<00:15,  1.42s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     80%|█████████████████████████████████████▌         | 40/50 [01:15<00:12,  1.25s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     82%|██████████████████████████████████████▌        | 41/50 [01:16<00:10,  1.20s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     84%|███████████████████████████████████████▍       | 42/50 [01:17<00:08,  1.12s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     86%|████████████████████████████████████████▍      | 43/50 [01:20<00:12,  1.81s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     88%|█████████████████████████████████████████▎     | 44/50 [01:23<00:12,  2.07s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     90%|██████████████████████████████████████████▎    | 45/50 [01:24<00:08,  1.70s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     92%|███████████████████████████████████████████▏   | 46/50 [01:25<00:05,  1.48s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     94%|████████████████████████████████████████████▏  | 47/50 [01:26<00:04,  1.38s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     96%|█████████████████████████████████████████████  | 48/50 [01:29<00:03,  1.72s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

     98%|██████████████████████████████████████████████ | 49/50 [01:31<00:01,  1.84s/trial, best loss: -0.8983750665956314]

    C:\Users\Julius\anaconda3\envs\tf\lib\site-packages\xgboost\sklearn.py:793: UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
      warnings.warn(
    
    

    100%|███████████████████████████████████████████████| 50/50 [01:34<00:00,  1.88s/trial, best loss: -0.8983750665956314]
    




    {'colsample_bytree': 0.9803142462584047,
     'gamma': 8.987297166404515,
     'max_depth': 12.0,
     'min_child_weight': 0.0,
     'reg_alpha': 180.0,
     'reg_lambda': 0.6069047121789143}




```python
best_hyperparams = space_eval(xgb_space, trials.argmin)
best_hyperparams
```




    {'colsample_bytree': 0.9803142462584047,
     'gamma': 8.987297166404515,
     'max_depth': 12,
     'min_child_weight': 0.0,
     'n_estimators': 180,
     'reg_alpha': 180.0,
     'reg_lambda': 0.6069047121789143,
     'seed': 0}




```python
xgb_clf = XGBClassifier(**best_hyperparams)

```


```python
xgb_clf.fit(X_train, y_train)

```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBClassifier(base_score=0.5, booster=&#x27;gbtree&#x27;, callbacks=None,
              colsample_bylevel=1, colsample_bynode=1,
              colsample_bytree=0.9803142462584047, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None,
              gamma=8.987297166404515, gpu_id=-1, grow_policy=&#x27;depthwise&#x27;,
              importance_type=None, interaction_constraints=&#x27;&#x27;,
              learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=12, max_leaves=0,
              min_child_weight=0.0, missing=nan, monotone_constraints=&#x27;()&#x27;,
              n_estimators=180, n_jobs=0, num_parallel_tree=1, predictor=&#x27;auto&#x27;,
              random_state=0, reg_alpha=180.0, reg_lambda=0.6069047121789143, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=0.5, booster=&#x27;gbtree&#x27;, callbacks=None,
              colsample_bylevel=1, colsample_bynode=1,
              colsample_bytree=0.9803142462584047, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None,
              gamma=8.987297166404515, gpu_id=-1, grow_policy=&#x27;depthwise&#x27;,
              importance_type=None, interaction_constraints=&#x27;&#x27;,
              learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=12, max_leaves=0,
              min_child_weight=0.0, missing=nan, monotone_constraints=&#x27;()&#x27;,
              n_estimators=180, n_jobs=0, num_parallel_tree=1, predictor=&#x27;auto&#x27;,
              random_state=0, reg_alpha=180.0, reg_lambda=0.6069047121789143, ...)</pre></div></div></div></div></div>



Now we've got a trained model. Let's see how well it does.


```python
xgb_preds = xgb_clf.predict(X_test)

```


```python
def get_metrics(y_test, y_pred):
    """
    Print standard sklearn metrics
    """
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"Precision: {precision_score(y_test, y_pred):.2%}")
    print(f"Recall: {recall_score(y_test, y_pred):.2%}")
    print(f"F1: {f1_score(y_test, y_pred):.2%}")
```


```python
get_metrics(y_test, xgb_preds)
```

    Accuracy: 89.84%
    Precision: 0.52%
    Recall: 66.67%
    F1: 1.04%
    

We have some metrics now, but how much can we trust them. Let's look at the number of positive examples.


```python
num_pos_test = np.sum(y_test)
print(f"{num_pos_test=}")
```

    num_pos_test=6
    

Only six! So all of our metrics are possibly pretty far off their true values. There aren't enough positives to get a good measure of the model quality.

Just like we did in [Part I](https://jss367.github.io/test-set-metrics-on-unbalanced-datasets-part-i.html), we'll additional test cases. Here, it'll be those that we split off earlier. This will reduce the error on our metrics because we have more samples.


```python
split_labels = split_rich_df.pop('class')
```


```python
X_test_added = pd.concat([X_test, split_rich_df])
y_test_added = pd.concat([y_test, split_labels])
```


```python
xgb_preds_added = xgb_clf.predict(X_test_added)

```


```python
get_metrics(y_test_added, xgb_preds_added)

```

    Accuracy: 89.75%
    Precision: 5.23%
    Recall: 75.00%
    F1: 9.78%
    

We got a metrics, but they are not all an apples-to-apples comparison. In particular, the precision (and therefore F1 score) is far higher. That's because we've changed the ratio of positives to negatives in the dataset. To get back the original values,  we'll have to use the tricks we did in [Part I](https://jss367.github.io/test-set-metrics-on-unbalanced-datasets-part-i.html).


```python
def get_model_stats(y_true, y_pred):
    pos_indices = [i for i , x in enumerate(y_true) if x == 1]
    neg_indices = [i for i , x in enumerate(y_true) if x == 0]
    preds_for_pos_labels = [y_pred[i] for i in pos_indices]
    preds_for_neg_labels = [y_pred[i] for i in neg_indices]
    percent_pos_correct = sum(preds_for_pos_labels) / len(preds_for_pos_labels)
    percent_neg_correct = np.sum(np.array(preds_for_neg_labels) == 0) / len(preds_for_neg_labels)
    return percent_pos_correct, percent_neg_correct
```


```python
percent_pos_correct, percent_neg_correct = get_model_stats(y_test_added, xgb_preds_added)
```


```python
print(f"{percent_pos_correct=}")
print(f"{percent_neg_correct=}")
```

    percent_pos_correct=0.75
    percent_neg_correct=0.8985603838976273
    

These are the stats for the model. If we wanted to convert them to f1 for a specific distribution, we could do that.


```python
# print(f"{num_pos=}")
# print(f"{num_neg=}")
```


```python
def np_float_to_int(x):
    return np.rint(x).astype(int)

def get_y_pred(num_pos, num_neg, percent_pos_correct, percent_neg_correct):
    return (
        [1] * np_float_to_int(num_pos * percent_pos_correct)
        + [0] * np_float_to_int(num_pos - num_pos * percent_pos_correct)
        + [0] * np_float_to_int(num_neg * percent_neg_correct)
        + [1] * np_float_to_int(num_neg - num_neg * percent_neg_correct)
    )
```


```python
y_pred_recreated = get_y_pred(num_pos_test, num_neg_test, percent_pos_correct, percent_neg_correct)
```


```python
y_test_recreated = [1] * num_pos_test + [0] * num_neg_test
```


```python
get_metrics(y_test_recreated, y_pred_recreated)
```

    Accuracy: 89.84%
    Precision: 0.52%
    Recall: 66.67%
    F1: 1.04%
    

These are better estimates of the model's preformance and are more precise than if it only used the true unbalanced test set.
