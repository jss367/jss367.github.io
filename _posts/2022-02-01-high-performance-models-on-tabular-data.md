---
layout: post
title: "High Performance Models on Tabular Data"
description: "This post shows how to get good performance on tabular data from models like xgboost and random forest. It relies on Bayesian optimization using Hyperopt to do so"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/red-backed_fairywren.jpg"
tags: [FastAI, Machine Learning, Neural Networks, Python, Scikit-Learn, XGBoost]
---

This blog posts shows some ways to get generally good performance on tabular data. Most of the work in getting high performance models from tabular data comes from cleaning the dataset, clever feature engineering, and other tasks specific to the data set. We won't be doing that here. However, there's still a need for some good baseline parameters to know you're getting the best out of your model. This post provides a way to use Bayesian optimization to find good hyperparameters and get good performance.


```python
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
```


```python
df = pd.read_csv(Path(os.getenv('DATA')) / 'stroke/healthcare-dataset-stroke-data.csv')
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
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9046</td>
      <td>Male</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51676</td>
      <td>Female</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>202.21</td>
      <td>NaN</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31112</td>
      <td>Male</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60182</td>
      <td>Female</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1665</td>
      <td>Female</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5110 entries, 0 to 5109
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   id                 5110 non-null   int64  
     1   gender             5110 non-null   object 
     2   age                5110 non-null   float64
     3   hypertension       5110 non-null   int64  
     4   heart_disease      5110 non-null   int64  
     5   ever_married       5110 non-null   object 
     6   work_type          5110 non-null   object 
     7   Residence_type     5110 non-null   object 
     8   avg_glucose_level  5110 non-null   float64
     9   bmi                4909 non-null   float64
     10  smoking_status     5110 non-null   object 
     11  stroke             5110 non-null   int64  
    dtypes: float64(3), int64(4), object(5)
    memory usage: 479.2+ KB
    


```python
df = df.drop('id', axis=1)
```


```python
df.isnull().sum()
```




    gender                 0
    age                    0
    hypertension           0
    heart_disease          0
    ever_married           0
    work_type              0
    Residence_type         0
    avg_glucose_level      0
    bmi                  201
    smoking_status         0
    stroke                 0
    dtype: int64




```python
df['stroke'].value_counts()
```




    0    4861
    1     249
    Name: stroke, dtype: int64



We have significantly unbalanced data. We'll have to use oversampling to adjust for this in the training data.


```python
le = LabelEncoder()
en_df = df.apply(le.fit_transform)
en_df.head()
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
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>88</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3850</td>
      <td>239</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>82</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>3588</td>
      <td>418</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>101</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2483</td>
      <td>198</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3385</td>
      <td>217</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>100</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>3394</td>
      <td>113</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Clean up dataset


```python
en_df_imputed = en_df
imputer = KNNImputer(n_neighbors=4, weights="uniform")
imputer.fit_transform(en_df_imputed)
```




    array([[  1.,  88.,   0., ..., 239.,   1.,   1.],
           [  0.,  82.,   0., ..., 418.,   2.,   1.],
           [  1., 101.,   0., ..., 198.,   2.,   1.],
           ...,
           [  0.,  56.,   0., ..., 179.,   2.,   0.],
           [  1.,  72.,   0., ..., 129.,   1.,   0.],
           [  0.,  65.,   0., ..., 135.,   0.,   0.]])




```python
en_df_imputed.isnull().sum()

```




    gender               0
    age                  0
    hypertension         0
    heart_disease        0
    ever_married         0
    work_type            0
    Residence_type       0
    avg_glucose_level    0
    bmi                  0
    smoking_status       0
    stroke               0
    dtype: int64




```python
features=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type',
       'smoking_status']
```


```python
from imblearn.over_sampling import SMOTE
X, y = en_df_imputed[features], en_df_imputed["stroke"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)
```

# Modeling


```python
from functools import partial

from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
from sklearn.metrics import accuracy_score, f1_score
```


```python
num_trials = 100
svm_trials = 10 # svm takes much longer, so you may want to limit this
```

## XGBoost


```python
from xgboost import XGBClassifier
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
def train_clf(clf, params):
    clf=clf(**params)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds>0.5)

    return {'loss': -accuracy, 'status': STATUS_OK}
```


```python
def train_xgb(params):
    """
    xgb needs eval_metric or lots of warnings
    """
    clf=XGBClassifier(**params)
    clf.fit(X_train, y_train, eval_metric='logloss')
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)

    return {'loss': -accuracy, 'status': STATUS_OK}
```


```python
trials = Trials()

fmin(fn = train_xgb,
    space = xgb_space,
    algo = tpe.suggest,
    max_evals = num_trials,
    trials = trials)
```

    100%|█████████████████████████████████████████████| 100/100 [00:25<00:00,  3.93trial/s, best loss: -0.7592954990215264]
    




    {'colsample_bytree': 0.664711277682662,
     'gamma': 1.2129835895645058,
     'max_depth': 15.0,
     'min_child_weight': 3.0,
     'reg_alpha': 44.0,
     'reg_lambda': 0.36894533857340944}




```python
best_hyperparams = space_eval(xgb_space, trials.argmin)
```


```python
best_hyperparams
```




    {'colsample_bytree': 0.664711277682662,
     'gamma': 1.2129835895645058,
     'max_depth': 15,
     'min_child_weight': 3.0,
     'n_estimators': 180,
     'reg_alpha': 44.0,
     'reg_lambda': 0.36894533857340944,
     'seed': 0}




```python
xgb_clf = XGBClassifier(**best_hyperparams)
```


```python
xgb_clf.fit(X_train, y_train)
```

    [22:30:08] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.5.1/src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.664711277682662,
                  enable_categorical=False, gamma=1.2129835895645058, gpu_id=-1,
                  importance_type=None, interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=15,
                  min_child_weight=3.0, missing=nan, monotone_constraints='()',
                  n_estimators=180, n_jobs=12, num_parallel_tree=1,
                  predictor='auto', random_state=0, reg_alpha=44.0,
                  reg_lambda=0.36894533857340944, scale_pos_weight=1, seed=0,
                  subsample=1, tree_method='exact', validate_parameters=1,
                  verbosity=None)




```python
xgb_preds = xgb_clf.predict(X_test)
```


```python
f1_score(y_test, xgb_preds)
```




    0.24074074074074073




```python
accuracy_score(y_test, xgb_preds)
```




    0.7592954990215264




```python
y_test
```




    4688    0
    4478    0
    3849    0
    4355    0
    3826    0
           ..
    3605    0
    4934    0
    4835    0
    4105    0
    2902    0
    Name: stroke, Length: 1022, dtype: int64



## Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rf_space = {
    "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 50)),
    "max_depth": hp.quniform("max_depth", 1, 15, 1),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}
```


```python
trials = Trials()

fmin(fn = partial(train_clf, RandomForestClassifier),
    space = rf_space,
    algo = tpe.suggest,
    max_evals = num_trials,
    trials = trials)
```

    100%|█████████████████████████████████████████████| 100/100 [01:54<00:00,  1.14s/trial, best loss: -0.8512720156555773]
    




    {'criterion': 0, 'max_depth': 15.0, 'n_estimators': 500.0}




```python
rf_best_hyperparams = space_eval(rf_space, trials.argmin)
```


```python
rf_best_hyperparams
```




    {'criterion': 'gini', 'max_depth': 15.0, 'n_estimators': 500}




```python
rf_clf = RandomForestClassifier(**rf_best_hyperparams)
```


```python
rf_clf.fit(X_train, y_train)
```




    RandomForestClassifier(max_depth=15.0, n_estimators=500)




```python
rf_preds = rf_clf.predict(X_test)
```


```python
f1_score(y_test, rf_preds)
```




    0.14364640883977903




```python
accuracy_score(y_test, rf_preds)
```




    0.8483365949119374



## SVM

You can also try support vector machines, but I usually skip these for very large datasets. They don't generally get the best performance and the training time is much higher than the others. Fundamentally, nonlinear SVM kernels are trying to solve a problem that is `O(n_samples^2 * n_features)`, so things quickly get out of hand with a lot of samples.


```python
from sklearn.svm import SVC
```


```python
svm_space = {
      'C': hp.lognormal('svm_C', 0, 1),
      'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly']),
      'degree':hp.choice('degree',[2,3,4]),
      'probability':hp.choice('probability',[True])
      }
```


```python
trials = Trials()

fmin(fn = partial(train_clf, SVC),
    space = svm_space,
    algo = tpe.suggest,
    max_evals = 4, # svm takes too long so reducing num trials
    trials = trials)
```

    100%|█████████████████████████████████████████████████| 4/4 [01:22<00:00, 20.66s/trial, best loss: -0.8003913894324853]
    




    {'degree': 2, 'kernel': 2, 'probability': 0, 'svm_C': 0.2371250621465351}




```python
svm_best_hyperparams = space_eval(svm_space, trials.argmin)
```


```python
svm_best_hyperparams
```




    {'C': 0.2371250621465351, 'degree': 4, 'kernel': 'poly', 'probability': True}




```python
svm_clf = SVC(**svm_best_hyperparams)
```


```python
svm_clf.fit(X_train, y_train)
```




    SVC(C=0.2371250621465351, degree=4, kernel='poly', probability=True)




```python
svm_preds = svm_clf.predict(X_test)
```


```python
f1_score(y_test, svm_preds)
```




    0.28671328671328666




```python
accuracy_score(y_test, svm_preds)
```




    0.8003913894324853



## Neural Network with FastAI

I've had poor-to-mixes results with neural networks and hyperopt. But I still included it because I thought it might be helpful.


```python
from fastai.tabular.all import *
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
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>202.21</td>
      <td>NaN</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
dep_var = 'stroke'
```

For FastAI, we'll combine the training and validation data into one DataFrame and then split them out later. It's just easier this way.


```python
full_X_df = pd.concat([X_train, X_test])
full_y_df = pd.concat([y_train, y_test])
```


```python
df = pd.merge(full_X_df, full_y_df, left_index=True, right_index=True)
```


```python
np.sum(y_test)
```




    62




```python
continuous_vars, categorical_vars = cont_cat_split(df, dep_var=dep_var)

```


```python
val_indices = list(range(len(X_train), len(X_train) + len(X_test)))
```


```python
ind_splitter = IndexSplitter(val_indices)
```


```python
splits = ind_splitter(df) 

```


```python
preprocessing = [Categorify, Normalize]
```


```python
to_nn = TabularPandas(df, preprocessing, categorical_vars, continuous_vars, splits=splits, y_names=dep_var)
```


```python
dls = to_nn.dataloaders(64)
```


```python
def my_acc(preds, gt):
    """
    The order that FAI and sklearn received inputs is flipped, so be careful.
    """
    return accuracy_score(gt.cpu(), np.rint(preds.cpu()))
```


```python
nn_space = [
    {'layer1': scope.int(hp.quniform('layer1', 2, 200, 1))},
    {'layer2': scope.int(hp.quniform('layer2', 2, 500, 2))},
    {'epochs': scope.int(hp.quniform('epochs', 1, 20, 1))},
    {'lr': hp.uniform('lr', 1e-7, 1e-1)},
]
```


```python
def objective(params):
    learn = tabular_learner(dls, y_range=(y.min(), y.max()), layers=[params[0]['layer1'],params[1]['layer2']], metrics=accuracy)
    learn.fit(params[2]['epochs'], params[3]['lr'])
    return {'loss': learn.recorder.losses[-1], 'status': STATUS_OK}
```


```python
trials = Trials()

best = fmin(objective,
    space=nn_space,
    algo=tpe.suggest,
    max_evals=num_trials,
           trials=trials)
print(best)
```

      0%|                                                                          | 0/100 [00:00<?, ?trial/s, best loss=?]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.266162</td>
      <td>0.367915</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


      1%|▍                                              | 1/100 [00:02<04:47,  2.90s/trial, best loss: 0.26616227626800537]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.173481</td>
      <td>0.275837</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.155806</td>
      <td>0.258449</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.152677</td>
      <td>0.201270</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.154663</td>
      <td>0.134104</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.147593</td>
      <td>0.216417</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149340</td>
      <td>0.209012</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.147586</td>
      <td>0.231209</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.145267</td>
      <td>0.229008</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.144006</td>
      <td>0.182196</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.143992</td>
      <td>0.247818</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.139587</td>
      <td>0.320102</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.146561</td>
      <td>0.271461</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.144931</td>
      <td>0.228534</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


      2%|▉                                              | 2/100 [00:19<18:03, 11.06s/trial, best loss: 0.14493143558502197]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.250243</td>
      <td>0.418120</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.260521</td>
      <td>0.380860</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.261869</td>
      <td>0.529354</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.248912</td>
      <td>0.447990</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.246130</td>
      <td>0.467184</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.287706</td>
      <td>0.574936</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.288179</td>
      <td>0.387476</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.259243</td>
      <td>0.411937</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.282903</td>
      <td>0.579255</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.261702</td>
      <td>0.605595</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.268408</td>
      <td>0.553799</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.259359</td>
      <td>0.443249</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.261318</td>
      <td>0.523483</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.251320</td>
      <td>0.436399</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.247600</td>
      <td>0.402228</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.268977</td>
      <td>0.595890</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.258441</td>
      <td>0.346379</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


      3%|█▍                                             | 3/100 [00:41<25:58, 16.06s/trial, best loss: 0.14493143558502197]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.278295</td>
      <td>0.273026</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.243224</td>
      <td>0.477241</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.294494</td>
      <td>0.411374</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.272634</td>
      <td>0.589032</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.284962</td>
      <td>0.813167</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.264567</td>
      <td>0.346040</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.251033</td>
      <td>0.466497</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.271383</td>
      <td>0.353754</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.254366</td>
      <td>0.440315</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


      4%|█▉                                             | 4/100 [00:53<23:06, 14.45s/trial, best loss: 0.14493143558502197]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.156448</td>
      <td>0.275086</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.150530</td>
      <td>0.202307</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


      5%|██▎                                            | 5/100 [00:56<16:08, 10.19s/trial, best loss: 0.14493143558502197]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.243554</td>
      <td>0.460086</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.229212</td>
      <td>0.385627</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.310834</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.340423</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.337066</td>
      <td>0.902151</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.283584</td>
      <td>0.624796</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.256619</td>
      <td>0.484743</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.267031</td>
      <td>0.512884</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.296712</td>
      <td>0.804305</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


      6%|██▊                                            | 6/100 [01:08<16:51, 10.76s/trial, best loss: 0.14493143558502197]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.293355</td>
      <td>0.615268</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.341735</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.333435</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.337697</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.347606</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.346039</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


      7%|███▎                                           | 7/100 [01:16<15:12,  9.81s/trial, best loss: 0.14493143558502197]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.155487</td>
      <td>0.218044</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151981</td>
      <td>0.239342</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.146046</td>
      <td>0.287531</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.144341</td>
      <td>0.243682</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.144937</td>
      <td>0.235739</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.144451</td>
      <td>0.249156</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


      8%|███▊                                           | 8/100 [01:24<14:32,  9.49s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.176433</td>
      <td>0.237168</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.160265</td>
      <td>0.226088</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.156498</td>
      <td>0.224849</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.154980</td>
      <td>0.196317</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


      9%|████▏                                          | 9/100 [01:29<12:19,  8.13s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.157053</td>
      <td>0.197585</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     10%|████▌                                         | 10/100 [01:31<09:01,  6.01s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.275249</td>
      <td>0.398403</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.273341</td>
      <td>0.221651</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.263803</td>
      <td>0.435332</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.256919</td>
      <td>0.341250</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.263162</td>
      <td>0.443249</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.293181</td>
      <td>0.447668</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.260664</td>
      <td>0.371308</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.299794</td>
      <td>0.247554</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.296996</td>
      <td>0.641913</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     11%|█████                                         | 11/100 [01:42<11:26,  7.72s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.339262</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.337378</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.346994</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.338631</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.339389</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.340772</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.338352</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.343501</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.338603</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.335032</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     12%|█████▌                                        | 12/100 [01:55<13:32,  9.24s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.336170</td>
      <td>0.881605</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.276627</td>
      <td>0.523429</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.252784</td>
      <td>0.646466</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.258481</td>
      <td>0.570531</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.257776</td>
      <td>0.421800</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.257819</td>
      <td>0.408801</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.339992</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.343089</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.344745</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.338805</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.340732</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.345180</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.342283</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.339067</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.334864</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     13%|█████▉                                        | 13/100 [02:14<17:49, 12.30s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.180439</td>
      <td>0.222340</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.155118</td>
      <td>0.239016</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.159105</td>
      <td>0.281732</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.157197</td>
      <td>0.261397</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.154369</td>
      <td>0.218101</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.155308</td>
      <td>0.216774</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.154302</td>
      <td>0.193289</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.152122</td>
      <td>0.253967</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.150959</td>
      <td>0.296858</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.147515</td>
      <td>0.325015</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     14%|██████▍                                       | 14/100 [02:27<17:58, 12.54s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.253645</td>
      <td>0.375376</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.240726</td>
      <td>0.391660</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.246171</td>
      <td>0.436927</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.243354</td>
      <td>0.449476</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.242847</td>
      <td>0.385568</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.243101</td>
      <td>0.356934</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.243012</td>
      <td>0.446050</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.237305</td>
      <td>0.465751</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.232671</td>
      <td>0.370742</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.246890</td>
      <td>0.419765</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.241139</td>
      <td>0.364561</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     15%|██████▉                                       | 15/100 [02:42<18:29, 13.05s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.156136</td>
      <td>0.235145</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.152974</td>
      <td>0.166320</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.155588</td>
      <td>0.248385</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.158243</td>
      <td>0.251783</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.155733</td>
      <td>0.204941</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.150630</td>
      <td>0.312261</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.154253</td>
      <td>0.208867</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.150499</td>
      <td>0.175292</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.150392</td>
      <td>0.279819</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.151221</td>
      <td>0.205292</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.145100</td>
      <td>0.279651</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.148908</td>
      <td>0.231631</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.150971</td>
      <td>0.171191</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.149624</td>
      <td>0.190666</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.144667</td>
      <td>0.210604</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.145435</td>
      <td>0.206589</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     16%|███████▎                                      | 16/100 [03:02<21:22, 15.27s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.164218</td>
      <td>0.172055</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.152978</td>
      <td>0.162864</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.154623</td>
      <td>0.277776</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.156921</td>
      <td>0.239833</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.159806</td>
      <td>0.305884</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     17%|███████▊                                      | 17/100 [03:09<17:27, 12.62s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.350829</td>
      <td>0.871817</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.341853</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.337545</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.339120</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.341450</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.337434</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     18%|████████▎                                     | 18/100 [03:16<15:17, 11.19s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.174311</td>
      <td>0.246915</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.159435</td>
      <td>0.205366</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.159499</td>
      <td>0.310583</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.157218</td>
      <td>0.226060</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.151534</td>
      <td>0.213838</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149286</td>
      <td>0.314008</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.150094</td>
      <td>0.169298</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.148730</td>
      <td>0.246668</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.151602</td>
      <td>0.195450</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.148981</td>
      <td>0.199057</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.146331</td>
      <td>0.239428</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.150930</td>
      <td>0.160327</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.146833</td>
      <td>0.141123</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     19%|████████▋                                     | 19/100 [03:37<18:49, 13.95s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.361006</td>
      <td>0.632094</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.352854</td>
      <td>0.635029</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.337417</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.341253</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.343200</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.336819</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.339660</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.340946</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.343185</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.341349</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.337541</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     20%|█████████▏                                    | 20/100 [03:53<19:40, 14.76s/trial, best loss: 0.14445063471794128]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.154664</td>
      <td>0.236320</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.150237</td>
      <td>0.241981</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.147206</td>
      <td>0.214496</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.147562</td>
      <td>0.239145</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.146072</td>
      <td>0.245978</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.148029</td>
      <td>0.222825</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.142559</td>
      <td>0.154123</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.142956</td>
      <td>0.233709</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.142027</td>
      <td>0.183550</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.145075</td>
      <td>0.226792</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.142600</td>
      <td>0.195049</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.139770</td>
      <td>0.178667</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.143580</td>
      <td>0.226856</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.138340</td>
      <td>0.211529</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     21%|█████████▋                                    | 21/100 [04:12<20:58, 15.93s/trial, best loss: 0.13834045827388763]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.153765</td>
      <td>0.278719</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.153772</td>
      <td>0.258929</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.151000</td>
      <td>0.212872</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.147318</td>
      <td>0.209615</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.145885</td>
      <td>0.199299</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.142274</td>
      <td>0.223472</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.144255</td>
      <td>0.203096</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.145988</td>
      <td>0.224949</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.143607</td>
      <td>0.236618</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.141989</td>
      <td>0.187356</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.146113</td>
      <td>0.220106</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.145425</td>
      <td>0.283698</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.145357</td>
      <td>0.193863</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.145125</td>
      <td>0.247975</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.143986</td>
      <td>0.247731</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.139645</td>
      <td>0.261960</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.141586</td>
      <td>0.214280</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.142369</td>
      <td>0.215255</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.139638</td>
      <td>0.230383</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.141621</td>
      <td>0.235287</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     22%|██████████                                    | 22/100 [04:39<25:05, 19.30s/trial, best loss: 0.13834045827388763]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.147303</td>
      <td>0.185964</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.150441</td>
      <td>0.166426</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.152341</td>
      <td>0.227136</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.150130</td>
      <td>0.242738</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.147471</td>
      <td>0.295437</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.142865</td>
      <td>0.209538</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.147723</td>
      <td>0.223869</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.147344</td>
      <td>0.262262</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.144863</td>
      <td>0.187405</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.141153</td>
      <td>0.173189</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.142041</td>
      <td>0.217065</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.142059</td>
      <td>0.214544</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.136707</td>
      <td>0.182034</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.140699</td>
      <td>0.206319</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.142137</td>
      <td>0.232617</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.139883</td>
      <td>0.216119</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.141062</td>
      <td>0.219791</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.140376</td>
      <td>0.207368</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.139918</td>
      <td>0.199749</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.141769</td>
      <td>0.287557</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     23%|██████████▌                                   | 23/100 [05:06<27:26, 21.39s/trial, best loss: 0.13834045827388763]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.155823</td>
      <td>0.249763</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.154308</td>
      <td>0.232878</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.150771</td>
      <td>0.235126</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.151651</td>
      <td>0.279654</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.146321</td>
      <td>0.180945</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.146027</td>
      <td>0.208467</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.142337</td>
      <td>0.219187</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.143279</td>
      <td>0.226393</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.145970</td>
      <td>0.246701</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.146652</td>
      <td>0.201769</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.143587</td>
      <td>0.165640</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.141493</td>
      <td>0.217778</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.141104</td>
      <td>0.189653</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.139685</td>
      <td>0.197084</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.144065</td>
      <td>0.225479</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.141336</td>
      <td>0.210238</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.137108</td>
      <td>0.239454</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.138519</td>
      <td>0.219511</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.141108</td>
      <td>0.191953</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.141612</td>
      <td>0.204788</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     24%|███████████                                   | 24/100 [05:32<28:51, 22.79s/trial, best loss: 0.13834045827388763]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.156233</td>
      <td>0.211124</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.154873</td>
      <td>0.236691</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.150434</td>
      <td>0.147784</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.149061</td>
      <td>0.232782</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.145674</td>
      <td>0.263141</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.146552</td>
      <td>0.222248</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.146280</td>
      <td>0.217669</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.145888</td>
      <td>0.216519</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.143332</td>
      <td>0.251550</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.144489</td>
      <td>0.236220</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.141832</td>
      <td>0.186139</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.141394</td>
      <td>0.163658</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.143748</td>
      <td>0.179892</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.140471</td>
      <td>0.224699</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.143984</td>
      <td>0.185071</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.144399</td>
      <td>0.269946</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.141358</td>
      <td>0.209971</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.139042</td>
      <td>0.175271</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     25%|███████████▌                                  | 25/100 [05:55<28:52, 23.10s/trial, best loss: 0.13834045827388763]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.158746</td>
      <td>0.247588</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.149693</td>
      <td>0.220640</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.150593</td>
      <td>0.154744</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.151016</td>
      <td>0.288088</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.151037</td>
      <td>0.273787</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.145842</td>
      <td>0.230033</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.147289</td>
      <td>0.214378</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.149331</td>
      <td>0.214324</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.147299</td>
      <td>0.215341</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.145268</td>
      <td>0.277632</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.147626</td>
      <td>0.291367</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.142261</td>
      <td>0.286495</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.144533</td>
      <td>0.252009</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.145535</td>
      <td>0.217124</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.143755</td>
      <td>0.155447</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.144600</td>
      <td>0.208635</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.142711</td>
      <td>0.242058</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.141821</td>
      <td>0.195013</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     26%|███████████▉                                  | 26/100 [06:19<28:37, 23.21s/trial, best loss: 0.13834045827388763]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.166382</td>
      <td>0.265137</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.152733</td>
      <td>0.199722</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.149169</td>
      <td>0.250594</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.150091</td>
      <td>0.303567</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.149431</td>
      <td>0.248212</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149944</td>
      <td>0.216271</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.146256</td>
      <td>0.244059</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.146906</td>
      <td>0.236853</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.146961</td>
      <td>0.175519</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.145007</td>
      <td>0.224877</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.146591</td>
      <td>0.155453</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.142774</td>
      <td>0.192929</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.142282</td>
      <td>0.239659</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.141746</td>
      <td>0.272025</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     27%|████████████▍                                 | 27/100 [06:37<26:22, 21.68s/trial, best loss: 0.13834045827388763]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.152193</td>
      <td>0.227492</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.149710</td>
      <td>0.205077</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.143902</td>
      <td>0.189282</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.140971</td>
      <td>0.201470</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.140075</td>
      <td>0.228513</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.138664</td>
      <td>0.232122</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.140360</td>
      <td>0.195164</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.141225</td>
      <td>0.208157</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.135310</td>
      <td>0.178094</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.139992</td>
      <td>0.204434</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.133911</td>
      <td>0.200988</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.133880</td>
      <td>0.175439</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.136770</td>
      <td>0.185651</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.131462</td>
      <td>0.197636</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.131260</td>
      <td>0.193190</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.135594</td>
      <td>0.209833</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.133908</td>
      <td>0.238912</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.132696</td>
      <td>0.202083</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     28%|████████████▉                                 | 28/100 [07:01<26:44, 22.28s/trial, best loss: 0.13269570469856262]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.154097</td>
      <td>0.257993</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.149661</td>
      <td>0.256499</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.145493</td>
      <td>0.214608</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.145870</td>
      <td>0.187691</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.140759</td>
      <td>0.197760</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.137866</td>
      <td>0.197585</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.139921</td>
      <td>0.205145</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.136736</td>
      <td>0.221008</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.140528</td>
      <td>0.203651</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.137804</td>
      <td>0.190685</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.136193</td>
      <td>0.165756</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.141448</td>
      <td>0.181642</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.134072</td>
      <td>0.193738</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.134319</td>
      <td>0.192154</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.135196</td>
      <td>0.214755</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.136968</td>
      <td>0.196824</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.134528</td>
      <td>0.203488</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     29%|█████████████▎                                | 29/100 [07:24<26:34, 22.45s/trial, best loss: 0.13269570469856262]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.168803</td>
      <td>0.192775</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.157617</td>
      <td>0.176765</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153256</td>
      <td>0.215707</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.149218</td>
      <td>0.203154</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.145423</td>
      <td>0.213807</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.145214</td>
      <td>0.215411</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.145220</td>
      <td>0.196809</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.143905</td>
      <td>0.204798</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.141523</td>
      <td>0.190106</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.143603</td>
      <td>0.214841</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.142043</td>
      <td>0.221993</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.137867</td>
      <td>0.205521</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.138485</td>
      <td>0.228096</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.138415</td>
      <td>0.236330</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.140400</td>
      <td>0.205180</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.140190</td>
      <td>0.188040</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.135287</td>
      <td>0.197370</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.139608</td>
      <td>0.233484</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     30%|█████████████▊                                | 30/100 [07:48<26:43, 22.91s/trial, best loss: 0.13269570469856262]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.172740</td>
      <td>0.178331</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.161982</td>
      <td>0.190741</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.155771</td>
      <td>0.220043</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.152206</td>
      <td>0.202981</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.145353</td>
      <td>0.203362</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.150603</td>
      <td>0.226631</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.144737</td>
      <td>0.199324</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.145750</td>
      <td>0.201224</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.142920</td>
      <td>0.212282</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.144028</td>
      <td>0.191417</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.142633</td>
      <td>0.210871</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.143444</td>
      <td>0.212476</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.142176</td>
      <td>0.207459</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.143133</td>
      <td>0.236489</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.141328</td>
      <td>0.223977</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.142438</td>
      <td>0.217697</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.140684</td>
      <td>0.222895</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.135982</td>
      <td>0.201001</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.139088</td>
      <td>0.202657</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     31%|██████████████▎                               | 31/100 [08:13<27:10, 23.63s/trial, best loss: 0.13269570469856262]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.155282</td>
      <td>0.185464</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.154073</td>
      <td>0.215248</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.149975</td>
      <td>0.254613</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.147053</td>
      <td>0.293687</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.143937</td>
      <td>0.244259</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.143146</td>
      <td>0.233673</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.143972</td>
      <td>0.191924</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.142591</td>
      <td>0.220638</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.141568</td>
      <td>0.203905</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.138342</td>
      <td>0.206792</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.137793</td>
      <td>0.214082</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.136722</td>
      <td>0.210435</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.137273</td>
      <td>0.223968</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.138774</td>
      <td>0.218468</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.140349</td>
      <td>0.195589</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.137034</td>
      <td>0.215439</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     32%|██████████████▋                               | 32/100 [08:34<25:59, 22.94s/trial, best loss: 0.13269570469856262]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.173252</td>
      <td>0.189628</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.156443</td>
      <td>0.182256</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.154529</td>
      <td>0.358543</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.155967</td>
      <td>0.189052</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.150421</td>
      <td>0.231179</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.148521</td>
      <td>0.257716</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.150336</td>
      <td>0.257137</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.148998</td>
      <td>0.160439</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.150426</td>
      <td>0.189264</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.148309</td>
      <td>0.224251</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.144120</td>
      <td>0.205692</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.146838</td>
      <td>0.207491</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.143809</td>
      <td>0.269314</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     33%|███████████████▏                              | 33/100 [08:51<23:42, 21.24s/trial, best loss: 0.13269570469856262]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.154794</td>
      <td>0.233754</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151464</td>
      <td>0.222442</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.144001</td>
      <td>0.243317</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.145985</td>
      <td>0.224214</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.148819</td>
      <td>0.239617</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.140543</td>
      <td>0.200690</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.142828</td>
      <td>0.214155</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.144841</td>
      <td>0.207711</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.138768</td>
      <td>0.193661</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.137209</td>
      <td>0.202891</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.137116</td>
      <td>0.209110</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.136444</td>
      <td>0.204970</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.133655</td>
      <td>0.205443</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.138244</td>
      <td>0.204844</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.135752</td>
      <td>0.192984</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.134214</td>
      <td>0.197434</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.131822</td>
      <td>0.184232</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     34%|███████████████▋                              | 34/100 [09:15<24:03, 21.88s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.159063</td>
      <td>0.236582</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.154198</td>
      <td>0.193831</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153534</td>
      <td>0.225321</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.153057</td>
      <td>0.230206</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.148201</td>
      <td>0.170835</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.146335</td>
      <td>0.218390</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.143253</td>
      <td>0.198693</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.146572</td>
      <td>0.211717</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.147383</td>
      <td>0.245262</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.147226</td>
      <td>0.254368</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.147929</td>
      <td>0.138648</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.145644</td>
      <td>0.210796</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.145915</td>
      <td>0.213291</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.145197</td>
      <td>0.334187</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.146940</td>
      <td>0.193975</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.148753</td>
      <td>0.141210</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     35%|████████████████                              | 35/100 [09:36<23:33, 21.75s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.155345</td>
      <td>0.288260</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.152017</td>
      <td>0.256684</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.151696</td>
      <td>0.241910</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.146543</td>
      <td>0.221550</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.148683</td>
      <td>0.201204</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.145590</td>
      <td>0.178282</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.141607</td>
      <td>0.209994</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.142562</td>
      <td>0.210429</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.143887</td>
      <td>0.204022</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.141721</td>
      <td>0.187138</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.140584</td>
      <td>0.211990</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.141388</td>
      <td>0.207041</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     36%|████████████████▌                             | 36/100 [09:53<21:28, 20.13s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.246884</td>
      <td>0.424440</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.189447</td>
      <td>0.290811</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.159058</td>
      <td>0.211703</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.155408</td>
      <td>0.173767</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.157130</td>
      <td>0.329416</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.155219</td>
      <td>0.206483</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.159268</td>
      <td>0.153505</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.153656</td>
      <td>0.224511</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.161282</td>
      <td>0.255212</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.156822</td>
      <td>0.230341</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.156328</td>
      <td>0.197216</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.155451</td>
      <td>0.281660</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.155256</td>
      <td>0.185582</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.154006</td>
      <td>0.226575</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.157036</td>
      <td>0.174495</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.153088</td>
      <td>0.176614</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.152645</td>
      <td>0.250904</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.161322</td>
      <td>0.246151</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.154371</td>
      <td>0.282314</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     37%|█████████████████                             | 37/100 [10:19<23:07, 22.02s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.150743</td>
      <td>0.262670</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.148315</td>
      <td>0.175862</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.144369</td>
      <td>0.203770</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.146928</td>
      <td>0.204422</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.144040</td>
      <td>0.193580</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.141452</td>
      <td>0.220236</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.142494</td>
      <td>0.189359</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.141252</td>
      <td>0.180693</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     38%|█████████████████▍                            | 38/100 [10:30<19:14, 18.62s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.160136</td>
      <td>0.245691</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.155130</td>
      <td>0.241279</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.148775</td>
      <td>0.238249</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.149933</td>
      <td>0.205525</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.149087</td>
      <td>0.186718</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149738</td>
      <td>0.226116</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.146311</td>
      <td>0.183605</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.144050</td>
      <td>0.262233</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.141272</td>
      <td>0.215512</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.143063</td>
      <td>0.216844</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.144875</td>
      <td>0.237433</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.143102</td>
      <td>0.242436</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.140912</td>
      <td>0.222113</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.141100</td>
      <td>0.220004</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.139470</td>
      <td>0.196540</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     39%|█████████████████▉                            | 39/100 [10:50<19:32, 19.22s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.157293</td>
      <td>0.247746</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.158140</td>
      <td>0.257612</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.154685</td>
      <td>0.235904</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.153787</td>
      <td>0.252253</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.152177</td>
      <td>0.235067</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149969</td>
      <td>0.250410</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.151193</td>
      <td>0.213838</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.150510</td>
      <td>0.232516</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.148226</td>
      <td>0.191187</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.151819</td>
      <td>0.199688</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.153376</td>
      <td>0.299692</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.149957</td>
      <td>0.208511</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.152504</td>
      <td>0.259198</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.146270</td>
      <td>0.274231</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.146561</td>
      <td>0.303533</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.146479</td>
      <td>0.244389</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.148680</td>
      <td>0.246239</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     40%|██████████████████▍                           | 40/100 [11:13<20:17, 20.29s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.158012</td>
      <td>0.226890</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151916</td>
      <td>0.232860</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.149677</td>
      <td>0.206090</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.144734</td>
      <td>0.233100</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.151136</td>
      <td>0.235108</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.142777</td>
      <td>0.255470</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.145379</td>
      <td>0.218923</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.143321</td>
      <td>0.219194</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.141863</td>
      <td>0.241724</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.136584</td>
      <td>0.218570</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.139436</td>
      <td>0.174956</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.140653</td>
      <td>0.228779</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.137948</td>
      <td>0.214799</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.135202</td>
      <td>0.220559</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.137401</td>
      <td>0.208216</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.138589</td>
      <td>0.173288</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.139371</td>
      <td>0.243769</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.140933</td>
      <td>0.206899</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.139471</td>
      <td>0.202851</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     41%|██████████████████▊                           | 41/100 [11:38<21:22, 21.73s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.332423</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.345930</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.340385</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.337550</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.347351</td>
      <td>0.899217</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.339183</td>
      <td>0.899217</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.343762</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.348075</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     42%|███████████████████▎                          | 42/100 [11:49<17:45, 18.38s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.237050</td>
      <td>0.425215</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.237031</td>
      <td>0.407816</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.247664</td>
      <td>0.379546</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.241951</td>
      <td>0.326600</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.236514</td>
      <td>0.532875</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.235113</td>
      <td>0.331721</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.228510</td>
      <td>0.537517</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.239876</td>
      <td>0.473150</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.231187</td>
      <td>0.490786</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.235308</td>
      <td>0.351272</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.229438</td>
      <td>0.424299</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.263603</td>
      <td>0.486271</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.228755</td>
      <td>0.355784</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.234382</td>
      <td>0.433196</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.258315</td>
      <td>0.518579</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     43%|███████████████████▊                          | 43/100 [12:09<17:57, 18.90s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.246108</td>
      <td>0.460641</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.253901</td>
      <td>0.270705</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.255882</td>
      <td>0.569604</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.238764</td>
      <td>0.512775</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.263987</td>
      <td>0.892422</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.267095</td>
      <td>0.314130</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.247272</td>
      <td>0.360573</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.249715</td>
      <td>0.367366</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.242132</td>
      <td>0.406578</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.279274</td>
      <td>0.281972</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.251984</td>
      <td>0.261234</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.287211</td>
      <td>0.276941</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     44%|████████████████████▏                         | 44/100 [12:25<16:54, 18.11s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.160809</td>
      <td>0.226070</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.150382</td>
      <td>0.180475</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153405</td>
      <td>0.221162</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.152996</td>
      <td>0.198301</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.148244</td>
      <td>0.208145</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149784</td>
      <td>0.296326</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.148703</td>
      <td>0.237478</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.147523</td>
      <td>0.201175</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.148707</td>
      <td>0.216504</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.145102</td>
      <td>0.247359</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.146010</td>
      <td>0.204565</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.148158</td>
      <td>0.205772</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.145678</td>
      <td>0.232518</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.145204</td>
      <td>0.230063</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.140693</td>
      <td>0.173337</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.141703</td>
      <td>0.330744</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.145206</td>
      <td>0.232764</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     45%|████████████████████▋                         | 45/100 [12:48<17:49, 19.44s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.168961</td>
      <td>0.185058</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.154495</td>
      <td>0.246349</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.155581</td>
      <td>0.208313</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     46%|█████████████████████▏                        | 46/100 [12:52<13:21, 14.85s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.247183</td>
      <td>0.223507</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.230896</td>
      <td>0.361645</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.255104</td>
      <td>0.401484</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.257039</td>
      <td>0.396429</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.239538</td>
      <td>0.307262</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.255740</td>
      <td>0.425995</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.320044</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.344555</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.342214</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.340454</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.332555</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.338530</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.337457</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.336823</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     47%|█████████████████████▌                        | 47/100 [13:13<14:47, 16.75s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.173845</td>
      <td>0.298617</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.156890</td>
      <td>0.244841</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.158763</td>
      <td>0.207617</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.152871</td>
      <td>0.221580</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.157735</td>
      <td>0.191269</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.150923</td>
      <td>0.172213</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.152691</td>
      <td>0.271112</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.154428</td>
      <td>0.269167</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.148110</td>
      <td>0.218908</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.148282</td>
      <td>0.288284</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.148495</td>
      <td>0.235379</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.147693</td>
      <td>0.204176</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     48%|██████████████████████                        | 48/100 [13:31<14:55, 17.23s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.153821</td>
      <td>0.286362</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151188</td>
      <td>0.209846</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.147163</td>
      <td>0.269045</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.143979</td>
      <td>0.244038</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.142889</td>
      <td>0.239065</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.141918</td>
      <td>0.284458</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.141082</td>
      <td>0.180191</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.138163</td>
      <td>0.172173</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.141739</td>
      <td>0.189356</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.139361</td>
      <td>0.176920</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.138686</td>
      <td>0.195629</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.136160</td>
      <td>0.177925</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.139418</td>
      <td>0.212399</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.134481</td>
      <td>0.184417</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.133755</td>
      <td>0.198286</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.135208</td>
      <td>0.227384</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.133264</td>
      <td>0.192577</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.134441</td>
      <td>0.212332</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.136094</td>
      <td>0.192591</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     49%|██████████████████████▌                       | 49/100 [13:56<16:33, 19.48s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.171493</td>
      <td>0.307974</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.159727</td>
      <td>0.256448</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153044</td>
      <td>0.253139</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.151634</td>
      <td>0.277208</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.148385</td>
      <td>0.346112</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.148824</td>
      <td>0.209925</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.148548</td>
      <td>0.206068</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.145310</td>
      <td>0.176523</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.148240</td>
      <td>0.162744</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.144528</td>
      <td>0.239187</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.149794</td>
      <td>0.227142</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.142287</td>
      <td>0.187652</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.145676</td>
      <td>0.225041</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.142436</td>
      <td>0.161590</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.147070</td>
      <td>0.236300</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.144679</td>
      <td>0.275914</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     50%|███████████████████████                       | 50/100 [14:17<16:41, 20.03s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.156895</td>
      <td>0.220744</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.158053</td>
      <td>0.248021</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.152955</td>
      <td>0.251755</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.154671</td>
      <td>0.190896</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.153168</td>
      <td>0.209386</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149205</td>
      <td>0.210760</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.147004</td>
      <td>0.255964</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.146596</td>
      <td>0.192808</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.148957</td>
      <td>0.255034</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.143660</td>
      <td>0.208286</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.143941</td>
      <td>0.213471</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.143350</td>
      <td>0.247071</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.144102</td>
      <td>0.266975</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.143282</td>
      <td>0.218795</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.143101</td>
      <td>0.224062</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.144097</td>
      <td>0.204343</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.143559</td>
      <td>0.165846</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.142237</td>
      <td>0.221315</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     51%|███████████████████████▍                      | 51/100 [14:42<17:22, 21.27s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.157653</td>
      <td>0.247831</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.148582</td>
      <td>0.196901</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.151208</td>
      <td>0.263215</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.147462</td>
      <td>0.172459</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.144541</td>
      <td>0.221669</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.144141</td>
      <td>0.240411</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.142273</td>
      <td>0.204199</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.145686</td>
      <td>0.223973</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.137941</td>
      <td>0.226674</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.143289</td>
      <td>0.184835</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     52%|███████████████████████▉                      | 52/100 [14:55<15:12, 19.00s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.264441</td>
      <td>0.437883</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.257675</td>
      <td>0.462606</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.251409</td>
      <td>0.387173</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.247953</td>
      <td>0.438541</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.306470</td>
      <td>0.451892</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.243176</td>
      <td>0.442219</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.243391</td>
      <td>0.461203</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.240064</td>
      <td>0.464902</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     53%|████████████████████████▍                     | 53/100 [15:07<13:03, 16.67s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.158647</td>
      <td>0.229139</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151459</td>
      <td>0.287066</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.150296</td>
      <td>0.222202</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.152756</td>
      <td>0.244538</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.149888</td>
      <td>0.200117</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149761</td>
      <td>0.231569</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.146061</td>
      <td>0.256620</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.147855</td>
      <td>0.257604</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.147081</td>
      <td>0.202776</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.146310</td>
      <td>0.181829</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.144591</td>
      <td>0.183731</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.144297</td>
      <td>0.244443</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.140974</td>
      <td>0.224717</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.145514</td>
      <td>0.175265</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.146905</td>
      <td>0.245321</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     54%|████████████████████████▊                     | 54/100 [15:27<13:45, 17.94s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.157493</td>
      <td>0.288796</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.153223</td>
      <td>0.253851</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.152635</td>
      <td>0.197879</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.154169</td>
      <td>0.167102</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.150852</td>
      <td>0.210536</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.145540</td>
      <td>0.197922</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.148030</td>
      <td>0.274831</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.146857</td>
      <td>0.183007</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.145608</td>
      <td>0.238674</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.142105</td>
      <td>0.215984</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.141070</td>
      <td>0.159369</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.139429</td>
      <td>0.230347</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.138801</td>
      <td>0.217004</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     55%|█████████████████████████▎                    | 55/100 [15:45<13:20, 17.80s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.163332</td>
      <td>0.261557</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.154173</td>
      <td>0.187998</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153658</td>
      <td>0.184663</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.156452</td>
      <td>0.254957</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.151602</td>
      <td>0.189183</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149897</td>
      <td>0.279922</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.147743</td>
      <td>0.270845</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     56%|█████████████████████████▊                    | 56/100 [15:54<11:07, 15.17s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.159095</td>
      <td>0.264581</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     57%|██████████████████████████▏                   | 57/100 [15:55<07:53, 11.02s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.180165</td>
      <td>0.210342</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.163273</td>
      <td>0.288876</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.155824</td>
      <td>0.233314</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.156679</td>
      <td>0.220685</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.155437</td>
      <td>0.162979</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149560</td>
      <td>0.266534</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.153071</td>
      <td>0.219283</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.149860</td>
      <td>0.206024</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.151972</td>
      <td>0.204734</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.146945</td>
      <td>0.233327</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.146792</td>
      <td>0.163857</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.146046</td>
      <td>0.277867</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.146032</td>
      <td>0.168512</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.145469</td>
      <td>0.294087</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.148592</td>
      <td>0.297565</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.147218</td>
      <td>0.213382</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.150206</td>
      <td>0.144574</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.149840</td>
      <td>0.290239</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.148486</td>
      <td>0.417301</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.146831</td>
      <td>0.153121</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     58%|██████████████████████████▋                   | 58/100 [16:22<10:58, 15.68s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.157723</td>
      <td>0.260646</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151339</td>
      <td>0.215021</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.150437</td>
      <td>0.264769</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.153812</td>
      <td>0.185279</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.149697</td>
      <td>0.269895</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.150786</td>
      <td>0.227686</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.147794</td>
      <td>0.205035</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.147266</td>
      <td>0.301543</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.147497</td>
      <td>0.264680</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.149048</td>
      <td>0.252611</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.146873</td>
      <td>0.225988</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     59%|███████████████████████████▏                  | 59/100 [16:37<10:37, 15.55s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.150114</td>
      <td>0.211779</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.149229</td>
      <td>0.191834</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.144642</td>
      <td>0.232656</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.142891</td>
      <td>0.202888</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.142173</td>
      <td>0.250319</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.138263</td>
      <td>0.228354</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.137183</td>
      <td>0.212892</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.140047</td>
      <td>0.196034</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.136555</td>
      <td>0.211375</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.137144</td>
      <td>0.209025</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.132023</td>
      <td>0.178118</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.133361</td>
      <td>0.209972</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.137618</td>
      <td>0.214112</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.135313</td>
      <td>0.179640</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.134066</td>
      <td>0.181832</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.133343</td>
      <td>0.184077</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.135623</td>
      <td>0.196577</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     60%|███████████████████████████▌                  | 60/100 [17:00<11:51, 17.78s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.253066</td>
      <td>0.332084</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.256963</td>
      <td>0.241125</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.279512</td>
      <td>0.463107</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.232852</td>
      <td>0.359488</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.234880</td>
      <td>0.505529</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.274058</td>
      <td>0.564065</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.253032</td>
      <td>0.391389</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.246357</td>
      <td>0.433142</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.244948</td>
      <td>0.415706</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.246513</td>
      <td>0.407045</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.252197</td>
      <td>0.575341</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.253608</td>
      <td>0.411538</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.266370</td>
      <td>0.548921</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.243686</td>
      <td>0.514469</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     61%|████████████████████████████                  | 61/100 [17:18<11:39, 17.93s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.170346</td>
      <td>0.215461</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.154027</td>
      <td>0.214488</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.151119</td>
      <td>0.236707</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.151689</td>
      <td>0.246854</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.149264</td>
      <td>0.224631</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149658</td>
      <td>0.187215</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.148319</td>
      <td>0.272298</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.148890</td>
      <td>0.338029</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.145224</td>
      <td>0.240500</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     62%|████████████████████████████▌                 | 62/100 [17:30<10:08, 16.02s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.204166</td>
      <td>0.271950</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.175599</td>
      <td>0.375426</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.157937</td>
      <td>0.196443</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.153027</td>
      <td>0.275219</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.157845</td>
      <td>0.136148</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     63%|████████████████████████████▉                 | 63/100 [17:36<08:07, 13.18s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.158826</td>
      <td>0.241166</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.158318</td>
      <td>0.194139</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     64%|█████████████████████████████▍                | 64/100 [17:39<05:59, 10.00s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.157341</td>
      <td>0.225554</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.149580</td>
      <td>0.259186</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.147683</td>
      <td>0.234329</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.144330</td>
      <td>0.249062</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.144074</td>
      <td>0.239697</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.143993</td>
      <td>0.194126</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.141518</td>
      <td>0.193074</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.141568</td>
      <td>0.202468</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.135758</td>
      <td>0.189105</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.139217</td>
      <td>0.208780</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.138828</td>
      <td>0.173057</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.135713</td>
      <td>0.173615</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.134857</td>
      <td>0.184310</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.136244</td>
      <td>0.212018</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.135268</td>
      <td>0.170393</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.135845</td>
      <td>0.195255</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.132422</td>
      <td>0.203451</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.132849</td>
      <td>0.181774</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.132764</td>
      <td>0.194380</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.133062</td>
      <td>0.184696</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     65%|█████████████████████████████▉                | 65/100 [18:05<08:41, 14.91s/trial, best loss: 0.13182219862937927]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.155540</td>
      <td>0.299459</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151102</td>
      <td>0.219402</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.149852</td>
      <td>0.291408</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.147771</td>
      <td>0.200572</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.143293</td>
      <td>0.213384</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.142957</td>
      <td>0.243924</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.143550</td>
      <td>0.194306</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.140811</td>
      <td>0.194070</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.141435</td>
      <td>0.201105</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.142030</td>
      <td>0.235396</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.138223</td>
      <td>0.193886</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.136174</td>
      <td>0.199317</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.135215</td>
      <td>0.184905</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.136846</td>
      <td>0.180991</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.132569</td>
      <td>0.205311</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.135513</td>
      <td>0.188601</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.137852</td>
      <td>0.216326</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.131547</td>
      <td>0.194653</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.133701</td>
      <td>0.245046</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.128472</td>
      <td>0.201047</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     66%|███████████████████████████████                | 66/100 [18:31<10:17, 18.17s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.155501</td>
      <td>0.217591</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.156229</td>
      <td>0.221910</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.152834</td>
      <td>0.192168</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.152483</td>
      <td>0.250933</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.152288</td>
      <td>0.182336</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.147326</td>
      <td>0.217590</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.145670</td>
      <td>0.242228</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.140424</td>
      <td>0.247831</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.144201</td>
      <td>0.183360</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.141159</td>
      <td>0.201359</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.138553</td>
      <td>0.201899</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.139737</td>
      <td>0.195013</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.140482</td>
      <td>0.179146</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.140578</td>
      <td>0.207775</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.138776</td>
      <td>0.246926</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.138158</td>
      <td>0.202262</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.135576</td>
      <td>0.220034</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.130071</td>
      <td>0.190983</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.138055</td>
      <td>0.206550</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     67%|███████████████████████████████▍               | 67/100 [18:55<11:00, 20.01s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.161326</td>
      <td>0.310213</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.156402</td>
      <td>0.221219</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.152422</td>
      <td>0.230495</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.149263</td>
      <td>0.189929</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.150233</td>
      <td>0.200791</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.147397</td>
      <td>0.269245</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.144896</td>
      <td>0.209916</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.145523</td>
      <td>0.207769</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.140481</td>
      <td>0.234083</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.140062</td>
      <td>0.213178</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.137958</td>
      <td>0.233813</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.139504</td>
      <td>0.250736</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.138811</td>
      <td>0.188865</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.140106</td>
      <td>0.238699</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.140343</td>
      <td>0.205934</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.138617</td>
      <td>0.231684</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.137745</td>
      <td>0.193538</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.134077</td>
      <td>0.188616</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     68%|███████████████████████████████▉               | 68/100 [19:19<11:09, 20.92s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.267547</td>
      <td>0.300391</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.277085</td>
      <td>0.449046</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.293805</td>
      <td>0.496086</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.263592</td>
      <td>0.516483</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.265555</td>
      <td>0.269080</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.248057</td>
      <td>0.534953</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.304453</td>
      <td>0.538158</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.275610</td>
      <td>0.552838</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.272681</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.334059</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.336455</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.343408</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.336350</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.339188</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.340943</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.338663</td>
      <td>0.902153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     69%|████████████████████████████████▍              | 69/100 [19:39<10:44, 20.79s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.157736</td>
      <td>0.260520</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151622</td>
      <td>0.224758</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.147007</td>
      <td>0.187997</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.145321</td>
      <td>0.212501</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.144857</td>
      <td>0.254330</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.140674</td>
      <td>0.186691</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.144916</td>
      <td>0.216182</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.137626</td>
      <td>0.193737</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.138999</td>
      <td>0.237396</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.139438</td>
      <td>0.183863</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.135435</td>
      <td>0.204566</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.132803</td>
      <td>0.206114</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.137220</td>
      <td>0.208404</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.134093</td>
      <td>0.185792</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.136084</td>
      <td>0.173625</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.134697</td>
      <td>0.206978</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.134734</td>
      <td>0.215849</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.137515</td>
      <td>0.204684</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.132993</td>
      <td>0.190313</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.133580</td>
      <td>0.178246</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     70%|████████████████████████████████▉              | 70/100 [20:05<11:09, 22.31s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.229075</td>
      <td>0.473461</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.233655</td>
      <td>0.430227</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.163440</td>
      <td>0.259717</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.163939</td>
      <td>0.183453</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.156469</td>
      <td>0.192956</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.153814</td>
      <td>0.185835</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.151819</td>
      <td>0.315321</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.155886</td>
      <td>0.251055</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.148714</td>
      <td>0.250517</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.152804</td>
      <td>0.265831</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.150479</td>
      <td>0.230882</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.149714</td>
      <td>0.208787</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.147738</td>
      <td>0.345984</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.147586</td>
      <td>0.180200</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.149125</td>
      <td>0.275903</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.146719</td>
      <td>0.212335</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.147617</td>
      <td>0.274636</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.149973</td>
      <td>0.240487</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     71%|█████████████████████████████████▎             | 71/100 [20:28<10:54, 22.58s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.170122</td>
      <td>0.210667</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.159748</td>
      <td>0.226774</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153383</td>
      <td>0.201024</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.150942</td>
      <td>0.225384</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.153627</td>
      <td>0.289551</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.152383</td>
      <td>0.219544</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.148827</td>
      <td>0.199067</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.151881</td>
      <td>0.219655</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.152861</td>
      <td>0.225627</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.144384</td>
      <td>0.243275</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.149242</td>
      <td>0.186624</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.147084</td>
      <td>0.185185</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.145789</td>
      <td>0.281587</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.144799</td>
      <td>0.219446</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.145285</td>
      <td>0.220932</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.144361</td>
      <td>0.174031</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.139898</td>
      <td>0.187600</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.142462</td>
      <td>0.194178</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.142123</td>
      <td>0.241620</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     72%|█████████████████████████████████▊             | 72/100 [20:52<10:47, 23.13s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.178587</td>
      <td>0.279955</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.162492</td>
      <td>0.239476</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.161906</td>
      <td>0.321966</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.156414</td>
      <td>0.172788</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.160301</td>
      <td>0.189626</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.153050</td>
      <td>0.175569</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.153128</td>
      <td>0.187838</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.153431</td>
      <td>0.242866</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.146023</td>
      <td>0.229427</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.149849</td>
      <td>0.242973</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.148618</td>
      <td>0.218369</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.148739</td>
      <td>0.227807</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.145071</td>
      <td>0.189745</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.145739</td>
      <td>0.189069</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.149408</td>
      <td>0.241293</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.146343</td>
      <td>0.195406</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.146253</td>
      <td>0.162499</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     73%|██████████████████████████████████▎            | 73/100 [21:14<10:14, 22.74s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.173591</td>
      <td>0.219883</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.163099</td>
      <td>0.277958</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.157043</td>
      <td>0.209593</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.152829</td>
      <td>0.188218</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.146659</td>
      <td>0.255433</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149111</td>
      <td>0.211376</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.144584</td>
      <td>0.195663</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.141980</td>
      <td>0.197883</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.143185</td>
      <td>0.251757</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.146071</td>
      <td>0.267708</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.144865</td>
      <td>0.202255</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.141214</td>
      <td>0.193495</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.138554</td>
      <td>0.248541</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.140811</td>
      <td>0.242374</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.143440</td>
      <td>0.242072</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.139925</td>
      <td>0.168577</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     74%|██████████████████████████████████▊            | 74/100 [21:35<09:34, 22.08s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.153032</td>
      <td>0.221690</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.147876</td>
      <td>0.230328</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.147366</td>
      <td>0.226665</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.143880</td>
      <td>0.253310</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.140226</td>
      <td>0.224625</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.139258</td>
      <td>0.196491</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.138328</td>
      <td>0.222148</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.137994</td>
      <td>0.202528</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.138494</td>
      <td>0.207461</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.135747</td>
      <td>0.185130</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.137127</td>
      <td>0.181658</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.135660</td>
      <td>0.191529</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.137720</td>
      <td>0.195204</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.135448</td>
      <td>0.196444</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.136115</td>
      <td>0.200697</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     75%|███████████████████████████████████▎           | 75/100 [21:54<08:50, 21.21s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.159804</td>
      <td>0.208951</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.154636</td>
      <td>0.179493</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.151680</td>
      <td>0.234854</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.150952</td>
      <td>0.294565</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.153985</td>
      <td>0.221764</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.148254</td>
      <td>0.235479</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.150284</td>
      <td>0.169156</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.147970</td>
      <td>0.292179</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.148535</td>
      <td>0.203459</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.143839</td>
      <td>0.227231</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.146574</td>
      <td>0.275721</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.147219</td>
      <td>0.225067</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.148094</td>
      <td>0.195455</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.144949</td>
      <td>0.214633</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.143415</td>
      <td>0.203496</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.143941</td>
      <td>0.198345</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.140696</td>
      <td>0.164368</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.139307</td>
      <td>0.197705</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     76%|███████████████████████████████████▋           | 76/100 [22:17<08:44, 21.86s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.164660</td>
      <td>0.313040</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.155191</td>
      <td>0.278368</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.154689</td>
      <td>0.245806</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.153683</td>
      <td>0.233953</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.148692</td>
      <td>0.262994</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149574</td>
      <td>0.263139</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.151503</td>
      <td>0.265389</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.149863</td>
      <td>0.245010</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.144595</td>
      <td>0.223546</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.143387</td>
      <td>0.241342</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.142564</td>
      <td>0.208424</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.144619</td>
      <td>0.272598</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.144066</td>
      <td>0.218189</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.143922</td>
      <td>0.185788</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.144208</td>
      <td>0.174134</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.143905</td>
      <td>0.200984</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.141248</td>
      <td>0.211444</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     77%|████████████████████████████████████▏          | 77/100 [22:41<08:31, 22.25s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.162829</td>
      <td>0.209370</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.156203</td>
      <td>0.190103</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.151018</td>
      <td>0.194223</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.145708</td>
      <td>0.237911</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.144040</td>
      <td>0.256700</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.143904</td>
      <td>0.221774</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.143911</td>
      <td>0.220618</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.148555</td>
      <td>0.225531</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.141519</td>
      <td>0.213376</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.146520</td>
      <td>0.215973</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.142564</td>
      <td>0.225153</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.142780</td>
      <td>0.185934</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.140231</td>
      <td>0.208553</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.141118</td>
      <td>0.236399</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     78%|████████████████████████████████████▋          | 78/100 [22:59<07:45, 21.16s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.162216</td>
      <td>0.223288</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.155741</td>
      <td>0.192309</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.154420</td>
      <td>0.227740</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.147545</td>
      <td>0.216497</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.141415</td>
      <td>0.290734</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.142642</td>
      <td>0.216740</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.143425</td>
      <td>0.232664</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.140409</td>
      <td>0.211519</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.141255</td>
      <td>0.225111</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.137399</td>
      <td>0.197963</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.137642</td>
      <td>0.223689</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.138260</td>
      <td>0.212562</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.137927</td>
      <td>0.234125</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     79%|█████████████████████████████████████▏         | 79/100 [23:16<06:57, 19.87s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.171477</td>
      <td>0.176975</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.153007</td>
      <td>0.194370</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153316</td>
      <td>0.205656</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.149611</td>
      <td>0.266549</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.147105</td>
      <td>0.288226</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.147515</td>
      <td>0.196881</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.149744</td>
      <td>0.198704</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.150582</td>
      <td>0.183226</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.145579</td>
      <td>0.211639</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.144083</td>
      <td>0.172527</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.144431</td>
      <td>0.352645</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.144537</td>
      <td>0.158452</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.145792</td>
      <td>0.220841</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.142366</td>
      <td>0.216108</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.143379</td>
      <td>0.205265</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.142152</td>
      <td>0.216405</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.144942</td>
      <td>0.244132</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.145334</td>
      <td>0.194484</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.143948</td>
      <td>0.161482</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.145573</td>
      <td>0.178681</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     80%|█████████████████████████████████████▌         | 80/100 [23:42<07:13, 21.68s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.158353</td>
      <td>0.264882</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.154315</td>
      <td>0.272942</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153292</td>
      <td>0.254367</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.150399</td>
      <td>0.237359</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.148141</td>
      <td>0.280630</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.149028</td>
      <td>0.241562</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.149246</td>
      <td>0.167919</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.147713</td>
      <td>0.173379</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.144428</td>
      <td>0.206009</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.145561</td>
      <td>0.215772</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.141641</td>
      <td>0.195185</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.141739</td>
      <td>0.201290</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.145909</td>
      <td>0.238764</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.143644</td>
      <td>0.205800</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.141499</td>
      <td>0.284205</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.140407</td>
      <td>0.283311</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.146844</td>
      <td>0.227282</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.146113</td>
      <td>0.215429</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.141568</td>
      <td>0.200848</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     81%|██████████████████████████████████████         | 81/100 [24:08<07:14, 22.84s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.155614</td>
      <td>0.247002</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.157359</td>
      <td>0.235293</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.152417</td>
      <td>0.204895</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.147466</td>
      <td>0.207997</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.151026</td>
      <td>0.226564</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.147734</td>
      <td>0.222262</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.142129</td>
      <td>0.192753</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.144636</td>
      <td>0.228871</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.142245</td>
      <td>0.237229</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.139847</td>
      <td>0.196317</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.142124</td>
      <td>0.224239</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.142674</td>
      <td>0.174575</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.138380</td>
      <td>0.175502</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.140942</td>
      <td>0.225575</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.136895</td>
      <td>0.220441</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     82%|██████████████████████████████████████▌        | 82/100 [24:28<06:37, 22.08s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.157337</td>
      <td>0.249441</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151500</td>
      <td>0.239586</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153041</td>
      <td>0.192254</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.151379</td>
      <td>0.204621</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.146789</td>
      <td>0.255131</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.144254</td>
      <td>0.171487</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.153341</td>
      <td>0.227715</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.147033</td>
      <td>0.224116</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.144085</td>
      <td>0.157258</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.144168</td>
      <td>0.256045</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.149276</td>
      <td>0.193362</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.144138</td>
      <td>0.233982</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.142950</td>
      <td>0.190231</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.145308</td>
      <td>0.198803</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.147310</td>
      <td>0.210522</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.141721</td>
      <td>0.163976</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     83%|███████████████████████████████████████        | 83/100 [24:49<06:08, 21.68s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.181813</td>
      <td>0.163422</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.168778</td>
      <td>0.178244</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.160145</td>
      <td>0.188355</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.155001</td>
      <td>0.182272</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.158035</td>
      <td>0.203945</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.153931</td>
      <td>0.198295</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.149336</td>
      <td>0.202840</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.150637</td>
      <td>0.207278</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.148302</td>
      <td>0.206075</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.146446</td>
      <td>0.213715</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.146725</td>
      <td>0.199607</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.144616</td>
      <td>0.242836</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.143932</td>
      <td>0.201375</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.143024</td>
      <td>0.208750</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.146025</td>
      <td>0.205245</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.144106</td>
      <td>0.219956</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.140911</td>
      <td>0.214144</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.143198</td>
      <td>0.203185</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.139806</td>
      <td>0.204141</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.135178</td>
      <td>0.220107</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     84%|███████████████████████████████████████▍       | 84/100 [25:14<06:06, 22.89s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.158195</td>
      <td>0.293139</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.149984</td>
      <td>0.209103</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.154251</td>
      <td>0.214952</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.147999</td>
      <td>0.193725</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.145975</td>
      <td>0.251989</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.147950</td>
      <td>0.164976</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.149815</td>
      <td>0.235850</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.141137</td>
      <td>0.228584</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.142144</td>
      <td>0.223392</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.140837</td>
      <td>0.247972</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.144186</td>
      <td>0.208596</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     85%|███████████████████████████████████████▉       | 85/100 [25:28<05:03, 20.24s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.178036</td>
      <td>0.352986</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.159556</td>
      <td>0.204007</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.155472</td>
      <td>0.233206</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.156447</td>
      <td>0.236531</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.153460</td>
      <td>0.220609</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.148102</td>
      <td>0.223294</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.151514</td>
      <td>0.203832</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.146117</td>
      <td>0.230025</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.147261</td>
      <td>0.230721</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.148438</td>
      <td>0.182332</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.143319</td>
      <td>0.260690</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.146273</td>
      <td>0.200521</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.154295</td>
      <td>0.207514</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.145402</td>
      <td>0.327131</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.145337</td>
      <td>0.239176</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.144472</td>
      <td>0.157616</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.142662</td>
      <td>0.341749</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.140941</td>
      <td>0.202400</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     86%|████████████████████████████████████████▍      | 86/100 [25:52<04:55, 21.12s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.155615</td>
      <td>0.255374</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151284</td>
      <td>0.224062</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.148673</td>
      <td>0.310907</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.147061</td>
      <td>0.187586</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.146439</td>
      <td>0.234508</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.146970</td>
      <td>0.141953</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.146122</td>
      <td>0.146411</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.148861</td>
      <td>0.245133</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.147604</td>
      <td>0.286413</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.145349</td>
      <td>0.157902</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.143190</td>
      <td>0.251472</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.146447</td>
      <td>0.225237</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     87%|████████████████████████████████████████▉      | 87/100 [26:07<04:12, 19.40s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.164023</td>
      <td>0.205276</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.159032</td>
      <td>0.239839</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153536</td>
      <td>0.249863</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.149002</td>
      <td>0.217152</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.145985</td>
      <td>0.219572</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.146896</td>
      <td>0.176600</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.148092</td>
      <td>0.248493</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.146142</td>
      <td>0.251647</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.144316</td>
      <td>0.210022</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.141234</td>
      <td>0.215150</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.141819</td>
      <td>0.251005</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.142850</td>
      <td>0.179933</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.142720</td>
      <td>0.200916</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.139155</td>
      <td>0.258709</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.142514</td>
      <td>0.221119</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.140951</td>
      <td>0.186505</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.142787</td>
      <td>0.197058</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.139257</td>
      <td>0.170101</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.139731</td>
      <td>0.207551</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     88%|█████████████████████████████████████████▎     | 88/100 [26:31<04:11, 20.94s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.165782</td>
      <td>0.307565</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.159521</td>
      <td>0.203050</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.159139</td>
      <td>0.236647</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.154882</td>
      <td>0.215954</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.152356</td>
      <td>0.189333</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.155493</td>
      <td>0.254085</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.151004</td>
      <td>0.241847</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.151246</td>
      <td>0.143864</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.150075</td>
      <td>0.272867</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.156463</td>
      <td>0.223162</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.149093</td>
      <td>0.186872</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.154221</td>
      <td>0.276457</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.152368</td>
      <td>0.221286</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.149890</td>
      <td>0.195851</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.150259</td>
      <td>0.219712</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.147333</td>
      <td>0.238082</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.152776</td>
      <td>0.272321</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     89%|█████████████████████████████████████████▊     | 89/100 [26:53<03:53, 21.23s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.153944</td>
      <td>0.206956</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151115</td>
      <td>0.208966</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.147915</td>
      <td>0.192316</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.144031</td>
      <td>0.218714</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.145411</td>
      <td>0.192293</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.137863</td>
      <td>0.255011</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.141614</td>
      <td>0.232586</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.140074</td>
      <td>0.201057</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.139104</td>
      <td>0.230463</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.140652</td>
      <td>0.223728</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.140672</td>
      <td>0.208345</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.139930</td>
      <td>0.248977</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.138507</td>
      <td>0.190464</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     90%|██████████████████████████████████████████▎    | 90/100 [27:10<03:18, 19.86s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.236117</td>
      <td>0.399745</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.237158</td>
      <td>0.625425</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.234372</td>
      <td>0.546390</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.256612</td>
      <td>0.381857</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.239939</td>
      <td>0.405503</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.235625</td>
      <td>0.396887</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.251816</td>
      <td>0.341511</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.250681</td>
      <td>0.528776</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.250177</td>
      <td>0.451383</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.244024</td>
      <td>0.397779</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.236951</td>
      <td>0.448139</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.242799</td>
      <td>0.547082</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.253530</td>
      <td>0.414641</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.233972</td>
      <td>0.376397</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.242488</td>
      <td>0.506568</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.351250</td>
      <td>0.784736</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     91%|██████████████████████████████████████████▊    | 91/100 [27:31<03:00, 20.05s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.169050</td>
      <td>0.270766</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.158961</td>
      <td>0.276935</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.152327</td>
      <td>0.224788</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.151558</td>
      <td>0.285743</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.151770</td>
      <td>0.205763</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.148828</td>
      <td>0.264407</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.145212</td>
      <td>0.233319</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.149067</td>
      <td>0.202937</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.145532</td>
      <td>0.201299</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.145015</td>
      <td>0.217843</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.146505</td>
      <td>0.267559</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.145327</td>
      <td>0.212896</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.145443</td>
      <td>0.230920</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.144743</td>
      <td>0.224075</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     92%|███████████████████████████████████████████▏   | 92/100 [27:48<02:35, 19.42s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.202683</td>
      <td>0.271230</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.166301</td>
      <td>0.296366</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153584</td>
      <td>0.193421</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.163716</td>
      <td>0.140913</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.152975</td>
      <td>0.228916</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.158370</td>
      <td>0.226082</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.151250</td>
      <td>0.244284</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     93%|███████████████████████████████████████████▋   | 93/100 [27:57<01:54, 16.29s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.161607</td>
      <td>0.216502</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.154714</td>
      <td>0.210289</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.153323</td>
      <td>0.199921</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.148578</td>
      <td>0.235819</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.150401</td>
      <td>0.213885</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.146584</td>
      <td>0.215642</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.146702</td>
      <td>0.240186</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.149475</td>
      <td>0.204254</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.147051</td>
      <td>0.318362</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     94%|████████████████████████████████████████████▏  | 94/100 [28:09<01:29, 14.93s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.167906</td>
      <td>0.259242</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.159706</td>
      <td>0.239828</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.155762</td>
      <td>0.173202</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.155204</td>
      <td>0.213238</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.152528</td>
      <td>0.163492</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.152705</td>
      <td>0.214428</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.148685</td>
      <td>0.279601</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.148834</td>
      <td>0.234686</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.145647</td>
      <td>0.199252</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.148625</td>
      <td>0.213249</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.145581</td>
      <td>0.179193</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.141956</td>
      <td>0.190743</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.144296</td>
      <td>0.238249</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.142755</td>
      <td>0.206510</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.145639</td>
      <td>0.167634</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.147816</td>
      <td>0.188332</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.143397</td>
      <td>0.231020</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.141398</td>
      <td>0.234943</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     95%|████████████████████████████████████████████▋  | 95/100 [28:33<01:27, 17.55s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.159412</td>
      <td>0.341526</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.152605</td>
      <td>0.250997</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.151615</td>
      <td>0.207556</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.149692</td>
      <td>0.361150</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.149652</td>
      <td>0.208080</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.151925</td>
      <td>0.342654</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.150777</td>
      <td>0.230835</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.146943</td>
      <td>0.207785</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.145015</td>
      <td>0.164238</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.146674</td>
      <td>0.219904</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.149418</td>
      <td>0.165297</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.146313</td>
      <td>0.189400</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.144023</td>
      <td>0.261744</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.144746</td>
      <td>0.286137</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.142060</td>
      <td>0.181350</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     96%|█████████████████████████████████████████████  | 96/100 [28:52<01:12, 18.06s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.155856</td>
      <td>0.336353</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.155230</td>
      <td>0.269974</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.148463</td>
      <td>0.184205</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.146481</td>
      <td>0.212799</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.143238</td>
      <td>0.184731</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.142828</td>
      <td>0.177256</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.145693</td>
      <td>0.201286</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.143152</td>
      <td>0.202496</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.140705</td>
      <td>0.229073</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.138355</td>
      <td>0.231279</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     97%|█████████████████████████████████████████████▌ | 97/100 [29:05<00:49, 16.50s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.251475</td>
      <td>0.355777</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.246147</td>
      <td>0.442853</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.244313</td>
      <td>0.376632</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.227486</td>
      <td>0.437353</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.217916</td>
      <td>0.361986</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.235194</td>
      <td>0.445982</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.228937</td>
      <td>0.371561</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.211378</td>
      <td>0.244907</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.161090</td>
      <td>0.204574</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.155508</td>
      <td>0.258912</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.157831</td>
      <td>0.193453</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.153758</td>
      <td>0.202941</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.153581</td>
      <td>0.377188</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.150705</td>
      <td>0.196924</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.153338</td>
      <td>0.272195</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.153483</td>
      <td>0.269553</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.155928</td>
      <td>0.235786</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.149785</td>
      <td>0.164070</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.150556</td>
      <td>0.232658</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     98%|██████████████████████████████████████████████ | 98/100 [29:29<00:37, 18.88s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.164508</td>
      <td>0.205700</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.153951</td>
      <td>0.214883</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.152736</td>
      <td>0.247493</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.145500</td>
      <td>0.245163</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.145266</td>
      <td>0.221146</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.150016</td>
      <td>0.201555</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.142347</td>
      <td>0.192550</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.146407</td>
      <td>0.167611</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.142471</td>
      <td>0.190586</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.136223</td>
      <td>0.275381</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.137557</td>
      <td>0.221906</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.139235</td>
      <td>0.210712</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.139729</td>
      <td>0.181575</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.137599</td>
      <td>0.211206</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.140031</td>
      <td>0.228419</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.139503</td>
      <td>0.194129</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.134222</td>
      <td>0.197152</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


     99%|██████████████████████████████████████████████▌| 99/100 [29:51<00:19, 19.79s/trial, best loss: 0.1284717321395874]


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.156397</td>
      <td>0.209976</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.152624</td>
      <td>0.315141</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.148414</td>
      <td>0.234133</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.147357</td>
      <td>0.201878</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.147269</td>
      <td>0.238779</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.144389</td>
      <td>0.197335</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.145433</td>
      <td>0.251422</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.140403</td>
      <td>0.184167</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.139279</td>
      <td>0.236334</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.137117</td>
      <td>0.219727</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.141830</td>
      <td>0.176662</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.136899</td>
      <td>0.172791</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.135654</td>
      <td>0.217206</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.139379</td>
      <td>0.179674</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.134881</td>
      <td>0.205049</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.136767</td>
      <td>0.212307</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.139150</td>
      <td>0.201167</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.135531</td>
      <td>0.228192</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.132643</td>
      <td>0.179707</td>
      <td>0.097847</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


    100%|██████████████████████████████████████████████| 100/100 [30:16<00:00, 18.17s/trial, best loss: 0.1284717321395874]
    {'epochs': 20.0, 'layer1': 150.0, 'layer2': 402.0, 'lr': 0.0047008749172631585}
    


```python
nn_best_hyperparams = space_eval(nn_space, trials.argmin)
nn_best_hyperparams
```




    ({'layer1': 150},
     {'layer2': 402},
     {'epochs': 20},
     {'lr': 0.0047008749172631585})




```python
learn = tabular_learner(dls, y_range=(y.min(), y.max()), 
                        layers=[nn_best_hyperparams[0]['layer1'], nn_best_hyperparams[1]['layer2']], metrics=my_acc)
learn.fit(nn_best_hyperparams[2]['epochs'], nn_best_hyperparams[3]['lr'])
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>my_acc</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.155174</td>
      <td>0.238023</td>
      <td>0.615460</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.148983</td>
      <td>0.246778</td>
      <td>0.633072</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.154161</td>
      <td>0.234980</td>
      <td>0.633072</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.148525</td>
      <td>0.267568</td>
      <td>0.559687</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.144647</td>
      <td>0.171860</td>
      <td>0.758317</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.147481</td>
      <td>0.176536</td>
      <td>0.783757</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.139489</td>
      <td>0.242890</td>
      <td>0.647750</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.137373</td>
      <td>0.223886</td>
      <td>0.662427</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.142005</td>
      <td>0.191750</td>
      <td>0.729941</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.140119</td>
      <td>0.197063</td>
      <td>0.732877</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.136069</td>
      <td>0.198501</td>
      <td>0.726027</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.134901</td>
      <td>0.231177</td>
      <td>0.670254</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.139858</td>
      <td>0.184374</td>
      <td>0.761252</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.138891</td>
      <td>0.194273</td>
      <td>0.754403</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.138725</td>
      <td>0.182043</td>
      <td>0.770059</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.133586</td>
      <td>0.203673</td>
      <td>0.737769</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.135633</td>
      <td>0.186135</td>
      <td>0.749511</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.130184</td>
      <td>0.187927</td>
      <td>0.756360</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.134350</td>
      <td>0.171455</td>
      <td>0.785714</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.135002</td>
      <td>0.193351</td>
      <td>0.746575</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>



```python
nn_preds, gt = learn.get_preds()
```






```python
my_acc(nn_preds, gt)
```




    0.7465753424657534



Note that the gt is different here than for the other classifiers.


```python
gt.sum(), len(gt)
```




    (tensor(922), 1022)




```python
y_test.value_counts()
```




    0    960
    1     62
    Name: stroke, dtype: int64



## Catboost


```python
from catboost import CatBoostClassifier
```


```python
cb_params = {'loss_function':'Logloss',
             'eval_metric':'AUC',
             'cat_features': categorical_vars,
             'verbose': 200,
             'random_seed': 42
            }
cb_clf = CatBoostClassifier(**cb_params)
cb_clf.fit(X_train, y_train,
          eval_set=(X_test, y_test),
          use_best_model=True,
          plot=True
         );
```


    MetricVisualizer(layout=Layout(align_self='stretch', height='500px'))


    Learning rate set to 0.052636
    0:	test: 0.7411374	best: 0.7411374 (0)	total: 175ms	remaining: 2m 54s
    200:	test: 0.7470514	best: 0.7940020 (9)	total: 5.79s	remaining: 23s
    400:	test: 0.7385249	best: 0.7940020 (9)	total: 12.2s	remaining: 18.2s
    600:	test: 0.7342070	best: 0.7940020 (9)	total: 18.2s	remaining: 12.1s
    800:	test: 0.7256384	best: 0.7940020 (9)	total: 24.1s	remaining: 5.97s
    999:	test: 0.7232863	best: 0.7940020 (9)	total: 30.4s	remaining: 0us
    
    bestTest = 0.7940020161
    bestIteration = 9
    
    Shrink model to first 10 iterations.
    


```python
cb_preds = cb_clf.predict(X_test)
```


```python
f1_score(y_test, cb_preds)
```




    0.2608695652173913




```python
accuracy_score(y_test, cb_preds)
```




    0.7504892367906066



## Ensembling

Some of these models are already ensemble models. But who says you can't ensemble ensemble models? No one that I listen to!

#### Averaging

Each of these classifiers can return the probabilities from the classifier. If you're going to do averaging, you'll want to use these. Let's get the probabilities from each classifier.


```python
xgb_probs = xgb_clf.predict_proba(X_test)
```


```python
xgb_probs[:5]
```




    array([[0.9609083 , 0.03909168],
           [0.9253093 , 0.07469067],
           [0.95745856, 0.04254143],
           [0.2907248 , 0.7092752 ],
           [0.32043564, 0.67956436]], dtype=float32)



You can see that the predictions are just the argmax of the probabilities.


```python
xgb_probs_labels = np.argmax(xgb_probs, axis=1)
```


```python
(xgb_probs_labels == xgb_preds).all()
```




    True



Let's get them for the other classifiers.


```python
rf_probs = rf_clf.predict_proba(X_test)
```


```python
svm_probs = svm_clf.predict_proba(X_test)
```


```python
cb_probs = cb_clf.predict_proba(X_test)
```


```python
ensemble_ave = np.argmax(xgb_probs + rf_probs + svm_probs + cb_probs, axis=1)
```


```python
f1_score(ensemble_ave, y_test)
```




    0.26053639846743293




```python
accuracy_score(y_test, ensemble_ave)
```




    0.8111545988258317



This is not always going to give the best result, but it can be something to keep in your back pocket.

#### Voting

`scikit-learn` also provides a voting mechanism for ensembling, which you can see here.


```python
from sklearn.ensemble import VotingClassifier
```


```python
clfs = [('xbg', xgb_clf), ('rf', rf_clf), ('svm', svm_clf), ('cb', cb_clf)]
ensemble = VotingClassifier(clfs, voting='hard')
```


```python
ensemble.fit(X_train, y_train)
```

    [23:48:45] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.5.1/src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    Learning rate set to 0.024769
    0:	total: 38.2ms	remaining: 38.1s
    200:	total: 6.34s	remaining: 25.2s
    400:	total: 12.9s	remaining: 19.3s
    600:	total: 19s	remaining: 12.6s
    800:	total: 25.4s	remaining: 6.32s
    999:	total: 32.3s	remaining: 0us
    




    VotingClassifier(estimators=[('xbg',
                                  XGBClassifier(base_score=0.5, booster='gbtree',
                                                colsample_bylevel=1,
                                                colsample_bynode=1,
                                                colsample_bytree=0.664711277682662,
                                                enable_categorical=False,
                                                gamma=1.2129835895645058, gpu_id=-1,
                                                importance_type=None,
                                                interaction_constraints='',
                                                learning_rate=0.300000012,
                                                max_delta_step=0, max_depth=15,
                                                min_child_weight=3.0, missing=n...
                                                predictor='auto', random_state=0,
                                                reg_alpha=44.0,
                                                reg_lambda=0.36894533857340944,
                                                scale_pos_weight=1, seed=0,
                                                subsample=1, tree_method='exact',
                                                validate_parameters=1,
                                                verbosity=None)),
                                 ('rf',
                                  RandomForestClassifier(max_depth=15.0,
                                                         n_estimators=500)),
                                 ('svm',
                                  SVC(C=0.2371250621465351, degree=4, kernel='poly',
                                      probability=True)),
                                 ('cb',
                                  <catboost.core.CatBoostClassifier object at 0x000001C06170FEB0>)])




```python
ensemble_preds = ensemble.predict(X_test)
```


```python
f1_score(ensemble_preds, y_test)
```




    0.22680412371134023




```python
accuracy_score(ensemble.predict(X_test), y_test)
```




    0.8033268101761253


