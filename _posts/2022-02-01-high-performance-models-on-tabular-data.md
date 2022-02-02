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
num_trials = 500
svm_trials = 100 # svm takes much longer, so you may want to limit this
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
    xgb needs eval_metric or it produces lots of warnings
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

    100%|██████████████████████████████████████████████| 500/500 [02:25<00:00,  3.43trial/s, best loss: -0.764187866927593]
    




    {'colsample_bytree': 0.5671189561452116,
     'gamma': 1.0071481663300468,
     'max_depth': 14.0,
     'min_child_weight': 2.0,
     'reg_alpha': 50.0,
     'reg_lambda': 0.6971844311013579}




```python
best_hyperparams = space_eval(xgb_space, trials.argmin)
```


```python
best_hyperparams
```




    {'colsample_bytree': 0.5671189561452116,
     'gamma': 1.0071481663300468,
     'max_depth': 14,
     'min_child_weight': 2.0,
     'n_estimators': 180,
     'reg_alpha': 50.0,
     'reg_lambda': 0.6971844311013579,
     'seed': 0}




```python
xgb_clf = XGBClassifier(**best_hyperparams)
```


```python
xgb_clf.fit(X_train, y_train)
```

    [00:09:42] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.5.1/src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.5671189561452116,
                  enable_categorical=False, gamma=1.0071481663300468, gpu_id=-1,
                  importance_type=None, interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=14,
                  min_child_weight=2.0, missing=nan, monotone_constraints='()',
                  n_estimators=180, n_jobs=12, num_parallel_tree=1,
                  predictor='auto', random_state=0, reg_alpha=50.0,
                  reg_lambda=0.6971844311013579, scale_pos_weight=1, seed=0,
                  subsample=1, tree_method='exact', validate_parameters=1,
                  verbosity=None)




```python
xgb_preds = xgb_clf.predict(X_test)
```


```python
f1_score(y_test, xgb_preds)
```




    0.25846153846153846




```python
accuracy_score(y_test, xgb_preds)
```




    0.764187866927593




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
    "n_estimators": hp.randint("n_estimators", 10, 700),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "max_depth": hp.randint('max_depth', 1, 100),
    "min_samples_split": hp.randint('min_samples_split', 2, 20),
    "min_samples_leaf": hp.randint('min_samples_leaf', 1, 10),
    "max_features": hp.choice('max_features', ['sqrt', 'log2']),
    "random_state": 42
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

    100%|█████████████████████████████████████████████| 500/500 [09:01<00:00,  1.08s/trial, best loss: -0.8679060665362035]
    




    {'criterion': 0,
     'max_depth': 98,
     'max_features': 1,
     'min_samples_leaf': 1,
     'min_samples_split': 9,
     'n_estimators': 535}




```python
rf_best_hyperparams = space_eval(rf_space, trials.argmin)
```


```python
rf_best_hyperparams
```




    {'criterion': 'gini',
     'max_depth': 98,
     'max_features': 'log2',
     'min_samples_leaf': 1,
     'min_samples_split': 9,
     'n_estimators': 535,
     'random_state': 42}




```python
rf_clf = RandomForestClassifier(**rf_best_hyperparams)
```


```python
rf_clf.fit(X_train, y_train)
```




    RandomForestClassifier(max_depth=98, max_features='log2', min_samples_split=9,
                           n_estimators=535, random_state=42)




```python
rf_preds = rf_clf.predict(X_test)
```


```python
f1_score(y_test, rf_preds)
```




    0.17177914110429446




```python
accuracy_score(y_test, rf_preds)
```




    0.8679060665362035



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
    max_evals = svm_trials,
    trials = trials)
```

    100%|█████████████████████████████████████████████| 100/100 [22:05<00:00, 13.25s/trial, best loss: -0.8111545988258317]
    




    {'degree': 2, 'kernel': 2, 'probability': 0, 'svm_C': 2.1465036697130855}




```python
svm_best_hyperparams = space_eval(svm_space, trials.argmin)
```


```python
svm_best_hyperparams
```




    {'C': 2.1465036697130855, 'degree': 4, 'kernel': 'poly', 'probability': True}




```python
svm_clf = SVC(**svm_best_hyperparams)
```


```python
svm_clf.fit(X_train, y_train)
```




    SVC(C=2.1465036697130855, degree=4, kernel='poly', probability=True)




```python
svm_preds = svm_clf.predict(X_test)
```


```python
f1_score(y_test, svm_preds)
```




    0.2771535580524344




```python
accuracy_score(y_test, svm_preds)
```




    0.8111545988258317



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
    with learn.no_bar(), learn.no_logging():
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

    100%|███████████████████████████████████████████| 500/500 [2:36:05<00:00, 18.73s/trial, best loss: 0.12682415544986725]
    {'epochs': 20.0, 'layer1': 181.0, 'layer2': 158.0, 'lr': 0.004167824772417915}
    


```python
nn_best_hyperparams = space_eval(nn_space, trials.argmin)
nn_best_hyperparams
```




    ({'layer1': 181},
     {'layer2': 158},
     {'epochs': 20},
     {'lr': 0.004167824772417915})




```python
learn = tabular_learner(dls, y_range=(y.min(), y.max()), 
                        layers=[nn_best_hyperparams[0]['layer1'], nn_best_hyperparams[1]['layer2']], metrics=my_acc)
learn.fit(nn_best_hyperparams[2]['epochs'], nn_best_hyperparams[3]['lr'], )
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
      <td>0.152055</td>
      <td>0.210501</td>
      <td>0.703523</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.149081</td>
      <td>0.241731</td>
      <td>0.587084</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.147788</td>
      <td>0.208247</td>
      <td>0.710372</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.144673</td>
      <td>0.194463</td>
      <td>0.737769</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.142503</td>
      <td>0.195599</td>
      <td>0.729941</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.143986</td>
      <td>0.247592</td>
      <td>0.614481</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.140043</td>
      <td>0.233124</td>
      <td>0.599804</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.140295</td>
      <td>0.193624</td>
      <td>0.726027</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.135865</td>
      <td>0.200070</td>
      <td>0.707436</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.134626</td>
      <td>0.198199</td>
      <td>0.737769</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.138697</td>
      <td>0.211755</td>
      <td>0.710372</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.133714</td>
      <td>0.199920</td>
      <td>0.726027</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.139617</td>
      <td>0.222729</td>
      <td>0.687867</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.131833</td>
      <td>0.192564</td>
      <td>0.747554</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.133367</td>
      <td>0.180420</td>
      <td>0.767123</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.133702</td>
      <td>0.213077</td>
      <td>0.709393</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.137790</td>
      <td>0.210081</td>
      <td>0.688845</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.134852</td>
      <td>0.158134</td>
      <td>0.800391</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.137128</td>
      <td>0.197334</td>
      <td>0.726027</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.130718</td>
      <td>0.185704</td>
      <td>0.740705</td>
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




    0.7407045009784736



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
    0:	test: 0.7928763	best: 0.7928763 (0)	total: 173ms	remaining: 2m 52s
    200:	test: 0.7582745	best: 0.8060484 (11)	total: 5.18s	remaining: 20.6s
    400:	test: 0.7407258	best: 0.8060484 (11)	total: 10.5s	remaining: 15.6s
    600:	test: 0.7307124	best: 0.8060484 (11)	total: 15.9s	remaining: 10.6s
    800:	test: 0.7226983	best: 0.8060484 (11)	total: 21.3s	remaining: 5.29s
    999:	test: 0.7188004	best: 0.8060484 (11)	total: 26.9s	remaining: 0us
    
    bestTest = 0.8060483871
    bestIteration = 11
    
    Shrink model to first 12 iterations.
    


```python
cb_preds = cb_clf.predict(X_test)
```


```python
f1_score(y_test, cb_preds)
```




    0.2580645161290323




```python
accuracy_score(y_test, cb_preds)
```




    0.7299412915851272



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




    array([[0.9548289 , 0.04517111],
           [0.9358052 , 0.06419478],
           [0.96357316, 0.03642686],
           [0.3344649 , 0.6655351 ],
           [0.23816943, 0.76183057]], dtype=float32)



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




    0.26562499999999994




```python
accuracy_score(y_test, ensemble_ave)
```




    0.8160469667318982



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

    [15:54:44] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.5.1/src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    Learning rate set to 0.024769
    0:	total: 47.2ms	remaining: 47.2s
    200:	total: 6.61s	remaining: 26.3s
    400:	total: 13.4s	remaining: 20s
    600:	total: 20.1s	remaining: 13.3s
    800:	total: 26.7s	remaining: 6.63s
    999:	total: 33.3s	remaining: 0us
    




    VotingClassifier(estimators=[('xbg',
                                  XGBClassifier(base_score=0.5, booster='gbtree',
                                                colsample_bylevel=1,
                                                colsample_bynode=1,
                                                colsample_bytree=0.5671189561452116,
                                                enable_categorical=False,
                                                gamma=1.0071481663300468, gpu_id=-1,
                                                importance_type=None,
                                                interaction_constraints='',
                                                learning_rate=0.300000012,
                                                max_delta_step=0, max_depth=14,
                                                min_child_weight=2.0, missing=...
                                                scale_pos_weight=1, seed=0,
                                                subsample=1, tree_method='exact',
                                                validate_parameters=1,
                                                verbosity=None)),
                                 ('rf',
                                  RandomForestClassifier(max_depth=98,
                                                         max_features='log2',
                                                         min_samples_split=9,
                                                         n_estimators=535,
                                                         random_state=42)),
                                 ('svm',
                                  SVC(C=2.1465036697130855, degree=4, kernel='poly',
                                      probability=True)),
                                 ('cb',
                                  <catboost.core.CatBoostClassifier object at 0x000001F10B3250A0>)])




```python
ensemble_preds = ensemble.predict(X_test)
```


```python
f1_score(ensemble_preds, y_test)
```




    0.25773195876288657




```python
accuracy_score(ensemble.predict(X_test), y_test)
```




    0.8590998043052838


