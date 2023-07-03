---
layout: post
title: "Test Set Metrics on Unbalanced Datasets: Part I"
description: "This post shows how to get accurate and precise metrics from a test set when you have an unbalanced dataset."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/bald_eagle1.jpg"
tags: [Deep Learning, Machine Learning, Unbalanced Data, Python]
---

In this post, I'm going to walk through how to solve a problem that you might run into when evaluating models on highly unbalanced datasets. Let's imagine you're classifying whether people have a really rare disease or not. You asked 100,000 people at random and only found 10 instances of the disease. How are you going to be able to get enough data to train a machine learning model? Fortunately, you know of a treatment center that treats this specific disease.

You can go to the treatment center and get lots of examples for your training set. This is fine, but what will you do when it comes time to test your model? If you include a bunch of data from the treatment center your distribution won't match the real world and your metrics will be off. But if you only use the data you collected in the real-world distribution you'll only have 10 instances.

The best solution is to label more data. But you can't imagine asking another 100,000 people just to get 10 more with the disease. You'd have to do even more to get a reasonable number.

Fortunately, there's another approach. You can add the treatment center data and calculate what your real-world precision and recall would be in the real-world distribution. Let's take a look.


```python
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
```

Let's say you make a classifier. Although you don't necessarily know these numbers before testing it, let's say it is 80% accurate on positive examples and 99.9% accurate on negative examples.


```python
percent_pos_correct = 0.8
percent_neg_correct = 0.999
```

Now we have all the information we need to make a simulated `y_true` and `y_pred`. Let's do that.


```python
num_pos = 10
num_neg = 99990
```


```python
def get_y_true(num_pos, num_neg):
    return [1] * num_pos + [0] * num_neg
```


```python
y_true = get_y_true(num_pos, num_neg)
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
y_pred = get_y_pred(num_pos, num_neg, percent_pos_correct, percent_neg_correct)
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
get_metrics(y_true, y_pred)
```

    Accuracy: 99.90%
    Precision: 7.41%
    Recall: 80.00%
    F1: 13.56%
    

How much can we trust these numbers? With so few positive examples, the precision and therefore F1 score are highly uncertain. This is where you add more positives. But that causes another problem. If we change the ratio of positives to negatives, we'll throw off our precision (and therefore F1 score). But all is not lost. The key is to find metrics that are invariant to the ratio of positives to negatives, then recover the precision from them.

Let's say we add 100 more examples to our positives.


```python
num_pos_expanded = 10 + 100
num_neg_expanded = 99990
```


```python
y_true_expanded = get_y_true(num_pos_expanded, num_neg_expanded)
```


```python
len(y_true_expanded)
```




    100100




```python
y_pred_expanded = get_y_pred(num_pos_expanded, num_neg_expanded, percent_pos_correct, percent_neg_correct)
```

Now let's get the metrics again.


```python
get_metrics(y_true_expanded, y_pred_expanded)
```

    Accuracy: 99.88%
    Precision: 46.81%
    Recall: 80.00%
    F1: 59.06%
    

As we expected, the precision and f1 scores are off, but the recall is right. Fortunately, we have enough information to recover what they should have been.


```python
def get_model_stats(y_true, y_pred):
    """
    Calculate the true positive rate and false positive rate from the predictions and labels.
    """
    pos_indices = [i for i , x in enumerate(y_true) if x == 1]
    neg_indices = [i for i , x in enumerate(y_true) if x == 0]
    preds_for_pos_labels = [y_pred[i] for i in pos_indices]
    preds_for_neg_labels = [y_pred[i] for i in neg_indices]
    percent_pos_correct = sum(preds_for_pos_labels) / len(preds_for_pos_labels)
    percent_neg_correct = np.sum(np.array(preds_for_neg_labels) == 0) / len(preds_for_neg_labels)
    return percent_pos_correct, percent_neg_correct
```


```python
percent_pos_correct, percent_neg_correct = get_model_stats(y_true, y_pred)
```

Let's make sure we've recovered the right values.


```python
print(f"{percent_pos_correct=}")
print(f"{percent_neg_correct=}")
```

    percent_pos_correct=0.8
    percent_neg_correct=0.998999899989999
    

Exactly as we expected (given floating point precision). Now we can recreate the precision and recall for any data distribution.


```python
y_pred_recreated = get_y_pred(num_pos, num_neg, percent_pos_correct, percent_neg_correct)
```


```python
get_metrics(y_true, y_pred_recreated)
```

    Accuracy: 99.90%
    Precision: 7.41%
    Recall: 80.00%
    F1: 13.56%
    

We got our original value back. Don't believe this would work on a real dataset? In [Part II](https://jss367.github.io/test-set-metrics-on-unbalanced-datasets-part-ii.html), we'll explore that.
