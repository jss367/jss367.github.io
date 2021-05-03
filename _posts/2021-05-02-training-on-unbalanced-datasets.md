---
layout: post
title: "Training on Unbalanced Datasets"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/water_text.jpg"
tags: [Python, TensorFlow]
---

In this post I'm going to look at a couple different methods of dealing with unbalanced data. For this, we'll use the cats vs dogs dataset. Let's assume we have there are 10 times as many cats as there are dogs in the population. What's the best way to deal with this?

We're going to build the model to answer the question, "Is it a dog", so a true positive would be considered a dog image correctly displayed as a dog.

<b>Table of contents</b>
* TOC
{:toc}


```python
import os
from os import listdir
from pathlib import Path
from typing import List

import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
```

Because we'll be working with images, I'm going to make sure my GPU doesn't run out of memory.


```python
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed(42)
```

## Prepare the Data

OK, now let's look at the dataset.


```python
root_dir = Path('E:/Data/Raw/cats_vs_dogs_dataset')
image_dir = root_dir / 'train'
```


```python
class_names = listdir(image_dir)
print(class_names)
```

    ['cats', 'dogs']
    

Let's see how many images we have.


```python
cat_dir = image_dir / 'cats'
dog_dir = image_dir / 'dogs'
```


```python
num_cat_train_im = len(listdir(cat_dir))
num_dog_train_im = len(listdir(dog_dir))
print(num_cat_train_im)
print(num_dog_train_im)
```

    4000
    4000
    

We have the same number of each dataset. Now, let's say we're looking for cats in a sea of dog images... how should we go about this?
Let's say we only have 50 cat images and 5000 dogs images.


```python
cat_list_ds = tf.data.Dataset.list_files(str(cat_dir/'*'), shuffle=False, seed=42)
dog_list_ds = tf.data.Dataset.list_files(str(dog_dir/'*'), shuffle=False, seed=42)
```


```python
def subset_dataset(dataset: tf.data.Dataset, splits: List) -> List[tf.data.Dataset]:
    """
    Split a dataset into any number of non-overlapping subdatasets of size listed in `splits`
    """
    assert sum(splits) <= tf.data.experimental.cardinality(dataset).numpy(), "Total number of images in splits exceeds dataset size"
    datasets = []
    total_used = 0
    for i, val in enumerate(splits):
        ds = dataset.skip(total_used).take(val)
        total_used += val
        datasets.append(ds)

    return datasets
```


```python
BATCH_SIZE = 32
NUM_EPOCHS = 4
img_height = 64
img_width = 64
num_channels = 3
```


```python
def prep_image(filename):
    img = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(img, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image = tf.image.resize(image, (img_height, img_width))
    return image

def prep_label(filename):
    parts = tf.strings.split(filename, sep=os.path.sep)
    one_hot_label = parts[-2] == class_names
    label = tf.argmax(one_hot_label)
    return label

def parse_file(filename):
    image = prep_image(filename)
    label = prep_label(filename)
    return image, label

def prepare_dataset(*datasets):
    dataset = tf.data.experimental.sample_from_datasets(datasets, seed=42)
    dataset = dataset.map(parse_file)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset
```

## Determine Metrics

Now we'll have to decide on what metrics to use. These will be important so we'll use a lot of them.


```python
all_metrics = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'), 
      metrics.BinaryAccuracy(name='accuracy'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall'),
]
```

## Create Model

OK, now we have to make a model. This post isn't about the model so I'm going to make a simple CNN.


```python
def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='same', strides=(1,1), kernel_initializer='he_uniform', input_shape=(img_height, img_width, num_channels), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), kernel_initializer='he_uniform', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', strides=(1,1), kernel_initializer='he_uniform', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model
```

## Compile Model


```python
model = get_model()
```


```python
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=all_metrics)
```

## Visualize Results

We're going to need some functions to visualize the results, so let's look at those here.


```python
def plot_loss(history, label):
    plt.plot(history.epoch, history.history['loss'],
               label='Train ' + label)
    plt.plot(history.epoch, history.history['val_loss'],
               label='Val ' + label,
               linestyle="--")
    plt.xlabel('Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.cm.rocket_r)
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Truth label')
    plt.xlabel('Predicted label')

```

## Experiment - Do you want balanced or unbalanced train data?

OK. Out first experiment we'll make a couple train datasets. One options is to have a balanced dataset, the other is to allow it to be unbalanced to match the "real world". Let's see which one produces better results.


```python
cat_list_train_balanced, cat_list_train_unbalanced, cat_list_val, cat_list_test = subset_dataset(cat_list_ds, [1000, 1000, 1000, 1000])
dog_list_train_balanced, dog_list_train_unbalanced, dog_list_val, dog_list_test = subset_dataset(dog_list_ds, [1000, 100, 100, 100])
```


```python
train_ds_balanced = prepare_dataset(cat_list_train_balanced, dog_list_train_balanced)
train_ds_unbalanced = prepare_dataset(cat_list_train_unbalanced, dog_list_train_unbalanced)
val_ds = prepare_dataset(cat_list_val, dog_list_val)
test_ds = prepare_dataset(cat_list_test, dog_list_test)
```

Great. Now let's train the models.


```python
model_balanced = get_model()
model_balanced.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_balanced = model_balanced.fit(train_ds_balanced, epochs=NUM_EPOCHS, validation_data=val_ds)
```

    Epoch 1/4
    35/35 [==============================] - 7s 151ms/step - loss: 1.4320 - tp: 48.8889 - fp: 62.4167 - tn: 1006.2778 - fn: 144.4167 - accuracy: 0.8283 - precision: 0.4514 - recall: 0.2488 - val_loss: 1.6646 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/4
    35/35 [==============================] - 4s 119ms/step - loss: 3.8962 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 2.5516 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/4
    35/35 [==============================] - 4s 118ms/step - loss: 6.2847 - tp: 0.0000e+00 - fp: 2.0556 - tn: 494.6389 - fn: 93.3056 - accuracy: 0.7581 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.2343 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/4
    35/35 [==============================] - 4s 119ms/step - loss: 3.0792 - tp: 0.0000e+00 - fp: 10.1944 - tn: 486.5000 - fn: 93.3056 - accuracy: 0.7474 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3537 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    


```python
model_unbalanced = get_model()
model_unbalanced.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_unbalanced = model_unbalanced.fit(train_ds_unbalanced, epochs=NUM_EPOCHS, validation_data=val_ds)
```

    Epoch 1/4
    35/35 [==============================] - 6s 138ms/step - loss: 2.2610 - tp: 26.5000 - fp: 109.3889 - tn: 1387.3056 - fn: 166.8056 - accuracy: 0.8356 - precision: 0.2379 - recall: 0.1367 - val_loss: 1.4492 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/4
    35/35 [==============================] - 4s 119ms/step - loss: 3.7224 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.0898 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/4
    35/35 [==============================] - 4s 122ms/step - loss: 2.8557 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.1095 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/4
    35/35 [==============================] - 4s 118ms/step - loss: 2.9751 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.1628 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    

## Evaluation


```python
plot_loss(history_balanced, "Balanced Training")
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_39_0.png)
    



```python
plot_loss(history_unbalanced, "Unbalanced Training")
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_40_0.png)
    



```python
eval_balanced = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
eval_unbalanced = model_unbalanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_balanced.metrics_names, eval_balanced):
    print(name, ': ', value)
```

    loss :  0.6205962300300598
    tp :  41.0
    fp :  233.0
    tn :  767.0
    fn :  59.0
    accuracy :  0.7345454692840576
    precision :  0.14963503181934357
    recall :  0.4099999964237213
    


```python
for name, value in zip(model_unbalanced.metrics_names, eval_unbalanced):
    print(name, ': ', value)
```

    loss :  0.47038525342941284
    tp :  0.0
    fp :  0.0
    tn :  1000.0
    fn :  100.0
    accuracy :  0.9090909361839294
    precision :  0.0
    recall :  0.0
    

OK, so the model trained on unbalanced has a higher accuracy, but that's because it predicted the majority class for everything! It has **zero** precision and recall.


```python
balanced_preds = model_balanced.predict(test_ds)
unbalanced_preds = model_unbalanced.predict(test_ds)
true_labels = tf.concat([y for x, y in test_ds], axis=0)
```


```python
plot_cm(true_labels, balanced_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_46_0.png)
    



```python
plot_cm(true_labels, unbalanced_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_47_0.png)
    


## Second model fitting

OK, so we know balanced data is better than unbalanced, but what if we can't get balanced data... what then? Let's try using class weights. We know we have 10 times as many cats as dogs, so we'll weight the dogs 10X as much.


```python
class_weight = {0:1, 1:10}
```


```python
model_weighted = get_model()
model_weighted.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_weighted = model_weighted.fit(train_ds_unbalanced, epochs=NUM_EPOCHS, validation_data=val_ds, class_weight=class_weight)
```

    Epoch 1/4
    35/35 [==============================] - 6s 131ms/step - loss: 3.1908 - tp: 80.7778 - fp: 201.1944 - tn: 1295.5000 - fn: 112.5278 - accuracy: 0.8155 - precision: 0.3168 - recall: 0.4123 - val_loss: 2.3255 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/4
    35/35 [==============================] - 4s 113ms/step - loss: 53.5416 - tp: 0.0000e+00 - fp: 48.0278 - tn: 448.6667 - fn: 93.3056 - accuracy: 0.7001 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.0043 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/4
    35/35 [==============================] - 4s 113ms/step - loss: 25.1313 - tp: 5.7222 - fp: 63.5278 - tn: 433.1667 - fn: 87.5833 - accuracy: 0.6827 - precision: 0.1736 - recall: 0.0639 - val_loss: 0.7753 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/4
    35/35 [==============================] - 4s 113ms/step - loss: 16.3211 - tp: 1.8611 - fp: 157.3889 - tn: 339.3056 - fn: 91.4444 - accuracy: 0.5429 - precision: 0.1796 - recall: 0.0196 - val_loss: 0.6090 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    


```python
eval_weighted = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_weighted.metrics_names, eval_weighted):
    print(name, ': ', value)
```

    loss :  0.3512849807739258
    tp :  0.0
    fp :  0.0
    tn :  1000.0
    fn :  100.0
    accuracy :  0.9090909361839294
    precision :  0.0
    recall :  0.0
    


```python
weighted_preds = model_weighted.predict(test_ds)
```


```python
plot_cm(true_labels, weighted_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_55_0.png)
    


Interestingly, even if you undo the difference in the unbalanced data by adjusting the weights, it goes *too* far.


```python

```

## Class Weights Round 2


```python
class_weight2 = {0:1, 1:25}
```


```python
model_weighted2 = get_model()
model_weighted2.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_weighted2 = model_weighted2.fit(train_ds_unbalanced, epochs=NUM_EPOCHS, validation_data=val_ds, class_weight=class_weight)
```

    Epoch 1/4
    35/35 [==============================] - 6s 132ms/step - loss: 2.6198 - tp: 97.9722 - fp: 193.0278 - tn: 1303.6667 - fn: 95.3333 - accuracy: 0.8273 - precision: 0.3420 - recall: 0.5012 - val_loss: 1.1990 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/4
    35/35 [==============================] - 4s 113ms/step - loss: 28.7468 - tp: 0.0000e+00 - fp: 3.1667 - tn: 493.5278 - fn: 93.3056 - accuracy: 0.7572 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.2067 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/4
    35/35 [==============================] - 4s 114ms/step - loss: 27.8534 - tp: 0.0000e+00 - fp: 221.7778 - tn: 274.9167 - fn: 93.3056 - accuracy: 0.4749 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.4057 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/4
    35/35 [==============================] - 4s 115ms/step - loss: 7.8004 - tp: 1.7222 - fp: 375.3611 - tn: 121.3333 - fn: 91.5833 - accuracy: 0.2794 - precision: 0.0352 - recall: 0.0172 - val_loss: 0.6893 - val_tp: 14.0000 - val_fp: 218.0000 - val_tn: 782.0000 - val_fn: 86.0000 - val_accuracy: 0.7236 - val_precision: 0.0603 - val_recall: 0.1400
    


```python
eval_weighted = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_weighted.metrics_names, eval_weighted):
    print(name, ': ', value)
```

    loss :  0.3512849807739258
    tp :  0.0
    fp :  0.0
    tn :  1000.0
    fn :  100.0
    accuracy :  0.9090909361839294
    precision :  0.0
    recall :  0.0
    


```python
weighted_preds = model_weighted.predict(test_ds)
```


```python
plot_cm(true_labels, weighted_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_64_0.png)
    


## Trying with oversampling

I generally prefer oversampling. Here's how to do it with tf.data datasets.

We know it's a 10:1 ratio, so we have to repeat the dogs 10 times.


```python
resampled_ds = tf.data.experimental.sample_from_datasets([cat_list_train_unbalanced, dog_list_train_unbalanced.repeat(10)], weights=[0.5, 0.5], seed=42)
resampled_ds = resampled_ds.map(parse_file)
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
```

Just to make sure it's what we want we can


```python
resampled_true_labels = tf.concat([y for x, y in resampled_ds], axis=0)
```


```python
resampled_true_labels
```




    <tf.Tensor: shape=(2000,), dtype=int64, numpy=array([0, 0, 0, ..., 0, 0, 0], dtype=int64)>




```python
sum(resampled_true_labels)
```




    <tf.Tensor: shape=(), dtype=int64, numpy=1000>



The sum is half the total, so half are 1 and half are 0, which is what we expected.


```python
model_oversampled = get_model()
model_oversampled.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_oversampled = model_oversampled.fit(resampled_ds, epochs=NUM_EPOCHS, validation_data=val_ds)
```

    Epoch 1/4
    63/63 [==============================] - 8s 97ms/step - loss: 1.5180 - tp: 370.6094 - fp: 324.2344 - tn: 1174.8125 - fn: 269.3438 - accuracy: 0.7398 - precision: 0.5395 - recall: 0.5091 - val_loss: 0.6722 - val_tp: 2.0000 - val_fp: 18.0000 - val_tn: 982.0000 - val_fn: 98.0000 - val_accuracy: 0.8945 - val_precision: 0.1000 - val_recall: 0.0200
    Epoch 2/4
    63/63 [==============================] - 5s 86ms/step - loss: 0.6945 - tp: 86.1250 - fp: 61.8125 - tn: 437.2344 - fn: 453.8281 - accuracy: 0.4953 - precision: 0.6239 - recall: 0.1160 - val_loss: 0.6541 - val_tp: 1.0000 - val_fp: 3.0000 - val_tn: 997.0000 - val_fn: 99.0000 - val_accuracy: 0.9073 - val_precision: 0.2500 - val_recall: 0.0100
    Epoch 3/4
    63/63 [==============================] - 5s 86ms/step - loss: 0.6894 - tp: 176.5625 - fp: 124.4062 - tn: 374.6406 - fn: 363.3906 - accuracy: 0.5175 - precision: 0.6864 - recall: 0.2443 - val_loss: 0.5669 - val_tp: 1.0000 - val_fp: 2.0000 - val_tn: 998.0000 - val_fn: 99.0000 - val_accuracy: 0.9082 - val_precision: 0.3333 - val_recall: 0.0100
    Epoch 4/4
    63/63 [==============================] - 5s 86ms/step - loss: 0.7019 - tp: 31.3125 - fp: 19.6562 - tn: 479.3906 - fn: 508.6406 - accuracy: 0.4873 - precision: 0.8064 - recall: 0.0440 - val_loss: 0.6052 - val_tp: 7.0000 - val_fp: 36.0000 - val_tn: 964.0000 - val_fn: 93.0000 - val_accuracy: 0.8827 - val_precision: 0.1628 - val_recall: 0.0700
    


```python
eval_oversampled = model_oversampled.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_oversampled.metrics_names, eval_oversampled):
    print(name, ': ', value)
```

    loss :  0.6051812767982483
    tp :  13.0
    fp :  45.0
    tn :  955.0
    fn :  87.0
    accuracy :  0.8799999952316284
    precision :  0.22413793206214905
    recall :  0.12999999523162842
    


```python
oversampled_preds = model_oversampled.predict(test_ds)
```


```python
plot_cm(true_labels, oversampled_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_78_0.png)
    


So even though it's not perfect, we got a better result. This does cause a problem though because we've overfit one side and not the other. Depending on how much data we have, we could undersample as well. But I think it would be better to just add data augmentation.


```python

```
