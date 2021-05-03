---
layout: post
title: "Training on Unbalanced Datasets"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/water_text.jpg"
tags: [Deep Learning, Python, TensorFlow]
---

In this post I'm going to look at different methods of train models on unbalanced populations. This post is paired with my [post on evaluating models with unbalanced datasets](https://jss367.github.io/training-on-unbalanced-datasets.html). For these experiments, we'll use the Kaggle [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats). The dataset has the same number of cat images as dog images, so we'll have to subset the dataset to run the experiment. We're going to pretend that there are 10 times as many cats as there are dogs in the "real world" population. We want to build a model that answers the question, "Is this an image of a dog?"

We'll answer a number of questions along the way. The first is, given that there are more cats than dogs in our population, should there also be more dogs than cats in the training data? That is, should we have unbalanced training data? Or should we find a way to balance it. And, if we want to balance the data, what's the best way to do it? The two most popular methods for this are adding more weight to the less common image or oversampling it. Which is better?

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
NUM_EPOCHS = 10
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

We're going to need some functions to visualize the results, so let's build those here.


```python
def plot_loss(history, label):
    """
    Plot the train and val loss from a TensorFlow train
    """
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
    """
    Plot a confusion matrix
    """
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.cm.rocket_r)
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Truth label')
    plt.xlabel('Predicted label')
```

## Experiment #1 - Should the Training Data Be Balanced or Unbalanced?

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

    Epoch 1/10
    63/63 [==============================] - 9s 109ms/step - loss: 1.6369 - tp: 293.2969 - fp: 295.4844 - tn: 203.5625 - fn: 246.6562 - accuracy: 0.4679 - precision: 0.4893 - recall: 0.5101 - val_loss: 0.6770 - val_tp: 19.0000 - val_fp: 112.0000 - val_tn: 888.0000 - val_fn: 81.0000 - val_accuracy: 0.8245 - val_precision: 0.1450 - val_recall: 0.1900
    Epoch 2/10
    63/63 [==============================] - 5s 87ms/step - loss: 0.6907 - tp: 281.4688 - fp: 220.7031 - tn: 278.3438 - fn: 258.4844 - accuracy: 0.5224 - precision: 0.5552 - recall: 0.4333 - val_loss: 0.5043 - val_tp: 0.0000e+00 - val_fp: 1.0000 - val_tn: 999.0000 - val_fn: 100.0000 - val_accuracy: 0.9082 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/10
    63/63 [==============================] - 5s 88ms/step - loss: 0.7129 - tp: 286.7812 - fp: 220.8438 - tn: 278.2031 - fn: 253.1719 - accuracy: 0.5352 - precision: 0.6190 - recall: 0.4376 - val_loss: 0.5921 - val_tp: 22.0000 - val_fp: 155.0000 - val_tn: 845.0000 - val_fn: 78.0000 - val_accuracy: 0.7882 - val_precision: 0.1243 - val_recall: 0.2200
    Epoch 4/10
    63/63 [==============================] - 6s 89ms/step - loss: 0.6850 - tp: 359.1250 - fp: 261.7500 - tn: 237.2969 - fn: 180.8281 - accuracy: 0.5650 - precision: 0.6063 - recall: 0.5758 - val_loss: 0.5707 - val_tp: 26.0000 - val_fp: 167.0000 - val_tn: 833.0000 - val_fn: 74.0000 - val_accuracy: 0.7809 - val_precision: 0.1347 - val_recall: 0.2600
    Epoch 5/10
    63/63 [==============================] - 5s 87ms/step - loss: 0.6792 - tp: 373.0156 - fp: 263.8281 - tn: 235.2188 - fn: 166.9375 - accuracy: 0.5697 - precision: 0.5898 - recall: 0.6037 - val_loss: 0.5330 - val_tp: 32.0000 - val_fp: 148.0000 - val_tn: 852.0000 - val_fn: 68.0000 - val_accuracy: 0.8036 - val_precision: 0.1778 - val_recall: 0.3200
    Epoch 6/10
    63/63 [==============================] - 5s 86ms/step - loss: 0.6640 - tp: 366.6250 - fp: 213.1406 - tn: 285.9062 - fn: 173.3281 - accuracy: 0.6221 - precision: 0.6631 - recall: 0.6151 - val_loss: 0.5484 - val_tp: 25.0000 - val_fp: 138.0000 - val_tn: 862.0000 - val_fn: 75.0000 - val_accuracy: 0.8064 - val_precision: 0.1534 - val_recall: 0.2500
    Epoch 7/10
    63/63 [==============================] - 5s 86ms/step - loss: 0.6587 - tp: 352.2812 - fp: 216.5469 - tn: 282.5000 - fn: 187.6719 - accuracy: 0.5978 - precision: 0.6519 - recall: 0.5715 - val_loss: 0.5705 - val_tp: 54.0000 - val_fp: 261.0000 - val_tn: 739.0000 - val_fn: 46.0000 - val_accuracy: 0.7209 - val_precision: 0.1714 - val_recall: 0.5400
    Epoch 8/10
    63/63 [==============================] - 5s 86ms/step - loss: 0.6279 - tp: 363.0938 - fp: 185.9531 - tn: 313.0938 - fn: 176.8594 - accuracy: 0.6466 - precision: 0.6876 - recall: 0.6219 - val_loss: 0.4938 - val_tp: 31.0000 - val_fp: 118.0000 - val_tn: 882.0000 - val_fn: 69.0000 - val_accuracy: 0.8300 - val_precision: 0.2081 - val_recall: 0.3100
    Epoch 9/10
    63/63 [==============================] - 5s 87ms/step - loss: 0.6273 - tp: 346.9219 - fp: 179.2969 - tn: 319.7500 - fn: 193.0312 - accuracy: 0.6283 - precision: 0.6866 - recall: 0.5725 - val_loss: 0.4830 - val_tp: 37.0000 - val_fp: 114.0000 - val_tn: 886.0000 - val_fn: 63.0000 - val_accuracy: 0.8391 - val_precision: 0.2450 - val_recall: 0.3700
    Epoch 10/10
    63/63 [==============================] - 5s 86ms/step - loss: 0.5958 - tp: 374.5156 - fp: 165.8125 - tn: 333.2344 - fn: 165.4375 - accuracy: 0.6745 - precision: 0.7131 - recall: 0.6510 - val_loss: 0.5675 - val_tp: 62.0000 - val_fp: 263.0000 - val_tn: 737.0000 - val_fn: 38.0000 - val_accuracy: 0.7264 - val_precision: 0.1908 - val_recall: 0.6200
    


```python
model_unbalanced = get_model()
model_unbalanced.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_unbalanced = model_unbalanced.fit(train_ds_unbalanced, epochs=NUM_EPOCHS, validation_data=val_ds)
```

    Epoch 1/10
    35/35 [==============================] - 6s 131ms/step - loss: 0.7253 - tp: 103.1944 - fp: 365.5000 - tn: 1131.1944 - fn: 90.1111 - accuracy: 0.7239 - precision: 0.2206 - recall: 0.5357 - val_loss: 1.8799 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/10
    35/35 [==============================] - 4s 112ms/step - loss: 4.7056 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.4484 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/10
    35/35 [==============================] - 4s 113ms/step - loss: 3.4133 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.8440 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/10
    35/35 [==============================] - 4s 113ms/step - loss: 2.3146 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.4698 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 5/10
    35/35 [==============================] - 4s 112ms/step - loss: 0.5496 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3351 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 6/10
    35/35 [==============================] - 4s 111ms/step - loss: 0.9683 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3327 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 7/10
    35/35 [==============================] - 4s 111ms/step - loss: 0.7072 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3245 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 8/10
    35/35 [==============================] - 4s 111ms/step - loss: 0.8402 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3247 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 9/10
    35/35 [==============================] - 4s 110ms/step - loss: 0.8093 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3231 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 10/10
    35/35 [==============================] - 4s 110ms/step - loss: 0.7270 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 496.6944 - fn: 93.3056 - accuracy: 0.7612 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3280 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    

#### Experiment #1 Evaluation


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

    loss :  0.5695805549621582
    tp :  63.0
    fp :  279.0
    tn :  721.0
    fn :  37.0
    accuracy :  0.7127272486686707
    precision :  0.18421052396297455
    recall :  0.6299999952316284
    


```python
for name, value in zip(model_unbalanced.metrics_names, eval_unbalanced):
    print(name, ': ', value)
```

    loss :  0.32628217339515686
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
    


## Experiment #2 - Using class_weight

OK, so we know balanced data is better than unbalanced, but what if we can't get balanced data... what then? Let's try using class weights. We know we have 10 times as many cats as dogs, so we'll weight the dogs 10X as much.


```python
class_weight = {0:1, 1:10}
```


```python
model_weighted = get_model()
model_weighted.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_weighted = model_weighted.fit(train_ds_unbalanced, epochs=NUM_EPOCHS, validation_data=val_ds, class_weight=class_weight)
```

    Epoch 1/10
    35/35 [==============================] - 6s 131ms/step - loss: 5.3339 - tp: 77.2222 - fp: 239.3611 - tn: 1257.3333 - fn: 116.0833 - accuracy: 0.7927 - precision: 0.2871 - recall: 0.3934 - val_loss: 1.8418 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/10
    35/35 [==============================] - 4s 110ms/step - loss: 46.7652 - tp: 0.0000e+00 - fp: 1.1111 - tn: 495.5833 - fn: 93.3056 - accuracy: 0.7598 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.3167 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/10
    35/35 [==============================] - 4s 111ms/step - loss: 27.7310 - tp: 0.0000e+00 - fp: 141.7500 - tn: 354.9444 - fn: 93.3056 - accuracy: 0.5591 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.0065 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/10
    35/35 [==============================] - 4s 110ms/step - loss: 23.4990 - tp: 0.0000e+00 - fp: 303.7222 - tn: 192.9722 - fn: 93.3056 - accuracy: 0.3906 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.6946 - val_tp: 55.0000 - val_fp: 684.0000 - val_tn: 316.0000 - val_fn: 45.0000 - val_accuracy: 0.3373 - val_precision: 0.0744 - val_recall: 0.5500
    Epoch 5/10
    35/35 [==============================] - 4s 110ms/step - loss: 2.2057 - tp: 36.3889 - fp: 453.7222 - tn: 42.9722 - fn: 56.9167 - accuracy: 0.1943 - precision: 0.1469 - recall: 0.3787 - val_loss: 0.7014 - val_tp: 99.0000 - val_fp: 999.0000 - val_tn: 1.0000 - val_fn: 1.0000 - val_accuracy: 0.0909 - val_precision: 0.0902 - val_recall: 0.9900
    Epoch 6/10
    35/35 [==============================] - 4s 110ms/step - loss: 2.1641 - tp: 93.3056 - fp: 492.6111 - tn: 4.0833 - fn: 0.0000e+00 - accuracy: 0.2471 - precision: 0.2413 - recall: 1.0000 - val_loss: 0.7013 - val_tp: 98.0000 - val_fp: 993.0000 - val_tn: 7.0000 - val_fn: 2.0000 - val_accuracy: 0.0955 - val_precision: 0.0898 - val_recall: 0.9800
    Epoch 7/10
    35/35 [==============================] - 4s 110ms/step - loss: 2.1644 - tp: 93.3056 - fp: 476.6944 - tn: 20.0000 - fn: 0.0000e+00 - accuracy: 0.2727 - precision: 0.2474 - recall: 1.0000 - val_loss: 0.7021 - val_tp: 98.0000 - val_fp: 975.0000 - val_tn: 25.0000 - val_fn: 2.0000 - val_accuracy: 0.1118 - val_precision: 0.0913 - val_recall: 0.9800
    Epoch 8/10
    35/35 [==============================] - 4s 110ms/step - loss: 2.1670 - tp: 90.4444 - fp: 463.3056 - tn: 33.3889 - fn: 2.8611 - accuracy: 0.2873 - precision: 0.2478 - recall: 0.9693 - val_loss: 0.7018 - val_tp: 98.0000 - val_fp: 969.0000 - val_tn: 31.0000 - val_fn: 2.0000 - val_accuracy: 0.1173 - val_precision: 0.0918 - val_recall: 0.9800
    Epoch 9/10
    35/35 [==============================] - 4s 110ms/step - loss: 2.1651 - tp: 89.6111 - fp: 443.3333 - tn: 53.3611 - fn: 3.6944 - accuracy: 0.3121 - precision: 0.2515 - recall: 0.9603 - val_loss: 0.7003 - val_tp: 95.0000 - val_fp: 914.0000 - val_tn: 86.0000 - val_fn: 5.0000 - val_accuracy: 0.1645 - val_precision: 0.0942 - val_recall: 0.9500
    Epoch 10/10
    35/35 [==============================] - 4s 110ms/step - loss: 2.1727 - tp: 79.8333 - fp: 432.6944 - tn: 64.0000 - fn: 13.4722 - accuracy: 0.3193 - precision: 0.2496 - recall: 0.8496 - val_loss: 0.6937 - val_tp: 89.0000 - val_fp: 850.0000 - val_tn: 150.0000 - val_fn: 11.0000 - val_accuracy: 0.2173 - val_precision: 0.0948 - val_recall: 0.8900
    

#### Experiment #2 Evaluation


```python
eval_weighted = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_weighted.metrics_names, eval_weighted):
    print(name, ': ', value)
```

    loss :  0.5695805549621582
    tp :  63.0
    fp :  279.0
    tn :  721.0
    fn :  37.0
    accuracy :  0.7127272486686707
    precision :  0.18421052396297455
    recall :  0.6299999952316284
    


```python
weighted_preds = model_weighted.predict(test_ds)
```


```python
plot_cm(true_labels, weighted_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_56_0.png)
    


Interestingly, even if you undo the difference in the unbalanced data by adjusting the weights, it goes *too* far.

## Experiment #2b - More class_weight


```python
class_weight2 = {0:1, 1:25}
```


```python
model_weighted2 = get_model()
model_weighted2.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_weighted2 = model_weighted2.fit(train_ds_unbalanced, epochs=NUM_EPOCHS, validation_data=val_ds, class_weight=class_weight)
```

    Epoch 1/10
    35/35 [==============================] - 6s 130ms/step - loss: 7.2505 - tp: 141.3056 - fp: 405.4167 - tn: 1091.2778 - fn: 52.0000 - accuracy: 0.7222 - precision: 0.2574 - recall: 0.7272 - val_loss: 2.1460 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/10
    35/35 [==============================] - 4s 113ms/step - loss: 50.8568 - tp: 0.0000e+00 - fp: 24.6667 - tn: 472.0278 - fn: 93.3056 - accuracy: 0.7230 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.5504 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/10
    35/35 [==============================] - 4s 112ms/step - loss: 33.8207 - tp: 0.0000e+00 - fp: 145.5556 - tn: 351.1389 - fn: 93.3056 - accuracy: 0.5539 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.5051 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/10
    35/35 [==============================] - 4s 111ms/step - loss: 12.4551 - tp: 0.0000e+00 - fp: 161.8611 - tn: 334.8333 - fn: 93.3056 - accuracy: 0.5483 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3295 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 5/10
    35/35 [==============================] - 4s 112ms/step - loss: 5.9104 - tp: 4.3333 - fp: 303.2500 - tn: 193.4444 - fn: 88.9722 - accuracy: 0.3504 - precision: 0.0489 - recall: 0.0434 - val_loss: 0.5748 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 6/10
    35/35 [==============================] - 4s 112ms/step - loss: 2.7828 - tp: 14.6111 - fp: 383.1389 - tn: 113.5556 - fn: 78.6944 - accuracy: 0.2804 - precision: 0.1289 - recall: 0.1506 - val_loss: 0.7046 - val_tp: 99.0000 - val_fp: 995.0000 - val_tn: 5.0000 - val_fn: 1.0000 - val_accuracy: 0.0945 - val_precision: 0.0905 - val_recall: 0.9900
    Epoch 7/10
    35/35 [==============================] - 4s 112ms/step - loss: 2.1691 - tp: 70.6944 - fp: 445.5278 - tn: 51.1667 - fn: 22.6111 - accuracy: 0.2614 - precision: 0.2151 - recall: 0.7559 - val_loss: 0.7038 - val_tp: 99.0000 - val_fp: 995.0000 - val_tn: 5.0000 - val_fn: 1.0000 - val_accuracy: 0.0945 - val_precision: 0.0905 - val_recall: 0.9900
    Epoch 8/10
    35/35 [==============================] - 4s 113ms/step - loss: 2.1738 - tp: 70.1667 - fp: 402.8056 - tn: 93.8889 - fn: 23.1389 - accuracy: 0.3480 - precision: 0.2460 - recall: 0.7564 - val_loss: 0.7029 - val_tp: 99.0000 - val_fp: 987.0000 - val_tn: 13.0000 - val_fn: 1.0000 - val_accuracy: 0.1018 - val_precision: 0.0912 - val_recall: 0.9900
    Epoch 9/10
    35/35 [==============================] - 4s 113ms/step - loss: 2.1742 - tp: 63.3889 - fp: 380.3889 - tn: 116.3056 - fn: 29.9167 - accuracy: 0.3552 - precision: 0.2328 - recall: 0.6786 - val_loss: 0.7020 - val_tp: 98.0000 - val_fp: 982.0000 - val_tn: 18.0000 - val_fn: 2.0000 - val_accuracy: 0.1055 - val_precision: 0.0907 - val_recall: 0.9800
    Epoch 10/10
    35/35 [==============================] - 4s 114ms/step - loss: 2.1708 - tp: 62.0833 - fp: 350.5000 - tn: 146.1944 - fn: 31.2222 - accuracy: 0.4027 - precision: 0.2512 - recall: 0.6588 - val_loss: 0.7007 - val_tp: 98.0000 - val_fp: 966.0000 - val_tn: 34.0000 - val_fn: 2.0000 - val_accuracy: 0.1200 - val_precision: 0.0921 - val_recall: 0.9800
    

#### Experiment #2b Evaluation


```python
eval_weighted = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_weighted.metrics_names, eval_weighted):
    print(name, ': ', value)
```

    loss :  0.5695805549621582
    tp :  63.0
    fp :  279.0
    tn :  721.0
    fn :  37.0
    accuracy :  0.7127272486686707
    precision :  0.18421052396297455
    recall :  0.6299999952316284
    


```python
weighted_preds = model_weighted.predict(test_ds)
```


```python
plot_cm(true_labels, weighted_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_65_0.png)
    


This definitely got more of the dog images. But I find the results using `class_weight` very inconsistent, so it's not my preferred approach.

## Experiment #3 Using Oversampling

I generally prefer oversampling. Here's how to do it with tf.data datasets. We know it's a 10:1 ratio, so we have to repeat the dogs 10 times.


```python
resampled_ds = tf.data.experimental.sample_from_datasets([cat_list_train_unbalanced, dog_list_train_unbalanced.repeat(10)], weights=[0.5, 0.5], seed=42)
resampled_ds = resampled_ds.map(parse_file)
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
```

Just to make sure it's what we want we can take a look at the labels.


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



The sum is half the total, so half are 1 and half are 0, which is what we expected. Let's train the model.


```python
model_oversampled = get_model()
model_oversampled.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_oversampled = model_oversampled.fit(resampled_ds, epochs=NUM_EPOCHS, validation_data=val_ds)
```

    Epoch 1/10
    63/63 [==============================] - 8s 99ms/step - loss: 1.6733 - tp: 394.0000 - fp: 588.6250 - tn: 910.4219 - fn: 245.9531 - accuracy: 0.6184 - precision: 0.3718 - recall: 0.5877 - val_loss: 0.4905 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/10
    63/63 [==============================] - 5s 86ms/step - loss: 0.7255 - tp: 303.0000 - fp: 247.6406 - tn: 251.4062 - fn: 236.9531 - accuracy: 0.5220 - precision: 0.5472 - recall: 0.4450 - val_loss: 0.6257 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/10
    63/63 [==============================] - 5s 86ms/step - loss: 0.6872 - tp: 228.0625 - fp: 147.6562 - tn: 351.3906 - fn: 311.8906 - accuracy: 0.5334 - precision: 0.6290 - recall: 0.3161 - val_loss: 0.3200 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/10
    63/63 [==============================] - 6s 88ms/step - loss: 0.8273 - tp: 320.6562 - fp: 197.0000 - tn: 302.0469 - fn: 219.2969 - accuracy: 0.5749 - precision: 0.6009 - recall: 0.4940 - val_loss: 0.3626 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 5/10
    63/63 [==============================] - 5s 87ms/step - loss: 0.8374 - tp: 401.3750 - fp: 226.6562 - tn: 272.3906 - fn: 138.5781 - accuracy: 0.6184 - precision: 0.5987 - recall: 0.6505 - val_loss: 0.4203 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 6/10
    63/63 [==============================] - 5s 87ms/step - loss: 0.7276 - tp: 425.2969 - fp: 182.4844 - tn: 316.5625 - fn: 114.6562 - accuracy: 0.6814 - precision: 0.7202 - recall: 0.6953 - val_loss: 0.4304 - val_tp: 8.0000 - val_fp: 19.0000 - val_tn: 981.0000 - val_fn: 92.0000 - val_accuracy: 0.8991 - val_precision: 0.2963 - val_recall: 0.0800
    Epoch 7/10
    63/63 [==============================] - 5s 87ms/step - loss: 0.4909 - tp: 450.7969 - fp: 135.5000 - tn: 363.5469 - fn: 89.1562 - accuracy: 0.7604 - precision: 0.7821 - recall: 0.7752 - val_loss: 0.4514 - val_tp: 18.0000 - val_fp: 49.0000 - val_tn: 951.0000 - val_fn: 82.0000 - val_accuracy: 0.8809 - val_precision: 0.2687 - val_recall: 0.1800
    Epoch 8/10
    63/63 [==============================] - 5s 87ms/step - loss: 0.3279 - tp: 463.5625 - fp: 65.0625 - tn: 433.9844 - fn: 76.3906 - accuracy: 0.8505 - precision: 0.8889 - recall: 0.8223 - val_loss: 0.4783 - val_tp: 16.0000 - val_fp: 45.0000 - val_tn: 955.0000 - val_fn: 84.0000 - val_accuracy: 0.8827 - val_precision: 0.2623 - val_recall: 0.1600
    Epoch 9/10
    63/63 [==============================] - 5s 87ms/step - loss: 0.2250 - tp: 493.6094 - fp: 40.9375 - tn: 458.1094 - fn: 46.3438 - accuracy: 0.9063 - precision: 0.9305 - recall: 0.8899 - val_loss: 0.5454 - val_tp: 23.0000 - val_fp: 64.0000 - val_tn: 936.0000 - val_fn: 77.0000 - val_accuracy: 0.8718 - val_precision: 0.2644 - val_recall: 0.2300
    Epoch 10/10
    63/63 [==============================] - 5s 86ms/step - loss: 0.1744 - tp: 507.1719 - fp: 27.1094 - tn: 471.9375 - fn: 32.7812 - accuracy: 0.9409 - precision: 0.9517 - recall: 0.9350 - val_loss: 0.6756 - val_tp: 23.0000 - val_fp: 65.0000 - val_tn: 935.0000 - val_fn: 77.0000 - val_accuracy: 0.8709 - val_precision: 0.2614 - val_recall: 0.2300
    

#### Experiment #3 Evaluation


```python
eval_oversampled = model_oversampled.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_oversampled.metrics_names, eval_oversampled):
    print(name, ': ', value)
```

    loss :  0.6102479100227356
    tp :  29.0
    fp :  56.0
    tn :  944.0
    fn :  71.0
    accuracy :  0.8845454454421997
    precision :  0.34117648005485535
    recall :  0.28999999165534973
    


```python
oversampled_preds = model_oversampled.predict(test_ds)
```


```python
plot_cm(true_labels, oversampled_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_80_0.png)
    


So even though it's not perfect, we got a decent result. This does cause a problem though because we've overfit one side and not the other. Depending on how much data we have, we could undersample as well. But I think it would be better to just add data augmentation.

## Conclusion

You'll notice that no result was always better than the other results. It depended on the exact parameters and what metrics are most important to you. If you only care about recall, you may want to weigh the target labels extra heavily. You can also combine `class_weight` and oversampling and tweak both to your liking.
