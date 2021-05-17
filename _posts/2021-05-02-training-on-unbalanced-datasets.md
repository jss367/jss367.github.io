---
layout: post
title: "Training on Unbalanced Datasets"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/water_text.jpg"
tags: [Deep Learning, Python, TensorFlow]
---

This post is in a series on doing machine learning with unbalanced datasets. This post focuses on the training aspect in particular. For background, please see the [setup](https://jss367.github.io/experiements-on-unbalanced-datasets-setup.html) post.

When we think about training a machine learning model on an unbalanced dataset, we need to answer a number of questions along the way. The first is, given that there are more cats than dogs in our population, should there also be more dogs than cats in the training data? That is, should we have unbalanced training data? Or is it better to have it balanced? And, if we want to balance the data, what's the best way to do it? If we can't add more data, the two most popular methods for rebalancing are adding more weight to the less common image or oversampling it. Which is better?

<b>Table of contents</b>
* TOC
{:toc}


```python
%run 2021-05-01-prep-for-experiements-on-unbalanced-datasets.ipynb
```

    Classes:
     ['cats', 'dogs']
    There are a total of 5000 cat images in the entire dataset.
    There are a total of 5000 dog images in the entire dataset.
    

## Experiment #1 - Should the Training Data Be Balanced or Unbalanced?

For our first experiment we'll make a couple train datasets. One option is to have a balanced dataset, the other is to allow it to be unbalanced to match the "real world". Let's see which one produces better results.


```python
cat_list_train_balanced, cat_list_train_unbalanced, cat_list_val, cat_list_test = subset_dataset(cat_list_ds, [1500, 1500, 1000, 1000])
dog_list_train_balanced, dog_list_train_unbalanced, dog_list_val, dog_list_test = subset_dataset(dog_list_ds, [1500, 150, 100, 100])
```


```python
train_ds_balanced = prepare_dataset(cat_list_train_balanced, dog_list_train_balanced)
train_ds_unbalanced = prepare_dataset(cat_list_train_unbalanced, dog_list_train_unbalanced)
val_ds = prepare_dataset(cat_list_val, dog_list_val)
test_ds = prepare_dataset(cat_list_test, dog_list_test)
```

Now let's train the models.


```python
model_balanced = get_model()
model_balanced.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_balanced = model_balanced.fit(train_ds_balanced, epochs=NUM_EPOCHS, validation_data=val_ds)
```

    Epoch 1/10
    94/94 [==============================] - 12s 103ms/step - loss: 1.4082 - tp: 449.8421 - fp: 435.1474 - tn: 309.6632 - fn: 340.8421 - accuracy: 0.4812 - precision: 0.4984 - recall: 0.5356 - val_loss: 0.5143 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/10
    94/94 [==============================] - 8s 85ms/step - loss: 0.7084 - tp: 366.4947 - fp: 322.8211 - tn: 421.9895 - fn: 424.1895 - accuracy: 0.5059 - precision: 0.5714 - recall: 0.3808 - val_loss: 0.6856 - val_tp: 52.0000 - val_fp: 409.0000 - val_tn: 591.0000 - val_fn: 48.0000 - val_accuracy: 0.5845 - val_precision: 0.1128 - val_recall: 0.5200
    Epoch 3/10
    94/94 [==============================] - 8s 85ms/step - loss: 0.6840 - tp: 503.5474 - fp: 406.3474 - tn: 338.4632 - fn: 287.1368 - accuracy: 0.5333 - precision: 0.5455 - recall: 0.6017 - val_loss: 0.6361 - val_tp: 66.0000 - val_fp: 427.0000 - val_tn: 573.0000 - val_fn: 34.0000 - val_accuracy: 0.5809 - val_precision: 0.1339 - val_recall: 0.6600cy: 0.5202 - precision: 0.5391  - ETA: 0s - loss: 0.6852 - tp: 421.4375 - fp: 343.5375 - tn: 281.6625 - fn: 249.3625 - accuracy: 0.5278 - precision: 0.5429 - 
    Epoch 4/10
    94/94 [==============================] - 8s 85ms/step - loss: 0.6765 - tp: 474.8105 - fp: 351.6316 - tn: 393.1789 - fn: 315.8737 - accuracy: 0.5585 - precision: 0.5858 - recall: 0.5548 - val_loss: 0.6090 - val_tp: 68.0000 - val_fp: 401.0000 - val_tn: 599.0000 - val_fn: 32.0000 - val_accuracy: 0.6064 - val_precision: 0.1450 - val_recall: 0.6800 238.5614 - fn: 213.2807 - accuracy: 0.5482
    Epoch 5/10
    94/94 [==============================] - 8s 85ms/step - loss: 0.6674 - tp: 428.9263 - fp: 284.4737 - tn: 460.3368 - fn: 361.7579 - accuracy: 0.5648 - precision: 0.6149 - recall: 0.4685 - val_loss: 0.3975 - val_tp: 4.0000 - val_fp: 10.0000 - val_tn: 990.0000 - val_fn: 96.0000 - val_accuracy: 0.9036 - val_precision: 0.2857 - val_recall: 0.0400
    Epoch 6/10
    94/94 [==============================] - 8s 84ms/step - loss: 0.7003 - tp: 415.5368 - fp: 286.9579 - tn: 457.8526 - fn: 375.1474 - accuracy: 0.5487 - precision: 0.6381 - recall: 0.4146 - val_loss: 0.6125 - val_tp: 82.0000 - val_fp: 512.0000 - val_tn: 488.0000 - val_fn: 18.0000 - val_accuracy: 0.5182 - val_precision: 0.1380 - val_recall: 0.8200
    Epoch 7/10
    94/94 [==============================] - 8s 84ms/step - loss: 0.6503 - tp: 581.7789 - fp: 341.5053 - tn: 403.3053 - fn: 208.9053 - accuracy: 0.6416 - precision: 0.6351 - recall: 0.7337 - val_loss: 0.5782 - val_tp: 70.0000 - val_fp: 302.0000 - val_tn: 698.0000 - val_fn: 30.0000 - val_accuracy: 0.6982 - val_precision: 0.1882 - val_recall: 0.70004.4407 - tn: 244.5763 - fn: 131.2373 - accuracy: 0.6412 -
    Epoch 8/10
    94/94 [==============================] - 8s 85ms/step - loss: 0.6527 - tp: 535.9263 - fp: 331.4526 - tn: 413.3579 - fn: 254.7579 - accuracy: 0.6123 - precision: 0.6257 - recall: 0.6477 - val_loss: 0.6014 - val_tp: 75.0000 - val_fp: 425.0000 - val_tn: 575.0000 - val_fn: 25.0000 - val_accuracy: 0.5909 - val_precision: 0.1500 - val_recall: 0.7500
    Epoch 9/10
    94/94 [==============================] - 8s 84ms/step - loss: 0.6366 - tp: 541.8842 - fp: 263.9895 - tn: 480.8211 - fn: 248.8000 - accuracy: 0.6612 - precision: 0.6804 - recall: 0.6656 - val_loss: 0.4841 - val_tp: 56.0000 - val_fp: 192.0000 - val_tn: 808.0000 - val_fn: 44.0000 - val_accuracy: 0.7855 - val_precision: 0.2258 - val_recall: 0.5600
    Epoch 10/10
    94/94 [==============================] - 8s 85ms/step - loss: 0.6268 - tp: 551.7053 - fp: 267.6316 - tn: 477.1789 - fn: 238.9789 - accuracy: 0.6589 - precision: 0.6776 - recall: 0.6637 - val_loss: 0.4209 - val_tp: 45.0000 - val_fp: 118.0000 - val_tn: 882.0000 - val_fn: 55.0000 - val_accuracy: 0.8427 - val_precision: 0.2761 - val_recall: 0.4500
    


```python
model_unbalanced = get_model()
model_unbalanced.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_unbalanced = model_unbalanced.fit(train_ds_unbalanced, epochs=NUM_EPOCHS, validation_data=val_ds)
```

    Epoch 1/10
    52/52 [==============================] - 8s 120ms/step - loss: 0.6336 - tp: 123.7547 - fp: 243.4528 - tn: 1480.0943 - fn: 115.5660 - accuracy: 0.8100 - precision: 0.3371 - recall: 0.5146 - val_loss: 3.0452 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/10
    52/52 [==============================] - 5s 105ms/step - loss: 6.7396 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 723.5472 - fn: 139.3208 - accuracy: 0.7552 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 5.4944 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/10
    52/52 [==============================] - 5s 105ms/step - loss: 9.4548 - tp: 0.0000e+00 - fp: 1.5094 - tn: 722.0377 - fn: 139.3208 - accuracy: 0.7535 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 2.6275 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/10
    52/52 [==============================] - 5s 106ms/step - loss: 4.9902 - tp: 0.0000e+00 - fp: 22.9811 - tn: 700.5660 - fn: 139.3208 - accuracy: 0.7299 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 2.0154 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00fn: 133.8286 - accuracy: 0.6570 - precisio
    Epoch 5/10
    52/52 [==============================] - 5s 104ms/step - loss: 3.9752 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 723.5472 - fn: 139.3208 - accuracy: 0.7552 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 2.1523 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 6/10
    52/52 [==============================] - 5s 105ms/step - loss: 3.8912 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 723.5472 - fn: 139.3208 - accuracy: 0.7552 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.5992 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+0000e+00 - fp: 0.0000e+00 - tn: 157.3750 - fn: 114.6250 - accuracy: 0.5338 - precision: 0.0000e+00 - recall: 0.000 - ETA: 1s - loss: 6.6593 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 228.9524 - fn: 123.0476 - accuracy: 0.5857 - precision: 0.0000e+00 - recall - ETA: 1s - loss: 5.1668 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 411.1515 - fn: 132.8485 - accuracy: 0.6734 - precision: 0.0000e+00 -
    Epoch 7/10
    52/52 [==============================] - 5s 106ms/step - loss: 1.5379 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 723.5472 - fn: 139.3208 - accuracy: 0.7552 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.7308 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 8/10
    52/52 [==============================] - 5s 105ms/step - loss: 1.7880 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 723.5472 - fn: 139.3208 - accuracy: 0.7552 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3189 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 9/10
    52/52 [==============================] - 5s 106ms/step - loss: 0.7203 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 723.5472 - fn: 139.3208 - accuracy: 0.7552 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3168 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 10/10
    52/52 [==============================] - 5s 106ms/step - loss: 0.8390 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 723.5472 - fn: 139.3208 - accuracy: 0.7552 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3125 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    

#### Experiment #1 Evaluation


```python
plot_loss(history_balanced, "Balanced Training")
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_13_0.png)
    



```python
plot_loss(history_unbalanced, "Unbalanced Training")
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_14_0.png)
    



```python
eval_balanced = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
eval_unbalanced = model_unbalanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_balanced.metrics_names, eval_balanced):
    print(name, ': ', value)
```

    loss :  0.44630274176597595
    tp :  40.0
    fp :  147.0
    tn :  853.0
    fn :  60.0
    accuracy :  0.8118181824684143
    precision :  0.2139037400484085
    recall :  0.4000000059604645
    


```python
for name, value in zip(model_unbalanced.metrics_names, eval_unbalanced):
    print(name, ': ', value)
```

    loss :  0.31001996994018555
    tp :  0.0
    fp :  0.0
    tn :  1000.0
    fn :  100.0
    accuracy :  0.9090909361839294
    precision :  0.0
    recall :  0.0
    

OK, so the model trained on the unbalanced dataset has a higher accuracy, but that's because it predicted the majority class for everything! It has **zero** precision and recall.


```python
balanced_preds = model_balanced.predict(test_ds)
unbalanced_preds = model_unbalanced.predict(test_ds)
true_labels = tf.concat([y for x, y in test_ds], axis=0)
```


```python
plot_cm(true_labels, balanced_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_20_0.png)
    



```python
plot_cm(true_labels, unbalanced_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_21_0.png)
    


## Experiment #2 - Using class_weight

We know from the above experiments that training on balanced data is better than training on unbalanced, but what if we can't get balanced data... what then? There are many ways to try to get around this. One of the most popular is my using adjusting the weights of the less common images so the model learns more from them. We know we have 10 times as many cats as dogs, so we'll weigh the dogs 10X as much.


```python
class_weight = {0:1, 1:10}
```


```python
model_weighted = get_model()
model_weighted.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_weighted = model_weighted.fit(train_ds_unbalanced, epochs=NUM_EPOCHS, validation_data=val_ds, class_weight=class_weight)
```

    Epoch 1/10
    52/52 [==============================] - 8s 117ms/step - loss: 2.0352 - tp: 133.5472 - fp: 167.3585 - tn: 1556.1887 - fn: 105.7736 - accuracy: 0.8569 - precision: 0.4539 - recall: 0.5473 - val_loss: 2.9559 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/10
    52/52 [==============================] - 5s 105ms/step - loss: 53.0102 - tp: 0.0000e+00 - fp: 38.4528 - tn: 685.0943 - fn: 139.3208 - accuracy: 0.7115 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 3.1960 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00.0000e+00 - fp: 36.0000 - tn: 531.1628 - fn: 136.8372 - accuracy: 0.6744 - precision: 0.0000e+00 - recall: 0.0000e+ - ETA: 0s - loss: 59.5521 - tp: 0.0000e+00 - fp: 36.5778 - tn: 562.0000 - fn: 137.4222 - accuracy: 0.6826 - precision: 0.0000e+00 - recall: 0.0
    Epoch 3/10
    52/52 [==============================] - 5s 104ms/step - loss: 53.7068 - tp: 0.8491 - fp: 255.1887 - tn: 468.3585 - fn: 138.4717 - accuracy: 0.4927 - precision: 0.0077 - recall: 0.0057 - val_loss: 1.0481 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 fn: 133.9189 - accuracy: 0.4286 - prec
    Epoch 4/10
    52/52 [==============================] - 5s 104ms/step - loss: 20.9313 - tp: 0.0000e+00 - fp: 353.7547 - tn: 369.7925 - fn: 139.3208 - accuracy: 0.4423 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.6004 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00: 261.9500 - fn: 135.8500 - accuracy: 0.4349 - precision: 0.0000e
    Epoch 5/10
    52/52 [==============================] - 5s 105ms/step - loss: 11.1606 - tp: 9.5849 - fp: 271.3585 - tn: 452.1887 - fn: 129.7358 - accuracy: 0.4745 - precision: 0.1153 - recall: 0.0644 - val_loss: 0.5341 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00- fn: 122.6333 - accuracy: 0.3689 - precision: 0.1814  - ETA: 0s - loss: 13.9626 - tp: 9.0263 - fp: 231.8158 - tn: 257.0789 - fn: 126.0789 - accuracy: 0.4055 - precis
    Epoch 6/10
    52/52 [==============================] - 5s 103ms/step - loss: 11.1932 - tp: 1.8302 - fp: 592.8302 - tn: 130.7170 - fn: 137.4906 - accuracy: 0.2248 - precision: 0.0594 - recall: 0.0132 - val_loss: 0.7219 - val_tp: 100.0000 - val_fp: 999.0000 - val_tn: 1.0000 - val_fn: 0.0000e+00 - val_accuracy: 0.0918 - val_precision: 0.0910 - val_recall: 1.0000
    Epoch 7/10
    52/52 [==============================] - 5s 105ms/step - loss: 2.1647 - tp: 139.3208 - fp: 723.0189 - tn: 0.5283 - fn: 0.0000e+00 - accuracy: 0.2453 - precision: 0.2449 - recall: 1.0000 - val_loss: 0.7188 - val_tp: 100.0000 - val_fp: 999.0000 - val_tn: 1.0000 - val_fn: 0.0000e+00 - val_accuracy: 0.0918 - val_precision: 0.0910 - val_recall: 1.0000
    Epoch 8/10
    52/52 [==============================] - 5s 105ms/step - loss: 2.1660 - tp: 139.3208 - fp: 722.0377 - tn: 1.5094 - fn: 0.0000e+00 - accuracy: 0.2474 - precision: 0.2457 - recall: 1.0000 - val_loss: 0.7207 - val_tp: 100.0000 - val_fp: 999.0000 - val_tn: 1.0000 - val_fn: 0.0000e+00 - val_accuracy: 0.0918 - val_precision: 0.0910 - val_recall: 1.0000
    Epoch 9/10
    52/52 [==============================] - 5s 104ms/step - loss: 2.0718 - tp: 139.3208 - fp: 453.0189 - tn: 270.5283 - fn: 0.0000e+00 - accuracy: 0.4496 - precision: 0.2857 - recall: 1.0000 - val_loss: 2.0206 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 10/10
    52/52 [==============================] - 5s 105ms/step - loss: 29.4499 - tp: 22.5283 - fp: 628.2264 - tn: 95.3208 - fn: 116.7925 - accuracy: 0.2076 - precision: 0.1026 - recall: 0.1504 - val_loss: 0.7271 - val_tp: 100.0000 - val_fp: 1000.0000 - val_tn: 0.0000e+00 - val_fn: 0.0000e+00 - val_accuracy: 0.0909 - val_precision: 0.0909 - val_recall: 1.0000789 - fn: 113.9474 - accuracy: 0.2551 - precision: 0 - ETA: 0s - loss: 31.1127 - tp: 22.2449 - fp: 566.6939 - tn: 94.8571 - fn: 116.2041 - accuracy: 0.2182 - precision: 0.1095 - recall: 
    

#### Experiment #2 Evaluation


```python
eval_weighted = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_weighted.metrics_names, eval_weighted):
    print(name, ': ', value)
```

    loss :  0.44630274176597595
    tp :  40.0
    fp :  147.0
    tn :  853.0
    fn :  60.0
    accuracy :  0.8118181824684143
    precision :  0.2139037400484085
    recall :  0.4000000059604645
    


```python
weighted_preds = model_weighted.predict(test_ds)
```


```python
plot_cm(true_labels, weighted_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_30_0.png)
    


Interestingly, if you undo the difference in the unbalanced data by adjusting the weights, it goes *too* far.

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
    52/52 [==============================] - 8s 117ms/step - loss: 5.4867 - tp: 159.8679 - fp: 290.6038 - tn: 1432.9434 - fn: 79.4528 - accuracy: 0.8031 - precision: 0.3519 - recall: 0.6603 - val_loss: 1.6818 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/10
    52/52 [==============================] - 5s 106ms/step - loss: 32.6174 - tp: 1.7736 - fp: 82.1698 - tn: 641.3774 - fn: 137.5472 - accuracy: 0.6695 - precision: 0.1169 - recall: 0.0119 - val_loss: 1.4399 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+009.5307 - tp: 0.6667 - fp: 0.0000e+00 - tn: 72.8889 - fn: 86.4444 - accuracy: 0.4613 - precision: 0.444 - ETA: 1s - loss: 59.5299 - tp: 1.4545 - fp: 42.9545 - tn: 200.7727 - fn: 122.8182 - accuracy: 0.5214 - precision: - ETA: 0s - loss: 36.0998 - tp: 1.7391 - fp: 77.9348 - tn: 536.3696 - fn: 135.9565 - accuracy: 0.6439 - precision: 0.1320 - recall: 0.01 - ETA: 0s - loss: 35.0198 - tp: 1.7500 - fp: 79.2708 - tn: 566.5208 - fn: 136.4583 - accuracy: 0.6517 - precision: 0.1272 - recall: 0.
    Epoch 3/10
    52/52 [==============================] - 5s 104ms/step - loss: 23.4215 - tp: 5.1132 - fp: 155.8868 - tn: 567.6604 - fn: 134.2075 - accuracy: 0.5824 - precision: 0.0402 - recall: 0.0341 - val_loss: 2.0155 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/10
    52/52 [==============================] - 5s 104ms/step - loss: 28.7920 - tp: 38.4340 - fp: 171.0000 - tn: 552.5472 - fn: 100.8868 - accuracy: 0.6034 - precision: 0.1804 - recall: 0.2569 - val_loss: 1.7657 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 5/10
    52/52 [==============================] - 5s 106ms/step - loss: 29.8723 - tp: 1.7170 - fp: 432.5472 - tn: 291.0000 - fn: 137.6038 - accuracy: 0.3384 - precision: 0.0271 - recall: 0.0115 - val_loss: 0.4318 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 6/10
    52/52 [==============================] - 5s 105ms/step - loss: 3.1502 - tp: 56.5472 - fp: 660.6604 - tn: 62.8868 - fn: 82.7736 - accuracy: 0.2167 - precision: 0.1800 - recall: 0.4089 - val_loss: 0.7229 - val_tp: 100.0000 - val_fp: 1000.0000 - val_tn: 0.0000e+00 - val_fn: 0.0000e+00 - val_accuracy: 0.0909 - val_precision: 0.0909 - val_recall: 1.0000
    Epoch 7/10
    52/52 [==============================] - 5s 105ms/step - loss: 2.1631 - tp: 139.3208 - fp: 723.5472 - tn: 0.0000e+00 - fn: 0.0000e+00 - accuracy: 0.2448 - precision: 0.2448 - recall: 1.0000 - val_loss: 0.7223 - val_tp: 100.0000 - val_fp: 1000.0000 - val_tn: 0.0000e+00 - val_fn: 0.0000e+00 - val_accuracy: 0.0909 - val_precision: 0.0909 - val_recall: 1.0000 - tn: 0.0000e+00 - fn: 0.0000e+00 - accuracy: 0.3511 - precision: 0.3511 - recall:  - ETA: 1s - loss: 2.5777 - tp: 133.8286 - fp: 442.1714 - tn: 0.0000e+00 - fn: 0.0000e+00 - accuracy: 0.3157 - precision: 0.3157
    Epoch 8/10
    52/52 [==============================] - 5s 105ms/step - loss: 2.1642 - tp: 139.3208 - fp: 723.5472 - tn: 0.0000e+00 - fn: 0.0000e+00 - accuracy: 0.2448 - precision: 0.2448 - recall: 1.0000 - val_loss: 0.7217 - val_tp: 100.0000 - val_fp: 1000.0000 - val_tn: 0.0000e+00 - val_fn: 0.0000e+00 - val_accuracy: 0.0909 - val_precision: 0.0909 - val_recall: 1.0000
    Epoch 9/10
    52/52 [==============================] - 5s 105ms/step - loss: 2.1655 - tp: 139.3208 - fp: 723.5472 - tn: 0.0000e+00 - fn: 0.0000e+00 - accuracy: 0.2448 - precision: 0.2448 - recall: 1.0000 - val_loss: 0.7211 - val_tp: 100.0000 - val_fp: 1000.0000 - val_tn: 0.0000e+00 - val_fn: 0.0000e+00 - val_accuracy: 0.0909 - val_precision: 0.0909 - val_recall: 1.0000
    Epoch 10/10
    52/52 [==============================] - 5s 105ms/step - loss: 2.1667 - tp: 139.3208 - fp: 723.5472 - tn: 0.0000e+00 - fn: 0.0000e+00 - accuracy: 0.2448 - precision: 0.2448 - recall: 1.0000 - val_loss: 0.7204 - val_tp: 100.0000 - val_fp: 1000.0000 - val_tn: 0.0000e+00 - val_fn: 0.0000e+00 - val_accuracy: 0.0909 - val_precision: 0.0909 - val_recall: 1.0000
    

#### Experiment #2b Evaluation


```python
eval_weighted = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_weighted.metrics_names, eval_weighted):
    print(name, ': ', value)
```

    loss :  0.44630274176597595
    tp :  40.0
    fp :  147.0
    tn :  853.0
    fn :  60.0
    accuracy :  0.8118181824684143
    precision :  0.2139037400484085
    recall :  0.4000000059604645
    


```python
weighted_preds = model_weighted.predict(test_ds)
```


```python
plot_cm(true_labels, weighted_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_39_0.png)
    


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




    <tf.Tensor: shape=(3000,), dtype=int64, numpy=array([0, 0, 0, ..., 0, 0, 0], dtype=int64)>




```python
sum(resampled_true_labels)
```




    <tf.Tensor: shape=(), dtype=int64, numpy=1500>



The sum is half the total, so half are 1 and half are 0, which is what we expected. Let's train the model.


```python
model_oversampled = get_model()
model_oversampled.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_oversampled = model_oversampled.fit(resampled_ds, epochs=NUM_EPOCHS, validation_data=val_ds)
```

    Epoch 1/10
    94/94 [==============================] - 10s 94ms/step - loss: 1.4639 - tp: 542.7053 - fp: 637.8000 - tn: 1107.0105 - fn: 347.9789 - accuracy: 0.6424 - precision: 0.4321 - recall: 0.5671 - val_loss: 0.6605 - val_tp: 3.0000 - val_fp: 12.0000 - val_tn: 988.0000 - val_fn: 97.0000 - val_accuracy: 0.9009 - val_precision: 0.2000 - val_recall: 0.0300
    Epoch 2/10
    94/94 [==============================] - 8s 87ms/step - loss: 0.6921 - tp: 478.7684 - fp: 372.0211 - tn: 372.7895 - fn: 311.9158 - accuracy: 0.5395 - precision: 0.5583 - recall: 0.5344 - val_loss: 0.5639 - val_tp: 5.0000 - val_fp: 16.0000 - val_tn: 984.0000 - val_fn: 95.0000 - val_accuracy: 0.8991 - val_precision: 0.2381 - val_recall: 0.0500fp: 16.3636 - tn: 70.2727 - fn: 83.9091 - accuracy: 0.4647 - precision: 0.5336 - - ETA: 4s - loss: 0.7066 - tp: 78.5652 - fp: 62.6087 - tn: 117.6522 - fn: 125.1739 - accuracy: 0.4927 - p - ETA: 2s - loss: 0.6992 - tp: 228.7843 - fp: 178.0980 - tn: 219.0784 - fn: 206.0392 - accuracy: 
    Epoch 3/10
    94/94 [==============================] - 8s 86ms/step - loss: 0.6519 - tp: 520.3368 - fp: 321.3263 - tn: 423.4842 - fn: 270.3474 - accuracy: 0.6032 - precision: 0.6437 - recall: 0.5866 - val_loss: 0.3287 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00- precision: 0.6457 - recall: - ETA: 0s - loss: 0.6527 - tp: 508.3441 - fp: 314.1935 - tn: 414.3763 - fn: 267.0860 - accuracy: 0.6024 - precision: 0.6441 - recall: 0.583
    Epoch 4/10
    94/94 [==============================] - 8s 87ms/step - loss: 0.6947 - tp: 576.1895 - fp: 265.2737 - tn: 479.5368 - fn: 214.4947 - accuracy: 0.6676 - precision: 0.6886 - recall: 0.6683 - val_loss: 0.3570 - val_tp: 1.0000 - val_fp: 3.0000 - val_tn: 997.0000 - val_fn: 99.0000 - val_accuracy: 0.9073 - val_precision: 0.2500 - val_recall: 0.01008.1250 - fn: 118.9250 - accuracy: 0.6356 - precision: 0.6953 - rec - ETA: 2s - loss: 0.7862 - tp: 294.2549 - fp: 137.2745 - tn: 259.9020 - fn: 140.5686 - accuracy: 0.6442 - precision: 0.6917 - recall:  - ETA: 2s - loss: 0.7682 - tp: 332.7719 - fp: 155.4737 - tn: 288.1754 - fn: 151.5789 - accuracy: 0.6481 - precision: 0.6905 - recall:  - ETA: 1s - loss: 0.7529 - tp: 371.5556 - fp: 173.8730 - tn: 316.3810 - fn: 162.1905 - accuracy: 0.6515 - pre
    Epoch 5/10
    94/94 [==============================] - 8s 87ms/step - loss: 0.5646 - tp: 645.3053 - fp: 245.8421 - tn: 498.9684 - fn: 145.3789 - accuracy: 0.7215 - precision: 0.7151 - recall: 0.7667 - val_loss: 0.4652 - val_tp: 5.0000 - val_fp: 30.0000 - val_tn: 970.0000 - val_fn: 95.0000 - val_accuracy: 0.8864 - val_precision: 0.1429 - val_recall: 0.0500
    Epoch 6/10
    94/94 [==============================] - 8s 87ms/step - loss: 0.3855 - tp: 673.3158 - fp: 132.2842 - tn: 612.5263 - fn: 117.3684 - accuracy: 0.8197 - precision: 0.8377 - recall: 0.8171 - val_loss: 0.5273 - val_tp: 6.0000 - val_fp: 28.0000 - val_tn: 972.0000 - val_fn: 94.0000 - val_accuracy: 0.8891 - val_precision: 0.1765 - val_recall: 0.06005600 - fp: 38.3200 - tn: 157.6400 - fn: 49.4800 - accuracy: 0.7751 - precision: 0.8516 - - ETA: 3s - loss: 0.4351 - tp: 278.4250 - fp: 63.5250 - tn: 248.6000 - fn: 65.4500 - accuracy: 0.7892 - precision: 0.8370 - recall: 0.760 - ETA: 3s - loss: 0.4338 - tp: 285.5366 - fp: 65.0976 - tn: 254.7561 - fn: 66.
    Epoch 7/10
    94/94 [==============================] - 8s 86ms/step - loss: 0.2794 - tp: 713.3474 - fp: 91.8842 - tn: 652.9263 - fn: 77.3368 - accuracy: 0.8770 - precision: 0.8830 - recall: 0.8817 - val_loss: 0.5698 - val_tp: 15.0000 - val_fp: 40.0000 - val_tn: 960.0000 - val_fn: 85.0000 - val_accuracy: 0.8864 - val_precision: 0.2727 - val_recall: 0.1500n: 430.9531 - fn: 57.9688 - accuracy: 0.8670 - precision: 0.8785 - recall: 0. - ETA: 1s - loss: 0.2953 - tp: 513.3235 - fp: 70.6324 - tn: 459.2206 - fn: 60.8235 - accuracy: 0.8684 - precision: 0.8790 - recall: 0 - ETA: 1s - loss: 0.2921 - tp: 550.2192 - fp: 75.0548 - tn: 494.5616 - fn: 64.1644 - accuracy: 0.8700 - precision: 0.8795 - recall: 0 - ETA: 1s - loss: 0.2890 - tp: 587.3974 - fp: 79.1538 - tn: 530.1282 - fn: 67.3205 - accuracy: 0.8717 - precision: 0.8802 
    Epoch 8/10
    94/94 [==============================] - 8s 87ms/step - loss: 0.1790 - tp: 748.2211 - fp: 54.7158 - tn: 690.0947 - fn: 42.4632 - accuracy: 0.9289 - precision: 0.9304 - recall: 0.9339 - val_loss: 0.5990 - val_tp: 15.0000 - val_fp: 47.0000 - val_tn: 953.0000 - val_fn: 85.0000 - val_accuracy: 0.8800 - val_precision: 0.2419 - val_recall: 0.150038.0253 - accuracy: 0.9256 - precision: 0.9288 
    Epoch 9/10
    94/94 [==============================] - 8s 86ms/step - loss: 0.1020 - tp: 768.8211 - fp: 32.0000 - tn: 712.8105 - fn: 21.8632 - accuracy: 0.9615 - precision: 0.9599 - recall: 0.9667 - val_loss: 0.6806 - val_tp: 14.0000 - val_fp: 46.0000 - val_tn: 954.0000 - val_fn: 86.0000 - val_accuracy: 0.8800 - val_precision: 0.2333 - val_recall: 0.1400.1500 - tn: 14
    Epoch 10/10
    94/94 [==============================] - 8s 87ms/step - loss: 0.0664 - tp: 776.3895 - fp: 21.1895 - tn: 723.6211 - fn: 14.2947 - accuracy: 0.9774 - precision: 0.9757 - recall: 0.9810 - val_loss: 0.8054 - val_tp: 17.0000 - val_fp: 36.0000 - val_tn: 964.0000 - val_fn: 83.0000 - val_accuracy: 0.8918 - val_precision: 0.3208 - val_recall: 0.1700
    

#### Experiment #3 Evaluation


```python
eval_oversampled = model_oversampled.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_oversampled.metrics_names, eval_oversampled):
    print(name, ': ', value)
```

    loss :  0.6757662296295166
    tp :  22.0
    fp :  40.0
    tn :  960.0
    fn :  78.0
    accuracy :  0.892727255821228
    precision :  0.35483869910240173
    recall :  0.2199999988079071
    


```python
oversampled_preds = model_oversampled.predict(test_ds)
```


```python
plot_cm(true_labels, oversampled_preds)
```


    
![png](2021-05-02-training-on-unbalanced-datasets_files/2021-05-02-training-on-unbalanced-datasets_54_0.png)
    


So even though it's not perfect, we got a decent result. This does cause a problem though because we've overfit one side and not the other. Depending on how much data we have, we could undersample as well. But I think it would be better to just add data augmentation.

## Conclusion

You'll notice that no result was always better than the other results. It depended on the exact parameters and what metrics are most important. If you only care about recall, you may want to weigh the target labels extra heavily. You can also combine `class_weight` and oversampling and tweak both to your liking.
