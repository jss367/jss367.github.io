---
layout: post
title: "Validation on Unbalanced Datasets"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/water_text.jpg"
tags: [Deep Learning, Python, TensorFlow]
---

In this post I'm going to look at different methods of dealing with unbalanced data. For these, we'll use the Kaggle [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats). The dataset has the same number of cat images as dog images, so we'll have to subset the dataset to run the experiment. We're going to pretend that there are 10 times as many cats as there are dogs in our population, and we want to build a model that answers the question, "Is this an image of a dog?"

This notebook focus on the makeup of the validation set.

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
    

Can you fix things with a custom loss function?

## Experiment #1 - Should the Validation Data Be Balanced or Representative?

OK. Out first experiment we'll make a couple train datasets. One options is to have a balanced dataset, the other is to allow it to be unbalanced to match the "real world". Let's see which one produces better results.


```python
cat_list_train, cat_list_val_balanced, cat_list_val_representative, cat_list_test = subset_dataset(cat_list_ds, [2000, 1000, 1000, 1000])
dog_list_train, dog_list_val_balanced, dog_list_val_representative, dog_list_test = subset_dataset(dog_list_ds, [2000, 1000, 100, 1000])
```


```python
train_ds = prepare_dataset(cat_list_train, dog_list_train)
val_ds_balanced = prepare_dataset(cat_list_val_balanced, dog_list_val_balanced)
val_ds_representative = prepare_dataset(cat_list_val_representative, dog_list_val_representative)
test_ds = prepare_dataset(cat_list_test, dog_list_test)
```

Great. Now let's train the models.


```python
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(patience=2), tf.keras.callbacks.EarlyStopping(patience=4)]
```


```python
model_balanced = get_model()
model_balanced.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_balanced = model_balanced.fit(train_ds, epochs=NUM_EPOCHS, validation_data=val_ds_balanced, callbacks=callbacks)
```

    Epoch 1/20
    125/125 [==============================] - 15s 103ms/step - loss: 1.2744 - tp: 608.8095 - fp: 570.6508 - tn: 417.7063 - fn: 434.5794 - accuracy: 0.4903 - precision: 0.5052 - recall: 0.5535 - val_loss: 0.6878 - val_tp: 175.0000 - val_fp: 91.0000 - val_tn: 909.0000 - val_fn: 825.0000 - val_accuracy: 0.5420 - val_precision: 0.6579 - val_recall: 0.1750
    Epoch 2/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.6767 - tp: 591.4921 - fp: 423.8810 - tn: 564.4762 - fn: 451.8968 - accuracy: 0.5543 - precision: 0.5912 - recall: 0.4868 - val_loss: 0.6774 - val_tp: 306.0000 - val_fp: 180.0000 - val_tn: 820.0000 - val_fn: 694.0000 - val_accuracy: 0.5630 - val_precision: 0.6296 - val_recall: 0.3060
    Epoch 3/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.6667 - tp: 594.2222 - fp: 374.7619 - tn: 613.5952 - fn: 449.1667 - accuracy: 0.5759 - precision: 0.6235 - recall: 0.4858 - val_loss: 0.7402 - val_tp: 27.0000 - val_fp: 6.0000 - val_tn: 994.0000 - val_fn: 973.0000 - val_accuracy: 0.5105 - val_precision: 0.8182 - val_recall: 0.0270
    Epoch 4/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.6663 - tp: 688.2857 - fp: 391.3095 - tn: 597.0476 - fn: 355.1032 - accuracy: 0.6106 - precision: 0.6263 - recall: 0.5868 - val_loss: 0.7847 - val_tp: 76.0000 - val_fp: 11.0000 - val_tn: 989.0000 - val_fn: 924.0000 - val_accuracy: 0.5325 - val_precision: 0.8736 - val_recall: 0.0760435 - fp: 288.7609 - tn: 431.4891 - fn: 273.7065 - accuracy: 0.5980 - 
    Epoch 5/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.6706 - tp: 579.5159 - fp: 223.9683 - tn: 764.3889 - fn: 463.8730 - accuracy: 0.6365 - precision: 0.7621 - recall: 0.4659 - val_loss: 0.5982 - val_tp: 665.0000 - val_fp: 279.0000 - val_tn: 721.0000 - val_fn: 335.0000 - val_accuracy: 0.6930 - val_precision: 0.7044 - val_recall: 0.6650
    Epoch 6/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.5739 - tp: 741.2619 - fp: 279.9444 - tn: 708.4127 - fn: 302.1270 - accuracy: 0.7101 - precision: 0.7274 - recall: 0.7060 - val_loss: 0.5865 - val_tp: 614.0000 - val_fp: 218.0000 - val_tn: 782.0000 - val_fn: 386.0000 - val_accuracy: 0.6980 - val_precision: 0.7380 - val_recall: 0.6140
    Epoch 7/20
    125/125 [==============================] - 11s 89ms/step - loss: 0.5692 - tp: 718.7143 - fp: 268.8016 - tn: 719.5556 - fn: 324.6746 - accuracy: 0.7011 - precision: 0.7293 - recall: 0.6753 - val_loss: 0.5780 - val_tp: 611.0000 - val_fp: 209.0000 - val_tn: 791.0000 - val_fn: 389.0000 - val_accuracy: 0.7010 - val_precision: 0.7451 - val_recall: 0.6110
    Epoch 8/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.5492 - tp: 726.7143 - fp: 244.4762 - tn: 743.8810 - fn: 316.6746 - accuracy: 0.7181 - precision: 0.7496 - recall: 0.6861 - val_loss: 0.5689 - val_tp: 618.0000 - val_fp: 206.0000 - val_tn: 794.0000 - val_fn: 382.0000 - val_accuracy: 0.7060 - val_precision: 0.7500 - val_recall: 0.6180p: 179.1277 - tn: 556.7021 - fn: 247.0000 - accuracy: 0.7143 - pr
    Epoch 9/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.5429 - tp: 735.9127 - fp: 231.2540 - tn: 757.1032 - fn: 307.4762 - accuracy: 0.7308 - precision: 0.7640 - recall: 0.6970 - val_loss: 0.5589 - val_tp: 638.0000 - val_fp: 193.0000 - val_tn: 807.0000 - val_fn: 362.0000 - val_accuracy: 0.7225 - val_precision: 0.7677 - val_recall: 0.6380
    Epoch 10/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.5338 - tp: 735.5317 - fp: 237.3492 - tn: 751.0079 - fn: 307.8571 - accuracy: 0.7222 - precision: 0.7550 - recall: 0.6893 - val_loss: 0.5558 - val_tp: 608.0000 - val_fp: 168.0000 - val_tn: 832.0000 - val_fn: 392.0000 - val_accuracy: 0.7200 - val_precision: 0.7835 - val_recall: 0.6080
    Epoch 11/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.5195 - tp: 743.2302 - fp: 214.1508 - tn: 774.2063 - fn: 300.1587 - accuracy: 0.7363 - precision: 0.7757 - recall: 0.6924 - val_loss: 0.5484 - val_tp: 623.0000 - val_fp: 178.0000 - val_tn: 822.0000 - val_fn: 377.0000 - val_accuracy: 0.7225 - val_precision: 0.7778 - val_recall: 0.6230
    Epoch 12/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.5188 - tp: 752.5079 - fp: 230.0794 - tn: 758.2778 - fn: 290.8810 - accuracy: 0.7372 - precision: 0.7675 - recall: 0.7087 - val_loss: 0.5443 - val_tp: 622.0000 - val_fp: 166.0000 - val_tn: 834.0000 - val_fn: 378.0000 - val_accuracy: 0.7280 - val_precision: 0.7893 - val_recall: 0.6220940 - tn: 497.6867 - fn: 200.9398 - accuracy: 0.7315 - precision: 0.7680 - recall: 0. - ETA: 2s - loss: 0.5245 - tp: 517.6552 - fp: 158.4138 - tn: 522.7356 - fn: 209.1954 - accuracy: 0.73
    Epoch 13/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.5060 - tp: 768.3730 - fp: 223.3968 - tn: 764.9603 - fn: 275.0159 - accuracy: 0.7516 - precision: 0.7822 - recall: 0.7231 - val_loss: 0.5398 - val_tp: 636.0000 - val_fp: 168.0000 - val_tn: 832.0000 - val_fn: 364.0000 - val_accuracy: 0.7340 - val_precision: 0.7910 - val_recall: 0.6360
    Epoch 14/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.4906 - tp: 776.2222 - fp: 212.5952 - tn: 775.7619 - fn: 267.1667 - accuracy: 0.7581 - precision: 0.7850 - recall: 0.7356 - val_loss: 0.5334 - val_tp: 643.0000 - val_fp: 166.0000 - val_tn: 834.0000 - val_fn: 357.0000 - val_accuracy: 0.7385 - val_precision: 0.7948 - val_recall: 0.6430
    Epoch 15/20
    125/125 [==============================] - 11s 89ms/step - loss: 0.4973 - tp: 766.5476 - fp: 202.9048 - tn: 785.4524 - fn: 276.8413 - accuracy: 0.7569 - precision: 0.7924 - recall: 0.7204 - val_loss: 0.5294 - val_tp: 655.0000 - val_fp: 183.0000 - val_tn: 817.0000 - val_fn: 345.0000 - val_accuracy: 0.7360 - val_precision: 0.7816 - val_recall: 0.6550
    Epoch 16/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.4775 - tp: 781.5000 - fp: 196.0556 - tn: 792.3016 - fn: 261.8889 - accuracy: 0.7768 - precision: 0.8088 - recall: 0.7461 - val_loss: 0.5266 - val_tp: 675.0000 - val_fp: 192.0000 - val_tn: 808.0000 - val_fn: 325.0000 - val_accuracy: 0.7415 - val_precision: 0.7785 - val_recall: 0.6750
    Epoch 17/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.4711 - tp: 777.0079 - fp: 186.5635 - tn: 801.7937 - fn: 266.3810 - accuracy: 0.7702 - precision: 0.8091 - recall: 0.7300 - val_loss: 0.5269 - val_tp: 657.0000 - val_fp: 176.0000 - val_tn: 824.0000 - val_fn: 343.0000 - val_accuracy: 0.7405 - val_precision: 0.7887 - val_recall: 0.6570
    Epoch 18/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.4670 - tp: 800.0317 - fp: 187.3175 - tn: 801.0397 - fn: 243.3571 - accuracy: 0.7818 - precision: 0.8147 - recall: 0.7503 - val_loss: 0.5258 - val_tp: 654.0000 - val_fp: 173.0000 - val_tn: 827.0000 - val_fn: 346.0000 - val_accuracy: 0.7405 - val_precision: 0.7908 - val_recall: 0.6540fn: 233.0583 - accuracy: 0.7814 - precision: 0.8153 - recall: 
    Epoch 19/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.4546 - tp: 805.4206 - fp: 187.7222 - tn: 800.6349 - fn: 237.9683 - accuracy: 0.7906 - precision: 0.8187 - recall: 0.7662 - val_loss: 0.5220 - val_tp: 660.0000 - val_fp: 176.0000 - val_tn: 824.0000 - val_fn: 340.0000 - val_accuracy: 0.7420 - val_precision: 0.7895 - val_recall: 0.6600
    Epoch 20/20
    125/125 [==============================] - 11s 88ms/step - loss: 0.4481 - tp: 804.9365 - fp: 188.9921 - tn: 799.3651 - fn: 238.4524 - accuracy: 0.7876 - precision: 0.8177 - recall: 0.7608 - val_loss: 0.5258 - val_tp: 635.0000 - val_fp: 164.0000 - val_tn: 836.0000 - val_fn: 365.0000 - val_accuracy: 0.7355 - val_precision: 0.7947 - val_recall: 0.6350
    


```python
model_representative = get_model()
model_representative.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_representative = model_representative.fit(train_ds, epochs=NUM_EPOCHS, validation_data=val_ds_representative, callbacks=callbacks)
```

    Epoch 1/20
    125/125 [==============================] - 12s 82ms/step - loss: 0.7748 - tp: 1289.9683 - fp: 775.9603 - tn: 1212.3968 - fn: 753.4206 - accuracy: 0.6307 - precision: 0.6420 - recall: 0.6304 - val_loss: 0.3915 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/20
    125/125 [==============================] - 10s 82ms/step - loss: 0.7266 - tp: 596.1587 - fp: 548.8492 - tn: 439.5079 - fn: 447.2302 - accuracy: 0.4998 - precision: 0.4875 - recall: 0.4909 - val_loss: 0.4833 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/20
    125/125 [==============================] - 11s 87ms/step - loss: 0.7074 - tp: 515.7619 - fp: 385.4841 - tn: 602.8730 - fn: 527.6270 - accuracy: 0.5342 - precision: 0.5510 - recall: 0.3969 - val_loss: 0.5778 - val_tp: 36.0000 - val_fp: 117.0000 - val_tn: 883.0000 - val_fn: 64.0000 - val_accuracy: 0.8355 - val_precision: 0.2353 - val_recall: 0.3600acy:
    Epoch 4/20
    125/125 [==============================] - 11s 87ms/step - loss: 0.6616 - tp: 575.3968 - fp: 322.1984 - tn: 666.1587 - fn: 467.9921 - accuracy: 0.6078 - precision: 0.6681 - recall: 0.5039 - val_loss: 0.6349 - val_tp: 73.0000 - val_fp: 427.0000 - val_tn: 573.0000 - val_fn: 27.0000 - val_accuracy: 0.5873 - val_precision: 0.1460 - val_recall: 0.7300
    Epoch 5/20
    125/125 [==============================] - 10s 83ms/step - loss: 0.6363 - tp: 748.2063 - fp: 448.8254 - tn: 539.5317 - fn: 295.1825 - accuracy: 0.6272 - precision: 0.6247 - recall: 0.7063 - val_loss: 0.6064 - val_tp: 66.0000 - val_fp: 337.0000 - val_tn: 663.0000 - val_fn: 34.0000 - val_accuracy: 0.6627 - val_precision: 0.1638 - val_recall: 0.66009.1165 - fp: 364.3495 - tn: 442.3495 - fn: 248.1845 - accuracy: 0.6248 - precision: 
    

#### Experiment #1 Evaluation


```python
plot_loss(history_balanced, "Balanced Validation")
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_15_0.png)
    



```python
plot_loss(history_representative, "Representative Validation")
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_16_0.png)
    



```python
eval_balanced = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
eval_representative = model_representative.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_balanced.metrics_names, eval_balanced):
    print(name, ': ', value)
```

    loss :  0.5160902738571167
    tp :  661.0
    fp :  173.0
    tn :  827.0
    fn :  339.0
    accuracy :  0.7440000176429749
    precision :  0.7925659418106079
    recall :  0.6610000133514404
    


```python
for name, value in zip(model_representative.metrics_names, eval_representative):
    print(name, ': ', value)
```

    loss :  0.6362556219100952
    tp :  631.0
    fp :  332.0
    tn :  668.0
    fn :  369.0
    accuracy :  0.6495000123977661
    precision :  0.6552440524101257
    recall :  0.6309999823570251
    

OK, so the model trained on unbalanced has a higher accuracy, but that's because it predicted the majority class for everything! It has **zero** precision and recall.


```python
balanced_preds = model_balanced.predict(test_ds)
representative_preds = model_representative.predict(test_ds)
true_labels = tf.concat([y for x, y in test_ds], axis=0)
```


```python
plot_cm(true_labels, balanced_preds)
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_22_0.png)
    



```python
plot_cm(true_labels, representative_preds)
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_23_0.png)
    


## Experiment #2

## Conclusion

You'll notice that no result was always better than the other results. It depended on the exact parameters and what metrics are most important to you. If you only care about recall, you may want to weigh the target labels extra heavily. You can also combine `class_weight` and oversampling and tweak both to your liking.
