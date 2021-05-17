---
layout: post
title: "Validation on Unbalanced Datasets"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/windy_roo.jpg"
tags: [Deep Learning, Python, TensorFlow]
---

This post is in a series on doing machine learning with unbalanced datasets. This post focuses on the makeup of the validation set in particular. For background, please see the [setup](https://jss367.github.io/experiements-on-unbalanced-datasets-setup.html) post.

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
    

## Train

Let's start out by creating our datasets. We'll make a balanced train set and a representative test set and keep those constant. We'll also make **two** validation sets - one will be balanced between the classes and the other will be representative of the "real world". Let's see which one produces better results.


```python
cat_list_train, cat_list_val_balanced, cat_list_val_representative, cat_list_test = subset_dataset(cat_list_ds, [2000, 1000, 1000, 1000])
dog_list_train, dog_list_val_balanced, dog_list_val_representative, dog_list_test = subset_dataset(dog_list_ds, [2000, 1000, 100, 100])
```


```python
train_ds = prepare_dataset(cat_list_train, dog_list_train)
val_ds_balanced = prepare_dataset(cat_list_val_balanced, dog_list_val_balanced)
val_ds_representative = prepare_dataset(cat_list_val_representative, dog_list_val_representative)
test_ds = prepare_dataset(cat_list_test, dog_list_test)
```

Now let's train the models. The validation set is going to affect the way our models train because we're going to use callbacks that include the validation set. I do this in real-world scenarios so it only makes sense to do it here. Let's set up those callbacks now. 


```python
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(patience=2), tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
```


```python
NUM_EPOCHS = 50
```

I'll do three different experiments. They'll be the same other than the learning rate. I wanted to vary the learning rate because this greatly affects model convergence, and we want to make sure the results we see aren't an odd edge case.

## Experiment #1


```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
```


```python
model_balanced = get_model()
model_balanced.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=all_metrics)
history_balanced = model_balanced.fit(train_ds, epochs=NUM_EPOCHS, validation_data=val_ds_balanced, callbacks=callbacks)
```

    Epoch 1/50
    125/125 [==============================] - 15s 96ms/step - loss: 0.9327 - tp: 567.3254 - fp: 542.7698 - tn: 445.5873 - fn: 476.0635 - accuracy: 0.4927 - precision: 0.5079 - recall: 0.5577 - val_loss: 0.6918 - val_tp: 796.0000 - val_fp: 623.0000 - val_tn: 377.0000 - val_fn: 204.0000 - val_accuracy: 0.5865 - val_precision: 0.5610 - val_recall: 0.7960
    Epoch 2/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.6906 - tp: 280.0238 - fp: 241.5635 - tn: 746.7937 - fn: 763.3651 - accuracy: 0.4985 - precision: 0.5426 - recall: 0.2161 - val_loss: 0.6921 - val_tp: 30.0000 - val_fp: 25.0000 - val_tn: 975.0000 - val_fn: 970.0000 - val_accuracy: 0.5025 - val_precision: 0.5455 - val_recall: 0.0300
    Epoch 3/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.6883 - tp: 456.4683 - fp: 355.4841 - tn: 632.8730 - fn: 586.9206 - accuracy: 0.5383 - precision: 0.5883 - recall: 0.4059 - val_loss: 0.6909 - val_tp: 8.0000 - val_fp: 6.0000 - val_tn: 994.0000 - val_fn: 992.0000 - val_accuracy: 0.5010 - val_precision: 0.5714 - val_recall: 0.0080
    Epoch 4/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.7056 - tp: 326.8175 - fp: 269.8968 - tn: 718.4603 - fn: 716.5714 - accuracy: 0.5036 - precision: 0.5519 - recall: 0.2391 - val_loss: 0.6882 - val_tp: 625.0000 - val_fp: 404.0000 - val_tn: 596.0000 - val_fn: 375.0000 - val_accuracy: 0.6105 - val_precision: 0.6074 - val_recall: 0.6250
    Epoch 5/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.6814 - tp: 637.7937 - fp: 466.3175 - tn: 522.0397 - fn: 405.5952 - accuracy: 0.5663 - precision: 0.5775 - recall: 0.6081 - val_loss: 0.6838 - val_tp: 57.0000 - val_fp: 15.0000 - val_tn: 985.0000 - val_fn: 943.0000 - val_accuracy: 0.5210 - val_precision: 0.7917 - val_recall: 0.0570
    Epoch 6/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.6878 - tp: 552.3889 - fp: 355.0079 - tn: 633.3492 - fn: 491.0000 - accuracy: 0.5678 - precision: 0.6141 - recall: 0.4640 - val_loss: 0.6901 - val_tp: 101.0000 - val_fp: 37.0000 - val_tn: 963.0000 - val_fn: 899.0000 - val_accuracy: 0.5320 - val_precision: 0.7319 - val_recall: 0.1010
    Epoch 7/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.6888 - tp: 577.9127 - fp: 332.3333 - tn: 656.0238 - fn: 465.4762 - accuracy: 0.5929 - precision: 0.6335 - recall: 0.5055 - val_loss: 0.6603 - val_tp: 620.0000 - val_fp: 334.0000 - val_tn: 666.0000 - val_fn: 380.0000 - val_accuracy: 0.6430 - val_precision: 0.6499 - val_recall: 0.6200
    Epoch 8/50
    125/125 [==============================] - 11s 84ms/step - loss: 0.6584 - tp: 662.2619 - fp: 386.1429 - tn: 602.2143 - fn: 381.1270 - accuracy: 0.6117 - precision: 0.6355 - recall: 0.5951 - val_loss: 0.6472 - val_tp: 520.0000 - val_fp: 245.0000 - val_tn: 755.0000 - val_fn: 480.0000 - val_accuracy: 0.6375 - val_precision: 0.6797 - val_recall: 0.5200
    Epoch 9/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.6388 - tp: 706.6032 - fp: 386.6508 - tn: 601.7063 - fn: 336.7857 - accuracy: 0.6285 - precision: 0.6452 - recall: 0.6318 - val_loss: 0.6697 - val_tp: 257.0000 - val_fp: 71.0000 - val_tn: 929.0000 - val_fn: 743.0000 - val_accuracy: 0.5930 - val_precision: 0.7835 - val_recall: 0.2570
    Epoch 10/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.6340 - tp: 701.0794 - fp: 363.5635 - tn: 624.7937 - fn: 342.3095 - accuracy: 0.6300 - precision: 0.6494 - recall: 0.6203 - val_loss: 0.6179 - val_tp: 500.0000 - val_fp: 200.0000 - val_tn: 800.0000 - val_fn: 500.0000 - val_accuracy: 0.6500 - val_precision: 0.7143 - val_recall: 0.5000
    Epoch 11/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.6065 - tp: 716.2540 - fp: 325.4206 - tn: 662.9365 - fn: 327.1349 - accuracy: 0.6638 - precision: 0.6885 - recall: 0.6456 - val_loss: 0.6007 - val_tp: 517.0000 - val_fp: 169.0000 - val_tn: 831.0000 - val_fn: 483.0000 - val_accuracy: 0.6740 - val_precision: 0.7536 - val_recall: 0.5170
    Epoch 12/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.5895 - tp: 725.1905 - fp: 305.9683 - tn: 682.3889 - fn: 318.1984 - accuracy: 0.6822 - precision: 0.7132 - recall: 0.6551 - val_loss: 0.5901 - val_tp: 625.0000 - val_fp: 217.0000 - val_tn: 783.0000 - val_fn: 375.0000 - val_accuracy: 0.7040 - val_precision: 0.7423 - val_recall: 0.6250
    Epoch 13/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.5680 - tp: 757.8333 - fp: 289.4841 - tn: 698.8730 - fn: 285.5556 - accuracy: 0.7062 - precision: 0.7308 - recall: 0.6913 - val_loss: 0.5719 - val_tp: 615.0000 - val_fp: 191.0000 - val_tn: 809.0000 - val_fn: 385.0000 - val_accuracy: 0.7120 - val_precision: 0.7630 - val_recall: 0.6150
    Epoch 14/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.5460 - tp: 722.8571 - fp: 255.1111 - tn: 733.2460 - fn: 320.5317 - accuracy: 0.7068 - precision: 0.7495 - recall: 0.6585 - val_loss: 0.5580 - val_tp: 624.0000 - val_fp: 193.0000 - val_tn: 807.0000 - val_fn: 376.0000 - val_accuracy: 0.7155 - val_precision: 0.7638 - val_recall: 0.6240
    Epoch 15/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.5268 - tp: 772.2460 - fp: 265.5952 - tn: 722.7619 - fn: 271.1429 - accuracy: 0.7231 - precision: 0.7466 - recall: 0.7084 - val_loss: 0.5630 - val_tp: 592.0000 - val_fp: 156.0000 - val_tn: 844.0000 - val_fn: 408.0000 - val_accuracy: 0.7180 - val_precision: 0.7914 - val_recall: 0.5920
    Epoch 16/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.5180 - tp: 771.1587 - fp: 249.7540 - tn: 738.6032 - fn: 272.2302 - accuracy: 0.7288 - precision: 0.7544 - recall: 0.7087 - val_loss: 0.5640 - val_tp: 523.0000 - val_fp: 105.0000 - val_tn: 895.0000 - val_fn: 477.0000 - val_accuracy: 0.7090 - val_precision: 0.8328 - val_recall: 0.5230
    Epoch 17/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4947 - tp: 738.7698 - fp: 180.8889 - tn: 807.4683 - fn: 304.6190 - accuracy: 0.7464 - precision: 0.8062 - recall: 0.6738 - val_loss: 0.5278 - val_tp: 699.0000 - val_fp: 225.0000 - val_tn: 775.0000 - val_fn: 301.0000 - val_accuracy: 0.7370 - val_precision: 0.7565 - val_recall: 0.6990
    Epoch 18/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4595 - tp: 811.4048 - fp: 220.3968 - tn: 767.9603 - fn: 231.9841 - accuracy: 0.7755 - precision: 0.7862 - recall: 0.7787 - val_loss: 0.5246 - val_tp: 699.0000 - val_fp: 214.0000 - val_tn: 786.0000 - val_fn: 301.0000 - val_accuracy: 0.7425 - val_precision: 0.7656 - val_recall: 0.6990
    Epoch 19/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.4586 - tp: 815.5952 - fp: 221.2460 - tn: 767.1111 - fn: 227.7937 - accuracy: 0.7725 - precision: 0.7829 - recall: 0.7767 - val_loss: 0.5238 - val_tp: 691.0000 - val_fp: 209.0000 - val_tn: 791.0000 - val_fn: 309.0000 - val_accuracy: 0.7410 - val_precision: 0.7678 - val_recall: 0.6910
    Epoch 20/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.4592 - tp: 814.6587 - fp: 222.5079 - tn: 765.8492 - fn: 228.7302 - accuracy: 0.7709 - precision: 0.7815 - recall: 0.7751 - val_loss: 0.5214 - val_tp: 706.0000 - val_fp: 214.0000 - val_tn: 786.0000 - val_fn: 294.0000 - val_accuracy: 0.7460 - val_precision: 0.7674 - val_recall: 0.7060
    Epoch 21/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4525 - tp: 820.6508 - fp: 213.3730 - tn: 774.9841 - fn: 222.7381 - accuracy: 0.7809 - precision: 0.7932 - recall: 0.7810 - val_loss: 0.5220 - val_tp: 696.0000 - val_fp: 202.0000 - val_tn: 798.0000 - val_fn: 304.0000 - val_accuracy: 0.7470 - val_precision: 0.7751 - val_recall: 0.6960
    Epoch 22/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4614 - tp: 819.3413 - fp: 207.4365 - tn: 780.9206 - fn: 224.0476 - accuracy: 0.7815 - precision: 0.7945 - recall: 0.7804 - val_loss: 0.5202 - val_tp: 702.0000 - val_fp: 218.0000 - val_tn: 782.0000 - val_fn: 298.0000 - val_accuracy: 0.7420 - val_precision: 0.7630 - val_recall: 0.7020
    Epoch 23/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.4542 - tp: 831.7460 - fp: 216.4524 - tn: 771.9048 - fn: 211.6429 - accuracy: 0.7844 - precision: 0.7895 - recall: 0.7965 - val_loss: 0.5192 - val_tp: 698.0000 - val_fp: 203.0000 - val_tn: 797.0000 - val_fn: 302.0000 - val_accuracy: 0.7475 - val_precision: 0.7747 - val_recall: 0.6980
    Epoch 24/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.4448 - tp: 823.8016 - fp: 197.0000 - tn: 791.3571 - fn: 219.5873 - accuracy: 0.7903 - precision: 0.8056 - recall: 0.7848 - val_loss: 0.5188 - val_tp: 702.0000 - val_fp: 202.0000 - val_tn: 798.0000 - val_fn: 298.0000 - val_accuracy: 0.7500 - val_precision: 0.7765 - val_recall: 0.7020
    Epoch 25/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4344 - tp: 829.0317 - fp: 201.1111 - tn: 787.2460 - fn: 214.3571 - accuracy: 0.7942 - precision: 0.8073 - recall: 0.7919 - val_loss: 0.5180 - val_tp: 695.0000 - val_fp: 200.0000 - val_tn: 800.0000 - val_fn: 305.0000 - val_accuracy: 0.7475 - val_precision: 0.7765 - val_recall: 0.6950
    Epoch 26/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4276 - tp: 836.0714 - fp: 202.2063 - tn: 786.1508 - fn: 207.3175 - accuracy: 0.7977 - precision: 0.8074 - recall: 0.8005 - val_loss: 0.5172 - val_tp: 693.0000 - val_fp: 197.0000 - val_tn: 803.0000 - val_fn: 307.0000 - val_accuracy: 0.7480 - val_precision: 0.7787 - val_recall: 0.6930
    Epoch 27/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4295 - tp: 835.9524 - fp: 204.1508 - tn: 784.2063 - fn: 207.4365 - accuracy: 0.7914 - precision: 0.8001 - recall: 0.7970 - val_loss: 0.5172 - val_tp: 688.0000 - val_fp: 194.0000 - val_tn: 806.0000 - val_fn: 312.0000 - val_accuracy: 0.7470 - val_precision: 0.7800 - val_recall: 0.6880
    Epoch 28/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4383 - tp: 822.3571 - fp: 199.9048 - tn: 788.4524 - fn: 221.0317 - accuracy: 0.7892 - precision: 0.8050 - recall: 0.7829 - val_loss: 0.5145 - val_tp: 704.0000 - val_fp: 204.0000 - val_tn: 796.0000 - val_fn: 296.0000 - val_accuracy: 0.7500 - val_precision: 0.7753 - val_recall: 0.7040
    Epoch 29/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4300 - tp: 848.7619 - fp: 197.1270 - tn: 791.2302 - fn: 194.6270 - accuracy: 0.8030 - precision: 0.8070 - recall: 0.8165 - val_loss: 0.5143 - val_tp: 703.0000 - val_fp: 204.0000 - val_tn: 796.0000 - val_fn: 297.0000 - val_accuracy: 0.7495 - val_precision: 0.7751 - val_recall: 0.7030
    Epoch 30/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4231 - tp: 842.2302 - fp: 195.1667 - tn: 793.1905 - fn: 201.1587 - accuracy: 0.8038 - precision: 0.8122 - recall: 0.8084 - val_loss: 0.5124 - val_tp: 714.0000 - val_fp: 219.0000 - val_tn: 781.0000 - val_fn: 286.0000 - val_accuracy: 0.7475 - val_precision: 0.7653 - val_recall: 0.7140
    Epoch 31/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.4229 - tp: 846.2063 - fp: 200.1905 - tn: 788.1667 - fn: 197.1825 - accuracy: 0.8034 - precision: 0.8085 - recall: 0.8135 - val_loss: 0.5102 - val_tp: 718.0000 - val_fp: 220.0000 - val_tn: 780.0000 - val_fn: 282.0000 - val_accuracy: 0.7490 - val_precision: 0.7655 - val_recall: 0.7180
    Epoch 32/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.4132 - tp: 850.1429 - fp: 193.0000 - tn: 795.3571 - fn: 193.2460 - accuracy: 0.8093 - precision: 0.8141 - recall: 0.8194 - val_loss: 0.5122 - val_tp: 692.0000 - val_fp: 188.0000 - val_tn: 812.0000 - val_fn: 308.0000 - val_accuracy: 0.7520 - val_precision: 0.7864 - val_recall: 0.6920
    Epoch 33/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4183 - tp: 835.6349 - fp: 185.1984 - tn: 803.1587 - fn: 207.7540 - accuracy: 0.8032 - precision: 0.8178 - recall: 0.7981 - val_loss: 0.5119 - val_tp: 708.0000 - val_fp: 206.0000 - val_tn: 794.0000 - val_fn: 292.0000 - val_accuracy: 0.7510 - val_precision: 0.7746 - val_recall: 0.7080
    Epoch 34/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4034 - tp: 843.3333 - fp: 180.3968 - tn: 807.9603 - fn: 200.0556 - accuracy: 0.8136 - precision: 0.8223 - recall: 0.8166 - val_loss: 0.5122 - val_tp: 706.0000 - val_fp: 203.0000 - val_tn: 797.0000 - val_fn: 294.0000 - val_accuracy: 0.7515 - val_precision: 0.7767 - val_recall: 0.7060
    Epoch 35/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4053 - tp: 844.7460 - fp: 194.1429 - tn: 794.2143 - fn: 198.6429 - accuracy: 0.8056 - precision: 0.8106 - recall: 0.8152 - val_loss: 0.5122 - val_tp: 703.0000 - val_fp: 202.0000 - val_tn: 798.0000 - val_fn: 297.0000 - val_accuracy: 0.7505 - val_precision: 0.7768 - val_recall: 0.7030
    Epoch 36/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.4117 - tp: 839.8730 - fp: 188.5397 - tn: 799.8175 - fn: 203.5159 - accuracy: 0.8071 - precision: 0.8151 - recall: 0.8117 - val_loss: 0.5121 - val_tp: 704.0000 - val_fp: 205.0000 - val_tn: 795.0000 - val_fn: 296.0000 - val_accuracy: 0.7495 - val_precision: 0.7745 - val_recall: 0.7040
    


```python
model_representative = get_model()
model_representative.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=all_metrics)
history_representative = model_representative.fit(train_ds, epochs=NUM_EPOCHS, validation_data=val_ds_representative, callbacks=callbacks)
```

    Epoch 1/50
    125/125 [==============================] - 11s 77ms/step - loss: 0.7605 - tp: 1208.3016 - fp: 679.0952 - tn: 1309.2619 - fn: 835.0873 - accuracy: 0.6359 - precision: 0.6552 - recall: 0.5985 - val_loss: 0.7070 - val_tp: 53.0000 - val_fp: 590.0000 - val_tn: 410.0000 - val_fn: 47.0000 - val_accuracy: 0.4209 - val_precision: 0.0824 - val_recall: 0.5300
    Epoch 2/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.7319 - tp: 554.8968 - fp: 508.4603 - tn: 479.8968 - fn: 488.4921 - accuracy: 0.5071 - precision: 0.5243 - recall: 0.5299 - val_loss: 0.7026 - val_tp: 50.0000 - val_fp: 555.0000 - val_tn: 445.0000 - val_fn: 50.0000 - val_accuracy: 0.4500 - val_precision: 0.0826 - val_recall: 0.5000
    Epoch 3/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.7326 - tp: 560.4762 - fp: 522.0079 - tn: 466.3492 - fn: 482.9127 - accuracy: 0.5055 - precision: 0.5224 - recall: 0.5334 - val_loss: 0.7010 - val_tp: 47.0000 - val_fp: 539.0000 - val_tn: 461.0000 - val_fn: 53.0000 - val_accuracy: 0.4618 - val_precision: 0.0802 - val_recall: 0.4700
    Epoch 4/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.7166 - tp: 563.3175 - fp: 513.8571 - tn: 474.5000 - fn: 480.0714 - accuracy: 0.5137 - precision: 0.5305 - recall: 0.5386 - val_loss: 0.7008 - val_tp: 48.0000 - val_fp: 536.0000 - val_tn: 464.0000 - val_fn: 52.0000 - val_accuracy: 0.4655 - val_precision: 0.0822 - val_recall: 0.4800
    Epoch 5/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.7175 - tp: 563.0556 - fp: 522.4206 - tn: 465.9365 - fn: 480.3333 - accuracy: 0.5145 - precision: 0.5303 - recall: 0.5488 - val_loss: 0.6995 - val_tp: 48.0000 - val_fp: 529.0000 - val_tn: 471.0000 - val_fn: 52.0000 - val_accuracy: 0.4718 - val_precision: 0.0832 - val_recall: 0.4800
    Epoch 6/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.7389 - tp: 552.9921 - fp: 549.8810 - tn: 438.4762 - fn: 490.3968 - accuracy: 0.4841 - precision: 0.5021 - recall: 0.5287 - val_loss: 0.6981 - val_tp: 47.0000 - val_fp: 515.0000 - val_tn: 485.0000 - val_fn: 53.0000 - val_accuracy: 0.4836 - val_precision: 0.0836 - val_recall: 0.4700
    Epoch 7/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.7261 - tp: 538.9603 - fp: 526.2778 - tn: 462.0794 - fn: 504.4286 - accuracy: 0.4923 - precision: 0.5101 - recall: 0.5129 - val_loss: 0.6996 - val_tp: 48.0000 - val_fp: 531.0000 - val_tn: 469.0000 - val_fn: 52.0000 - val_accuracy: 0.4700 - val_precision: 0.0829 - val_recall: 0.4800
    Epoch 8/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.7228 - tp: 553.0476 - fp: 533.8810 - tn: 454.4762 - fn: 490.3413 - accuracy: 0.4961 - precision: 0.5132 - recall: 0.5282 - val_loss: 0.6992 - val_tp: 49.0000 - val_fp: 528.0000 - val_tn: 472.0000 - val_fn: 51.0000 - val_accuracy: 0.4736 - val_precision: 0.0849 - val_recall: 0.4900
    Epoch 9/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.7213 - tp: 565.5000 - fp: 517.7222 - tn: 470.6349 - fn: 477.8889 - accuracy: 0.5179 - precision: 0.5343 - recall: 0.5464 - val_loss: 0.6989 - val_tp: 49.0000 - val_fp: 526.0000 - val_tn: 474.0000 - val_fn: 51.0000 - val_accuracy: 0.4755 - val_precision: 0.0852 - val_recall: 0.4900
    Epoch 10/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.7204 - tp: 565.1825 - fp: 537.2143 - tn: 451.1429 - fn: 478.2063 - accuracy: 0.5017 - precision: 0.5182 - recall: 0.5443 - val_loss: 0.6988 - val_tp: 49.0000 - val_fp: 524.0000 - val_tn: 476.0000 - val_fn: 51.0000 - val_accuracy: 0.4773 - val_precision: 0.0855 - val_recall: 0.4900
    Epoch 11/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.7111 - tp: 580.4921 - fp: 520.3810 - tn: 467.9762 - fn: 462.8968 - accuracy: 0.5168 - precision: 0.5325 - recall: 0.5547 - val_loss: 0.6988 - val_tp: 49.0000 - val_fp: 523.0000 - val_tn: 477.0000 - val_fn: 51.0000 - val_accuracy: 0.4782 - val_precision: 0.0857 - val_recall: 0.4900
    

#### Experiment #1 Evaluation


```python
plot_loss(history_balanced, "Balanced Validation")
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_17_0.png)
    



```python
plot_loss(history_representative, "Representative Validation")
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_18_0.png)
    



```python
eval_balanced = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
eval_representative = model_representative.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_balanced.metrics_names, eval_balanced):
    print(name, ': ', value)
```

    loss :  0.4705812335014343
    tp :  67.0
    fp :  202.0
    tn :  798.0
    fn :  33.0
    accuracy :  0.7863636612892151
    precision :  0.24907062947750092
    recall :  0.6700000166893005
    


```python
for name, value in zip(model_representative.metrics_names, eval_representative):
    print(name, ': ', value)
```

    loss :  0.6952241659164429
    tp :  61.0
    fp :  502.0
    tn :  498.0
    fn :  39.0
    accuracy :  0.5081818103790283
    precision :  0.10834813863039017
    recall :  0.6100000143051147
    

OK, so the model trained on unbalanced has a higher accuracy, but that's because it predicted the majority class for everything! It has **zero** precision and recall.


```python
balanced_preds = model_balanced.predict(test_ds)
representative_preds = model_representative.predict(test_ds)
true_labels = tf.concat([y for x, y in test_ds], axis=0)
```


```python
plot_cm(true_labels, balanced_preds)
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_24_0.png)
    



```python
plot_cm(true_labels, representative_preds)
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_25_0.png)
    


## Experiment #2

In this case we'll use a higher learning rate. Everything else remains the same.


```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
```


```python
model_balanced = get_model()
model_balanced.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=all_metrics)
history_balanced = model_balanced.fit(train_ds, epochs=NUM_EPOCHS, validation_data=val_ds_balanced, callbacks=callbacks)
```

    Epoch 1/50
    125/125 [==============================] - 13s 90ms/step - loss: 1.8018 - tp: 777.5397 - fp: 1150.6905 - tn: 837.6667 - fn: 365.8492 - accuracy: 0.5153 - precision: 0.3725 - recall: 0.6669 - val_loss: 0.6929 - val_tp: 895.0000 - val_fp: 781.0000 - val_tn: 219.0000 - val_fn: 105.0000 - val_accuracy: 0.5570 - val_precision: 0.5340 - val_recall: 0.8950
    Epoch 2/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.6933 - tp: 222.4524 - fp: 213.0159 - tn: 775.3413 - fn: 820.9365 - accuracy: 0.4902 - precision: 0.5264 - recall: 0.1947 - val_loss: 0.6931 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 1000.0000 - val_accuracy: 0.5000 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.6947 - tp: 452.9683 - fp: 430.4444 - tn: 557.9127 - fn: 590.4206 - accuracy: 0.4908 - precision: 0.4497 - recall: 0.3350 - val_loss: 0.6938 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 1000.0000 - val_accuracy: 0.5000 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.6937 - tp: 45.2778 - fp: 33.2540 - tn: 955.1032 - fn: 998.1111 - accuracy: 0.4880 - precision: 0.5873 - recall: 0.0399 - val_loss: 0.6918 - val_tp: 126.0000 - val_fp: 58.0000 - val_tn: 942.0000 - val_fn: 874.0000 - val_accuracy: 0.5340 - val_precision: 0.6848 - val_recall: 0.1260
    Epoch 5/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.6912 - tp: 278.9048 - fp: 206.6667 - tn: 781.6905 - fn: 764.4841 - accuracy: 0.5140 - precision: 0.6027 - recall: 0.2138 - val_loss: 0.6876 - val_tp: 187.0000 - val_fp: 78.0000 - val_tn: 922.0000 - val_fn: 813.0000 - val_accuracy: 0.5545 - val_precision: 0.7057 - val_recall: 0.1870
    Epoch 6/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.6881 - tp: 612.4127 - fp: 496.2381 - tn: 492.1190 - fn: 430.9762 - accuracy: 0.5367 - precision: 0.5675 - recall: 0.5070 - val_loss: 0.6764 - val_tp: 720.0000 - val_fp: 511.0000 - val_tn: 489.0000 - val_fn: 280.0000 - val_accuracy: 0.6045 - val_precision: 0.5849 - val_recall: 0.7200
    Epoch 7/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.6795 - tp: 645.5317 - fp: 451.2937 - tn: 537.0635 - fn: 397.8571 - accuracy: 0.5748 - precision: 0.5981 - recall: 0.5694 - val_loss: 0.6607 - val_tp: 441.0000 - val_fp: 182.0000 - val_tn: 818.0000 - val_fn: 559.0000 - val_accuracy: 0.6295 - val_precision: 0.7079 - val_recall: 0.4410
    Epoch 8/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.6733 - tp: 628.2937 - fp: 409.7222 - tn: 578.6349 - fn: 415.0952 - accuracy: 0.5741 - precision: 0.6065 - recall: 0.5224 - val_loss: 0.6475 - val_tp: 355.0000 - val_fp: 126.0000 - val_tn: 874.0000 - val_fn: 645.0000 - val_accuracy: 0.6145 - val_precision: 0.7380 - val_recall: 0.3550
    Epoch 9/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.6530 - tp: 687.2857 - fp: 390.5238 - tn: 597.8333 - fn: 356.1032 - accuracy: 0.6159 - precision: 0.6453 - recall: 0.5946 - val_loss: 0.6333 - val_tp: 403.0000 - val_fp: 147.0000 - val_tn: 853.0000 - val_fn: 597.0000 - val_accuracy: 0.6280 - val_precision: 0.7327 - val_recall: 0.4030
    Epoch 10/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.6459 - tp: 679.1984 - fp: 332.1667 - tn: 656.1905 - fn: 364.1905 - accuracy: 0.6387 - precision: 0.6853 - recall: 0.5829 - val_loss: 0.6202 - val_tp: 467.0000 - val_fp: 166.0000 - val_tn: 834.0000 - val_fn: 533.0000 - val_accuracy: 0.6505 - val_precision: 0.7378 - val_recall: 0.4670
    Epoch 11/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.6291 - tp: 673.3492 - fp: 321.7063 - tn: 666.6508 - fn: 370.0397 - accuracy: 0.6422 - precision: 0.6821 - recall: 0.5889 - val_loss: 0.6096 - val_tp: 500.0000 - val_fp: 183.0000 - val_tn: 817.0000 - val_fn: 500.0000 - val_accuracy: 0.6585 - val_precision: 0.7321 - val_recall: 0.5000
    Epoch 12/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.6091 - tp: 681.3175 - fp: 303.0238 - tn: 685.3333 - fn: 362.0714 - accuracy: 0.6601 - precision: 0.7035 - recall: 0.6072 - val_loss: 0.6087 - val_tp: 468.0000 - val_fp: 149.0000 - val_tn: 851.0000 - val_fn: 532.0000 - val_accuracy: 0.6595 - val_precision: 0.7585 - val_recall: 0.4680
    Epoch 13/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.6052 - tp: 678.1905 - fp: 273.4603 - tn: 714.8968 - fn: 365.1984 - accuracy: 0.6708 - precision: 0.7218 - recall: 0.6043 - val_loss: 0.5952 - val_tp: 548.0000 - val_fp: 198.0000 - val_tn: 802.0000 - val_fn: 452.0000 - val_accuracy: 0.6750 - val_precision: 0.7346 - val_recall: 0.5480
    Epoch 14/50
    125/125 [==============================] - 11s 84ms/step - loss: 0.5889 - tp: 708.2381 - fp: 283.6508 - tn: 704.7063 - fn: 335.1508 - accuracy: 0.6875 - precision: 0.7250 - recall: 0.6478 - val_loss: 0.5942 - val_tp: 497.0000 - val_fp: 161.0000 - val_tn: 839.0000 - val_fn: 503.0000 - val_accuracy: 0.6680 - val_precision: 0.7553 - val_recall: 0.4970
    Epoch 15/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.5748 - tp: 709.9127 - fp: 258.6905 - tn: 729.6667 - fn: 333.4762 - accuracy: 0.6938 - precision: 0.7339 - recall: 0.6451 - val_loss: 0.6080 - val_tp: 397.0000 - val_fp: 80.0000 - val_tn: 920.0000 - val_fn: 603.0000 - val_accuracy: 0.6585 - val_precision: 0.8323 - val_recall: 0.3970
    Epoch 16/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.5694 - tp: 721.7302 - fp: 253.4921 - tn: 734.8651 - fn: 321.6587 - accuracy: 0.6969 - precision: 0.7415 - recall: 0.6429 - val_loss: 0.5650 - val_tp: 617.0000 - val_fp: 202.0000 - val_tn: 798.0000 - val_fn: 383.0000 - val_accuracy: 0.7075 - val_precision: 0.7534 - val_recall: 0.6170
    Epoch 17/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.5692 - tp: 739.1190 - fp: 252.8175 - tn: 735.5397 - fn: 304.2698 - accuracy: 0.7137 - precision: 0.7493 - recall: 0.6768 - val_loss: 0.5615 - val_tp: 599.0000 - val_fp: 188.0000 - val_tn: 812.0000 - val_fn: 401.0000 - val_accuracy: 0.7055 - val_precision: 0.7611 - val_recall: 0.5990
    Epoch 18/50
    125/125 [==============================] - 11s 90ms/step - loss: 0.5417 - tp: 744.1746 - fp: 240.3651 - tn: 747.9921 - fn: 299.2143 - accuracy: 0.7242 - precision: 0.7615 - recall: 0.6850 - val_loss: 0.5515 - val_tp: 634.0000 - val_fp: 202.0000 - val_tn: 798.0000 - val_fn: 366.0000 - val_accuracy: 0.7160 - val_precision: 0.7584 - val_recall: 0.6340
    Epoch 19/50
    125/125 [==============================] - 12s 92ms/step - loss: 0.5422 - tp: 749.5397 - fp: 236.6746 - tn: 751.6825 - fn: 293.8492 - accuracy: 0.7261 - precision: 0.7642 - recall: 0.6862 - val_loss: 0.5460 - val_tp: 671.0000 - val_fp: 243.0000 - val_tn: 757.0000 - val_fn: 329.0000 - val_accuracy: 0.7140 - val_precision: 0.7341 - val_recall: 0.6710
    Epoch 20/50
    125/125 [==============================] - 11s 88ms/step - loss: 0.5248 - tp: 758.0079 - fp: 236.3889 - tn: 751.9683 - fn: 285.3810 - accuracy: 0.7319 - precision: 0.7658 - recall: 0.6989 - val_loss: 0.5395 - val_tp: 643.0000 - val_fp: 188.0000 - val_tn: 812.0000 - val_fn: 357.0000 - val_accuracy: 0.7275 - val_precision: 0.7738 - val_recall: 0.6430
    Epoch 21/50
    125/125 [==============================] - 11s 88ms/step - loss: 0.5178 - tp: 755.0714 - fp: 228.2302 - tn: 760.1270 - fn: 288.3175 - accuracy: 0.7330 - precision: 0.7711 - recall: 0.6925 - val_loss: 0.5357 - val_tp: 679.0000 - val_fp: 225.0000 - val_tn: 775.0000 - val_fn: 321.0000 - val_accuracy: 0.7270 - val_precision: 0.7511 - val_recall: 0.6790
    Epoch 22/50
    125/125 [==============================] - 11s 90ms/step - loss: 0.5047 - tp: 760.7460 - fp: 208.1667 - tn: 780.1905 - fn: 282.6429 - accuracy: 0.7501 - precision: 0.7948 - recall: 0.7020 - val_loss: 0.5339 - val_tp: 656.0000 - val_fp: 205.0000 - val_tn: 795.0000 - val_fn: 344.0000 - val_accuracy: 0.7255 - val_precision: 0.7619 - val_recall: 0.6560
    Epoch 23/50
    125/125 [==============================] - 11s 88ms/step - loss: 0.4945 - tp: 784.7381 - fp: 209.8889 - tn: 778.4683 - fn: 258.6508 - accuracy: 0.7563 - precision: 0.7863 - recall: 0.7290 - val_loss: 0.5251 - val_tp: 662.0000 - val_fp: 184.0000 - val_tn: 816.0000 - val_fn: 338.0000 - val_accuracy: 0.7390 - val_precision: 0.7825 - val_recall: 0.6620
    Epoch 24/50
    125/125 [==============================] - 11s 89ms/step - loss: 0.4808 - tp: 795.3016 - fp: 200.4444 - tn: 787.9127 - fn: 248.0873 - accuracy: 0.7721 - precision: 0.8054 - recall: 0.7417 - val_loss: 0.5215 - val_tp: 690.0000 - val_fp: 207.0000 - val_tn: 793.0000 - val_fn: 310.0000 - val_accuracy: 0.7415 - val_precision: 0.7692 - val_recall: 0.6900
    Epoch 25/50
    125/125 [==============================] - 11s 86ms/step - loss: 0.4676 - tp: 792.0000 - fp: 186.1429 - tn: 802.2143 - fn: 251.3889 - accuracy: 0.7794 - precision: 0.8162 - recall: 0.7438 - val_loss: 0.5185 - val_tp: 709.0000 - val_fp: 215.0000 - val_tn: 785.0000 - val_fn: 291.0000 - val_accuracy: 0.7470 - val_precision: 0.7673 - val_recall: 0.7090
    Epoch 26/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.4490 - tp: 813.4762 - fp: 180.1429 - tn: 808.2143 - fn: 229.9127 - accuracy: 0.7908 - precision: 0.8191 - recall: 0.7664 - val_loss: 0.5199 - val_tp: 696.0000 - val_fp: 197.0000 - val_tn: 803.0000 - val_fn: 304.0000 - val_accuracy: 0.7495 - val_precision: 0.7794 - val_recall: 0.6960
    Epoch 27/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.4444 - tp: 810.4444 - fp: 178.7063 - tn: 809.6508 - fn: 232.9444 - accuracy: 0.7863 - precision: 0.8157 - recall: 0.7603 - val_loss: 0.5159 - val_tp: 667.0000 - val_fp: 172.0000 - val_tn: 828.0000 - val_fn: 333.0000 - val_accuracy: 0.7475 - val_precision: 0.7950 - val_recall: 0.6670
    Epoch 28/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.4243 - tp: 816.8889 - fp: 165.4841 - tn: 822.8730 - fn: 226.5000 - accuracy: 0.7993 - precision: 0.8318 - recall: 0.7688 - val_loss: 0.5318 - val_tp: 658.0000 - val_fp: 173.0000 - val_tn: 827.0000 - val_fn: 342.0000 - val_accuracy: 0.7425 - val_precision: 0.7918 - val_recall: 0.6580
    Epoch 29/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.4217 - tp: 802.0635 - fp: 154.4444 - tn: 833.9127 - fn: 241.3254 - accuracy: 0.7948 - precision: 0.8411 - recall: 0.7460 - val_loss: 0.5221 - val_tp: 666.0000 - val_fp: 171.0000 - val_tn: 829.0000 - val_fn: 334.0000 - val_accuracy: 0.7475 - val_precision: 0.7957 - val_recall: 0.6660
    Epoch 30/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.3876 - tp: 828.8254 - fp: 134.7857 - tn: 853.5714 - fn: 214.5635 - accuracy: 0.8262 - precision: 0.8605 - recall: 0.7931 - val_loss: 0.5234 - val_tp: 723.0000 - val_fp: 205.0000 - val_tn: 795.0000 - val_fn: 277.0000 - val_accuracy: 0.7590 - val_precision: 0.7791 - val_recall: 0.7230
    Epoch 31/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.3846 - tp: 844.2222 - fp: 149.6667 - tn: 838.6905 - fn: 199.1667 - accuracy: 0.8217 - precision: 0.8453 - recall: 0.8039 - val_loss: 0.5246 - val_tp: 729.0000 - val_fp: 209.0000 - val_tn: 791.0000 - val_fn: 271.0000 - val_accuracy: 0.7600 - val_precision: 0.7772 - val_recall: 0.7290
    Epoch 32/50
    125/125 [==============================] - 11s 85ms/step - loss: 0.3848 - tp: 851.8730 - fp: 151.8730 - tn: 836.4841 - fn: 191.5159 - accuracy: 0.8238 - precision: 0.8402 - recall: 0.8157 - val_loss: 0.5237 - val_tp: 734.0000 - val_fp: 209.0000 - val_tn: 791.0000 - val_fn: 266.0000 - val_accuracy: 0.7625 - val_precision: 0.7784 - val_recall: 0.7340
    


```python
model_representative = get_model()
model_representative.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=all_metrics)
history_representative = model_representative.fit(train_ds, epochs=NUM_EPOCHS, validation_data=val_ds_representative, callbacks=callbacks)
```

    Epoch 1/50
    125/125 [==============================] - 11s 79ms/step - loss: 0.9498 - tp: 1278.5317 - fp: 730.7222 - tn: 1257.6349 - fn: 764.8571 - accuracy: 0.6405 - precision: 0.6489 - recall: 0.6355 - val_loss: 0.6491 - val_tp: 25.0000 - val_fp: 268.0000 - val_tn: 732.0000 - val_fn: 75.0000 - val_accuracy: 0.6882 - val_precision: 0.0853 - val_recall: 0.2500
    Epoch 2/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.7136 - tp: 514.0476 - fp: 422.7302 - tn: 565.6270 - fn: 529.3413 - accuracy: 0.5224 - precision: 0.5467 - recall: 0.4548 - val_loss: 0.6638 - val_tp: 30.0000 - val_fp: 287.0000 - val_tn: 713.0000 - val_fn: 70.0000 - val_accuracy: 0.6755 - val_precision: 0.0946 - val_recall: 0.3000
    Epoch 3/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.6958 - tp: 497.8413 - fp: 392.8651 - tn: 595.4921 - fn: 545.5476 - accuracy: 0.5415 - precision: 0.5718 - recall: 0.4650 - val_loss: 0.6707 - val_tp: 33.0000 - val_fp: 306.0000 - val_tn: 694.0000 - val_fn: 67.0000 - val_accuracy: 0.6609 - val_precision: 0.0973 - val_recall: 0.3300
    Epoch 4/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.6838 - tp: 595.3492 - fp: 459.5238 - tn: 528.8333 - fn: 448.0397 - accuracy: 0.5557 - precision: 0.5706 - recall: 0.5757 - val_loss: 0.6691 - val_tp: 31.0000 - val_fp: 292.0000 - val_tn: 708.0000 - val_fn: 69.0000 - val_accuracy: 0.6718 - val_precision: 0.0960 - val_recall: 0.3100
    Epoch 5/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.6883 - tp: 554.2063 - fp: 438.2937 - tn: 550.0635 - fn: 489.1825 - accuracy: 0.5457 - precision: 0.5657 - recall: 0.5324 - val_loss: 0.6662 - val_tp: 30.0000 - val_fp: 271.0000 - val_tn: 729.0000 - val_fn: 70.0000 - val_accuracy: 0.6900 - val_precision: 0.0997 - val_recall: 0.3000
    Epoch 6/50
    125/125 [==============================] - 9s 71ms/step - loss: 0.6838 - tp: 565.7381 - fp: 421.4365 - tn: 566.9206 - fn: 477.6508 - accuracy: 0.5603 - precision: 0.5794 - recall: 0.5496 - val_loss: 0.6660 - val_tp: 30.0000 - val_fp: 271.0000 - val_tn: 729.0000 - val_fn: 70.0000 - val_accuracy: 0.6900 - val_precision: 0.0997 - val_recall: 0.3000
    

#### Experiment #2 Evaluation


```python
plot_loss(history_balanced, "Balanced Validation")
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_32_0.png)
    



```python
plot_loss(history_representative, "Representative Validation")
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_33_0.png)
    



```python
eval_balanced = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
eval_representative = model_representative.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_balanced.metrics_names, eval_balanced):
    print(name, ': ', value)
```

    loss :  0.4689180552959442
    tp :  66.0
    fp :  198.0
    tn :  802.0
    fn :  34.0
    accuracy :  0.7890909314155579
    precision :  0.25
    recall :  0.6600000262260437
    


```python
for name, value in zip(model_representative.metrics_names, eval_representative):
    print(name, ': ', value)
```

    loss :  0.6475046873092651
    tp :  29.0
    fp :  270.0
    tn :  730.0
    fn :  71.0
    accuracy :  0.6899999976158142
    precision :  0.09698996692895889
    recall :  0.28999999165534973
    


```python
balanced_preds = model_balanced.predict(test_ds)
representative_preds = model_representative.predict(test_ds)
true_labels = tf.concat([y for x, y in test_ds], axis=0)
```


```python
plot_cm(true_labels, balanced_preds)
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_38_0.png)
    



```python
plot_cm(true_labels, representative_preds)
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_39_0.png)
    


## Experiment #3


```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00008)
```


```python
model_balanced = get_model()
model_balanced.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=all_metrics)
history_balanced = model_balanced.fit(train_ds, epochs=NUM_EPOCHS, validation_data=val_ds_balanced, callbacks=callbacks)
```

    Epoch 1/50
    125/125 [==============================] - 13s 95ms/step - loss: 0.8509 - tp: 662.9286 - fp: 844.5238 - tn: 1143.8333 - fn: 480.4603 - accuracy: 0.5870 - precision: 0.4059 - recall: 0.5491 - val_loss: 0.6875 - val_tp: 247.0000 - val_fp: 166.0000 - val_tn: 834.0000 - val_fn: 753.0000 - val_accuracy: 0.5405 - val_precision: 0.5981 - val_recall: 0.2470
    Epoch 2/50
    125/125 [==============================] - 11s 88ms/step - loss: 0.6978 - tp: 237.4762 - fp: 197.1587 - tn: 791.1984 - fn: 805.9127 - accuracy: 0.5007 - precision: 0.5621 - recall: 0.1782 - val_loss: 0.6897 - val_tp: 60.0000 - val_fp: 61.0000 - val_tn: 939.0000 - val_fn: 940.0000 - val_accuracy: 0.4995 - val_precision: 0.4959 - val_recall: 0.0600
    Epoch 3/50
    125/125 [==============================] - 11s 86ms/step - loss: 0.6885 - tp: 623.2143 - fp: 510.6111 - tn: 477.7460 - fn: 420.1746 - accuracy: 0.5392 - precision: 0.5588 - recall: 0.5541 - val_loss: 0.6856 - val_tp: 419.0000 - val_fp: 272.0000 - val_tn: 728.0000 - val_fn: 581.0000 - val_accuracy: 0.5735 - val_precision: 0.6064 - val_recall: 0.4190
    Epoch 4/50
    125/125 [==============================] - 11s 88ms/step - loss: 0.6889 - tp: 528.9365 - fp: 431.7460 - tn: 556.6111 - fn: 514.4524 - accuracy: 0.5284 - precision: 0.5572 - recall: 0.4590 - val_loss: 0.6937 - val_tp: 15.0000 - val_fp: 15.0000 - val_tn: 985.0000 - val_fn: 985.0000 - val_accuracy: 0.5000 - val_precision: 0.5000 - val_recall: 0.01504153 - fp: 402.1949 - tn: 522.5169 - fn: 487.8729 - accuracy: 0.5271 - precision: 0.5574 - recall
    Epoch 5/50
    125/125 [==============================] - 11s 89ms/step - loss: 0.6926 - tp: 543.7698 - fp: 409.0794 - tn: 579.2778 - fn: 499.6190 - accuracy: 0.5463 - precision: 0.5759 - recall: 0.4788 - val_loss: 0.6834 - val_tp: 442.0000 - val_fp: 283.0000 - val_tn: 717.0000 - val_fn: 558.0000 - val_accuracy: 0.5795 - val_precision: 0.6097 - val_recall: 0.4420
    Epoch 6/50
    125/125 [==============================] - 11s 88ms/step - loss: 0.6842 - tp: 643.3333 - fp: 531.6508 - tn: 456.7063 - fn: 400.0556 - accuracy: 0.5368 - precision: 0.5511 - recall: 0.5830 - val_loss: 0.6820 - val_tp: 417.0000 - val_fp: 272.0000 - val_tn: 728.0000 - val_fn: 583.0000 - val_accuracy: 0.5725 - val_precision: 0.6052 - val_recall: 0.4170
    Epoch 7/50
    125/125 [==============================] - 11s 88ms/step - loss: 0.6837 - tp: 695.0714 - fp: 552.7619 - tn: 435.5952 - fn: 348.3175 - accuracy: 0.5461 - precision: 0.5546 - recall: 0.6228 - val_loss: 0.6822 - val_tp: 128.0000 - val_fp: 91.0000 - val_tn: 909.0000 - val_fn: 872.0000 - val_accuracy: 0.5185 - val_precision: 0.5845 - val_recall: 0.1280 tn: 393.7699 - fn: 320.6637 - accuracy: 0.5439 - precision: 0.5542 - r
    Epoch 8/50
    125/125 [==============================] - 11s 86ms/step - loss: 0.6945 - tp: 691.0952 - fp: 573.4206 - tn: 414.9365 - fn: 352.2937 - accuracy: 0.5397 - precision: 0.5514 - recall: 0.6172 - val_loss: 0.6861 - val_tp: 234.0000 - val_fp: 179.0000 - val_tn: 821.0000 - val_fn: 766.0000 - val_accuracy: 0.5275 - val_precision: 0.5666 - val_recall: 0.2340
    Epoch 9/50
    125/125 [==============================] - 11s 87ms/step - loss: 0.6849 - tp: 623.8254 - fp: 457.8016 - tn: 530.5556 - fn: 419.5635 - accuracy: 0.5600 - precision: 0.5787 - recall: 0.5578 - val_loss: 0.6791 - val_tp: 704.0000 - val_fp: 480.0000 - val_tn: 520.0000 - val_fn: 296.0000 - val_accuracy: 0.6120 - val_precision: 0.5946 - val_recall: 0.7040
    Epoch 10/50
    125/125 [==============================] - 11s 86ms/step - loss: 0.6703 - tp: 677.1587 - fp: 470.3571 - tn: 518.0000 - fn: 366.2302 - accuracy: 0.5877 - precision: 0.5931 - recall: 0.6509 - val_loss: 0.6775 - val_tp: 630.0000 - val_fp: 421.0000 - val_tn: 579.0000 - val_fn: 370.0000 - val_accuracy: 0.6045 - val_precision: 0.5994 - val_recall: 0.6300
    Epoch 11/50
    125/125 [==============================] - 11s 87ms/step - loss: 0.6738 - tp: 639.3889 - fp: 452.3651 - tn: 535.9921 - fn: 404.0000 - accuracy: 0.5804 - precision: 0.5935 - recall: 0.6060 - val_loss: 0.6769 - val_tp: 671.0000 - val_fp: 446.0000 - val_tn: 554.0000 - val_fn: 329.0000 - val_accuracy: 0.6125 - val_precision: 0.6007 - val_recall: 0.6710
    Epoch 12/50
    125/125 [==============================] - 11s 89ms/step - loss: 0.6704 - tp: 667.6746 - fp: 457.1508 - tn: 531.2063 - fn: 375.7143 - accuracy: 0.5941 - precision: 0.6002 - recall: 0.6466 - val_loss: 0.6755 - val_tp: 686.0000 - val_fp: 453.0000 - val_tn: 547.0000 - val_fn: 314.0000 - val_accuracy: 0.6165 - val_precision: 0.6023 - val_recall: 0.6860
    Epoch 13/50
    125/125 [==============================] - 11s 88ms/step - loss: 0.6720 - tp: 658.4444 - fp: 430.8492 - tn: 557.5079 - fn: 384.9444 - accuracy: 0.6012 - precision: 0.6132 - recall: 0.6257 - val_loss: 0.6746 - val_tp: 662.0000 - val_fp: 439.0000 - val_tn: 561.0000 - val_fn: 338.0000 - val_accuracy: 0.6115 - val_precision: 0.6013 - val_recall: 0.6620
    Epoch 14/50
    125/125 [==============================] - 11s 88ms/step - loss: 0.6655 - tp: 676.1111 - fp: 468.2698 - tn: 520.0873 - fn: 367.2778 - accuracy: 0.5841 - precision: 0.5925 - recall: 0.6341 - val_loss: 0.6740 - val_tp: 666.0000 - val_fp: 445.0000 - val_tn: 555.0000 - val_fn: 334.0000 - val_accuracy: 0.6105 - val_precision: 0.5995 - val_recall: 0.6660
    Epoch 15/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.6717 - tp: 671.9286 - fp: 466.9683 - tn: 521.3889 - fn: 371.4603 - accuracy: 0.5856 - precision: 0.5947 - recall: 0.6321 - val_loss: 0.6743 - val_tp: 662.0000 - val_fp: 444.0000 - val_tn: 556.0000 - val_fn: 338.0000 - val_accuracy: 0.6090 - val_precision: 0.5986 - val_recall: 0.6620
    Epoch 16/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.6709 - tp: 626.7619 - fp: 439.8889 - tn: 548.4683 - fn: 416.6270 - accuracy: 0.5799 - precision: 0.5943 - recall: 0.5984 - val_loss: 0.6746 - val_tp: 684.0000 - val_fp: 461.0000 - val_tn: 539.0000 - val_fn: 316.0000 - val_accuracy: 0.6115 - val_precision: 0.5974 - val_recall: 0.6840
    Epoch 17/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.6665 - tp: 664.2460 - fp: 455.6270 - tn: 532.7302 - fn: 379.1429 - accuracy: 0.5929 - precision: 0.6000 - recall: 0.6418 - val_loss: 0.6744 - val_tp: 679.0000 - val_fp: 459.0000 - val_tn: 541.0000 - val_fn: 321.0000 - val_accuracy: 0.6100 - val_precision: 0.5967 - val_recall: 0.6790
    Epoch 18/50
    125/125 [==============================] - 10s 83ms/step - loss: 0.6684 - tp: 657.6111 - fp: 436.9048 - tn: 551.4524 - fn: 385.7778 - accuracy: 0.5998 - precision: 0.6102 - recall: 0.6307 - val_loss: 0.6742 - val_tp: 680.0000 - val_fp: 458.0000 - val_tn: 542.0000 - val_fn: 320.0000 - val_accuracy: 0.6110 - val_precision: 0.5975 - val_recall: 0.6800
    Epoch 19/50
    125/125 [==============================] - 10s 84ms/step - loss: 0.6673 - tp: 662.5794 - fp: 435.3492 - tn: 553.0079 - fn: 380.8095 - accuracy: 0.6005 - precision: 0.6088 - recall: 0.6397 - val_loss: 0.6742 - val_tp: 679.0000 - val_fp: 457.0000 - val_tn: 543.0000 - val_fn: 321.0000 - val_accuracy: 0.6110 - val_precision: 0.5977 - val_recall: 0.6790
    


```python
model_representative = get_model()
model_representative.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=all_metrics)
history_representative = model_representative.fit(train_ds, epochs=NUM_EPOCHS, validation_data=val_ds_representative, callbacks=callbacks)
```

    Epoch 1/50
    125/125 [==============================] - 11s 77ms/step - loss: 1.0594 - tp: 1530.9365 - fp: 1279.2063 - tn: 709.1508 - fn: 512.4524 - accuracy: 0.5617 - precision: 0.5508 - recall: 0.7456 - val_loss: 1.0676 - val_tp: 99.0000 - val_fp: 987.0000 - val_tn: 13.0000 - val_fn: 1.0000 - val_accuracy: 0.1018 - val_precision: 0.0912 - val_recall: 0.9900
    Epoch 2/50
    125/125 [==============================] - 9s 72ms/step - loss: 0.9010 - tp: 743.1270 - fp: 681.1905 - tn: 307.1667 - fn: 300.2619 - accuracy: 0.5232 - precision: 0.5287 - recall: 0.7249 - val_loss: 0.9581 - val_tp: 97.0000 - val_fp: 961.0000 - val_tn: 39.0000 - val_fn: 3.0000 - val_accuracy: 0.1236 - val_precision: 0.0917 - val_recall: 0.9700
    Epoch 3/50
    125/125 [==============================] - 9s 73ms/step - loss: 0.8892 - tp: 607.5873 - fp: 599.3968 - tn: 388.9603 - fn: 435.8016 - accuracy: 0.4900 - precision: 0.5073 - recall: 0.5771 - val_loss: 0.9175 - val_tp: 95.0000 - val_fp: 939.0000 - val_tn: 61.0000 - val_fn: 5.0000 - val_accuracy: 0.1418 - val_precision: 0.0919 - val_recall: 0.9500
    Epoch 4/50
    125/125 [==============================] - 9s 73ms/step - loss: 0.8753 - tp: 637.8810 - fp: 582.1349 - tn: 406.2222 - fn: 405.5079 - accuracy: 0.5116 - precision: 0.5245 - recall: 0.6139 - val_loss: 0.8894 - val_tp: 95.0000 - val_fp: 929.0000 - val_tn: 71.0000 - val_fn: 5.0000 - val_accuracy: 0.1509 - val_precision: 0.0928 - val_recall: 0.9500
    Epoch 5/50
    125/125 [==============================] - 9s 73ms/step - loss: 0.8858 - tp: 583.7937 - fp: 562.2381 - tn: 426.1190 - fn: 459.5952 - accuracy: 0.4900 - precision: 0.5072 - recall: 0.5596 - val_loss: 0.8765 - val_tp: 94.0000 - val_fp: 918.0000 - val_tn: 82.0000 - val_fn: 6.0000 - val_accuracy: 0.1600 - val_precision: 0.0929 - val_recall: 0.9400
    Epoch 6/50
    125/125 [==============================] - 9s 73ms/step - loss: 0.8985 - tp: 579.7302 - fp: 553.1667 - tn: 435.1905 - fn: 463.6587 - accuracy: 0.4971 - precision: 0.5135 - recall: 0.5578 - val_loss: 0.8679 - val_tp: 93.0000 - val_fp: 910.0000 - val_tn: 90.0000 - val_fn: 7.0000 - val_accuracy: 0.1664 - val_precision: 0.0927 - val_recall: 0.9300
    Epoch 7/50
    125/125 [==============================] - 9s 72ms/step - loss: 0.8698 - tp: 563.2460 - fp: 520.3254 - tn: 468.0317 - fn: 480.1429 - accuracy: 0.5047 - precision: 0.5209 - recall: 0.5462 - val_loss: 0.8641 - val_tp: 93.0000 - val_fp: 908.0000 - val_tn: 92.0000 - val_fn: 7.0000 - val_accuracy: 0.1682 - val_precision: 0.0929 - val_recall: 0.9300
    Epoch 8/50
    125/125 [==============================] - 9s 72ms/step - loss: 0.8443 - tp: 586.7143 - fp: 524.5714 - tn: 463.7857 - fn: 456.6746 - accuracy: 0.5255 - precision: 0.5390 - recall: 0.5749 - val_loss: 0.8589 - val_tp: 92.0000 - val_fp: 905.0000 - val_tn: 95.0000 - val_fn: 8.0000 - val_accuracy: 0.1700 - val_precision: 0.0923 - val_recall: 0.9200
    Epoch 9/50
    125/125 [==============================] - 9s 72ms/step - loss: 0.8527 - tp: 565.3254 - fp: 513.4286 - tn: 474.9286 - fn: 478.0635 - accuracy: 0.5077 - precision: 0.5246 - recall: 0.5321 - val_loss: 0.8560 - val_tp: 92.0000 - val_fp: 901.0000 - val_tn: 99.0000 - val_fn: 8.0000 - val_accuracy: 0.1736 - val_precision: 0.0926 - val_recall: 0.9200
    Epoch 10/50
    125/125 [==============================] - 9s 72ms/step - loss: 0.8595 - tp: 563.7857 - fp: 525.2222 - tn: 463.1349 - fn: 479.6032 - accuracy: 0.5032 - precision: 0.5195 - recall: 0.5443 - val_loss: 0.8515 - val_tp: 92.0000 - val_fp: 899.0000 - val_tn: 101.0000 - val_fn: 8.0000 - val_accuracy: 0.1755 - val_precision: 0.0928 - val_recall: 0.9200
    Epoch 11/50
    125/125 [==============================] - 9s 72ms/step - loss: 0.8673 - tp: 546.1349 - fp: 517.9048 - tn: 470.4524 - fn: 497.2540 - accuracy: 0.5009 - precision: 0.5181 - recall: 0.5273 - val_loss: 0.8487 - val_tp: 92.0000 - val_fp: 899.0000 - val_tn: 101.0000 - val_fn: 8.0000 - val_accuracy: 0.1755 - val_precision: 0.0928 - val_recall: 0.9200
    Epoch 12/50
    125/125 [==============================] - 9s 72ms/step - loss: 0.8546 - tp: 558.7222 - fp: 529.8571 - tn: 458.5000 - fn: 484.6667 - accuracy: 0.4994 - precision: 0.5161 - recall: 0.5360 - val_loss: 0.8464 - val_tp: 91.0000 - val_fp: 897.0000 - val_tn: 103.0000 - val_fn: 9.0000 - val_accuracy: 0.1764 - val_precision: 0.0921 - val_recall: 0.9100
    Epoch 13/50
    125/125 [==============================] - 9s 72ms/step - loss: 0.8368 - tp: 549.1429 - fp: 515.4524 - tn: 472.9048 - fn: 494.2460 - accuracy: 0.5014 - precision: 0.5183 - recall: 0.5295 - val_loss: 0.8465 - val_tp: 91.0000 - val_fp: 897.0000 - val_tn: 103.0000 - val_fn: 9.0000 - val_accuracy: 0.1764 - val_precision: 0.0921 - val_recall: 0.9100
    Epoch 14/50
    125/125 [==============================] - 9s 73ms/step - loss: 0.8578 - tp: 541.7063 - fp: 511.3492 - tn: 477.0079 - fn: 501.6825 - accuracy: 0.5002 - precision: 0.5179 - recall: 0.5190 - val_loss: 0.8484 - val_tp: 93.0000 - val_fp: 900.0000 - val_tn: 100.0000 - val_fn: 7.0000 - val_accuracy: 0.1755 - val_precision: 0.0937 - val_recall: 0.9300
    Epoch 15/50
    125/125 [==============================] - 9s 75ms/step - loss: 0.8463 - tp: 540.9841 - fp: 508.4921 - tn: 479.8651 - fn: 502.4048 - accuracy: 0.5073 - precision: 0.5248 - recall: 0.5214 - val_loss: 0.8486 - val_tp: 93.0000 - val_fp: 901.0000 - val_tn: 99.0000 - val_fn: 7.0000 - val_accuracy: 0.1745 - val_precision: 0.0936 - val_recall: 0.9300
    Epoch 16/50
    125/125 [==============================] - 9s 73ms/step - loss: 0.8295 - tp: 557.1508 - fp: 502.5238 - tn: 485.8333 - fn: 486.2381 - accuracy: 0.5124 - precision: 0.5288 - recall: 0.5376 - val_loss: 0.8486 - val_tp: 93.0000 - val_fp: 900.0000 - val_tn: 100.0000 - val_fn: 7.0000 - val_accuracy: 0.1755 - val_precision: 0.0937 - val_recall: 0.9300
    Epoch 17/50
    125/125 [==============================] - 9s 73ms/step - loss: 0.8344 - tp: 569.0635 - fp: 523.8968 - tn: 464.4603 - fn: 474.3254 - accuracy: 0.5132 - precision: 0.5291 - recall: 0.5509 - val_loss: 0.8486 - val_tp: 93.0000 - val_fp: 900.0000 - val_tn: 100.0000 - val_fn: 7.0000 - val_accuracy: 0.1755 - val_precision: 0.0937 - val_recall: 0.9300
    

#### Experiment #3 Evaluation


```python
plot_loss(history_balanced, "Balanced Validation")
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_45_0.png)
    



```python
plot_loss(history_representative, "Representative Validation")
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_46_0.png)
    



```python
eval_balanced = model_balanced.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
eval_representative = model_representative.evaluate(test_ds, batch_size=BATCH_SIZE, verbose=0)
```


```python
for name, value in zip(model_balanced.metrics_names, eval_balanced):
    print(name, ': ', value)
```

    loss :  0.6559638381004333
    tp :  59.0
    fp :  452.0
    tn :  548.0
    fn :  41.0
    accuracy :  0.5518181920051575
    precision :  0.1154598817229271
    recall :  0.5899999737739563
    


```python
for name, value in zip(model_representative.metrics_names, eval_representative):
    print(name, ': ', value)
```

    loss :  0.8531972169876099
    tp :  87.0
    fp :  903.0
    tn :  97.0
    fn :  13.0
    accuracy :  0.16727273166179657
    precision :  0.08787878602743149
    recall :  0.8700000047683716
    


```python
balanced_preds = model_balanced.predict(test_ds)
representative_preds = model_representative.predict(test_ds)
true_labels = tf.concat([y for x, y in test_ds], axis=0)
```


```python
plot_cm(true_labels, balanced_preds)
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_51_0.png)
    



```python
plot_cm(true_labels, representative_preds)
```


    
![png](2021-05-03-validation-on-unbalanced-datasets_files/2021-05-03-validation-on-unbalanced-datasets_52_0.png)
    


## Conclusion

We ran it three times and most the time we got better results when using the balanced validation dataset instead of the representative one. The exception was in the third experiment, where our recall was better with the representative dataset. However, the accuracy was much lower in that model, so I don't think it should really be considered better.

Overall, I think this suggests perhaps the validation set should be balanced and not representative if it's being used in callbacks. This is in conflict with most guidance on creating a validation set.

In another post, I'll discuss whether this can be fixed by using a different loss function, such as focal loss.
