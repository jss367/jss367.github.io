---
layout: post
title: "Kangaroos and Wallabies III: Classifying the Data"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/ki_kangs.jpg"
tags: [Computer Vision, Convolutional Neural Networks, Python, TensorFlow, Wildlife]
---

In this notebook, we're going to take our [augmented dataset](https://jss367.github.io/kangaroos-and-wallabies-ii-augmenting-the-data.html) and build a convolutional neural network to classify the images.

This is part three of a three-post series on creating your own dataset and classifying it using transfer learning.
* [Preparing the Data](https://jss367.github.io/kangaroos-and-wallabies-i-preparing-the-data.html)
* [Augmenting the Data](https://jss367.github.io/kangaroos-and-wallabies-ii-augmenting-the-data.html)
* [Classifying the Data](https://jss367.github.io/kangaroos-and-wallabies-iii-classifying-the-data.html)

<b>Table of contents</b>
* TOC
{:toc}

## Introduction

What model should we use? A simple model, such as logistic regression, is able to do fairly well on a simple image classification task like [MNIST](http://yann.lecun.com/exdb/mnist/) or even cat vs non-cat, but won't really cut it for this task. Because kangaroos and wallabies are so similar, even a basic neural network doesn't score particularly well on this dataset.

So we're just going to skip those and go straight to the cutting edge of computer vision classification: convolutional neural networks. We're going to take a huge model known as VGG-19 and apply it to our dataset. But we're not just going to take the model, we're going to take the actual [VGG19 ImageNet weights](https://github.com/flyyufelix/cnn_finetune/blob/master/vgg19.py) that have been developed by weeks of training it on the ImageNet dataset. Then we will freeze everything except the last five layers and retrain them with our dataset. This is known as transfer learning and it allows us to benefit from the highly tuned convolutional layers that are so good at object detection while allowing us to tweak the last layers specifically for our problem.


```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from keras import regularizers
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
```

## Preparing the data

We'll continue to use the [class for preparing images](https://jss367.github.io/class-for-preparing-images.html) to help organize the data.


```python
%run "Class_for_Preparing_Images.py"
my_data = GatherData()
```

Per the [original paper by Simonyan and Zisserman](https://arxiv.org/pdf/1409.1556.pdf), the images are supposed to be 224 X 224. We'll make sure we set them to that size.


```python
kangaroo_train_path = 'I:/data/train/kangaroos/'
wallaby_train_path = 'I:/data/train/wallabies/'
kangaroo_test_path = 'I:/data/test/kangaroos/'
wallaby_test_path =  'I:/data/test/wallabies/'
my_data.train_test_sets(kangaroo_train_path,
                        wallaby_train_path, 
                        kangaroo_test_path, 
                        wallaby_test_path, 
                        new_size=224, 
                        verbose=True,
                        standardization = 'rescale',
                        flatten=False)
```

    getting images
    zipping
    shuffling
    standardizing
    


```python
my_data.image_size_
```




    (224, 224)




```python
img_width, img_height = (my_data.image_size_)
```

## Establishing a baseline

Before we get started building a complex model, we should develop a baseline so we can gauge our model. As I said in the [first post](https://jss367.github.io/kangaroos-and-wallabies-i-preparing-the-data.html), there are more images of kangaroos than of wallabies, so we need to take that into consideration. For example, if our model accurately predicted 75% of images, that might sound good. But if we just guessed that every image is of a kangaroo and got 70% correct, then maybe the model is not very good. So what's the baseline? How many would be great from pure guessing? To answer that, we'll use a dummy classifier. Our dummy classifier will find the most common label and predicts that every image will be that label.


```python
X_train = my_data.X_train_
X_test = my_data.X_test_
y_train = my_data.y_train_
y_test = my_data.y_test_
```


```python
clf = DummyClassifier(strategy='most_frequent')
clf.fit(X_train[:, 0, 0], y_train)
```




    DummyClassifier(constant=None, random_state=None, strategy='most_frequent')



Let's see what the accuracy of the dummy model is. This will give us something to compare our final result to.


```python
print("By just selecting the most common label, the model was able to get {:.1%} accurate.".format(accuracy_score(y_test, clf.predict(X_test[:, 0, 0]))))
```

    By just selecting the most common label, the model was able to get 56.0% accurate.
    

## Preparing the model

We have to provide some basic characteristics of the network and how we want to train it.


```python
batch_size = 4 # Find batch size appropriate for GPU
nb_train_samples = my_data.num_train_images_ // batch_size
nb_validation_samples = my_data.num_test_images_ // batch_size
```


```python
train_data_dir = "I:/data/train"
validation_data_dir = "I:/data/validate"
epochs = 50
model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
```

We will freeze all layers except the last five. Then we'll add our own layers at the end. For the final activation function, I'm going to use [softmax](https://en.wikipedia.org/wiki/Softmax_function). I could just use a sigmoid function, but using softmax makes it easier to scale the model to multiple classes, even though they're mathematically equivalent in the case of two classes.

One problem we're sure to run into with such a large and complex neural network is overfitting. This is when the model finds a small number of features that work well in the dataset but might not generalize to all images. For example, say all the images in the training set show the ears of the kangaroos and wallabies really well and the model learns how to distinguish them based on that. Well, maybe the ears are behind a branch in some other images, then how is the model going to decide? We want to model to look at many aspects of the image and use them all to classify it.

There are several different types of regularization that we could use, but we'll discuss just two of them: L1 and L2. They rely on the same concept - penalizing the network for weights that are too large. This forces each individual parameter to be low and therefore prevents the model from relying too much on a single weight or feature. L1 regularization penalizes based on the magnitude of the weights, and L2 penalizes based on the <i>square</i> of the magnitude of the weights. There are good reasons to use one over the other which we won't get into, but in this case, we'll use L2 because it's more common and generally seems to give better performance in most cases.


```python
for layer in model.layers[:5]:
    layer.trainable = False

# Add our own layers at the end 
x = model.output
x = Flatten()(x)
x = Dense(1024,
          activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024,
          activation="relu",
          kernel_regularizer=regularizers.l2(0.01))(x)
predictions = Dense(2, activation="softmax")(x) # Only two outputs in this case
```

Create the model using the inputs from VGG19.


```python
model_final = Model(inputs = model.input, outputs = predictions)
```

Now we compile the model. The compiler automatically determines how to split up the data between the CPU and GPU. 
We have to specify the loss function, as well as an optimizer.


```python
# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
```

We'll use the same augmentation techniques described in the [augmentation notebook](https://jss367.github.io/kangaroos-and-wallabies-ii-augmenting-the-data.html). In this case we won't save the images to disk though.


```python
# Initiate the train and validation generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")
```

    Found 3043 images belonging to 2 classes.
    


```python
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical")
```

    Found 468 images belonging to 2 classes.
    


```python
# Save the model according to the conditions
checkpoint = ModelCheckpoint("vgg19.h5", monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0,
                      patience=4, verbose=1, mode='auto')
```

## Training the model


```python
# Train the model 
model_final.fit_generator(
train_generator,
steps_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = validation_generator,
validation_steps = nb_validation_samples,
callbacks = [checkpoint, early])
```

    Epoch 1/50
    760/760 [==============================] - 255s 336ms/step - loss: 10.7671 - acc: 0.5652 - val_loss: 10.6022 - val_acc: 0.5374
    
    Epoch 00001: val_acc improved from -inf to 0.53740, saving model to vgg19.h5
    Epoch 2/50
    760/760 [==============================] - 253s 333ms/step - loss: 10.3804 - acc: 0.6525 - val_loss: 10.1274 - val_acc: 0.7279
    
    Epoch 00002: val_acc improved from 0.53740 to 0.72785, saving model to vgg19.h5
    Epoch 3/50
    760/760 [==============================] - 253s 333ms/step - loss: 9.9841 - acc: 0.7390 - val_loss: 9.9047 - val_acc: 0.6981
    
    Epoch 00003: val_acc did not improve from 0.72785
    Epoch 4/50
    760/760 [==============================] - 246s 324ms/step - loss: 9.5925 - acc: 0.8187 - val_loss: 9.6218 - val_acc: 0.7426
    
    Epoch 00004: val_acc improved from 0.72785 to 0.74262, saving model to vgg19.h5
    Epoch 5/50
    760/760 [==============================] - 257s 338ms/step - loss: 9.2314 - acc: 0.8670 - val_loss: 9.3943 - val_acc: 0.7394
    
    Epoch 00005: val_acc did not improve from 0.74262
    Epoch 6/50
    760/760 [==============================] - 256s 337ms/step - loss: 8.8870 - acc: 0.9000 - val_loss: 8.9388 - val_acc: 0.7960
    
    Epoch 00006: val_acc improved from 0.74262 to 0.79601, saving model to vgg19.h5
    Epoch 7/50
    760/760 [==============================] - 253s 333ms/step - loss: 8.5896 - acc: 0.9211 - val_loss: 8.7437 - val_acc: 0.8034
    
    Epoch 00007: val_acc improved from 0.79601 to 0.80340, saving model to vgg19.h5
    Epoch 8/50
    760/760 [==============================] - 254s 334ms/step - loss: 8.2752 - acc: 0.9473 - val_loss: 8.4920 - val_acc: 0.8027
    
    Epoch 00008: val_acc did not improve from 0.80340
    Epoch 9/50
    760/760 [==============================] - 255s 336ms/step - loss: 8.0207 - acc: 0.9480 - val_loss: 8.6310 - val_acc: 0.7788
    
    Epoch 00009: val_acc did not improve from 0.80340
    Epoch 10/50
    760/760 [==============================] - 254s 334ms/step - loss: 7.7678 - acc: 0.9536 - val_loss: 8.0634 - val_acc: 0.8091
    
    Epoch 00010: val_acc improved from 0.80340 to 0.80906, saving model to vgg19.h5
    Epoch 11/50
    760/760 [==============================] - 256s 336ms/step - loss: 7.5297 - acc: 0.9614 - val_loss: 8.6194 - val_acc: 0.6405
    
    Epoch 00011: val_acc did not improve from 0.80906
    Epoch 12/50
    141/760 [====>.........................] - ETA: 2:37 - loss: 7.3791 - acc: 0.9574


    ---------------------------------------------------------------------------

    KeyboardInterrupt: 


Note how much higher the training accuracy is than the validation accuracy. That means we're overfitting the training data. We'll go over how to correct for that in a future notebook.

## Reloading the weights

The good thing about saving the weights as you train is that you can always interrupt the kernel if you need to. Then when you want to resume you can load the saved weights like so.


```python
model_final = load_model('vgg19.h5')
```


```python
# Do another 10 epochs
epochs = 10
```


```python
# Resume training the model 
model_final.fit_generator(
train_generator,
steps_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = validation_generator,
validation_steps = nb_validation_samples,
callbacks = [checkpoint, early])
```

## Testing the accuracy

Note that we're only saving the model when it improves the validation set. Since we keep checking the accuracy on our validation set, we could actually be overfitting the validation set as well. That's why we reserved a test set that, so far, we haven't even looked at. We'll use that to compute the model's accuracy. Accuracy isn't the most comprehensive way to measure model quality, especially in the case of multiple classes, but it's quick and simple, so we'll use it.


```python
prediction_probabilities = model_final.predict(X_test, batch_size=4)
```


```python
predictions = []
for i in range(len(prediction_probabilities)):
    predictions.append(np.argmax(prediction_probabilities[i]))
```


```python
print("Our final accuracy is {:.2%}".format(1 - sum(np.abs(np.asarray(predictions) - y_test)) / len(y_test)))
```

    Our final accuracy is 85.09%
    

If we really were going to deploy this model and didn't need to know the accuracy precisely, we could retrain it on all the data. Then we wouldn't know exactly how good the model is, except that we could expect it to be better than the model trained with only part of the data.

And that's it. We've taken our own homemade dataset and used transfer learning to classify it better than 85%. Not bad for such a difficult dataset.
