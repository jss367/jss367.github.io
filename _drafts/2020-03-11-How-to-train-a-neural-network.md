---
layout: post
title: "How to Train a Neural Network"
---

This is a collection of my thoughts and tips on how to train a neural network.

<b>Table of contents</b>
* TOC
{:toc}


# Starting Small

* Do not jump into it. I know this is the opposite of most tutorials, where they just throw a ResNet50 at it, but that only works once someone has gone over the data and the model and everything is perfect. As in, that never works in "real" life. Once a dataset is clean and the pipeline is perfect, sure, it will work. Or if you are using a curated dataset, but I'm talking about real-world problems.

* Visualize everything possible - in the beginning, after augmentation, right before it goes into the network, what the weights and biases look like inside the network. Everything you can think to visualize, you should.


* Start at the shallow end
** we don't want to add a bunch of complexity at once- you will inevitably, but you'd like to limit it, don't try for a walk-off grand slam. swing the bat in the on-deck circle, test it, make sure you know it's swing


shuffle your training data

# Overfit First

Overfit first. If you can't overfit, your model is not capable of solving the task. Maybe that's because of the model or maybe it's because of the data, but either way you know you need to solve this before moving on. Maybe you need a bigger network or cleaner data or something else.


* have a baseline cnn that you use - set random seed- no data augmentation - i like to turn that on later - AFTER I OVERFIT

* train on a small subset - overfitthen train on a larger subset - overfit less


* Overfit and underfit



** You want to know where you dataset and model's limits are


* you should know what your first loss looks like. a sign of overfitting is when the loss drops significantly from the end of the first epoch to the beginning of the second


* when your model is small, you will underfit on it. increase the model capacity and watch the underfit go away

* as soon as you are able to scale, get a good metric. In general, the simple ones (f1 score, IOU, etc.) work just fine.

* especially if you're working with a lot of models (more common in industry than research), resnet50 is great.


Building a model

# Regularize

Now that you've overfit your data, you need to make it overfit less. Here are some ways to do that:

1. more data
2. data augmentation
3. Generalizable archtecture - more batch norm, densenets
4. regularlization - weight decay, dropout
5. reduce architecture complexity if nothing else works - less layers, less activations

You can always use data augmentation, L1, L2, or dropout to regularlize if you need to


overfit - use a small weight decay, such as 0.01
then you can gradually increase it
weight decay is a regularization technique









Thoughts on training models




I'm pretty convinced that you should use transfer learning whenever you can, and you almost always can.

Any kind of image, I would transfer learn. THe first filters are mostly color separators and Gabor filters (link to distil paper). You're going to need them no matter what. Even for geospatial analytics, I use transfer learning. Use it all the time.


Use some form of normalization. If you have large batch size, use Batch norm. If small, maybe try group norm or instance norm.






You can also contrain your weights directly (model.add(Conv2D(64, kernel_constraint=max_norm(2.)))) but I recommend weight decay (better than L2?)






