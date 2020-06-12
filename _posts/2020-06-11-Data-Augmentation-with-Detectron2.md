---
layout: post
title: "Data Augmentation in Detectron2"
description: "A walkthrough of the different augmentation methods available in detectron2"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/bear_with_fish.jpg"
tags: [Detectron2, Computer Vision]
---

This post is a quick walkthrough of the different data augmentation methods available in [Detectron2](https://github.com/facebookresearch/detectron2) and how to implement them. I'm also going to talk about them and how they relate to geospatial analytics.

* TOC
{:toc}

### Data Augmentation Methods

Detectron2 has a [large list of available data augmentation methods](https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/transforms/transform_gen.py).


#### RandomApply

The is a wrapper around the other augmentation methods so that you can turn a group on or off together with a specified probability. I don't generally use it for my purposes, but it could definitely be helpful in some cases.

#### RandomFlip

You can only flip horizontally or vertically, so for overhead imagery you should include two of these, one for each. You can specify the probability.
#### Resize and ResizeShortestEdge
There are two resize options. If your images vary in shape and you don't want to distort them, `ResizeShortestEdge` is the one to use because it won't change the image's aspect ratio. It increases the size (preserving the aspect ratio) until the shortest edge size is met, then it checks if the longest edge is larger than the limit. If so, it reduces the image to fit. This means that the shortest side may be shorter than the value provided, which is a little counterintuitive, but on the plus size you can make sure your images aren't too long (or just not provide a maximum value).
One thing I don't like about this is that it upsamples different images differently, depending on their shape
* RandomRotation - 
* RandomCrop - the cropping is supported by default in the configuration files so I would prefer to do it there and leave this alone.

#### RandomExtent
Crops a random "subrect" of the image

#### RandomContrast, RandomBrightness, and RandomSaturation

These are all pretty straightforward. You can provide them ranges of the form `(min, max)` where using 1 would an identity function (no change). These augmentations are very important in cases where different classes might have variation in one of these. If all the pictures of cats were taken in brighter rooms than all the pictures of dogs, then it would be important to use a light of `RandomBrightness` so their brightnesses of the classes overlaps, forcing the model to actually learn the classes and not just some shortcut. Even in the case when the classes are from the same distribution, these are still good data augmentation techniques.


* RandomLighting

What this does is a bit tricky. It uses a [PCA](https://jss367.github.io/Principal-Component-Analysis.html) lighting trick that was used in [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) with great results. Here's the relevant section of the paper:

![AlexNet]({{site.baseurl}}/assets/img/alexnet_pca_lighting.png "AlexNet PCA Lighting")


I'm not sure what a good value for the scale is, but I will update this once I get a good feel for it.

### Implementing

Fortunately, Detectron2 makes implementation very easy. There are a few different ways to do it, but I would start by copying [their `DatasetMapper`](https://github.com/facebookresearch/detectron2/blob/01dab47ecc85434c31bd55460b7c72553fc35a7b/detectron2/data/dataset_mapper.py#L19) and tweaking it. You could subclass it but it's only a simple class with a `__init__` and `__call__` method, so I just copy the whole thing.

From there, you have two options. One is to add the transforms to the config file, and the other is to simply extend the transforms through code. Extending it through code is faster but the config option makes it much easier to save your settings, which is super important.

Either way, you'll need to find in the `DatasetMapper` the line that [builds the transforms](https://github.com/facebookresearch/detectron2/blob/01dab47ecc85434c31bd55460b7c72553fc35a7b/detectron2/data/dataset_mapper.py#L43). It should look something like `self.tfm_gens = utils.build_transform_gen(cfg, is_train)`. If you've added your augmentation methods to the config, you're already done. If not, you just need to add them here:

`self.tfm_gens.extend(augmentations)`

Then you need to pass in the augmentations you like into the `DatasetMapper`. Make sure to pass the augmentation to your trainer (subclass the `DefaultTrainer`) and pass in your mapper.


```python
train_augmentations = [
    T.RandomBrightness(0.5, 2),
    T.RandomContrast(0.5, 2),
    T.RandomSaturation(0.5, 2),
    T.RandomRotation([0, 90]),
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
]
```