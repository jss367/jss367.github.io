---
layout: post
title: "Where to Find Deep Learning Code Implementations"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/trees.jpg"
tags: [Python, Computer Vision, Deep Learning, Keras, TensorFlow, PyTorch]
---

There are a lot of great resources on the web that contain implementations of deep learning architectures, but they can be a little hard to find. This post aims to highlight and categorize some of the best resources that I have found. Most repositories have either Tensorflow or PyTorch implementations, so this post is divided by frameworks.

<b>Table of contents</b>
* TOC
{:toc}

## Framework-agnostic Resources

### Papers with Code

Although *most* places have code for a specific framework, one of the best places to find code has implementations for every framework. It's called [Papers with Code](https://paperswithcode.com/). This is absolutely the best way to find implementations in places that you wouldn't normally look.  For example, if you go to the [page for RetinaNet](https://paperswithcode.com/paper/focal-loss-for-dense-object-detection), it shows the excellent [fizyr implementation of RetinaNet](https://github.com/fizyr/keras-retinanet), which I wouldn't have found otherwise. It ranks all the implementations by GitHub stars, so it's easy to find some good ones. As well as PyTorch and Tensorflow, it also often has implementations for [MXNet](https://mxnet.apache.org/).

### Model Zoo

There is also a [Model Zoo website](https://modelzoo.co/) that contains implementations of every type from with every framework. There's a lot of great variety here.

## Framework-specific Resources

### Tensorflow

The first place to go to find Tensorflow implementations is probably the [Tensorflow models repo](https://github.com/tensorflow/models). There's tons of hidden stuff inside this repo. For starters, I would direct people to the `research` folder. There are a lot of implementations there, and, within that, there's another set of models in the [slim directory](https://github.com/tensorflow/models/tree/master/research/slim). Also inside the `research` directory is a [section on keras models](https://github.com/tensorflow/models/tree/master/research/object_detection/models/keras_models) that has implementations of many well-known models, such as inception, mobilenet, and resnet.

The author of Keras, [Francois Chollet](https://fchollet.com/), also has his own [repo with some very easy to understand examples](https://github.com/fchollet/deep-learning-models).

There are also great resources that other people have compiled. Github user [qubvel](https://github.com/qubvel) has created repos that host TensorFlow [classification](https://github.com/qubvel/classification_models) and [segmentation](https://github.com/qubvel/segmentation_models) models.

##### Object Detection

If you're particularly interested in object detection, [tensorpack](https://github.com/tensorpack/tensorpack) is a great place to look. If you don't know where to start, take a look at the `examples` folder.

##### Reinforcement Learning

A great place to look for reinforcement learning code is [Stable Baselines](https://github.com/hill-a/stable-baselines), which is based on the [baselines](https://github.com/openai/baselines) code of [OpenAI](https://openai.com/).

##### Overhead imagery

Some repositories focus on collecting domain-specific models, such as those for overhead imagery. For Tensorflow, [CosmiQ Works](https://www.cosmiqworks.org/)' [simrdwn repo](https://github.com/CosmiQ/simrdwn) is worth checking out.


### PyTorch

PyTorch doesn't have as many pre-trained models as Tensorflow but there's still a lot there. If you're interested in computer vision [Torchvision Models](https://pytorch.org/docs/stable/torchvision/models.html) is a good place to start.

Github user [qubvel](https://github.com/qubvel) has also created repos for PyTorch [segmentation](https://github.com/qubvel/segmentation_models.pytorch) models.

##### Object Detection

The best place to look for PyTorch object detection models is certainly Facebook's [Detectron2 repository](https://github.com/facebookresearch/detectron2). There's a lot in here but if you're looking for pre-trained networks to download for use or transfer learning, definitely check out the [Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md).

##### Overhead imagery

For PyTorch models focused on overhead imagery, one [really good repository](https://github.com/dingjiansw101/AerialDetection) is that of [Jian Ding](https://dingjiansw101.github.io/), one of the authors on the [DOTA dataset](https://captain-whu.github.io/DOTA/).

### FastAI

Finding code for [FastAI](https://www.fast.ai/) is tricky at the moment because 1. the user base is much, much smaller than either Tensorflow or PyTorch (although PyTorch code *is* compatible, it's just not smooth) and 2. the code is still in a state of rapid change. Most code from a few years ago will be very difficult to run on today's [FastAI codebase](https://github.com/fastai/fastai) (and [fastai2](https://github.com/fastai/fastai2) is coming along soon). However, if you are watching the FastAI videos, you can follow along with the [notebooks used in the latest course](https://github.com/fastai/course-v3/tree/master/nbs). The notebooks have changed so you may have to tweak things but it's a good place to get started.

##### Object Detection

I have found that object detection resources are the most lacking, so I've tried to focus on those. Most of the code is in notebooks due to the teaching style of the course. The [most recent official object detection notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl2/pascal.ipynb) I've found is from version 3 of the course. It does basic object detection and has a [video lecture](https://www.youtube.com/watch?v=Z0ssNAbe81M) associated with it.

There's also a [notebook on multiclass object detection](https://github.com/fastai/fastai/blob/master/courses/dl2/pascal-multi.ipynb), although it's even older and I don't believe is included in the courses anymore. Both of these notebooks use the [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) dataset. There's also a [video to go along with it](https://www.youtube.com/watch?v=0frKXR-2PBY).

Outside of that, some other notebooks in the repository are worth looking at. [This notebook](https://github.com/fastai/fastai_dev/blob/master/dev_nb/102a_coco.ipynb) is more recent than the others, but it's not officially part of the course (as far as I can tell). And although [FastAI2](https://github.com/fastai/fastai2) isn't ready yet, there are a lot of [good notebooks in the repo](https://github.com/fastai/fastai2/tree/master/nbs).

If you want to find the latest information on object detection notebooks in FastAI, [this thread](https://forums.fast.ai/t/object-detection-in-fast-ai-v1/29266) is a good place to look.

There are also student notebooks that are worth looking at. This student implemented [RetinaNet using FastAI](https://github.com/ChristianMarzahl/ObjectDetection). The code is out of date with the most recent version of FastAI but it's still worth looking at. There's an [implementation of SSD here](https://github.com/rohitgeo/singleshotdetector/blob/master/SingleShotDetector%20on%20Pascal.ipynb), although I haven't had a chance to go through it yet. This repo used [FastAI for Whale Identification](https://github.com/radekosmulski/whale/). These [notebooks have instance segmentation](https://github.com/lgvaz/mantisshrimp/tree/master/examples), which I haven't been able to find much of.

I would like to create my own implementations of some of these object detection algorithms using FastAI and update this post with them, but I'm not sure when I'll get to that.