---
layout: post
title: "FastAI Data Tutorial - Semantic Segmentation"
description: "This tutorial describes how to work with the FastAI library for semantic segmentation"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/brown_falcon.jpg"
tags: [FastAI, Python]
---

In this tutorial, I will be looking at how to prepare a semantic segmentation dataset for use with FastAI. I will be using the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle as an example. This post focuses on the components that are specific to semantic segmentation. To see tricks and tips for using FastAI with data in general, see my [FastAI Data Tutorial - Image Classification](https://jss367.github.io/fastai-data-tutorial-image-classification.html).


```python
from fastai.vision.all import *
```


```python
root_path = Path('/home/julius/data/kaggle/chest-xray-pneumonia/chest_xray')
```

I have download the data from kaggle. It comes split into a train, val, and test set. The val set is oddly small so I'm going to combine it with the test set here.


```python
train_path = root_path / 'train'
val_path = root_path / 'val'
test_path = root_path / 'test'
```


```python
image_files = get_image_files(root_path)

```


```python
train_idxs = [i for i, file_name in enumerate(image_files) if "train" in str(file_name)]
val_idxs = [i for i, file_name in enumerate(image_files) if "val" in str(file_name) or "test" in str(file_name)]
```


```python
len(train_idxs), len(val_idxs)
```




    10432




```python
dblock = DataBlock(blocks=(ImageBlock(cls=PILImage), CategoryBlock),
                   splitter=IndexSplitter(valid_idx=val_idxs),
                   get_y=parent_label,
                   item_tfms=Resize(512),
                   )
```


```python
#dblock.summary(image_files) # this is printing out every path, so I've commented it out
```


```python
dls = dblock.dataloaders(image_files)
dls.show_batch()
```


    
![png](2022-01-02-fastai-data-tutorial-semantic-segmentation_files/2022-01-02-fastai-data-tutorial-semantic-segmentation_11_0.png)
    



```python

```
