---
layout: post
title: "FastAI Data Tutorial - Semantic Segmentation"
description: "This tutorial describes how to work with the FastAI library for semantic segmentation"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/brown_falcon.jpg"
tags: [FastAI, Python]
---

In this tutorial, I will be looking at how to prepare a semantic segmentation dataset for use with FastAI. I will be using the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle as an example. This post focuses on the components that are specific to semantic segmentation. To see tricks and tips for using FastAI with data in general, see my [FastAI Data Tutorial - Image Classification](https://jss367.github.io/fastai-data-tutorial-image-classification.html).

<b>Table of Contents</b>
* TOC
{:toc}


```python
from fastai.vision.all import *
```


```python
path = Path('/home/julius/data/kaggle/lgg-mri-segmentation/kaggle_3m/')
```


```python
def get_images(path):
    all_files = get_image_files(path)
    images = [i for i in all_files if 'mask' not in str(i)]
    return images
```


```python
def get_label(im_path):
    return im_path.parent / (im_path.stem + '_mask' + im_path.suffix)
```


```python
all_images = get_images(path)
```


```python
examp_im_path = all_images[0]
examp_im_path
```




    Path('/home/julius/data/kaggle/lgg-mri-segmentation/kaggle_3m/TCGA_DU_7309_19960831/TCGA_DU_7309_19960831_21.tif')




```python
im = PILImage.create(examp_im_path)
im
```




    
![png](2022-01-02-fastai-data-tutorial-semantic-segmentation_files/2022-01-02-fastai-data-tutorial-semantic-segmentation_8_0.png)
    




```python
mask = PILImage.create(get_label(examp_im_path))
mask.show(figsize=(5,5), alpha=1)
```




    <AxesSubplot:>




    
![png](2022-01-02-fastai-data-tutorial-semantic-segmentation_files/2022-01-02-fastai-data-tutorial-semantic-segmentation_9_1.png)
    



```python
mask.shape
```




    (256, 256)




```python
codes = ['n', 'y']
bs=16
```

Now we determine which blocks to use. For segmentation tasks, we'll generally use `MaskBlock`.


```python
blocks = (ImageBlock, MaskBlock(codes))
```

Now we create the `DataBlock`.


```python
dblock = DataBlock(blocks    = blocks,
                   get_items = get_images,
                   get_y     = get_label,
                   splitter  = RandomSplitter(),
                   item_tfms = Resize(224))
```


```python
dblock.summary(path)
```

    Setting-up type transforms pipelines
    Collecting items from /home/julius/data/kaggle/lgg-mri-segmentation/kaggle_3m
    Found 3929 items
    2 datasets of sizes 3144,785
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: get_label -> PILBase.create
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: get_label -> PILBase.create
    
    Building one sample
      Pipeline: PILBase.create
        starting from
          /home/julius/data/kaggle/lgg-mri-segmentation/kaggle_3m/TCGA_DU_7304_19930325/TCGA_DU_7304_19930325_14.tif
        applying PILBase.create gives
          PILImage mode=RGB size=256x256
      Pipeline: get_label -> PILBase.create
        starting from
          /home/julius/data/kaggle/lgg-mri-segmentation/kaggle_3m/TCGA_DU_7304_19930325/TCGA_DU_7304_19930325_14.tif
        applying get_label gives
          /home/julius/data/kaggle/lgg-mri-segmentation/kaggle_3m/TCGA_DU_7304_19930325/TCGA_DU_7304_19930325_14_mask.tif
        applying PILBase.create gives
          PILMask mode=L size=256x256
    
    Final sample: (PILImage mode=RGB size=256x256, PILMask mode=L size=256x256)
    
    
    Collecting items from /home/julius/data/kaggle/lgg-mri-segmentation/kaggle_3m
    Found 3929 items
    2 datasets of sizes 3144,785
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: get_label -> PILBase.create
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: get_label -> PILBase.create
    Setting up after_item: Pipeline: AddMaskCodes -> Resize -- {'size': (224, 224), 'method': 'crop', 'pad_mode': 'reflection', 'resamples': (2, 0), 'p': 1.0} -> ToTensor
    Setting up before_batch: Pipeline: 
    Setting up after_batch: Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
    
    Building one batch
    Applying item_tfms to the first sample:
      Pipeline: AddMaskCodes -> Resize -- {'size': (224, 224), 'method': 'crop', 'pad_mode': 'reflection', 'resamples': (2, 0), 'p': 1.0} -> ToTensor
        starting from
          (PILImage mode=RGB size=256x256, PILMask mode=L size=256x256)
        applying AddMaskCodes gives
          (PILImage mode=RGB size=256x256, PILMask mode=L size=256x256)
        applying Resize -- {'size': (224, 224), 'method': 'crop', 'pad_mode': 'reflection', 'resamples': (2, 0), 'p': 1.0} gives
          (PILImage mode=RGB size=224x224, PILMask mode=L size=224x224)
        applying ToTensor gives
          (TensorImage of size 3x224x224, TensorMask of size 224x224)
    
    Adding the next 3 samples
    
    No before_batch transform to apply
    
    Collating items in a batch
    
    Applying batch_tfms to the batch built
      Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
        starting from
          (TensorImage of size 4x3x224x224, TensorMask of size 4x224x224)
        applying IntToFloatTensor -- {'div': 255.0, 'div_mask': 1} gives
          (TensorImage of size 4x3x224x224, TensorMask of size 4x224x224)
    

    /home/julius/miniconda3/envs/fai/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448234945/work/aten/src/ATen/native/BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)
    


```python
dls = dblock.dataloaders(path)

```


```python
dls.train.show_batch(max_n=4, nrows=1)

```


    
![png](2022-01-02-fastai-data-tutorial-semantic-segmentation_files/2022-01-02-fastai-data-tutorial-semantic-segmentation_18_0.png)
    

