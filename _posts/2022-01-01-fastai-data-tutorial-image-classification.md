---
layout: post
title: "FastAI Data Tutorial - Image Classification"
description: "This tutorial describes how to work with the FastAI library for image classification"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/peregrine_falcon.jpg"
tags: [FastAI, Python]
---

This post is a tutorial for loading data with [FastAI](https://github.com/fastai/fastai). The interface has changed a lot since I originally wrote a FastAI data tutorial, so I deleted that one and I'm starting from scratch and making a brand new one. I'll try to keep this up-to-date with the latest version. FastAI seems to be quite stable at the moment, so hopefully this will continue to work with the latest version.

## Using your Own Data

There are already [tutorials on the website](https://docs.fast.ai/tutorial.vision.html) for how to work with the provided data, so I thought I would talk about how to work with data that is saved on your disk. We'll use the Kangaroos and Wallabies dataset that I discuss in [this post](https://jss367.github.io/kangaroos-and-wallabies-i-preparing-the-data.html).

To start with, we do the standard FastAI imports.


```python
from fastai.data.all import *
from fastai.vision.all import *
```

To generate a dataset, you'll need to create a `DataBlock` and a `DataLoader`. The `DataBlock` is the first and main thing required to generate a dataset. A datablock explains what you are going to do with your data. DataBlocks are the building blocks of DataLoaders.

First, you'll need to specify what the input and labels look like. For standard use-cases the tools you need are already built into FastAI. For image data, you use an `ImageBlock` and for categorical labels you use a `CategoryBlock`.


```python
blocks = ImageBlock, CategoryBlock
```

You'll need to tell it where to get your items. `fastai` comes with a nice little function, `get_image_files`, that makes pulling files from a disk easy.

Then, we need to explain how to get the label. In our case, the label name come right from the folder name. 

Then, you can add a method to split between train and validation data, as well as any transform you want.

Putting it all together, it will look like this:


```python
get_image_files
```




    <function fastai.data.transforms.get_image_files(path, recurse=True, folders=None)>




```python
dblock = DataBlock(blocks    = blocks,
                   get_items = get_image_files,
                   get_y     = parent_label,
                   splitter  = GrandparentSplitter('train', 'val'),
                   item_tfms = Resize(224))
```

Note that we haven't actually told it where our images are on disk. That's because a `DataBlock` exists irrespective of underlying images. You will pass it a path of images to use it.


```python
if sys.platform == 'linux':
    path = Path(r'/home/julius/data/WallabiesAndRoosFullSize')
else:
    path = Path(r'E:/Data/Raw/WallabiesAndRoosFullSize')
```

The best way to see if you have made a valid `DataBlock` is to use the `.summary()` method.


```python
dblock.summary(path)
```

    Setting-up type transforms pipelines
    Collecting items from /home/julius/data/WallabiesAndRoosFullSize
    Found 4721 items
    2 datasets of sizes 3653,567
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: parent_label -> Categorize -- {'vocab': None, 'sort': True, 'add_na': False}
    
    Building one sample
      Pipeline: PILBase.create
        starting from
          /home/julius/data/WallabiesAndRoosFullSize/train/wallaby/wallaby-385.jpg
        applying PILBase.create gives
          PILImage mode=RGB size=4000x4000
      Pipeline: parent_label -> Categorize -- {'vocab': None, 'sort': True, 'add_na': False}
        starting from
          /home/julius/data/WallabiesAndRoosFullSize/train/wallaby/wallaby-385.jpg
        applying parent_label gives
          wallaby
        applying Categorize -- {'vocab': None, 'sort': True, 'add_na': False} gives
          TensorCategory(1)
    
    Final sample: (PILImage mode=RGB size=4000x4000, TensorCategory(1))
    
    
    Collecting items from /home/julius/data/WallabiesAndRoosFullSize
    Found 4721 items
    2 datasets of sizes 3653,567
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: parent_label -> Categorize -- {'vocab': None, 'sort': True, 'add_na': False}
    Setting up after_item: Pipeline: Resize -- {'size': (224, 224), 'method': 'crop', 'pad_mode': 'reflection', 'resamples': (2, 0), 'p': 1.0} -> ToTensor
    Setting up before_batch: Pipeline: 
    Setting up after_batch: Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
    
    Building one batch
    Applying item_tfms to the first sample:
      Pipeline: Resize -- {'size': (224, 224), 'method': 'crop', 'pad_mode': 'reflection', 'resamples': (2, 0), 'p': 1.0} -> ToTensor
        starting from
          (PILImage mode=RGB size=4000x4000, TensorCategory(1))
        applying Resize -- {'size': (224, 224), 'method': 'crop', 'pad_mode': 'reflection', 'resamples': (2, 0), 'p': 1.0} gives
          (PILImage mode=RGB size=224x224, TensorCategory(1))
        applying ToTensor gives
          (TensorImage of size 3x224x224, TensorCategory(1))
    
    Adding the next 3 samples
    
    No before_batch transform to apply
    
    Collating items in a batch
    
    Applying batch_tfms to the batch built
      Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
        starting from
          (TensorImage of size 4x3x224x224, TensorCategory([1, 1, 1, 1], device='cuda:0'))
        applying IntToFloatTensor -- {'div': 255.0, 'div_mask': 1} gives
          (TensorImage of size 4x3x224x224, TensorCategory([1, 1, 1, 1], device='cuda:0'))
    

Once you've got a `DataBlock`, you can convert it into either a dataset using `dblock.datasets` or a dataloader using `dblock.dataloaders`. In this case, we'll do the `DataLoader`.

## DataLoaders

Creating `DataLoaders` from a `DataBlock` is trivially simple - all you do is pass a path.


```python
dls = dblock.dataloaders(path)
```


```python
dls.train.show_batch(max_n=4, nrows=1)
```


    
![png](2022-01-01-fastai-data-tutorial-image-classification_files/2022-01-01-fastai-data-tutorial-image-classification_23_0.png)
    



```python
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](2022-01-01-fastai-data-tutorial-image-classification_files/2022-01-01-fastai-data-tutorial-image-classification_24_0.png)
    


We can get an example of a batch like so:


```python
images, labels = first(dls.train)
```

Let's look at the shape to make sure it's what we expect. PyTorch uses channels first, so it should be N X C X H X W.


```python
print(images.shape, labels.shape)
```

    torch.Size([64, 3, 224, 224]) torch.Size([64])
    

## Creating New DataBlocks

It's easy to create new `DataBlock`s. Let's say you want to add some transformations. Here's one way to do that.


```python
dblock = dblock.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'), batch_tfms=aug_transforms(mult=2))
dls = dblock.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1)
```

    /home/julius/miniconda3/envs/fai/lib/python3.8/site-packages/torch/_tensor.py:1023: UserWarning: torch.solve is deprecated in favor of torch.linalg.solveand will be removed in a future PyTorch release.
    torch.linalg.solve has its arguments reversed and does not return the LU factorization.
    To get the LU factorization see torch.lu, which can be used with torch.lu_solve or torch.lu_unpack.
    X = torch.solve(B, A).solution
    should be replaced with
    X = torch.linalg.solve(A, B) (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448234945/work/aten/src/ATen/native/BatchLinearAlgebra.cpp:760.)
      ret = func(*args, **kwargs)
    


    
![png](2022-01-01-fastai-data-tutorial-image-classification_files/2022-01-01-fastai-data-tutorial-image-classification_31_1.png)
    


## Oversampling

Let's also say we wanted to add oversampling to the train set. Here's how we do that. First, let's see how many more we have of one type than the other.


```python
train_files = get_image_files(path / 'train')
val_files = get_image_files(path / 'val')
```


```python
wallaby_files = [f for f in train_files if 'wallaby' in str(f)]
kangaroo_files = [f for f in train_files if 'kangaroo' in str(f)]
```


```python
len(wallaby_files), len(kangaroo_files)
```




    (1465, 2188)



Now let's say we want to double the number of wallaby files.


```python
oversampled_files = wallaby_files * 2 + kangaroo_files
```


```python
len(oversampled_files), len(val_files)
```




    (5118, 567)



OK, now we've got 5118 files and these are all train files. Fortunately, the same splitter that we used before will work here, so we can use that.


```python

```


```python
dblock = DataBlock(blocks    = blocks,
                   get_items = get_image_files,
                   get_y     = parent_label,
                   splitter  = GrandparentSplitter('train', 'val'),
                   item_tfms = Resize(224))
```


```python

```

## Normalizing


```python
means = [x.mean(dim=(0, 2, 3)) for x, y in dls.train]
stds = [x.std(dim=(0, 2, 3)) for x, y in dls.train]
mean = torch.stack(means).mean(dim=0)
std = torch.stack(stds).mean(dim=0)
print(mean, std)
```

    TensorImage([0.4947, 0.4527, 0.4030], device='cuda:0') TensorImage([0.2462, 0.2288, 0.2235], device='cuda:0')
    


```python
augs = [RandomResizedCropGPU(size=224, min_scale=0.75), Zoom()]
augs += [Normalize.from_stats(mean, std)]
```


```python
dblock = DataBlock(blocks    = blocks,
                   get_items = get_image_files,
                   get_y     = parent_label,
                   splitter  = GrandparentSplitter('train', 'val'),
                   item_tfms = Resize(224),
                   batch_tfms=augs)
```


```python
dls = dblock.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1)
```


    
![png](2022-01-01-fastai-data-tutorial-image-classification_files/2022-01-01-fastai-data-tutorial-image-classification_48_0.png)
    


## Exploring DataLoaders

Let's look in more detail at the dataloaders. First, what kind of object are they?


```python
dls
```




    <fastai.data.core.DataLoaders at 0x7f0f2c315790>



The `DataLoaders` class is a wrapper around multiple, you guessed it, `DataLoader` classes. This is particularly useful when using a train and a test set. Let's see what `DataLoader`s we have here.


```python
dls.loaders
```




    [<fastai.data.core.TfmdDL at 0x7f0f2e0572e0>,
     <fastai.data.core.TfmdDL at 0x7f0f2c3731c0>]




```python
dl = dls.loaders[0]
```


```python
dl
```




    <fastai.data.core.TfmdDL at 0x7f0f2e0572e0>



Let's see what it spits out.


```python
item = next(iter(dl))
```


```python
len(item)
```




    2



It's got two items - the first is a PyTorch Tensor and the second is the label.


```python
type(item[0])
```




    fastai.torch_core.TensorImage




```python
type(item[1])
```




    fastai.torch_core.TensorCategory




```python
item[0].shape
```




    torch.Size([64, 3, 224, 224])



There's a default batch size of 64, so that's why we have 64 items.


```python
item[1]
```




    TensorCategory([1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
            0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], device='cuda:0')


