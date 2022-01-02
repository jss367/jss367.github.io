---
layout: post
title: "FastAI Data Tutorial - Object Detection"
description: "This tutorial describes how to work with the FastAI library for object detection"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/australian_hobby.jpg"
tags: [FastAI, Python]
---

In this tutorial, I will be looking at how to prepare an object detection dataset for use with PyTorch and FastAI. I will be using the [DOTA](https://captain-whu.github.io/DOTA/dataset.html) dataset as an example. I will prepare the same data for both PyTorch and FastAI to illustrate the differences. This post focuses on the components that are specific to object detection. To see tricks and tips for using FastAI with data in general, see my [FastAI Data Tutorial - Image Classification](https://jss367.github.io/fastai-data-tutorial-image-classification.html).


```python
from fastai.data.all import *
from fastai.vision.all import *
from torchvision.datasets.vision import VisionDataset
from pycocotools.coco import COCO
from pyxtend import struct # pyxtend is available on pypi
```

## DOTA - PyTorch

PyTorch Datasets are SUPER simple. So simple that they don't actually do anything. It's just a format. You can [see the code here](https://github.com/pytorch/pytorch/blob/ce86881afadc0fea628c7e47d64a4073f3e09894/torch/utils/data/dataset.py#L51), but basically the only thing that makes something a PyTorch Dataset is that it has a `__getitem__` method. This gives us incredible flexibility, but the lack of structure can also be difficult at first. For example, it's not even clear what data type `__getitem__` should return. Although it's commonly a tuple, sometimes returning a dictionary can be useful too.


```python
class DOTADataset(VisionDataset):
    """
    Is there a separate dataset for train, test, and val?
    """
    def __init__(self, image_root, annotations, transforms=None):
        super().__init__(image_root, annotations, transforms)
        #self.root = image_root don't need this cause super?
        self.coco = COCO(annotations)
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))
        
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # don't want to return a pil image
        img = np.array(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self):
        return len(self.ids)
```


```python
dota_path = Path(r'E:\Data\Processed\DOTACOCO')
dota_train_annotations = dota_path / 'dota2coco_train.json'
dota_train_images = Path(r'E:\Data\Raw\DOTA\train\images\all\images')
```


```python
dota_dataset = DOTADataset(dota_train_images, dota_train_annotations)
```

    loading annotations into memory...
    Done (t=1.47s)
    creating index...
    index created!
    

Because it's a VisionDataset, we have a nice repr response.


```python
dota_dataset
```




    Dataset DOTADataset
        Number of datapoints: 1411
        Root location: E:\Data\Raw\DOTA\train\images\all\images



It's easy to plot the images.


```python
plt.imshow(dota_dataset[0][0])
```




    <matplotlib.image.AxesImage at 0x2388054dac0>




    
![png](2022-01-02-fastai-data-tutorial-object-detection_files/2022-01-02-fastai-data-tutorial-object-detection_11_1.png)
    


Let's build a simple way to look at images with labels.


```python
def show_annotations(image, annotations, figsize=(20,20), axis_off=True):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    if axis_off:
        plt.axis('off')
    coco.showAnns(annotations)
```


```python
coco = dota_dataset.coco
```


```python
show_annotations(*dota_dataset[0])
```


    
![png](2022-01-02-fastai-data-tutorial-object-detection_files/2022-01-02-fastai-data-tutorial-object-detection_15_0.png)
    


## DOTA - FastAI

OK, we've got the PyTorch part working. Now let's plug it into FastAI

We need a way to get the images for the respective blocks. This will be a list of three functions, like so:


```python
imgs, lbl_bbox = get_annotations(dota_train_annotations)
```


```python
imgs[:5]
```




    ['P0000.png', 'P0001.png', 'P0002.png', 'P0005.png', 'P0008.png']



`lbl_bbox` contains lots of elements, so let's take a look at the structure of it.


```python
struct(lbl_bbox)
```




    {list: [{tuple: [{list: [{list: [float, float, float, '...4 total']},
          {list: [float, float, float, '...4 total']},
          {list: [float, float, float, '...4 total']},
          '...323 total']},
        {list: [str, str, str, '...323 total']}]},
      {tuple: [{list: [{list: [float, float, float, '...4 total']},
          {list: [float, float, float, '...4 total']},
          {list: [float, float, float, '...4 total']},
          '...40 total']},
        {list: [str, str, str, '...40 total']}]},
      {tuple: [{list: [{list: [float, float, float, '...4 total']},
          {list: [float, float, float, '...4 total']},
          {list: [float, float, float, '...4 total']},
          '...288 total']},
        {list: [str, str, str, '...288 total']}]},
      '...1410 total']}



Now we need a function to pass to `get_items` inside the datablock. Because we already have a list of all the items, all we need to do is write a function that returns that list.


```python
def get_train_imgs(noop):
    return imgs
```

Given an we need to get the correct annotation. Fortunately, we can look it up in a dictionary.


```python
img2bbox = dict(zip(imgs, lbl_bbox))
```

Now, we put all that together in our `getters`.


```python
getters = [lambda o: dota_train_images/o,
           lambda o: img2bbox[o][0],
           lambda o: img2bbox[o][1]]
```

We can add any transforms we want.


```python
item_tfms = [Resize(128, method='pad'),]
batch_tfms = [Rotate(), Flip(), Normalize.from_stats(*imagenet_stats)]
```

Now, we turn it into a `DataBlock`.


```python
dota_dblock = DataBlock(blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),
                 splitter=RandomSplitter(),
                 get_items=get_train_imgs,
                 getters=getters,
                 item_tfms=item_tfms,
                 batch_tfms=batch_tfms,
                 n_inp=1)
```

From from there we create our `DataLoaders`.


```python
dls = dota_dblock.dataloaders(dota_train_images)
```

    Due to IPython and Windows limitation, python multiprocessing isn't available now.
    So `number_workers` is changed to 0 to avoid getting stuck
    

As you can see, the `show_batch` method doesn't work as well with many labels, as is often the case with aerial imagery. However, you can see use it to get a general sense.


```python
dls.show_batch()
```


    
![png](2022-01-02-fastai-data-tutorial-object-detection_files/2022-01-02-fastai-data-tutorial-object-detection_36_0.png)
    


That's all there is to it!
