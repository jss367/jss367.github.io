---
layout: post
title: "Siamese Networks with FastAI"
description: "This tutorial describes how to work with the FastAI library for siamese networks"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/log.jpg"
tags: [FastAI, Neural Networks, Python]
---

This post walks through how to create a Siamese network using FastAI. It is based on [a tutorial from FastAI[(https://docs.fast.ai/tutorial.siamese.html), so some of the classes and function are directly copied from there, but I've adding things as well.

First we import the FastAI library.


```python
from fastai.vision.all import *
```

Then make sure GPU support is available.


```python
torch.cuda.is_available()
```




    True



First we download the images. We'll use the cats and dogs dataset because it's built into FastAI.


```python
path = untar_data(URLs.PETS)
files = get_image_files(path/"images")
```

Now we'll need a class to deal with the Siamese images. It'll make everything easier.


```python
class ImageTuple(fastuple):
    @classmethod
    def create(cls, file_paths):
        """
        creates a tuple of two file paths
        """
        return cls(tuple(PILImage.create(f) for f in file_paths))
    
    def show(self, ctx=None, **kwargs): 
        t1, t2 = self
        if not isinstance(t1, Tensor) or not isinstance(t2, Tensor) or t1.shape != t2.shape:
            return ctx
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2], dim=2), ctx=ctx, **kwargs)
```

Let's look at an image.


```python
PILImage.create(files[0])
```




    
![png](2022-04-11-siamese-networks-with-fastai_files/2022-04-11-siamese-networks-with-fastai_11_0.png)
    



Now let's look at a tuple.


```python
img = ImageTuple.create((files[0], files[1]))
tst = ToTensor()(img)
type(tst[0]),type(tst[1])
```




    (fastai.torch_core.TensorImage, fastai.torch_core.TensorImage)



FastAI has a class `ToTensor` that converts items into tensors. We can use it to visualize the image.


```python
img1 = Resize(224)(img)
tst = ToTensor()(img1)
tst.show();
```


    
![png](2022-04-11-siamese-networks-with-fastai_files/2022-04-11-siamese-networks-with-fastai_15_0.png)
    


Now we create an image tuple block. It's really simple. It just returns a TransformBlock where we can the function to create a Siamese image.


```python
def ImageTupleBlock():
    return TransformBlock(type_tfms=ImageTuple.create, batch_tfms=IntToFloatTensor)

```

Now we have to split the data. We can do a random split.


```python
splits = RandomSplitter()(files)
splits_files = [files[splits[i]] for i in range(2)]
splits_sets = mapped(set, splits_files)
```


```python
def get_split(f):
    for i,s in enumerate(splits_sets):
        if f in s:
            return i
    raise ValueError(f'File {f} is not presented in any split.')
```

Now we need a function to return the label given an item. Fortunately, the label is in the filename, so we can use regular expressions to extract it.


```python
def label_func(fname: str):
    """
    Extract the label from the file name.
    """
    return re.match(r'^(.*)_\d+.jpg$', fname.name).groups()[0]

label_func(files[0])
```




    'havanese'




```python
labels = list(set(files.map(label_func)))
```


```python
splbl2files = [{l: [f for f in s if label_func(f) == l] for l in labels} for s in splits_sets]

```


```python
def splitter(items):
    """
    This is the function that we actually pass to the DataBlock. All the others are helpers of some form.
    """
    def get_split_files(i):
        return [j for j,(f1,f2,same) in enumerate(items) if get_split(f1)==i]
    return get_split_files(0),get_split_files(1)
```


```python
def draw_other(f):
    """
    Find the pair for the other Siamese image.
    """
    same = random.random() < 0.5
    cls = label_func(f)
    split = get_split(f)
    if not same:
        cls = random.choice(L(l for l in labels if l != cls)) 
    return random.choice(splbl2files[split][cls]),same
```


```python
def get_tuples(files):
    """
    This function turns a list of files into a list of inputs and label combos.
    So each element in this list contains paths of two images and the associated label
    """
    return [[f, *draw_other(f)] for f in files]

```


```python
def get_x(t):
    return t[:2]
def get_y(t):
    return t[2]
```

New we can build the DataBlock.


```python
siamese = DataBlock(
    blocks=(ImageTupleBlock, CategoryBlock),
    get_items=get_tuples,
    get_x=get_x, get_y=get_y,
    item_tfms=Resize(224),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)]
)
```


```python
# siamese.summary(files) # long output
```

## Create Dataloaders

Everything looks good. Now we can create our `Dataloaders`.


```python
dls = siamese.dataloaders(files)
```

Let's make sure it looks good.


```python
b = dls.one_batch()
explode_types(b)
```




    {tuple: [{__main__.ImageTuple: [fastai.torch_core.TensorImage,
        fastai.torch_core.TensorImage]},
      fastai.torch_core.TensorCategory]}



Now we can make a function to show a batch.


```python
@typedispatch
def show_batch(x:ImageTuple, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None:
        figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None:
        ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    ctxs = show_batch[object](x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs)
    return ctxs
```


```python
dls.show_batch()
```


    
![png](2022-04-11-siamese-networks-with-fastai_files/2022-04-11-siamese-networks-with-fastai_39_0.png)
    


One of my favorite things to do with data in FastAI is turn it into a pandas DataFrame. I find it so much easy to clean or modify the data in the format. In this case, we'll balance the dataset using a DataFrame.

## Now let's turn it into a DataFrame


```python
def df_label_func(fname):
    return fname[2]

```


```python
def create_dataframe(data, label_func, is_valid=False) -> pd.DataFrame:
    """
    Create pandas DataFrame from DataLoaders
    """
    items = [x[:2] for x in data.valid.items] if is_valid else [x[:2] for x in data.train.items]
    labels = [x[2] for x in data.valid.items] if is_valid else [x[2] for x in data.train.items]
    is_valid = [is_valid] * len(items)
    return pd.DataFrame({'items': items, 'label': labels, 'is_valid':is_valid})
```


```python
train_df = create_dataframe(dls, df_label_func, is_valid=False)
valid_df = create_dataframe(dls, df_label_func, is_valid=True)
```

Let's see what we have.


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>items</th>
      <th>label</th>
      <th>is_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/Ragdoll_117.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/leonberger_79.jpg]</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/Maine_Coon_214.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/samoyed_40.jpg]</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/havanese_55.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/havanese_30.jpg]</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/english_cocker_spaniel_37.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/english_cocker_spaniel_90.jpg]</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/chihuahua_165.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/english_setter_91.jpg]</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Now we have it as a dataframe, it's easier to clean or manipulate the data.

Now let's oversample the data


```python
train_df['label'].value_counts()
```




    True     2971
    False    2941
    Name: label, dtype: int64




```python
def oversample(df: pd.DataFrame) -> pd.DataFrame:
    """
    We will use pd.DataFrame.sample to oversample from the rows with labels that need to be oversampled.
    """
    num_max_labels = df['label'].value_counts().max()
    dfs = [df]
    for class_index, group in df.groupby('label'):
        num_new_samples = num_max_labels - len(group)
        dfs.append(group.sample(num_new_samples, replace=True, random_state=42))
    return pd.concat(dfs)
```


```python
oversampled_train_df = oversample(train_df)
oversampled_train_df['label'].value_counts()
```




    False    2971
    True     2971
    Name: label, dtype: int64



Great. Now we'll combine it with the validation data again. Note that we didn't mess with the validation dataset's label balance.


```python
full_df = pd.concat([oversampled_train_df, valid_df])
```

Now we make another DataBlock that can read from a csv. Fortunately, FastAI's support for this is great.


```python
new_dblock = DataBlock(
    blocks    = (ImageTupleBlock, CategoryBlock),
    get_x=ColReader(0),
    get_y=ColReader('label'),
    splitter  = ColSplitter(),
    item_tfms = Resize(224),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)]
    )
```


```python
new_data = new_dblock.dataloaders(full_df)
```

## Modeling

Now we build a simple Siamese Model.


```python
class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder, self.head = encoder, head

    def forward(self, x):
        """
        This takes x1 and x2 as two separate things. But if we change it to x, maybe we're OK?
        """
        x1, x2 = x
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)
```


```python
model_meta[resnet34]
```




    {'cut': -2,
     'split': <function fastai.vision.learner._resnet_split(m)>,
     'stats': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}




```python
encoder = create_body(resnet34, cut=-2)
head = create_head(512*2, 2, ps=0.5)
model = SiameseModel(encoder, head)
```


```python
def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]

def loss_func(out, targ):
    return CrossEntropyLossFlat()(out, targ.long())
```


```python
learn = Learner(new_data, model, loss_func=loss_func, splitter=siamese_splitter, metrics=accuracy)
```


```python
learn.freeze()
```


```python
learn.lr_find()
```





    /home/julius/miniconda3/envs/pt/lib/python3.8/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448234945/work/c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    




    SuggestedLRs(valley=0.0006918309954926372)




    
![png](2022-04-11-siamese-networks-with-fastai_files/2022-04-11-siamese-networks-with-fastai_65_3.png)
    



```python
learn.fit_one_cycle(4, 1e-3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.616689</td>
      <td>0.371314</td>
      <td>0.854533</td>
      <td>00:24</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.372786</td>
      <td>0.235297</td>
      <td>0.917456</td>
      <td>00:24</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.246533</td>
      <td>0.197698</td>
      <td>0.930988</td>
      <td>00:24</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.185981</td>
      <td>0.194794</td>
      <td>0.928281</td>
      <td>00:24</td>
    </tr>
  </tbody>
</table>


There you go! A fully trained Siamese network.
