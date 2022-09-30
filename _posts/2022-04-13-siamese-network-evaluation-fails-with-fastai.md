---
layout: post
title: "Siamese Network Evaluation Fails with FastAI"
description: "This tutorial describes how not to evaluate siamese networks with the FastAI library"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/platypus3.jpg"
tags: [FastAI, Neural Networks, Python]
---

This post summarizes some of the paths I went down trying to figure out how to evaluate things in FastAI. I'll start it off correctly and let you know when I go down a bad path.


```python
import ast
import dill
from fastai.vision.all import *
```


```python
learner = load_learner(Path(os.getenv('MODELS')) / 'siamese_catsvdogs.pkl', cpu=False, pickle_module=dill)
```


```python
df = pd.read_csv('siamese_data.csv', index_col=0)
```


```python
df['items'] = df['items'].apply(ast.literal_eval)
```


```python
df.head()
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
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/keeshond_13.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/leonberger_184.jpg]</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/Russian_Blue_175.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Russian_Blue_24.jpg]</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/german_shorthaired_145.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/german_shorthaired_147.jpg]</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/beagle_17.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/american_bulldog_110.jpg]</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/leonberger_11.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Persian_5.jpg]</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



OK. Now let's talk about how to run some items. There are a lot of things to be aware of.


```python
df['is_valid'].value_counts()
```




    False    6118
    True     1478
    Name: is_valid, dtype: int64



The correct way to evaluate it would be to create a DataLoader from your learner, like this:


```python
test_dl = learner.dls.test_dl(df)
```

Let's check if the DataLoader is good.


```python
test_dl.bs
```




    64




```python
test_dl.device
```




    device(type='cuda', index=0)



You can see that you have bad data here if you try to iterate over it.


```python
try:
    next(iter(test_dl))
except AttributeError as err:
    print(err)
```

    Caught AttributeError in DataLoader worker process 0.
    Original Traceback (most recent call last):
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/PIL/Image.py", line 3050, in open
        fp.seek(0)
    AttributeError: 'list_iterator' object has no attribute 'seek'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", line 302, in _worker_loop
        data = fetcher.fetch(index)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 39, in fetch
        data = next(self.dataset_iter)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 140, in create_batches
        yield from map(self.do_batch, self.chunkify(res))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/basics.py", line 230, in chunked
        res = list(itertools.islice(it, chunk_sz))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 155, in do_item
        try: return self.after_item(self.create_item(s))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 162, in create_item
        if self.indexed: return self.dataset[s or 0]
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 557, in __getitem__
        res = tuple([tl[it] for tl in self.tls])
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 557, in <listcomp>
        res = tuple([tl[it] for tl in self.tls])
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 509, in __getitem__
        return self._after_item(res) if is_indexer(idx) else res.map(self._after_item)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 455, in _after_item
        return self.tfms(o)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 299, in __call__
        return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 234, in compose_tfms
        x = f(x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 117, in __call__
        return self._call("encodes", x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 132, in _call
        return self._do_call(getattr(self, fn), x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 139, in _do_call
        return retain_type(f(x, **kwargs), x, ret)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/dispatch.py", line 161, in __call__
        return f(*args, **kwargs)
      File "/tmp/ipykernel_15574/1954726133.py", line 235, in create
      File "/tmp/ipykernel_15574/1954726133.py", line -1, in <genexpr>
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/vision/core.py", line 185, in create
        return cls(load_image(fn, **merge(cls._open_args, kwargs)))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/vision/core.py", line 151, in load_image
        im = Image.open(fn)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/PIL/Image.py", line 3052, in open
        fp = io.BytesIO(fp.read())
    AttributeError: 'list_iterator' object has no attribute 'read'
    
    

Since you have a bad DataLoader if you try to make a prediction, it will fail.


```python
try:
    learner.get_preds(dl=test_dl)
except AttributeError as err:
    print(err)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='0' class='' max='119' style='width:300px; height:20px; vertical-align: middle;'></progress>
  0.00% [0/119 00:00&lt;?]
</div>



    Caught AttributeError in DataLoader worker process 0.
    Original Traceback (most recent call last):
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/PIL/Image.py", line 3050, in open
        fp.seek(0)
    AttributeError: 'list_iterator' object has no attribute 'seek'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", line 302, in _worker_loop
        data = fetcher.fetch(index)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 39, in fetch
        data = next(self.dataset_iter)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 140, in create_batches
        yield from map(self.do_batch, self.chunkify(res))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/basics.py", line 230, in chunked
        res = list(itertools.islice(it, chunk_sz))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 155, in do_item
        try: return self.after_item(self.create_item(s))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 162, in create_item
        if self.indexed: return self.dataset[s or 0]
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 557, in __getitem__
        res = tuple([tl[it] for tl in self.tls])
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 557, in <listcomp>
        res = tuple([tl[it] for tl in self.tls])
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 509, in __getitem__
        return self._after_item(res) if is_indexer(idx) else res.map(self._after_item)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 455, in _after_item
        return self.tfms(o)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 299, in __call__
        return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 234, in compose_tfms
        x = f(x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 117, in __call__
        return self._call("encodes", x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 132, in _call
        return self._do_call(getattr(self, fn), x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 139, in _do_call
        return retain_type(f(x, **kwargs), x, ret)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/dispatch.py", line 161, in __call__
        return f(*args, **kwargs)
      File "/tmp/ipykernel_15574/1954726133.py", line 235, in create
      File "/tmp/ipykernel_15574/1954726133.py", line -1, in <genexpr>
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/vision/core.py", line 185, in create
        return cls(load_image(fn, **merge(cls._open_args, kwargs)))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/vision/core.py", line 151, in load_image
        im = Image.open(fn)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/PIL/Image.py", line 3052, in open
        fp = io.BytesIO(fp.read())
    AttributeError: 'list_iterator' object has no attribute 'read'
    
    

## Recreating the DataBlock

Another incorrect way to do this is to create a new DataBlock using the same code that you did to create the model in the first place, like so:

I've copied the functions that we used here:


```python
def ImageTupleBlock():
    return TransformBlock(type_tfms=ImageTuple.create, batch_tfms=IntToFloatTensor)

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


```python
dblock = DataBlock(
    blocks    = (ImageTupleBlock, CategoryBlock),
    get_x=ColReader(0),
    get_y=ColReader('label'),
    splitter  = ColSplitter(),
    item_tfms = Resize(224),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)]
    )
```


```python
dls = dblock.dataloaders(df)
```

We can make `DataLoaders` like that. But imagine our data was a new set that we wanted to run. Let's imagine it's all validation data.


```python
df['is_valid'] = True
```


```python
df.iat[0, 2]
```




    True




```python
try:
    dls = dblock.dataloaders(df, device='cuda')
except IndexError as err:
    print(err)
```

    single positional indexer is out-of-bounds
    

We got an `IndexError` and a not-very-helpful error message. The problem is that you must have training data for it to work.

So let's set it to training data.


```python
df['is_valid'] = False
```


```python
dls = dblock.dataloaders(df, device='cuda')
```


```python
try:
    preds, labels, decoded = learner.get_preds(dl=dls.valid, with_decoded=True)
except ValueError as err:
    print(err)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







    /home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastprogress/fastprogress.py:73: UserWarning: Your generator is empty.
      warn("Your generator is empty.")
    

    not enough values to unpack (expected 3, got 2)
    

This fails because you don't have any data, so it returns (None, None). Oddly it doesn't return a value for when `with_decode` is true, so you get a `ValueError`.


```python
learner.get_preds(dl=dls.valid, with_decoded=True)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    (None, None)



For it to work you need some of both kinds of data. Let's set it all to validation data except for one, which we'll set to train.


```python
# have to have some of both
df['is_valid'] = True
```


```python
df.tail()
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
      <th>1473</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/Bengal_162.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Bengal_47.jpg]</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1474</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/Bombay_203.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Bombay_88.jpg]</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1475</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/great_pyrenees_133.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Sphynx_47.jpg]</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1476</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/american_pit_bull_terrier_102.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/leonberger_138.jpg]</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1477</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/Persian_155.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Russian_Blue_137.jpg]</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iat[-1, 2] = False
```


```python
df.tail()
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
      <th>1473</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/Bengal_162.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Bengal_47.jpg]</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1474</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/Bombay_203.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Bombay_88.jpg]</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1475</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/great_pyrenees_133.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Sphynx_47.jpg]</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1476</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/american_pit_bull_terrier_102.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/leonberger_138.jpg]</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1477</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/Persian_155.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Russian_Blue_137.jpg]</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



(Note how hacky all of this is. That's because this is WRONG.)

Make sure it's only one row.


```python
df[~df['is_valid']]
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
      <th>1477</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/Persian_155.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Russian_Blue_137.jpg]</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
dls = dblock.dataloaders(df, device='cuda')
```


```python
try:
    next(iter(dls.train))
except StopIteration as err:
    print(f"Not a full batch so gives an error with no other details: {err=}")
```

    Not a full batch so gives an error with no other details: err=StopIteration()
    

You can, however, still get a prediction from it


```python
learner.get_preds(dl=dls.train)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    (TensorBase([[1.2749, 0.0387]]), TensorCategory([0]))



Even though we got a prediction out, this isn't the right way to do it. Let's see what happens if we run it again.


```python
learner.get_preds(dl=dls.train)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    (TensorBase([[1.1535, 0.2051]]), TensorCategory([0]))



We got a different answer. This is because we're using data augmentation in the pipeline, even if you didn't explicitly put it there (FastAI likes to do lots of things under the hood).

A way around this is to put them all in the validation set and one in the test set (keep in mind, this is all wrong though).


```python
df['is_valid'] = False
df.iat[-1, 2] = True
```


```python
df[df['is_valid']]
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
      <th>1477</th>
      <td>[/home/julius/.fastai/data/oxford-iiit-pet/images/Persian_155.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Russian_Blue_137.jpg]</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
dls = dblock.dataloaders(df, device='cuda')
```

Note that you can look at the data if it's in the validation set (which you can't do in the train set if it's not a full batch).


```python
next(iter(dls.valid))
```




    ((TensorImage([[[[ 2.0948,  2.0948,  2.0948,  ...,  2.1633,  2.1633,  2.1462],
                     [-0.0972, -0.0801, -0.0972,  ...,  1.6667,  1.2043,  1.2214],
                     [-1.2445, -1.2959, -1.3644,  ...,  1.4440,  0.9474,  0.8104],
                     ...,
                     [ 0.6906,  0.7077,  0.7077,  ...,  0.7077,  0.7077,  0.7077],
                     [ 1.8379,  1.8379,  1.8379,  ...,  1.8379,  1.8379,  1.8550],
                     [ 2.1975,  2.1975,  2.1975,  ...,  2.1975,  2.1804,  2.1804]],
      
                    [[ 2.2885,  2.2885,  2.2885,  ...,  2.3235,  2.3060,  2.3060],
                     [ 0.2052,  0.2577,  0.2577,  ...,  1.2031,  0.8179,  0.7654],
                     [-0.8978, -0.8627, -0.8627,  ...,  0.6604,  0.3102,  0.1702],
                     ...,
                     [ 0.8529,  0.8529,  0.8529,  ...,  0.8704,  0.8529,  0.8704],
                     [ 2.0084,  2.0084,  2.0084,  ...,  2.0084,  2.0084,  2.0259],
                     [ 2.3761,  2.3761,  2.3761,  ...,  2.3761,  2.3585,  2.3585]],
      
                    [[ 2.4483,  2.4657,  2.4831,  ...,  2.4483,  2.4657,  2.4483],
                     [ 0.0953,  0.1128,  0.1128,  ...,  0.1302,  0.0779,  0.0605],
                     [-1.4036, -1.4559, -1.4907,  ..., -1.1944, -1.2990, -1.2990],
                     ...,
                     [ 1.0539,  1.0714,  1.0714,  ...,  1.0539,  1.0539,  1.0888],
                     [ 2.2217,  2.2217,  2.2217,  ...,  2.2217,  2.2217,  2.2391],
                     [ 2.5877,  2.5877,  2.5877,  ...,  2.5877,  2.5703,  2.5703]]]],
                  device='cuda:0'),
      TensorImage([[[[-0.1999, -0.3541, -0.4739,  ..., -0.6281,  0.2111,  0.2624],
                     [-0.2171, -0.3027, -0.4054,  ..., -0.4739, -0.0287,  0.1254],
                     [-0.1999, -0.2513, -0.3198,  ..., -0.3712, -0.4054,  0.0912],
                     ...,
                     [ 0.0227, -0.2856, -0.6109,  ..., -0.2513, -0.2171, -0.2342],
                     [ 0.0569, -0.2171, -0.5938,  ..., -0.2684, -0.2856, -0.2171],
                     [ 0.0741, -0.1486, -0.5253,  ..., -0.2342, -0.2513, -0.2342]],
      
                    [[-0.8803, -1.0028, -1.0903,  ..., -1.0728, -0.4426, -0.5301],
                     [-0.8452, -0.9328, -1.0028,  ..., -0.9328, -0.6352, -0.6527],
                     [-0.7752, -0.8627, -0.9328,  ..., -1.0028, -0.9678, -0.6527],
                     ...,
                     [-0.6527, -0.8803, -1.1429,  ..., -0.1975, -0.1625, -0.2150],
                     [-0.7227, -0.8627, -1.0903,  ..., -0.2325, -0.2325, -0.2150],
                     [-0.7227, -0.8803, -1.0203,  ..., -0.2150, -0.1975, -0.2500]],
      
                    [[-1.0201, -1.1073, -1.1596,  ..., -1.3164, -1.0376, -1.1247],
                     [-1.0027, -1.0550, -1.1421,  ..., -1.3861, -1.2641, -1.3513],
                     [-0.9156, -1.0376, -1.1073,  ..., -1.3861, -1.4907, -1.3687],
                     ...,
                     [-0.8284, -1.0201, -1.2293,  ..., -0.0615, -0.0441, -0.1487],
                     [-0.7936, -1.0550, -1.2816,  ..., -0.0790, -0.0790, -0.0790],
                     [-0.7936, -1.0376, -1.2641,  ..., -0.0441, -0.0267, -0.0964]]]],
                  device='cuda:0')),
     TensorCategory([0], device='cuda:0'))




```python
learner.get_preds(dl=dls.valid)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    (TensorBase([[ 1.4939, -0.1861]]), TensorCategory([0]))



If we do it again, we'll get the same answer.


```python
dls = dblock.dataloaders(df, device='cuda')
```


```python
learner.get_preds(dl=dls.valid)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    (TensorBase([[ 1.4939, -0.1861]]), TensorCategory([0]))



However, this is still a super hacky method and not the right way to do it.

You can also make predictions without using a `DataBlock`, but I don't recommend this. For one, you'll have to be careful to prepare the data correctly. You'll need to make sure you do all the item and batch transforms. In addition, if you try to manually load the data into a `DataLoader`, you'll have to use the `ImageTuple` class from the same place in memory. That means you'll have to pull it out of your Jupyter Notebook and put it in a file and load in from there in both the notebooks. If you don't, you'll get an error like:

```python
AssertionError: Expected an input of type in 
  - <class 'pandas.core.series.Series'>
  - <class 'list'>
  - <class '__main__.ImageTuple'>
 but got <class '__main__.ImageTuple'>
```


```python

```
