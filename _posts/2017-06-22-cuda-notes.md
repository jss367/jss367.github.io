---
layout: post
title: "CUDA Notes"
feature-img: "assets/img/rainbow.jpg"
tags: [CUDA, TensorFlow, PyTorch]
---


## Specify Which GPU to Use

Just like in TensorFlow, you can specify which GPU to use. If you're going to do this from the command line, you can do:

```bash
CUDA_VISIBLE_DEVICES="0" python -m my_trainer
```

Or you could do this within Python. If you do, be sure to do this before you import TensorFlow/PyTorch.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```


## Test if Tensorflow is working on the GPU

You can see all your physical devices like so:
``` python
import tensorflow as tf
tf.config.experimental.list_physical_devices()
```
and you can limit them to the GPU:
``` python
tf.config.experimental.list_physical_devices('GPU')
```
``` python
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```


## Accessing the GPU in PyTorch


```python
torch.cuda.current_device()
```




    0



How many are available?


```python
torch.cuda.device_count()
```




    1



What's the name of the GPU I'm using?


```python
torch.cuda.get_device_name(0)
```




    'NVIDIA GeForce GTX 960'



Is a GPU available?


```python
torch.cuda.is_available()
```




    True



How much memory is being used?


```python
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
```

    Allocated: 0.0 GB
    
