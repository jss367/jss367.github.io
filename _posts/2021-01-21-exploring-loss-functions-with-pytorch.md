---
layout: post
title: "Exploring Loss Functions with PyTorch"
description: "A guide to loss functions and how to implement them in PyTorch"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/frogmouth_staring.jpg"
tags: [Neural Networks, Python, PyTorch]
---

```python
import torch
from torch import nn
import numpy as np
```

How does one divide up loss functions? They could be divided by regression and classification loss functions.

# Notes

I wanted to start with a few notes on loss functions in PyTorch. We'll talk about 

## Requires_grad

Loss functions exist to help compute the gradients for trainable parameters. We want to find how the loss changes with each parameter, that is, $$\frac{\partial loss}{\partial x}$$ for each parameter $ x $.

So what you will see is this


```python
y_pred = torch.randn(3, 5, requires_grad=True)
y_true = torch.randn(3, 5)

mae_loss = nn.L1Loss()
output = mae_loss(y_pred, y_true)
print(output)
output.backward()
```

    tensor(1.1468, grad_fn=<L1LossBackward>)
    

For our purposes, we're just going to be stopping at creating the value, so we can skip `requires_grad` for some of the demo.

### Numpy

Note that you can't directly put numpy arrays in a loss function. PyTorch losses rely on being able to call a `.size()` method, which doesn't exist for numpy arrays.


```python
np.random.randn(3,5)
```




    array([[-0.12188827,  0.51116533, -0.23330246, -0.26176837,  0.35833209],
           [ 0.17633687,  0.10065291, -0.44017884, -1.42619676, -0.95035462],
           [-2.09363552,  0.85312672, -1.69011215,  0.03842335,  0.86324746]])




```python
y_pred = np.random.randn(3,5)
y_true = np.random.randn(3,5)
```


```python
loss = nn.L1Loss()
```


```python
try:
    output = loss(y_pred, y_true)
except TypeError:
    pass
```

If you have a numpy array, you'll need to convert it to a numpy tensor first.


```python
y_pred_np = np.array([0.1, 0.3, 0.5, 0.9])
y_pred_np
```




    array([0.1, 0.3, 0.5, 0.9])




```python
y_true_np = np.array([0, 1, 1, 1])
```


```python
y_pred = torch.from_numpy(y_pred_np)
y_pred
```




    tensor([0.1000, 0.3000, 0.5000, 0.9000], dtype=torch.float64)




```python
y_true = torch.from_numpy(y_true_np)
y_true
```




    tensor([0, 1, 1, 1], dtype=torch.int32)




```python
loss(y_pred, y_true)
```




    tensor(0.3500, dtype=torch.float64)



OK, let's get to the loss functions.

# Loss Functions

## Inputs

Your inputs are going to look different based on the task. For a regression task, you'll generally have `y_pred` and `y_true` tensors that are the same size. But for image classification tasks, you'll have a prediction probability associated with each class, so your `y_pred` tensor will have an extra dimension of size `N` where `N` is the number of possible classes.

### Regression

For regression, the most commons losses are L1 and L2. Let's take a look at them.

##### L1


```python
y_pred = torch.tensor([1, 2.5, 4, 0.5])
```


```python
y_pred
```




    tensor([1.0000, 2.5000, 4.0000, 0.5000])




```python
y_true = torch.tensor([2, 2.5, 2, 1])
```


```python
y_true
```




    tensor([2.0000, 2.5000, 2.0000, 1.0000])




```python
mae_loss = nn.L1Loss()
```


```python
mae_loss(y_pred, y_true)
```




    tensor(0.8750)



##### L2

The same can be done with L2 loss.


```python
mse_loss = nn.MSELoss()
```


```python
mse_loss(y_pred, y_true)
```




    tensor(1.3125)



#### Image Classification

Let's say we're predicting a batch of four items that are each one of three classes (0-2)


```python
y_true = torch.IntTensor(np.array([1,2,0,1]))
```


```python
y_true
```




    tensor([1, 2, 0, 1], dtype=torch.int32)




```python
y_true.shape
```




    torch.Size([4])



But your predictions will have an extra dimension in them. So they might look like


```python
y_pred = torch.FloatTensor(np.array([[0.1, 0.5,0.4,],
                                    [0.1, 0.2, 0.7],
                                    [0.3, 0.25, 0.55],
                                    [0.3, 0.4, 0.1]]))
```


```python
y_pred
```




    tensor([[0.1000, 0.5000, 0.4000],
            [0.1000, 0.2000, 0.7000],
            [0.3000, 0.2500, 0.5500],
            [0.3000, 0.4000, 0.1000]])



To get the actual value predicted, we have to take the argmax


```python
y_pred_labels = torch.argmax(y_pred, dim=1)
```


```python
y_pred_labels
```




    tensor([1, 2, 2, 1])



###### Negative Log Loss

NLL is used in multiclass classification problems. It uses a softmax


```python
nll = nn.NLLLoss()
```


```python
y_pred = torch.tensor([[0.5153, 0.7051, 0.4947, 0.3446,  0.5288],
        [0.3464, 0.2458, 0.8569, 0.4821, 0.3244],
        [0.4474, 0.6615,  0.0062,  0.6603, 0.2461]])
```


```python
y_true = torch.tensor([1, 0, 4])
```


```python
softmax = nn.LogSoftmax(dim=1)
```


```python
nll(softmax(y_pred), y_true)
```




    tensor(1.6554)



Note that it's important that we remember to do a softmax first. If we don't, we won't get an error, we'll just get the wrong answer.


```python
nll(y_pred, y_true)
```




    tensor(-0.4325)




```python

```
