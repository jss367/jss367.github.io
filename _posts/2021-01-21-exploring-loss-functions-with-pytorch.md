---
layout: post
title: "Exploring Loss Functions with PyTorch"
description: "A guide to loss functions and how to implement them in PyTorch"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/frogmouth_staring.jpg"
tags: [Neural Networks, Python, PyTorch]
---

In this blog post, I will discuss how to use loss functions in PyTorch. I will cover how loss functions work in both regression and classification tasks, how to work with numpy arrays, the expected shape and type of loss functions in PyTorch, and demonstrate some types of losses.


```python
import numpy as np
import torch
from torch import nn
```

<b>Table of Contents</b>
* TOC
{:toc}

# Working with PyTorch

I wanted to start with a few notes on loss functions in PyTorch. We'll talk about things that people switching from TensorFlow should be aware of.

## Requires_grad

It's important to understand the role of `requires_grad` in loss functions and gradient computation. Our goal is to find how the loss changes with each parameter, i.e., $$\frac{\partial loss}{\partial x}$$ for each parameter x.

`requires_grad` is a flag that can be set on a tensor to indicate that gradients need to be computed for this tensor during the backward pass. When you create a tensor with `requires_grad=True`, PyTorch will track all the operations performed on that tensor and store the gradients when the backward pass is called.

Here's an example demonstrating the use of requires_grad:


```python
y_pred = torch.randn(3, 5, requires_grad=True)
y_true = torch.randn(3, 5)

mae_loss = nn.L1Loss()
output = mae_loss(y_pred, y_true)
print(output)

# Compute gradients
output.backward()

print("Gradients:", y_pred.grad)

```

    tensor(1.6688, grad_fn=<L1LossBackward>)
    Gradients: tensor([[ 0.0667, -0.0667, -0.0667,  0.0667, -0.0667],
            [-0.0667, -0.0667, -0.0667, -0.0667, -0.0667],
            [-0.0667,  0.0667, -0.0667, -0.0667,  0.0667]])
    

In this example, we create an input tensor `y_pred` with `requires_grad=True` and a target tensor `y_true`. We compute the L1 loss between the two tensors and call the `backward()` method on the output tensor to compute gradients for `y_pred`.

In this notebook, I'll be focused on calculating the loss values and will skip the `requires_grad` flag and the `backward()` method. But if we were training a model, we would require gradients for optimization and include the `requires_grad=True` flag.

## Numpy Arrays

PyTorch primarily works with tensors, but it provides easy interoperability with numpy arrays. You can convert a numpy array to a PyTorch tensor using `torch.from_numpy()` and convert a tensor back to a numpy array using the `.numpy()` method. It is important to note that PyTorch expects input tensors to be of type float and target tensors to be of type long for classification tasks.

This means that you can't directly put numpy arrays in a loss function. PyTorch losses rely on being able to call a `.size()` method, which doesn't exist for numpy arrays.


```python
y_pred_np = np.random.randn(3, 5)
y_true_np = np.random.randn(3, 5)

# Using PyTorch loss function directly with numpy arrays (will raise an error)
loss = nn.L1Loss()
try:
    output = loss(y_pred_np, y_true_np)
except TypeError:
    print("TypeError: PyTorch loss functions expect tensors, not numpy arrays")

# Converting numpy arrays to PyTorch tensors
y_pred = torch.from_numpy(y_pred_np)
y_true = torch.from_numpy(y_true_np)

# Now, we can calculate the loss
output = loss(y_pred, y_true)
print("L1 Loss:", output.item())
```

    TypeError: PyTorch loss functions expect tensors, not numpy arrays
    L1 Loss: 1.0592096183606834
    

OK, let's get to the loss functions.

# Loss Functions

## Inputs

Your inputs are going to look different based on the task. For a regression task, you'll generally have `y_pred` and `y_true` tensors that are the same size. But for image classification tasks, you'll have a prediction probability associated with each class, so your `y_pred` tensor will have an extra dimension of size `N` where `N` is the number of possible classes.

### Regression

For regression, the most commons losses are L1 and L2. Let's take a look at them.

##### L1 and L2 Loss


```python
input_tensor = torch.tensor([1, 2.5, 4, 0.5])
target_tensor = torch.tensor([2, 2.5, 2, 1])
 
l1_loss = nn.L1Loss()
l1_output = l1_loss(input_tensor, target_tensor)
print("L1 (Mean Absolute Error) Loss:", l1_output.item())

l2_loss = nn.MSELoss()
l2_output = l2_loss(input_tensor, target_tensor)
print("L2 (Mean Squared Error) Loss:", l2_output.item())

```

    L1 (Mean Absolute Error) Loss: 0.875
    L2 (Mean Squared Error) Loss: 1.3125
    

### Image Classification

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
input_tensor = torch.randn(5, 3)
target_tensor = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)

# Softmax to convert input tensor to probabilities
softmax = nn.Softmax(dim=1)
input_prob = softmax(input_tensor)

# Negative Log Loss (Negative Log Likelihood)
nll_loss = nn.NLLLoss()
nll_output = nll_loss(torch.log(input_prob), target_tensor)
print("Negative Log Loss:", nll_output.item())
```

    Negative Log Loss: 0.952965259552002
    

Note that you can also use `LogSoftmax`.


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
nll_loss(softmax(y_pred), y_true)
```




    tensor(1.6554)



Note that it's important that we remember to do a softmax first. If we don't, we won't get an error, we'll just get the wrong answer.


```python
nll_loss(y_pred, y_true)
```




    tensor(-0.4325)


