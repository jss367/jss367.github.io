---
layout: post
title: "Jupyter Environments and Kernels in Windows"
description: "A discussion of how to set up Jupyter environments both inside Docker containers and outside"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/feral_pigeon.jpg"
tags: [Jupyter Notebooks]
---

The post aims to show how to create Jupyter environments and how to debug any issues. It also provides some commands that are good for general debugging.


<b>Table of Contents</b>
* TOC
{:toc}

## Debugging Commands

Here are some commands I've found useful when debugging issues with my Jupyter environment. You'll see them used in the examples below.

#### Conda environments and Jupyter kernels

Two things you'll want to know when debugging Jupyter environment problems are what conda environments you have and what kernels are available to Jupyter. You can find your environments with either

`conda env list` or `conda info --envs`

They provide the same information.

![png]({{site.baseurl}}/assets/img/windows_conda_envs.png)

#### What Python interpreter am I using?

In Python:

``` python
import sys
sys.executable
```

`C:\\Users\\Julius\\anaconda3\\envs\\tf\\python.exe`

#### What kernels are available to Jupyter?

`jupyter kernelspec list`

![png]({{site.baseurl}}/assets/img/kernels2.png)

#### Where are my kernels located?

![png]({{site.baseurl}}/assets/img/windows_kernels.png)

You should also be aware that different kernels will have different paths. For example, `import my_package` may work in one kernel but not in another. Check `sys.path` to see which paths are being called.

Keep in mind that you can run all of the commands either from a terminal or from a Jupyter notebook. To run a Unix command from Jupyter, you simply need to add a `!` before it. So you can run `!conda info --envs` or `!jupyter kernelspec list` from a notebook.

![png]({{site.baseurl}}/assets/img/jupyter_commands.png)

OK, now let's talk about some examples where you might need these.

#### What is the active kernel?

Here's the only way I've found how to do this ([source](https://stackoverflow.com/questions/43759543/how-to-get-active-kernel-name-in-jupyter-notebook)):
```
%%javascript
var kernel = Jupyter.notebook.kernel
kernel.execute('kernel_name = ' + '"' + kernel.name + '"')
```

```
print(kernel_name)
# my-kernel
```

```
from jupyter_client import kernelspec
spec = kernelspec.get_kernel_spec(kernel_name)
print(spec.resource_dir)
```

#### Removing Kernels

You can remove a kernel like so:

`jupyter kernelspec uninstall my_kernel`

## Windows

For Windows users, you can use the Anaconda Prompt

`jupyter kernelspec list`

Should see something like `C:\Users\Julius\anaconda3\share\jupyter\kernels\python3`

`cd` there

You should see a `kernel.json` file. You can look inside it with `type kernel.json` (`type` is the Windows version of the Unix command `cat`)

```
{
 "argv": [
  "C:/Users/Julius/anaconda3\\python.exe",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python 3",
 "language": "python"
}
```

Now you can [add the kernel as shown above](https://jss367.github.io/jupyter-environments-and-kernels.html#adding-kernels).

## Python and environment not matching

If you're in your PT environment but that's not the Python distribution you're seeing when you `sys.executable`, you'll need to edit your `kernel.json` file that you found with `jupyter kernelspec list`.
