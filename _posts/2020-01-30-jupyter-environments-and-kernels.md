---
layout: post
title: "Jupyter Environments and Kernels"
description: "A discussion of how to set up jupyter environments both inside Docker containers and outside"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/feral_pigeon.jpg"
tags: [Jupyter Notebooks]
---

The post aims to show how to create Jupyter environments and how to debug any issues. It also provides some commands that are good for general debugging.


<b>Table of contents</b>
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

The response on Linux should say something like this:

`/opt/conda/envs/my_env/bin/python`

On a Mac it will look like this:

'/Users/julius/opt/anaconda3/envs/my_env/bin/python'

And here's Windows:

`'C:\\Users\\Julius\\anaconda3\\envs\\tf\\python.exe'`

#### What kernels are available to Jupyter?

`jupyter kernelspec list`

Mac:

![png]({{site.baseurl}}/assets/img/jupyter_kernelspec_mac.png)

Windows:

![png]({{site.baseurl}}/assets/img/kernels2.png)

#### Where are my kernels located?

The exact location may vary, but for Mac/Linux users, it should look something like this:

`ls /root/.local/share/jupyter/kernels/`

Here's what you might see on Windows:

![png]({{site.baseurl}}/assets/img/windows_kernels.png)

You should also be aware that different kernels will have different paths. For example, `import my_package` may work in one kernel but not in another. Check `sys.path` to see which paths are being called.

Keep in mind that you can run all of the commands either from a terminal or from a Jupyter notebook. To run a Unix command from Jupyter, you simply need to add a `!` before it. So you can run `!conda info --envs` or `!jupyter kernelspec list` from a notebook.

![png]({{site.baseurl}}/assets/img/jupyter_commands.png)

OK, now let's talk about some examples where you might need these.

#### Adding Kernels

You'll need to be in the envrionment that you launch Jupyter for this to work. Once you're there, you can add kernels like so:

```
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
python -m ipykernel install --user --name pt --display-name "PyTorch"
python -m ipykernel install --user --name fai --display-name "FastAI"
python -m ipykernel install --user --name tf2 --display-name "TF2"
```

Now, if you reload your Jupyter Notebook page, your new kernels should be available (you shouldn't need to restart your Jupyter Notebook server).


#### Removing Kernels

You can remove a kernel like so:

`jupyter kernelspec uninstall my_kernel`

## Environment not showing up in Jupyter

Let's say you have three conda environments: `py2`, `py3_env1` and `py3_env2`. You activate `py3_env1` and then started a Jupyter Notebook in your environment. You start a notebook and everything looks right. But when you try to import packages that you *know* are in `py3_env1`, they are missing. Confused, you check which environment you're in.

`!conda env list`

    # conda environments:
    #
    base                     /opt/conda
    py2                      /opt/conda/envs/py2
    env1                  *  /opt/conda/envs/py3_env1
    env2                     /opt/conda/envs/py3_env2

This looks fine. So you check to see what interpreter you're using

    import sys
    print(sys.executable)

it says

`/opt/conda/envs/py3_env2/bin/python`

What? Why is it using the `py3_env2` Python interpreter? The answer is in the available kernels. The way to solve this depends on how exactly you got here. First, we'll look at it assuming this is inside a Docker container that you created.

#### Docker Containers

OK, let's assume you're in a Docker container and did something like this:

    # Create the environments
    RUN conda env create -f py2.yaml
    RUN conda env create -f py3_env1.yaml
    RUN conda env create -f py3_env2.yaml
    
    # Install the IPython kernel
    RUN /bin/bash -c "conda init bash && source /root/.bashrc && conda activate env1 && conda install -y notebook ipykernel && ipython kernel install --user && conda deactivate"
    RUN /bin/bash -c "conda init bash && source /root/.bashrc && conda activate env1 && conda install -y notebook ipykernel && ipython kernel install --user && conda deactivate"
    RUN /bin/bash -c "conda init bash && source /root/.bashrc && conda activate env2 && conda install -y notebook ipykernel && ipython kernel install --user && conda deactivate"

OK, let's jump back to debugging. The next thing you'll need to do is look at your Jupyter kernels:

`cd /root/.local/share/jupyter/kernels/`

`ls`

`python2  python3`

Two kernels. You have one for Python 2 and one for Python 3. When you look inside...

`cat python3/kernel.json`

You see

```
{
 "argv": [
  "/opt/conda/envs/py3_env2/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python 3",
 "language": "python"
}
```

You have two environments sharing the Python 3 kernel. Why is the python 3 kernel only using the `py3_env2` environment?


You can now change the kernel by selecting "Kernel" in the file menu, then "Change kernel".


Install the IPython kernel:

`pip install --user ipykernel`


There is a `kernel.json` file

If you don't get your env a name it will overwrite this

What happened is that the py3_enf2 kernel overwrote the py3_env1 kernel. 


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

If you're in your PT environment but that's not the Python distribution you're seeing when you `sys.executable`, you'll need to edit you `kernel.json` file that you found with `jupyter kernelspec list`.
