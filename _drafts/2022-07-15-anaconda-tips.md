---
layout: post
title: "Anaconda Tips"
description: "This post shows tips and tricks for working with Anaconda, especially if you are confused about the path."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/tiger_snake.jpg"
tags: [Git]
---

If you're struggling to understand what's going on with your Anaconda environment, this post has some hopefully helpful tips. The way Anaconda works with paths can be confusing at first and depends on your operating system, so let's examine that. First, `PATH` is an environment variable of a list of directories that contain executables. So when you execute a program it runs through all the directories in your `PATH` and executes the first one with the name you provided.


## Where is it installed?

The installation location will depend on your operating system and how it's installed. You will probably have a directory called `Anaconda3` somewhere. (Note that if you used this will be different for Miniconda.)


### Windows

It will be somewhere like:

`C:\Users\<USERNAME>\Anaconda3`

or 

`C:\Users\<USERNAME>\AppData\Local\Anaconda3`

### Mac

It will be somewhere like:

`/Users/<USERNAME>/Anaconda3`

If you install it with Homebrew:

`/opt/homebrew/anaconda3`

Also, might be in:

`/opt/anaconda3`

Sometimes conda will be in:

`/Users/<USERNAME>/opt/anaconda3`

in which case your executables will be here:

```
/Users/<USERNAME>/opt/anaconda3/bin/conda
/Users/<USERNAME>/opt/anaconda3/condabin/conda
```

### Linux

For linux, it's similar to mac except that instead of `/Users/<USERNAME>/Anaconda3` it will be `/home/<USERNAME>/Anaconda3`.

## Finding it

If you can execute `conda`, you can find where it is.

### Windows:

`where conda`

### Unix:

`which conda`

## Adding to Path

Sometimes Anaconda is placed in your path, sometimes not. There are pros and cons to putting it in your path. See [this SO answer](https://stackoverflow.com/questions/52664293/why-or-why-not-add-anaconda-to-path) for more. If you choose to, I recommend you place it in the front of your path so it picks up any calls to `python`.

### Mac

`export PATH="/Users/username/miniconda3/bin:$PATH"`

### Linux

`export PATH=/home/<USERNAME>/anaconda3/bin:$PATH`

### Windows

In Windows, you'll have to open up your environment variables and edit it from there.
