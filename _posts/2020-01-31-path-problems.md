---
layout: post
title: "Path Problems"
description: "A guide to some of the path problems you may face on various operating systems"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/dark_path.jpg"
tags: [Linux, Python, Windows]
---

Path problems are some of the most common and annoying problems machine learning engineers face, especially when frequently switching between operating systems. There are so many different issues ways to have path problems that no post could cover them all, but in this post, I'll try to provide some background on possible issues and how to resolve them.

<b>Table of Contents</b>
* TOC
{:toc}

The first thing you need to realize are that there are multiple different things that are at times called a "path". The two main ones that Python programmers will run into are the system path and the Python path. 

# System Path

The first step is being able to find out what's on your path. You can do this by looking at the `$PATH` environmental variable. You can do this from the command line, but the exact command depends on which operation system you're using.

## Viewing Your Path

#### Unix Systems

If you're using either a Mac or Linux, including Windows Subsystem for Linux, it's as simple as:

`echo $PATH`

The result might be a little hard to read, so if you want a more readable version you can use:

`echo "${PATH//:/$'\n'}"`

#### Windows

In Windows, it's not as simple to view your environmental variables because it depends on what terminal emulator you're using. If you are using [ConEmu](https://conemu.github.io/) or [Cmder](https://cmder.net/), you can

`echo %PATH%`

![png]({{site.baseurl}}/assets/img/windows_path.png)

Again, the resulting text can be hard to read so to make it easier you can: `echo %PATH:;=&echo.%`

![png]({{site.baseurl}}/assets/img/windows_path_simple.png)

However, if you're using [Windows PowerShell](https://docs.microsoft.com/en-us/powershell/scripting/overview), you'll have to:

`echo $Env:PATH`

Windows PowerShell is the default terminal in VSCode, so this is what you'll need to use there as well.

## Adding to Your Path

#### Unix

If you want to temporarily add to your path, type the following in the terminal:

`export PATH=$PATH:/path/to/directory`

If you want to permanently add to your path, simply take that command and place it in your `.bashrc` file (or wherever you keep your environmental variables - [I recommend `.profile`](https://jss367.github.io/shell-and-environment-setup.html))


#### Windows

If you want to temporarily add to your path:
```
set PATH="%PATH%;C:\path\to\directory\"
```

If you want to permanently add to your path:
`setx path "%PATH%;C:\path\to\directory\"`

# Python Path

If you can't import a module you wrote, it's probably because that location is missing from your path. But it's your *Python* path, not your system path, that it's missing from.

```
import my_module
ModuleNotFoundError: No module named 'my_module'
```

## Viewing your Python Path

#### Unix

`echo $PYTHONPATH`

#### Windows

On Windows, if you echo `$PYTHONPATH` you get nothing, but you can use Python to see what's in your path

#### Python

When inside Python you can `print(sys.path)` to see your path:
```
['C:\\Users\\Julius\\Documents\\GitHub\\ResNetFromScratch', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\python37.zip', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\DLLs', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib', 'C:\\Users\\Julius\\anaconda3\\envs\\tf', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\win32', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\win32\\lib', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\Pythonwin']
```


## Adding to your PYTHONPATH
```
Just go to system properties, environment variables, new...
Environment variables:
Hit windows key
type environment
Click on Environment Variables in the Advanced tab.
c:\\Users\\Julius\\Documents\\GitHub\\DataManager
- for some reason this doesn't always add to the python path though
```
Let's test this out. As we saw above, you can test out your PYTHONPATH directly from the command line:

`python -c "import sys; print(sys.path)"`


Remember that your paths are hierarchical on Windows

If you have `C:\Users\Julius\AppData\Local\Microsoft\WindowsApps` it will try then then tell you to go to the Windows store. But you can move the one you want up in the list.

You can have this problem where it works if you are debugging but not if you're not debugging.


add BOTH conda and python to your path

open conda prompt then `where conda`

then `where python`

should say
`C:\Users\Julius\anaconda3\python.exe`

so use `C:\Users\Julius\anaconda3`




I use the User variables for this



Right-click anywhere in the editor window and select Run Python File in Terminal (which saves the file automatically):

- doesn't use selected interpreter





On a brand new Mac your path is:

```
julius@Juliuss-MacBook-Pro ~ % echo $PATH
/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
```



on Windows can also

```
echo $PYTHONPATH
set PYTHONPATH=C:\\Users\Julius\Documents\GitHub\facv\src\;C:\\Users\Julius\Documents\GitHub\fastai\fastai
python -c "import sys; print(sys.path)"
```

but in Windows it's `echo $PYTHONPATH$`




Note:
If you change the environment variables you'll need to restart VSCode or Jupyter Notebooks. Once you restart you'll see the addition in `sys.path`


## Finding your Python Interpreter

From within Python, you can do:

``` python
import sys
sys.executable
```


## Jupyter Notebook problems

Let's say you set you can variable and can see it in `PYTHONPATH`.


you `import sys; print(sys.version)` from a command line and it's there, but when you run the same command in a Jupyter Notebook it's not there. That's because Jupyter has its own `PATH` variable called `JUPYTER_PATH`.

If you need to quickly add to it, you can start a notebook with:

``` python
import sys
sys.path.append('C:\\Users\\Julius\\Documents\\GitHub\\cv_dataclass\\src')
```

## Other

If you want to know more about how the Python `sys.path` is initialized, there's a [detailed StackOverflow answer](https://stackoverflow.com/questions/897792/where-is-pythons-sys-path-initialized-from).
