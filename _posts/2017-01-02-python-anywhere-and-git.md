---
layout: post
title: "Python Anywhere and Git"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/london_bridge.jpg"
tags: [Git]
---
PythonAnywhere has an excellent [tutorial for setting up a website with Flask](https://blog.pythonanywhere.com/121/). The tutorial shows how to set up a website and put it under version control with Git. But it only shows how to do that by initiating a new repository; it doesn't show you how to connect with an existing repo. In this post, I will show how to connect Pythonanywhere to an existing Github account and some basic Git commands to manage your PythonAnywhere app.<!--more-->


## Starting a new project

1. Go to [Github.com](https://github.com/) and login

1. Click on "New repository" on the right or open an existing repository if you already have one you want to work with

1. If it's new, give it a title and click "Create repository"

1. Find the URL for the remote repository by clicking on the green "Clone or download" button, then click the "Copy to Clipboard" button to copy the URL. It should end with .git. Something like: `https://github.com/jss367/pythonanywhere.git`

1. Then go to your bash console on PythonAnywhere inside the folder you want to connect to GitHub. This is possibly your `mysite` folder. You'll want to initiate it as a git repository. Enter `git init`

1. Type: `git remote add origin <remote_repository_URL>`. Your remote_repository_URL is the thing we copied above that looks like `https://github.com/jss367/pythonanywhere.git`

1. Then test out that the connection works. Type: `git remote -v`. You should get a response verifying the remote URL. It should look something like this:

    ```bash
    origin  https://github.com/jss367/pythonanywhere.git (fetch)
    origin  https://github.com/jss367/pythonanywhere.git (push)
    ```

1. Then, you have to pull the remote repository to your local PythonAnywhere folder. Type: `git pull origin master`. This is one of the most common Git commands you will use. It is of the format `git pull <remote> <branch>`.

1. Now we'll want to make a `.gitignore` file. This is a file that contains all the stuff within your local git project that you don't want to commit with your project. Python cache files are a good example of a file that you don't want to commit each time. To create a `.gitignore` file, you can create one by typing `touch .gitignore`. Files that start with a `.` are normally hidden from the user. This is often how people store configuration files that they don't want to clutter up the users space with.

1. OK, let's add some things to it. We'll do that using [vim](https://www.vim.org/). Open the file for editing by typing `vim .gitignore`. vim is a great tool but it can be very difficult for beginners, so if you're not familiar with it try to type exactly and only what is written and if you get into a weird state you can always google for more instructions.

1. First you'll type `i` to go into INSERT mode.

1. Once you're in insert mode, type the following:
    ```bash
    *.pyc
    __pycache__
    ```
1. To save your work and escape from vim, you'll need to do a couple things. First, exit INSERT mode by hitting the `Escape` key. Then type in `:wq` to write your file and quit the application. Don't forget the colon at the beginning.

1. OK, you should be returned to the main console. To check on your work enter `cat .gitignore`. Look for the files you added.

1. The next thing you'll need to do is configure git with your information. Enter the following:
    ```
    git config --global user.email "Your Email"
    git config --global user.name "Your Name"
    ```
    Note that your `user.name` is NOT the same as your GitHub user name. My GitHub username is jss367, but my user.name is "Julius Simonelli".

1. To double check your changes, you can enter `git config --list`. You'll also see some other configurations that you didn't add, which is OK.

1. Next, enter `git status` to see what the status of your repo is. You should see both the `.gitignore` file and the `flask_app.py` listed under "Untracked files". You'll need to stage them both for a commit by entering `git add .`. By using a `git add .` you'll automatically add every file without typing them individually.

1. I like to check my `git status` frequently, just to make sure that I am exactly where I think I am. If you check `git status` again, it should say "Changes to be committed:" and then list .gitignore and flask_app.py as new files in green.

1. Then commit your new files with `git commit -m "Add initial flask app and .gitignore files"`

1. Finally, you'll push your updated repo to the remote repository. To do this, enter `git push origin master`. You may have to enter your GitHub username and password if you haven't already. This is your GitHub username, not your git user.name (I know, it's confusing).

#### Create a web application

The next thing you'll need to do is create a web application on PythonAnywhere to connect to your repo. PythonAnywhere makes this easy and it's addressed in the [tutorial on PythonAnywhere's website](https://blog.pythonanywhere.com/121/), so I won't go into it further. PythonAnywhere will create a WSGI (pronounced "whiz-gee" kind of like "whiskey") file for you. Just for reference, here's an example of what your WSGI file (`jss367_pythonanywhere_com_wsgi.py`) should look like:

```python
import sys
#
## The "/home/jss367" below specifies your home
## directory -- the rest should be the directory you uploaded your Flask
## code to underneath the home directory.  So if you just ran
## "git clone git@github.com/myusername/myproject.git"
## ...or uploaded files to the directory "myproject", then you should
## specify "/home/jss367/myproject"
path = '/home/jss367/pythonanywhere'
if path not in sys.path:
    sys.path.append(path)

from flask_app import app as application
```
