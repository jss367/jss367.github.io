---
layout: post
title: "Useful Git Commands"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/lake_sunrise.jpg"
tags: [Git, Cheat Sheet]
---


This post shows some of the most common Git commands. I find it very useful to have a quick reference with all of the most common tasks.


<b>Table of contents</b>
* TOC
{:toc}

## Using Git

A lot of people get frustrated with git (myself included) but I hope to lessen that with these tips. First, let's separate git from GitHub. Git can exist entirely without GitHub.


There are three primary things that people should know about git:
1. For the most part, **git is a local tool**. That means most of what happens in git happen only on your computer.

Many of the most common commands are completely local, such as: `status`, `commit`, `checkout`, `merge`, `rebase`, `diff`, and `log`. The *only* three that are remote are `push`, `pull`, and `fetch`.

Yes, that's right. When you `git status` or `git commit -m 'bug fix'`, nothing actually happens over the Internet. It's all local to your machine. The entire git history of your project is stored locally. You actually have a clone of the remote repositories on your local machine, so you don't need an Internet connection for most of the uses. If a Github repo is your remote, you have a clone of it locally. That means that to talk to `origin/master` you don't actually need an Internet connection - you're talking to your local clone of the remote respositiory. The downside here is that if `git status` say you are up to date, it means you are up to date with your local clone of the remote master, but not necessarily the actual master.

The good news is you can do so much of your workflow even without an Internet connection. You still still make git commits, change branches, make some more commits, merge a branch, and the check the git status at the end. The only thing you can't do is push those changes to GitHub.

`2`. There *is* excellent documentation, although most people skip it
I recommend trying to understand how git is working bit by bit, because the better you understand it the less frustrated you will be (in general). But even if you don't have time to read the whole [official Git book](https://git-scm.com/book/en/v2) (which I admit I haven't read completely either), here are some helpful commands to get basic usage out of git. (Maybe read chapter 2)

`3`. You can get really far with a few simple commands

## A Basic commit

### Staging Files

The commit process consists of a few easy steps. First, you have to "stage the commit" by telling git which files you want to commit. You do this by adding the files to the commit, and you have a lot of flexibility over how you do this. When you're editing a code base, you'll do some combination of adding files, changing existing files, and deleting files. Git gives you flexibility in which combinations of these you want to commit. You'll have to commit the changed files but you can choose whether or not to include the added or deleted files. Here are three options for staging your files:

1. `git add -A` - stage all files (added, changed, and deleted)

2. `git add .` - stage added and changed files only, not deleted

3. `git add -u` - stage changed and deleted files only, without new

### Commiting files

Now that the files are staged, you can create a commit. When you make a commit, you are expected to include a message about what your commit does. Here is how you do that:

`git commit -m "async bug fix"`

### Pushing to origin

You've made your commit, so the next step is completely optional. As I said, `git` can work entirely locally, and you've make a local commit. But many people (myself included) use GitHub to maintain their `git` repositories. To push to your remote respository, you can do this:

`git push origin master`

### Pulling from origin

Now let's say you want these changes pulled to another computer. The way to do that is:

`git pull origin master`

### Temporarily look at older commit (but don't plan to make changes)

You can just `git checkout <last seven characters or commit>`

This will put you in a detached HEAD state.

Then to get back you can `git checkout master` (or whatever branch you were on)

## Other Useful Commands

### Check out a remote branch

To checkout a branch you'll run `git checkout my_remote_branch`, but if the branch is remote only then you'll need to fetch it first:

`git fetch`
Then
`git checkout my_remote_branch`

### Undoing changes (i.e. returning to the previous commit)

`git checkout -- flask_app.py`

Or, if you want to revert everything: `git checkout .`

### I commited a file then added it to .gitignore - now it won't go away

I do this one all the time. It's like to need to commit at least one file before I remember to add it to .gitignore. What you need is:

`git rm --cached my_file_that_shouldnt_be_in_git`

#### But I did this to a bunch of files

First, commit any changes so that when you run `git status` everything is clean. Then, very carefully:

`git rm -r --cached .`

You've removed all your changed files, so now you need to add them again:

`git add .`

Then commit:

`git commit -m 'removed .gitignore files'`

### Staging and commiting all in one

If you want to combine the stage and commit steps into one you can:

`git commit -am "all files"`

Although this won't add the new files.

### Syncing your local with the remote branch

`git fetch origin`

### Seeing recent activity

`git log` shows recent commits, including the author and date.

### Finding changes

`git diff` is the command you'll need to find changes. There are different ways to use `git diff`.
1. Show changes you haven't committed yet: `git diff [filename]`
2. Show changes you already committed (but haven't sync'd): `git diff --cached [filename]`

### Syncing git with remote

If you want to sync your git with the remote one (like when you've added a branch): `git remote update`

### Working with branches

You can create a new branch on your local machine. Let's say your new branch is called database

`git checkout database`

You can make changes, break stuff, then switch back to your main branch at any time:

`git checkout master`

To see all your git branches:

What branch are you on: `git branch`

What branches are there: `git branch -a`

When you want to merge a branch back into it's master:

```
git checkout <master>
git pull origin <master>

git merge <branch>

git push origin <master>
```

If you want, you can delete the branch you just merged: `git branch -d <mergedbranch>`

### I just changed a branch, why did the old file changes come with me

### Syncing with upsteam branch

If you've created a branch and since then the master has been updated, you may get a message like: "This branch is X commits ahead, Y commits behind [masterbranchname]:master." Here are the steps to resolve this:

1. First, make sure you're upstream repo exists. To do this type:

`git remote add upstream git://github.com/barryclark/jekyll-now.git`

If this already exists, you'll get a message saying: "fatal: remote upstream already exists." That's OK. It just means you didn't need to do that step.

 1. Then type: `git fetch upstream`

 1. Then type: `git pull upstream master`

If both branches have editted the same file, you may get a conflict. If you do, follow the steps in [this article](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/).

### What are those weird strings of characters?

Those are SHA-1 hashs. They consistent of 40 characters although git often just shows the ones at the end.

### What can I do with my .git folder?

You should never touch that folder. Let git do its thing.

### How can I see my configuration?

`git config --list`

If your configuration is missing, you'll probably need to add your username and email. That should look something like this:

```
git config --global user.name <username>
git config --global user.email <email address>
```

This has your username and email, along with any other settings

### I just want to copy my changes then 

### I want to go to an older commit...

There are many ways to return to an older commit. In all of these options, you can either use HEAD~1 or the commit hash

#### just to take a look

`git checkout HEAD~1`

#### permanently

`git reset HEAD~1`

#### just to see the differences

`git diff HEAD~1`

#### take a look and maybe mess around with it, possibly keep some of the code

The way to do that is to make a new branch. Fortunately, git makes this easy with `git checkout -b my_branch_of_old_state HEAD~1`

## Help! I did something wrong and want to undo it! I...

Unfortunately, git is solely missing the `git undo` command which would undo whatever the last thing you did was. Instead, the way to undo a command depends on whatever the last thing you did was.

### added a file that I didn't mean to

If this is all you've done, it's an easy fix. All you have to do is `git reset my_file`. If you want to undo adding all files, you can `git reset`.

### made a commit that I shouldn't have
Warning: Do not do this if you have already pushed the commit to a central repo. Revert is much better in this case. But if you'll just made a local commit then realized you don't want to, you can:

#### I committed too early and want to undo the commit but not lose my work

`git reset --hard HEAD~1`

#### I committed and want to completely remove what I wrote

`git reset HEAD~1`

#### Everything I typed was fine but I want to go back to the second before I committed

`git reset --soft HEAD~1`

### made a commit and pushed it to remote

`git revert <bad-commit-sha1-id>` then `git push origin`

### have been committing to master but I should have committed to a branch

To fix this, you'll need to take your commits to the correct branch, and then undo the commits to master. First, take the commits to a new branch:

`git checkout my_branch_to_put_changes

git merge master`

Now undo those changes on master:

`git checkout master

git reset --keep HEAD~1
`
### Help! Everything is so messed up I want to delete everything and download a fresh copy

OK, [we've all been there before]((https://xkcd.com/1597/)). While that does work, you might be able to fix everything without doing that. First, you'll want to grab the latest version of your project without merging anything:

`git fetch --all`

Then reset to what you just fetched:

`git reset --hard origin/master`

You should be back with a clean copy of the most recent version of the repo.

You can also, however, save your local commits in a branch before doing this

`git checkout master`

`git branch my_branch_with_local_commits`

`git fetch --all`

`git reset --hard origin/master`

### How do I delete a branch?

If you want to delete a branch completely you will need to delete it both locally and remotely. The easiest way, by far, is to use the GitHub Desktop app. You just go to the branch you want to delete, then go to the Branch menu -> Delete -> Check checkbox to for "Yes, delete this branch on the remote" If that's not an option you'll need to do both independently like so:

#### Delete local branch

`git branch -D my_branch`

#### Delete remote branch

`git push origin --delete my_branch`

### Remove untracked files from working tree

You'll need to use `git clean`. There are a lot of options with `git clean`. For example, to see what files that would delete you can `git clean -n`. Note that `git clean` works in the folder you are in and the subfolders, so if you want to clean the entire repository you'll need to move to the root folder first.

Then to remove the files:

`git clean -f`

If there are files or directories still hanging around (like maybe they're being gitignored), you can be sure to delete them with:

`git clean -fdx`

## Git concepts to know

### Git States

Your files are always in one of three states with git: modified, staged, and committed

* Modified means you have changed it
* Staged means it will be commited during your next commit
* Commited means it is stored in your (local) database, ready to be modified again

### HEAD

HEAD is a reference to the last commit in the currently checked-out branch.

### Origin vs Remote

### Working Tree

### Working copy

### Index

## Using gitignore

You can create a .gitignore file automatically, and I recommend you do this.

Note that (at least on Windows) the direction of the slash matters. If you want to ignore all your VSCode files, `.vscode/*` will work, but `.vscode\*` won't.

## GUIs

While no GUI has all the features of the command line, they are still very helpful. The one I recommend is the [GitHub Desktop](https://desktop.github.com/) app. It's easy to use and offers support for Mac and Windows. I do wish they would add Linux to that list. But it brings up the problem with GUIs - you won't always have access to them, so don't rely on them. Say you're working through an SSH tunnel on some remote server. You'll often be in the case where you **can't** use a GUI, so I still recommend learning the command line. But the GUI definitely helps to make the initial steps more gentle. I still use the GUI when possible because I think the graphical inferface makes viewing the differences much easier.


## Problems with git

git blame - good tool  be should be renamed. Maybe `git credit`? It's a bit funny but in the end I don't think it's a good idea. I imagine it could have been done in jest but I think given the importance of git today and the difficulties of software engineering, not to mention imposter syndrome, it's not the right name.

## Configuring git

Supposed you wanted to add something that combined a few git commands, like `git pull` along with rebasing and autostashing. You could do something like this:

`git config --global alias.all 'pull --rebase --autostash'`

Now you can do `git all` to call this.