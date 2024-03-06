---
layout: post
title: "Windows Setup"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/turtle.jpg"
tags: [Software]
---
This post contains details of how I set up my shell and environment. I use Windows, Mac, and Linux on a daily basis, so I have different setups for different purposes, but I try to make them similar when I can.

<b>Table of Contents</b>
* TOC
{:toc}

## Packages

There are a few packages I use to improve my terminal experience.

### Pygments

* [Pygments](https://pygments.org/), a Python syntax highlighter. It's like `cat` with colors. I alias it to `c` (as seen below).

### Autojump

* [autojump](https://github.com/wting/autojump)

## Bash

I still use Bash fairly often, and because it doesn't come with all the same aliases that Oh My Zsh does, I have to add some of the most important ones manually. I use all the ones I use for zsh but I add these as well.

```bash


if [ $SHELL = "/bin/bash" ]
then
	shopt -s histappend                      # append to history, don't overwrite it
	bind '"\e[A":history-search-backward'
	bind '"\e[B":history-search-forward'
fi

alias ..='cd ..'
alias ...='cd ../../../'
alias ....='cd ../../../../'
```



```bash


conda activate "$DEFAULT_CONDA_ENVIRONMENT"
 
#color git branch:
parse_git_branch() {
     git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}
export PS1="\u@\h \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]$ "


# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi
```


I also like to customize the git prompt if it's not already done for me. Here's one I like:

```
parse_git_branch() {
     git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}
# customize the zsh prompt
PS1='%B%F{green}%(?.%F{green}√.%F{red}X:%?) %B%F{251} %1~ $(parse_git_branch)\ %# '
```

## Alias Notes

If you make a shortcut to your code base like so:

`export BASE='$HOME/git'`

then if you want to use it in an alias you'll have to use double quotes.

Instead of `alias cdh=cd $BASE'` you'll have to use `alias cdh="cd $BASE"`

However, if you were just doing it with `$HOME`, it seems single quotes work.

## Windows
 
For Windows, I sometimes use [Ubuntu](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6?activetab=pivot:overviewtab) as my command line. I do this because it's easiest for me to stick with Unix commands if I'm bouncing around between systems so much. In general, I try to run my Windows like a Linux system when I'm working with the command line a lot.
