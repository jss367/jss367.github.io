---
layout: post
title: "Linux Setup"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/turtle.jpg"
tags: [Software]
---
This post contains details of how I set up my shell and environment. I use Windows, Mac, and Linux on a daily basis, so I have different setups for different purposes, but I try to make them similar when I can. You can see the [softwhere I use](https://jss367.github.io/software-i-use.html) and [how I customize it](https://jss367.github.io/software-customizations.html) in the linked posts; this post will focus on setup.

<b>Table of Contents</b>
* TOC
{:toc}

## Shell

On Macs, I use [zsh](https://www.zsh.org/) as my main shell. It's the default shell now but older Macs will need to install it. My setup is based around `zsh`.

### Shell Configuration

I use [Oh My Zsh](https://ohmyz.sh/) to configure zsh and highly recommend it.

#### Development Environment

I have a particular way I set up my development environment. I store all of my aliases and environment variables other than my passwords in a `~/.profile` file. This way I can share it with a team and we can all have the same hotkeys. In `~/.profile`, I source a separate file called something like `.my_credentials`, which is where all my credentials are exported from.

I source `~/.profile` from whatever shell I'm using. If I'm using Oh My Zsh, I create a file that just says `source ~/.profile` and save it at `~/.oh-my-zsh/custom/profile.zsh`. I usually leave the `.zshrc` file alone, but you can customize the Oh My Zsh theme if you want. Sometimes other applications, like Anaconda, will modify the `.zshrc` file. This is fine, but for everything I add, I put it in `~.profile`.

The full chain looks like this:

`~/.zshrc` -> `~/.oh-my-zsh/custom/profile.zsh` -> `~/.profile` -> `~/.my_credentials`
* Also `.profile` will source `.bash_profile` if it exists

## Installing Homebrew

[Homebrew](https://brew.sh/) is the best package manager for Mac. It installs in `/usr/local` for macOS Intel and `/opt/homebrew` for Apple Silicon. You can run the right location either way with this:
```bash
# Set Homebrew path and run eval
HOMEBREW_PREFIX=$(brew --prefix)
if [[ -d "${HOMEBREW_PREFIX}" ]]; then
  eval "$("${HOMEBREW_PREFIX}/bin/brew" shellenv)"
fi
```

## Aliasing



### Additions to Other Files

Sometimes other applications will place information in your profile files. Some examples:

* brew puts something in `zprofile`
* conda adds to `.zshrc` or sometimes `.bash_profile` depending on how you install it.

## Packages

There are a few packages I use to improve my terminal experience.

### Pygments

* [Pygments](https://pygments.org/), a Python syntax highlighter. It's like `cat` with colors. I alias it to `c` (as seen below).

### Autojump

* [autojump](https://github.com/wting/autojump)

### ZSH Syntax Highlighting

* [zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting)

To activate the syntax highlighting, add the following at the end of your .zshrc:
  `source /opt/homebrew/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh`

If you receive "highlighters directory not found" error message,
you may need to add the following to your .zshenv:
  `export ZSH_HIGHLIGHT_HIGHLIGHTERS_DIR=/opt/homebrew/share/zsh-syntax-highlighting/highlighters`



## Exports
```bash
export BASE="$HOME/git"
export DATA="$HOME/data"
export MODELS="$HOME/models"
export FZF_DEFAULT_OPS="--extended"
```
## Aliases

```bash
# GENERAL ALIASES

## MOVING AROUND

alias cdh="cd $BASE"

alias please='sudo $(history -p !!)' # redo last command but with sudo
alias ff='find . -name' # find file
alias ftxt='grep -rnw . -e'
alias fpy='find . -name "*.py" | xargs grep --color'
alias grep='grep --color=auto'
alias hgrep='history | grep -v grep | grep '

alias ll='ls -GlAFh'
alias lls='ls -GlAFhS'


alias c='pygmentize -g' # like cat but with color
alias t='tail -v'

alias ckenv='printenv | grep -i' # check environmental variables
alias path='echo $PATH | tr ":" "\n"'
alias mkdir='mkdir -pv' # automatically make child directories


alias pu='popd'
alias pd='pushd'
alias c='clear'


## DATA SCIENCE

alias ip='ipython'
alias nb='jupyter notebook'
alias wgpu='watch -d -n 0.5 gpustat' # requires gpustat
alias ns='watch -d -n 0.5 $BASE/nvidia-htop.py'
alias catf='conda activate tf' # tensorflow environment
alias capt='conda activate pt' # pytorch environment


## TMUX ALIASES

alias tmn='tmux new-session'
alias tmk='tmux kill-session -t'
alias tma='tmux a -t'
alias tm='tmux ls'

## GIT ALIASES

alias gs='git status'
```



## Functions

```bash

# find text
function ft {
  grep -rn . -e "$1"
}

function cheat() {
      curl cht.sh/$1
  }

function extract () {
      if [ -f $1 ] ; then
        case $1 in
          *.tar.bz2)   tar xjf $1     ;;
          *.tar.gz)    tar xzf $1     ;;
          *.bz2)       bunzip2 $1     ;;
          *.rar)       unrar e $1     ;;
          *.gz)        gunzip $1      ;;
          *.tar)       tar xf $1      ;;
          *.tbz2)      tar xjf $1     ;;
          *.tgz)       tar xzf $1     ;;
          *.zip)       unzip $1       ;;
          *.Z)         uncompress $1  ;;
          *.7z)        7z x $1        ;;
          *)     echo "'$1' cannot be extracted via extract()" ;;
           esac
       else
           echo "'$1' is not a valid file"
       fi
     }

```

## Shell-specific

### ZSH

```bash
   alias rld='source ~/.zshrc' #reload profile

  # zsh syntax highlighting
  source /usr/local/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh

  ## allow autojump
  [ -f /usr/local/etc/profile.d/autojump.sh ] && . /usr/local/etc/profile.d/autojump.sh

  ## allow autojump - ubuntu
  [[ -s /home/julius/.autojump/etc/profile.d/autojump.sh ]] && source /home/julius/.autojump/etc/profile.d/autojump.sh

  autoload -U compinit && compinit -u

#### BASH
```bash
  # remote GPUs run bash
   alias rld='source ~/.bashrc' #reload profile

   [[ -s /usr/share/autojump/autojump.sh ]] && source /usr/share/autojump/autojump.sh

```


## My .zshrc

Conda will install the initialization script for conda inside `.zshrc` (for Macs). It will depend on whether you installed Anaconda or Miniconda, and on whether you installed in for a single user or for all users. If it's installed for all users it will be somewhere like `/opt/anaconda3/etc/profile.d/conda.sh`. If it's just installed for one user it will be somewhere like `/Users/$USER/opt/anaconda3/etc/profile.d/conda.sh`. The whole initialization looks like one of the following (depending on whether you use Anaconda or Miniconda): 

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/jsimonelli/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/jsimonelli/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/jsimonelli/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/julius/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/julius/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/julius/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/julius/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

It's fine to keep in there, but if you use `tmux`, you might run into a problem. `tmux` doesn't always source `.zshrc`. Sometimes it only sources `.profile`, so conda won't load in a tmux window. Even worse, it may pull Python from `/usr/bin/python`, which will be old Python 2 (use `which python` to see which python is being used). So you might want to cut and paste the initialization over to .profile.

I have found that if I don't include `conda activate $DEFAULT_CONDA_ENVIRONMENT` in my `.zshrc`, it doesn't activate my default profile, even though I have this in my `.profile`. So I leave it in `.zshrc`.

Other stuff is added to `.zshrc` automatically as well. Things like `[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh` automatically get added here. If you don't need it for `tmux`, you can leave it here. Otherwise I would recommend moving it all over to `.profile`.

## Working on Remote Linux Instances

If you're sshing into a remote Linux machine, you may have to set things up differently.

You're likely to have a `.bashrc` and that will have stuff like `export PS1="\u@\h \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]$ "` in it. It will also have your conda init. You will also want to put you `.my_credentials` file there, and may need to add a `export REMOTE_BASE="/home/me"` there.

### Adding credentials to `.my_credentials`

You might also want some environmental variables in your credentials file, especially if you are using it for other purposes. You can make it depend on the shell like so
```
export LOCAL_PATH="/Users/me"
export REMOTE_PATH="/home/me"

if [ -n "`$SHELL -c 'echo $ZSH_VERSION'`" ]; then
   # PUT YOUR LOCAL DATASET LOCATION HERE
  export TRUE_PATH=$LOCAL_PATH


elif [ -n "`$SHELL -c 'echo $BASH_VERSION'`" ]; then

  # remote GPUs run bash
   alias rld='source ~/.bashrc' #reload profile

   export TRUE_PATH=$REMOTE_PATH

else
   echo "Warning: Shell unknown"
fi
```

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

## Finding How Things Got in Environment

If you want to find how something got in your conda environment, you could grep it like this:

`grep 'how_did_this_get_here' ~/.bashrc ~/.bash_profile ~/.profile`



## Useful Git Commands

These are good to set to an alias
```
git log --pretty=format:'%C(yellow)%h %Cred%ad %Cblue%an%Cgreen%d %Creset%s' --date=short
```

Testing:
```bash
# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac
```


# Old

### Exports
```bash
export HISTSIZE=1000000
export HISTFILESIZE=1000000000
export HISTCONTROL=ignoredups:erasedups  # no duplicate entries
```



```bash
# general aliases


#redo last command but with sudo
alias psgrep='ps aux | grep -v grep | grep '


alias ccat='pygmentize -O style=monokai -f console256 -g'
alias c='pygmentize -g' # like cat but with color
alias pu='popd'
alias pd='pushd'
alias c='clear'
# See what's in your path


# Watch GPU usage
alias wgpu='watch -d -n 0.5 nvidia-smi'
alias ns='watch -d -n 0.5 $OI_BASE/core/nvidia-htop/nvidia-htop.py'
#alias wgpu='watch -d -n 0.5 gpustat' # requires gpustat
#alias ns='watch -d -n 0.5 nvidia-htop.py


# Moving around
alias cdh='cd ~/git'

# conda
alias catf='conda activate tf' # tensorflow environment
alias capt='conda activate pt' # pytorch environment

# git
alias gs='git status'
```
