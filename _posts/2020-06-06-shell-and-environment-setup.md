---
layout: post
title: "Shell and Environment Setup"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/turtle.jpg"
tags: [Software]
---

In this post, I'll talk about how I set up my shell and environment. I Use Windows, Mac, and Linux on a daily basis, so I have different setups for different purposes, but I try to make them similar when I can.

<b>Table of Contents</b>
* TOC
{:toc}

## Shell

I use [zsh](https://www.zsh.org/) as my main shell. It's now the default shell so newer Macs will have it but older ones will need to install it.

## Shell Configuration

I use [Oh My Zsh](https://ohmyz.sh/) to configure zsh and highly recommend it.

I store all of my environment variables in a `~/.profile` file. Then I source that file in whatever shell I'm using. This makes it much easier to work across a variety of environments.

## Oh My Zsh Configuration

* I usually leave the theme as `ZSH_THEME="robbyrussell"`

* Then I create a profile and put in it `~/.oh-my-zsh/custom/profile.zsh`

* That profile just says `source ~/.profile`

## Packages to Add

* [Pygments](https://pygments.org/), a Python syntax highlighter. It's like `cat` with colors. I alias it to `c` (as seen below).
* [autojump](https://github.com/wting/autojump)
* [zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting)

## My .profile Setup

* I usually share my `.profile` with others so that we can all share shortcuts. In order to do this without sharing passwords, I make a separate file called something like `.my_credentials` and export my credentials from there.

### Exports
```bash
export BASE="$HOME/git"
export DATA="$HOME/data"
export MODELS="$HOME/models"
export FZF_DEFAULT_OPS="--extended"
```
### Aliases

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

alias ckenv='printenv | grep -i' # lookup rabbit, lookup database, etc.
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



### Functions

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

### Shell-specific

#### ZSH

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

## conda init

This whole thing exists because for conda to fully work it needs to be initialized and activated. That's what this code block is doing.

Let's do over what the conda init command does
```bash
if [ $? -eq 0 ]; then
```
`$?` is a variable that is equal to the return value of the last command you ran. This is often a return code, which is 0 for a success and non-zero if there's been an error. SO this line is saying, if the last command ran successfully, then...


## My .zshrc

Conda will install the initialization script for conda inside `.zshrc` (for Macs). It usually looks like one of the following (depending on whether you use Anaconda or Miniconda): 

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

It's fine to keep in there, but if you use `tmux`, you might run into a problem. `tmux` doesn't source `.zshrc` - it only sources `.profile`, so conda won't load in a tmux window. Even worse, it may pull Python from `/usr/bin/python`, which will be old Python 2 (use `which python` to see which python is being used). So you might want to cut and paste the initialization over to .profile.

I have found that if I don't include `conda activate $DEFAULT_CONDA_ENVIRONMENT` in my `.zshrc`, it doesn't activate my default profile, even though I have this in my `.profile`. So I leave it in `.zshrc`.

Other stuff is added to `.zshrc` automatically as well. Things like `[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh` automatically get added here. If you don't need it for `tmux`, you can leave it here. Otherwise I would recommend moving it all over to `.profile`.



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

then if you want to use it an an alias you'll have to use double quotes.

Instead of `alias cdh=cd $BASE'` you'll have to use `alias cdh="cd $BASE"`

However, if you were just doing it with `$HOME`, it seems single quotes work.


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

## Windows
 
For Windows, I sometimes use [Ubuntu](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6?activetab=pivot:overviewtab) as my command line. I do this because it's easiest for me to stick with Unix commands if I'm bouncing around between systems so much. In general, I try to run my Windows like a Linux system when I'm working with the command line a lot.

# Old

### Exports
```bash
export HISTSIZE=1000000
export HISTFILESIZE=1000000000
export HISTCONTROL=ignoredups:erasedups  # no duplicate entries
```



```bash
# general aliases

alias rld='source ~/.zshrc' #reload, assume zsh

#redo last command but with sudo
alias please='sudo $(history -p !!)'
alias ff='find . -name'
alias findpy='find . -name "*.py" | xargs grep --color'
alias fpy='find . -name "*.py" | xargs grep --color'
alias grep='grep --color=auto'
alias hgrep='history | grep -v grep | grep '
alias psgrep='ps aux | grep -v grep | grep '

alias ll='ls -GlAFh'
# alias ll='ls -alF'
alias lls='ls -GlAFhS'
#alias la='ls -A'
#alias l='ls -CF'
# better ls (column detail and no meta-files (., .., etc))
alias ls='ls -lA'

alias showpath='echo $PATH | tr ":" "\n"'
alias path='echo -e ${PATH//:/\\n}'
alias wgpu='watch -d -n 0.5 gpustat' # requires gpustat
alias nb='jupyter notebook'

alias ckenv='printenv | grep -i' # lookup rabbit, lookup database, etc.

## Redo the last command but with sudo in front
alias please='sudo $(history -p !!)'

alias ccat='pygmentize -O style=monokai -f console256 -g'
alias c='pygmentize -g' # like cat but with color
alias pu='popd'
alias pd='pushd'
alias c='clear'
# See what's in your path


# tmux
alias tmn='tmux new-session'
alias tmk='tmux kill-session -t'
alias tma='tmux a -t'
alias tm='tmux ls'

# Python
alias ip='ipython'
#alias nb='jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.iopub_data_rate_limit=100000000000'
alias nb='jupyter notebook'

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
