In this post, I'll talk about how I set up my shell and environment. I Use Windows, Mac, and Linux on a daily basis, so I have different setups for different purposes, but I try to make them similar when I can.

I use [zsh](https://www.zsh.org/) as my main shell. It's now the default shell on Macs and I think deservedly so. I use [Oh My Zsh](https://ohmyz.sh/) to configure it and highly recommend it.

I store all of my environment variables in a `~/.profile` file. Then I source that file in whatever shell I'm using. This makes it much easier to work across a variety of environments.

## My .profile Setup

* I usually share my `.profile` with others in my company (or wherever I am) so that we can all share shortcuts. In order to do this without sharing passwords, I make a separate file called something like `.my_credentials` and export my credentials from there.



```
# general aliases

#alias myip='dig TXT +short o-o.myaddr.l.google.com @ns1.google.com'

# alias ll='ls -alF'

alias rld='source ~/.zshrc' #reload, assume zsh

alias please='sudo $(history -p !!)'
alias ff='find . -name'
alias fpy='find . -name "*.py" | xargs grep --color'
alias grep='grep --color=auto'
alias hgrep='history | grep -v grep | grep '
alias ll='ls -GlAFh'
alias lls='ls -GlAFhS'
alias showpath='echo $PATH | tr ":" "\n"'
alias wgpu='watch -d -n 0.5 gpustat' # requires gpustat
alias nb='jupyter notebook'
alias c='pygmentize -g' # like cat but with color
alias ckenv='printenv | grep -i' # lookup rabbit, lookup database, etc.
```


You'll want to find somewhere to put the conda init. Upon installation, Anaconda is automatically added to a resource file, `.zshrc` in the case of Macs. It usually looks like one of the following (depending on whether you use Anaconda or Miniconda): 

```
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

```
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

I usually leave this in my `.zshrc` file, but that's the only thing I allow there. As I said, everything else goes in `~/.profile`.

## Oh My Zsh Configuration

* I usually leave the theme as `ZSH_THEME="robbyrussell"`

* Then I create a profile and put in it `~/.oh-my-zsh/custom/profile.zsh`

* That profile just says `source ~/.profile`

## Bash

I still use Bash fairly often, and because it doesn't come with all the same aliases that Oh My Zsh does, I have to add some of the most important ones manually. Here are some I recommend.

```
alias ff='find . -name'
alias findpy='find . -name "*.py" | xargs grep --color'
alias grep='grep --color=auto'
alias hgrep='history | grep -v grep | grep '
alias ll='ls -GlAFh'
alias lls='ls -GlAFhS'
alias ..='cd..'
alias ...='cd../..'
```

```
alias gs='git status'

#redo last command but with sudo
please='sudo $(history -p !!)'

wgpu='watch -d -n 0.5 gpustat' # requires gpustat
ns='watch -d -n 0.5 nvidia-htop.py

nb='jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 ==allow-root --NotebookApp.iopub_data_rate_limit=1000000000'


#useful functions
function cheat() {
 curl cht.sh/$1
 }

function extract () {
 if [ -f $1 ] ; then
 case $1 in
 *.tar.bz2) tar xjf $1 ;;
 *.tar.gz) tar xzf $1 ;;
 *.bz2) bunzip2 $1 ;;
 *.rar) unrar e $1 ;;
 *.gz) gunzip $1 ;;
 *.tar) tar xf $1 ;;
 *.tbz2) tar xjf $1 ;;
 *.tgz) tar xzf $1 ;;
 *.zip) unzip $1 ;;
 *.Z) uncompress $1 ;;
 *.7z) 7z x $1 ;;
 *) echo "'$1' cannot be extracted via extract()" ;;
 esac
 else
 echo "'$1' is not a valid file"
 fi
 }
```

```
# reload user profile
alias rp='source ~/.profile'
alias rld='source ~/.bashrc' #reload

# better ls (column detail and no meta-files (., .., etc))
alias ls='ls -lA'

alias showpath='echo $PATH | tr ":" "\n"'
```

conda activate "$DEFAULT_CONDA_ENVIRONMENT"
 
color git branch:
```
parse_git_branch() {
     git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}
export PS1="\u@\h \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]$ "
```

```
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

```
# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
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

`git log --pretty=format:'%C(yellow)%h %Cred%ad %Cblue%an%Cgreen%d %Creset%s' --date=short`


Testing:
```
# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac
```

 ## Windows
 
 For Windows, I sometimes use [Ubuntu](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6?activetab=pivot:overviewtab) as my command line. I do this because it's easiest for me to stick with Unix commands if I'm bouncing around between systems so much. In general, I try to run my Windows like a Linux system when I'm working with the command line a lot.
