---
layout: post
title: "Linux Tricks and Tips"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/peacock.jpg"
tags: [Linux, Cheat Sheet]
---

This is a list of simple Linux tips and tricks.

<b>Table of contents</b>
* TOC
{:toc}

## Permissions and Ownership

Permissions are something that often get in the way for new users but are simple to fix. To see the permissions, you can type `ls -l`. This shows you the file type ("d" for directory, "-" for file), owner, group, and other permissions. After that you'll see the ownership. If you're having permission error when accessing a file, you may need to change the ownership. To do so you need to use the `chown` command and you'll probably need to `sudo` with it. It follows a format like this:

```
chown {user} {file}
```

For the {user} you can put your actual user name (something like jsimonelli) or simply use $USER to retrieve it.

```
sudo chown $USER my_file
```

You can also `chown` an entire directory

```
chown -R {user} {file}
```

```
sudo chown -R $USER my_dir
```

## Finding Files

There are two primary tools I use to find files on Linux, `locate` and `find`. `locate` is the newer and faster tool, but `find` is much more universal and can do a lot more, so I generally use `find`. `find` looks through the file system while `locate` looks through a database. As I said, this makes `locate` much faster but you'll have to update the database with `sudo updatedb` before you can find new files.


### Find Examples ###

For me the best documentation is some basic commands, so here are some examples:

`find` takes the format of:
`find location comparison-criteria search-term`

For example, you could do `find /usr/lib -name "*gdal*"`

Find jpg files:

`find . -name "*.jpg"`

find python core dumps:

`find . -name "core.*"` - note that this will also probably find non core dump files as well

You can also find and delete these with `find . -name "core.*" -exec rm {} +`

You can also search for file types, such as:

`find . -executable`

### fd ###

[fd](https://github.com/sharkdp/fd) is worth giving a try.

### ls ###

Don't forget `ls` can also be a great tool for this. Something as simple as `ls | grep rsa` to find rsa keys. Note that you don't need asterisks for this.

### Finding Text within Files ###

If you want to search for text within files, you can use `grep`:

`grep -d recurse "This Text" *`

One of the most useful flags for me is `-i` for ignore case:

`grep -i -d recurse "This Text" *`

## Viewing System Processes

`top` is the default tool and is great, but for something easier to view, try [htop](https://hisham.hm/htop/).


## Storage

[ncdu](https://dev.yorhel.nl/ncdu) is a great replacement for `du`. You might not have `ncdu` by default so you may have to `yum install ncdu`. If your storage is getting full:

`ncdu -x`

Another option for exploring storage is `df -h`.

## apt vs apt-get ##

Debian-based Linux distributions, like Ubuntu, started using apt. 

apt is in many ways a nicer version of apt-get. It takes the most commonly used part. 

so apt-get remove package is now apt remove package

## Other

Another way to find if a file exists is:
`[ -e myfile.txt ] && echo "Found" || echo "Not Found"`

You can also check if it is a regular file:
`[ -f myfile.txt ] && echo "Found" || echo "Not Found"`

Or a directory
`[ -d myfiles ] && echo "Found" || echo "Not Found"`

### tldr

Another package worth checking out is [tldr](https://tldr.sh/). It's like `man` but comes with examples.

### Colors

I found this [amazing script on AskUbuntu](https://askubuntu.com/questions/17299/what-do-the-different-colors-mean-in-ls).

```
eval $(echo "no:global default;fi:normal file;di:directory;ln:symbolic link;pi:named pipe;so:socket;do:door;bd:block device;cd:character device;or:orphan symlink;mi:missing file;su:set uid;sg:set gid;tw:sticky other writable;ow:other writable;st:sticky;ex:executable;"|sed -e 's/:/="/g; s/\;/"\n/g')           
{      
  IFS=:     
  for i in $LS_COLORS     
  do        
    echo -e "\e[${i#*=}m$( x=${i%=*}; [ "${!x}" ] && echo "${!x}" || echo "$x" )\e[m" 
  done       
}
```

## Quickly Adding to Files

Let's say you want to add something to your `.gitignore` file, and don't want to bother with [vim](https://www.vim.org/) at the moment. You can add what you need by typing `cat > .gitignore` then adding whatever you need. Then hit control + D to return to the bash prompt.

## Compressing and Decompressing files:

#### Compress

`tar -czvf my_directory.tar.gz /path/to/my_directory`

Here’s what those flags mean:

-c: Create a compressed file
-z: Zip with gzip
-v: Verbose mode
-f: Specify filename

#### Decompress

`tar -xzvf archive.tar.gz`

The only different is that we change the "-c" for Create to "-x" for eXtract
