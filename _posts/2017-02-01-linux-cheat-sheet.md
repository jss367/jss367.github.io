---
layout: post
title: "Linux Cheat Sheet"
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

The quick way to open up a directory to all people is to do `chmod -R 0777 my_dir`

# Searching

## Finding Files

There are two primary tools I use to find files on Linux, `locate` and `find`. `locate` is the newer and faster tool, but `find` is much more universal and can do a lot more, so I generally use `find`. `find` looks through the file system while `locate` looks through a database. This makes `locate` much faster but you'll have to update the database with `sudo updatedb` before you can find new files.

### find

`find` is automatically recursive, so it will search through your subdirectories with you needed to add a `-r` or  `-R` to it. 

`find` takes the format of:
`find location comparison-criteria search-term`

For example, you could do `find /usr/lib -name "*gdal*"`

#### find examples

Find jpg files:

`find . -name "*.jpg"`

find python core dumps:

`find . -name "core.*"` - note that this will also probably find non core dump files as well

##### Deleting with find

You can also find and delete these with `find . -name "core.*" -exec rm {} +`

##### Search By Directory or File Type

This will also work for directories. However, if you want to search only for directories, you can specify the `type`:

`find . -type d -name my_dir_name`

You can also search for file types, such as:

`find . -executable`

### fd

[fd](https://github.com/sharkdp/fd) is worth giving a try.

## Finding Text within Files

### grep

`grep` is a great tool. The basic usage is `grep [flags] [pattern] [filename]`

Here are the flags I use most often:
```
-i ignore case
-n show line numbers
-r recursive (search in folders); capitalize to add symlinks
```

Here's a way to find all the files with the word "tensorflow":

`grep -irl 'tensorflow' .`

Let's look at a `grep` command. Here's one I find particularly useful:

`grep -r password /etc`

The syntax of grep consists of four parts.

1. grep command
2. optional: option(s)
3. string to search
4. file, files, or path to be searched

```
grep -ir driver *   - done when in the folder
grep -r BBDatasetTrainer *
grep -ir precision_recall_curve /Users/juliussimonelli/Documents/pCloud
```

If you want to search for text within files, you can use `grep`:

`grep -d recurse "This Text" *`

One of the most useful flags for me is `-i` for ignore case:

`grep -i -d recurse "This Text" *`

### ls

Don't forget `ls` can also be a great tool for this. Something as simple as `ls | grep rsa` to find rsa keys. Note that you don't need asterisks for this.

Also, this question has a [great visualization of what the colors mean in ls](https://askubuntu.com/questions/17299/what-do-the-different-colors-mean-in-ls)

## Viewing System Processes

`top` is the default tool and is great, but for something easier to view, try [htop](https://hisham.hm/htop/).

# Storage

[ncdu](https://dev.yorhel.nl/ncdu) is a great replacement for `du`. You might not have `ncdu` by default so you may have to `yum install ncdu`. If your storage is getting full:

`ncdu -x`

Another option for exploring storage is `df -h`.

# apt vs apt-get ##

Debian-based Linux distributions, like Ubuntu, started using `apt`. 

`apt` is in many ways a nicer version of `apt-get`. It takes the most commonly used part. 

so `apt-get remove package` is now `apt remove package`

# Other

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

# Quickly Adding to Files

Let's say you want to add something to your `.gitignore` file, and don't want to bother with [vim](https://www.vim.org/) at the moment. You can add what you need by typing `cat > .gitignore` then adding whatever you need. Then hit `control + D` to return to the bash prompt.

## Compressing and Decompressing files

#### Compress

`tar -czvf my_directory.tar.gz /path/to/my_directory`

Here’s what those flags mean:

* -c: Create a compressed file
* -z: Zip with gzip
* -v: Verbose mode
* -f: Specify filename

#### Decompress

`tar -xzvf archive.tar.gz`

The only different is that we change the "-c" for Create to "-x" for eXtract

# Looking at directories, datasets

`tree --filelimit 10 --dirsfirst`

# Delete old files

## Just files

The command is of the form: `find /path/to/files* -mtime +5 -exec command {} \;`

`find . -mtime +5 -exec rm {} \;`

## Files and folders

`find . -mtime +30 -exec rm -rf {} \;`


