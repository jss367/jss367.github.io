---
layout: post
title: "Connecting VSCode to Google Cloud Platform Instances"
description: "This post describes how to connect VSCode to Google Cloud Platform instances"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/smokey_tree.jpg"
tags: [VSCode]
---


In this post, I'll demonstrate how to connect to [Google Cloud Platform](https://cloud.google.com/) (GCP) instances using VSCode's [Remote SSH extention](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh). We'll assume you already have working GCP instances that you can ssh into. To connect to a remote host, VSCode needs to know the HostName, User, and IdentityKey. In this guide, we'll go over how to find these. For simplicity, we'll assume you're trying to connect to an instance named `my_instance` and your zone is `europe-west4-b`. You'll need to find these values and change them in the instructions below.

## Project name

You'll need to know your project name to find your HostName. You can use `gcloud projects list` to list all of your project names. We'll assume that your project name is `my_project`.

## Username

First, let's get your username. You probably already know it, but just in case you don't you can find it by opening a terminal and entering `gcloud auth list --format="value(account)"`. This will give you your username and your domain in the format of `username@domain`. You'll only want the username here. Let's assume your username is `my_username`.

## HostName

Now we need to get your HostName. In a terminal, type to following: 
```
gcloud compute instances describe my_instance \
  --zone=europe-west4-b \
  --project=my_project \
  --format='value(networkInterfaces[0].accessConfigs[0].natIP)'
```

This should return an IP address. This is what you'll use as your HostName.

## IdentityFile

The IndentityFile will already be on your computer if you have been connecting to your instances already. It's not your normal ssh key - Google made a separate one for this. On Linux/Mac, it should be at `~/.ssh/google_compute_engine`. You can check if it's there but you don't need to do anything with it.

# Putting it All Together

That should be all you need to connect VSCode to your instance. I like to add a `LogLevel` of `verbose` to make debugging easier. You shouldn't have to specify a Port but you can also do so if you prefer to be explicit. In summary, it should look like this:

```
Host my_instance
  HostName XXX.XXX.XXX.XXX
  LogLevel verbose
  IdentityFile ~/.ssh/google_compute_engine
  User my_username
  Port 22
```
