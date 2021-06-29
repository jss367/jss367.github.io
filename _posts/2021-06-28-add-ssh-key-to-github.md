---
layout: post
title: "Adding SSH Keys to GitHub"
description: "A walkthrough of how to add SSH keys to a GitHub account"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/anzac_hill_night.jpg"
---

If you've ever tried to clone a repository from GitHub and gotten a "Permission denied (publickey)" error, you may need to create an ssh key and share it with GitHub. This post will walk through that process. The commands used in this post are for Mac and Linux.

![png]({{site.baseurl}}/assets/img/github-ssh/can_not_read.png)

The first thing you'll need to do is create an ssh key. To do so, enter `ssh-keygen` into your terminal. You will be asked a series of questions about where to save it and to enter a passphrase. The default location is usually fine and you don't need to enter a passphrase, so you can just hit `return` to skip those steps. If it works correctly it will create some nice ASCII art.

![png]({{site.baseurl}}/assets/img/github-ssh/ssh-keygen.png)

Now you've created both a public and a private key. The public key is `id_rsa.pub` and the private key is `id_rsa` (no file extension). You can take a look at your public key by entering `cat id_rsa.pub`.

![png]({{site.baseurl}}/assets/img/github-ssh/public_key.png)

This is the public key that you can share with external entities like GitHub. You can also look at your private key by entering `cat id_rsa`, although you shouldn't need to. That key is just for your system and not for sharing or posting on blogs :)

Now you need to go to [GitHub](https://github.com/) and enter your public key. Start by click on your icon on the top right and selecting `Settings` in the dropdown.

![png]({{site.baseurl}}/assets/img/github-ssh/github_settings.png)
![png]({{site.baseurl}}/assets/img/github-ssh/github_settings2.png)

Then under `Account Settings` click on `SSH and GPG keys`.

![png]({{site.baseurl}}/assets/img/github-ssh/github_ssh_menu.png)

Then enter a title and copy everything that printed when you ran `cat id_rsa.pub` and paste it into GitHub. This includes the `ssh-rsa` at the beginning and the `username@host` at the end.

![png]({{site.baseurl}}/assets/img/github-ssh/add_new_key.png)

Then all you need to do is click `Add SSH key`, enter your password, and you're all set!

