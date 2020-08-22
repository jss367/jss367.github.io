---
layout: post
title: "Stop Using Computers to Detect DeepFakes"
date: 2019-12-15
tags: [Artificial Intelligence]
---

There is a huge interest in building algorithms to detect DeepFakes. Many of the detection methods use machine learning themselves, although some don't. But pretty much everything I've seen has one thing in common - they are nearly always trivial to defeat. New research used phyiological signal analysis to distinguish between the two. Basically, they looked for the microsignatures of a heartbeat, which exists in humans but not in deepfakes. The paper [^1] is behind a paywall (gross)

https://www.spiedigitallibrary.org/journals/Journal-of-Electronic-Imaging/volume-29/issue-1/013009/Digital-human-face-detection-in-video-sequences-via-a-physiological/10.1117/1.JEI.29.1.013009.short?SSO=1


This one would be particularly easy. Because we know the underlying pattern it is looking for, we could add that to the image. But in a more general case, especially in one where you didn't know how the detector worked, you could use something like a SimGAN to add that last bit of realism to your model.


Here's a summary of it: https://www.eurekalert.org/pub_releases/2020-01/ssfo-nru012120.php

