---
layout: post
title: "Adversarial Attacks on Neural Networks"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/mountain.jpg"
tags: [Neural Networks]
---

![Skier or dog]({{site.baseurl}}/assets/img/skier_dog_attack.png "Skier to dog attack")

This is a picture of a dog. At least, that's what it is if you ask [Google Cloud Vision](https://cloud.google.com/vision). Really, it's an example of an adversarial attack on a neural network from that came out of [new research](https://arxiv.org/pdf/1804.08598.pdf) by a team at MIT and [LabSix](https://www.labsix.org/), an MIT student-run AI research group. To be clear, this prediction isn't a simple "goof" on Google's part that someone found, this was a targeted attempt to fool the neural network. And it works surprisingly well. Here are the resulting scores when this image is input into Google Cloud Vision.

![Adversarial attack on GCV]({{site.baseurl}}/assets/img/skier_dog_attack_result.png "Adversarial attack on GCV")

An adversarial attack is one where an input is slightly perturbed in a specific way in an attempt to fool an algorithm. We've previously seen many examples of such attacks against so-called "white-box models", where the attacker has full knowledge of the model. This allows the attacker to learn the gradient of the classifier's loss function and exploit it to move across a label boundary while minimizing the amount of change. This results in an image that, to a human, looks nearly identical to the original, but the classifier misclassifies it. However, that technique only works if you know everything about the model. What if you don't?

The key insight of this group's research is that, even if you don't know the loss function and can't calculate the exact gradient, you can approximate it. You do this by repeatedly querying the classifier and examining the results. Google Cloud Vision provides a score associated with the top `k` classes it sees in the image (where `k` seems to vary). In the original image, those were things like "Skiing", "Piste", and "Mountain Range". You can modify the image and see how those results change to get an estimate of the gradient. Gradient estimation, not descent, is the name of the game here. That's what allows this technique to be used against a black-box algorithm.

Let's walk through the steps:
- Start with an image of a dog
- Take a step towards blending in an image of the skiers
-- This causes a reduction in the classifier's score of "dog"
- Take a step that maximizes the likelihood of the dog class
-- This brings the score of "dog" back up
- Repeat over and over until you have an image that looks like a picture of skiers to a human but is classified as a dog by the model

You can read the whole paper [here](https://arxiv.org/pdf/1804.08598.pdf). Today marks one year to the day from the original release date of the paper, and the most remarkable thing is that the same image still tricks the classifier today. Even Google is having trouble with adversarial attacks. Will this still work in five years? My guess is no, but 1. that's just a guess and 2. I find it very interesting that it is still working an entire year later.

Admittedly, this only works by running many examples through the model. But the fact that it works without any insight into how the model is built is a significant achievement. So, what does this mean? Is the model bad? I don't think that's the right conclusion - at least given the current state of the field. This was a targeted attack meant to exploit a weakness in the system. Of course, there was going to be a weakness and someone was going to find it. So I don't think it's fair to think of the model as bad. But as black box neural networks play increasingly larger roles in our lives, it shows that there is good reason to proceed cautiously.
