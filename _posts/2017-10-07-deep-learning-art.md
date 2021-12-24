---
layout: post
title: "Deep Learning Art"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/neural_emu_thin.gif"
tags: [Deep Learning, Neural Networks]
---

Can AI create art? We're still far away from knowing what (if any) the limits of AI in the art world are, but a recent paper takes a sizeable step in that direction. The paper, [A neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) describes using deep neural networks to combine the content of one image with the style of another. This opens up incredible possibilities. An AI could be taught how to paint in the style of Picasso, but use your portrait of the subject. And, even cooler, another researcher developed an implementation using [Torch](http://torch.ch/) and [posted it on Github](https://github.com/jcjohnson/neural-style).

Here are a couple of my results from using it.


Content             |  Style           |  Result
:-------------------------:|:-------------------------: |:-------------------------:
![emu]({{site.baseurl}}/assets/img/neural_style/emu.jpg)  |  ![scream]({{site.baseurl}}/assets/img/neural_style/the_scream.jpg)  |  ![Neural style emu]({{site.baseurl}}/assets/img/neural_style/neural_emu.gif)


Content             |  Style           |  Result
:-------------------------:|:-------------------------: |:-------------------------:
![eagle]({{site.baseurl}}/assets/img/neural_style/wedge-tailed_eagle.jpg)  |  ![scream]({{site.baseurl}}/assets/img/neural_style/vangogh_self.jpg)  |  ![Neural style emu]({{site.baseurl}}/assets/img/neural_style/neural_eagle.gif)

