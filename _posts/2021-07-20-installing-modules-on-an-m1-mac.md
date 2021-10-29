---
layout: post
title: "Installing Modules on an M1 Mac"
description: "This post discusses how to install popular data science modules on Macs with an M1 chip"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/apple.jpg"
---

Apple's M1 chips were released to much fanfare and some [very impressive benchmarks](https://www.cpubenchmark.net/cpu.php?cpu=Apple+M1+8+Core+3200+MHz&id=4104). Unfortunately, all these gains don't come for free. The downside of such a significant change is that it causes incompatibility issues. And because the basic CPU and instruction set are different, there are a lot of thorny issues. In this post, I'll walk through some I ran into when installing standard data science libraries and how I solved them. In the end, I was able to get everything working, although it took quite a bit of patience. Hopefully, these incompatibilities get resolved relatively soon, but, until then, here's my guide to help you out.

I expect this process to get easier over time, but at the moment it's tricky. At the moment, Anaconda and M1 Macs aren't compatible, so I recommend people start with [`miniforge`](https://github.com/conda-forge/miniforge), which is a community-driven (like `conda-forge`), minimalisitic (like `miniconda`), package installer.

You'll also need [Rosetta 2](https://support.apple.com/en-us/HT211861) (also sometimes just called Rosetta), which translates binaries written for Intel processors to be compatible with Apple's chips.

To use Rosetta, you'll need to ensure you open your specific application with Rosetta. I recommend using iTerm2 as your terminal, so I'll give instructions assuming you use that. Go into Finder and then the Applications folder and right-click on iTerm2. Select "Get Info" and then under General check the checkbox for "Open using Rosetta". Then close down your terminals and open them again.

Now create your conda environment:

`conda create --name my_env python=3.8`

If you have a module that you usually install with pip, you're likely to run into problems. However, you can still get started that way. Even if you're used to installing everything with `pip install -r requirements.txt`, I would recommend using conda as much as possible for these.

If you try to install `scipy` with pip, you'll run into an error of no `lapack` and `blas` resources being found. For `shapely`, it's currently missing the `geos` library.

However, if you install these using conda it will bring these libraries with it. For `scipy`, it will include `liblapack`, `libblas`, `libgfortran`, and the rest.


`opencv` in particular is very difficult. I found that it would install but then it would give me an error when I imported it. What I had to do was to uninstall it using pip. It kept getting installed every time I install my own modules with `pip install -e .`, so every time afterwards I ran `pip uninstall opencv-python`. Then I was able to `import cv2` again.

Appendix:

I tried building packages through pip with things like `python -m pip install scipy --no-binary :all:` but I kept running into issues. For me the only way I got through it was relying on miniforge, so that's what I recommend at the moment.