---
layout: post
title: "Stylizing Jekyll"
description: "A guide for how to work with pages in jekyll"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/bridge.jpg"
tags: [Jekyll, Markdown, Cheat Sheet]
---
This is a quick cheatsheet for stylizing Jekyll blog posts. Jekyll uses Markdown for formatting, so all the Markdown commands work in Jekyll. There are also some additional things one can do with Jekyll.<!--more-->

<b>Table of Contents</b>
* TOC
{:toc}

# Headers

The largest header is a single `#`. For subsequently smaller headers, add another `#`

```
# Main header
## Secondary header
### Tertiary header...
```

# Main header
## Secondary header
### Tertiary header...

# Emphasis

```
*Single asterisk for italics*

**Double asterisk for bold**

***Triple asterisk for bold and italics***
```

*Single asterisk for italics*

**Double asterisk for bold**

***Triple asterisk for bold and italics***

# Code

Use a grave accent "\`" for `code` (it's to the left of 1 on your keyboard)

You can also use triple grave accents for block code and specify the language:

```html
<html>
  <head>
  </head>
  <body>
    <p>Hello, World! I am HTML </p>
  </body>
</html>
```

Here's the Jekyll way to do it:

{% highlight python %}
for i in range(10):
    print(i)
{% endhighlight %}

# HTML tags

You can also use HTML tags. `<b>Bold with HTML</b>` gives you:

<b>Bold with HTML</b>

# Lists

- Dash with a space for an unordered list (bullets)
- It works really well

The way ordered lists are down in Markdown is a bit quirky in my opinion. They are created as follows:
```
1. Step one
2. Step two
3. Step three
```
However, it doesn't actually matter what the numbers are, so you can also just use `1.` for each item. This makes it easier to add something into the middle of the list and everything else automatically updates.

```
1. Do this
1. Then do this
1. And finally this
```

1. Do this
1. Then do this
1. And finally this

# Bullets

```
* Use an asterisk and a space to make a bullet
  * Enter two spaces before the asterisk to make a subbullet
    * And four to make a subsubbullet
```

* Use an asterisk and a space to make a bullet
  * Enter two spaces before the asterisk to make a subbullet
    * And four to make a subsubbullet

# Links

```
[This Blog](https://jss367.github.io/)
```

[This Blog](https://jss367.github.io/)

# Images

`![title](images/image.jpg "text for mouse scroll over")`

For example:

`![emu]({{site.baseurl}}/assets/img/neural_style/emu.jpg "Picture of an emu")`

produces this:

![emu]({{site.baseurl}}/assets/img/neural_style/emu.jpg "Picture of an emu")

If you need more control over it (such as adjusting the size) you can always use standard HTML

`<img src="{{site.baseurl}}/assets/img/neural_style/emu.jpg" alt="Drawing" style="width: 400px;"/>`

<img src="{{site.baseurl}}/assets/img/neural_style/emu.jpg" alt="Drawing" style="width: 400px;"/>

# Tables

`Content             |  Style           |  Result
:-------------------------:|:-------------------------: |:-------------------------:
![eagle]({{site.baseurl}}/assets/img/neural_style/wedge-tailed_eagle.jpg)  |  ![scream]({{site.baseurl}}/assets/img/neural_style/vangogh_self.jpg)  |  ![Neural style emu]({{site.baseurl}}/assets/img/neural_style/neural_eagle.gif)`

Content             |  Style           |  Result
:-------------------------:|:-------------------------: |:-------------------------:
![eagle]({{site.baseurl}}/assets/img/neural_style/wedge-tailed_eagle.jpg)  |  ![scream]({{site.baseurl}}/assets/img/neural_style/vangogh_self.jpg)  |  ![Neural style emu]({{site.baseurl}}/assets/img/neural_style/neural_eagle.gif)

Here's another way to do a table specifically with Jekyll:

|Address               | City               | Zip Code            | State              |
|--------------------- | ------------------ | ------------------- | -------------------|
|123 Main St.          | Best City          | 12345               | AB |
|124 Main St.          | Best City          | 12345               | AB |
|125 Main St.          | Best City          | 12345               | AB |


Or you could do it in HTML like this:


`<table width="30%">`
`<tr>`
`<td><img src="{{site.baseurl}}/assets/img/neural_style/wedge-tailed_eagle.jpg"></td>`
`<td><p align="center"><img src="{{site.baseurl}}/assets/img/neural_style/vangogh_self.jpg"></p></td>`
`<td align="right"><img src="{{site.baseurl}}/assets/img/neural_style/neural_eagle.gif"></td>`
`</tr>`


<table width="30%">
<tr>
<td><img src="{{site.baseurl}}/assets/img/neural_style/wedge-tailed_eagle.jpg"></td>
<td><p align="center"><img src="{{site.baseurl}}/assets/img/neural_style/vangogh_self.jpg"></p></td>
<td align="right"><img src="{{site.baseurl}}/assets/img/neural_style/neural_eagle.gif"></td>
</tr>
</table>

# Mathematical Formulas

You can use [LaTeX](https://www.latex-project.org/) to write beautiful mathematical formulas. They can be in-line like $$ \sum_{k=0}^{n-1} e^{2 \pi i \frac{k}{n}} = 0 $$ or they can have their own line:

$$ \nabla \cdot \mathbf{E} = \frac {\rho} {\varepsilon_0} $$

$$ \nabla \cdot \mathbf{B} = 0 $$

$$ \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}} {\partial t} $$

$$ \nabla \times \mathbf{B} = \mu_0\left(\mathbf{J} + \varepsilon_0 \frac{\partial \mathbf{E}} {\partial t} \right) $$

You simply need to put `$$` around your equations. The theme I use relies on [KaTeX](https://katex.org/).

# More Math

Starting from Bayes’ Theorem:

$$ P(A|B) = \frac{P(B|A) \times P(A)}{P(B|A)P(A) + P(B|\neg A)P(\neg A)} $$


$$ P(A|B) = \frac{\frac{P(B|A)}{P(B|\neg A)} \times \frac{P(A)}{P(\neg A)}}{1 + \frac{P(B|A)}{P(B|\neg A)} \times \frac{P(A)}{P(\neg A)}} $$


You can't use them in bulleted lists, though, so you'll have to do something different.

Notice two ratios appear:

$$ \frac{P(A)}{P(\neg A)}: \text{ the prior odds.} $$

- $$ \frac{P(A)}{P(\neg A)}: \text{ the prior odds.} $$

- $$ \( \frac{P(A)}{P(\neg A)} \): the **prior odds**. $$
- $$ \( \frac{P(B|A)}{P(B|\neg A)} \): the **Bayes factor** (how strongly the evidence favors A over ¬A). $$


# MathJax Test

Inline math: \( P(A|B) = \frac{P(B|A)P(A)}{P(B)} \)

Display math:

\[
P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}
\]

If MathJax is working, you should see properly typeset equations above.  
If not, you'll just see the raw LaTeX code.

# Citations

This is a sentence with a citation[^1]. Go to this [StackOverflow question](https://stackoverflow.com/questions/50467557/jekyll-on-github-pages-how-to-include-external-content-inside-footnotes/50475226) to see how to set them up.

[^1]: 
    {% include citation.html key="ref1" %}
