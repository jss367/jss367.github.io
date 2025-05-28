---
layout: post
title: "How Many Steps Does it Take to See a Kangaroo?"
description: "A notebook visualizing a self-collected dataset of kangaroo sightings. The notebook is in Python using the Seaborn library"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/windy_roo.jpg"
tags: [Wildlife, Python, Seaborn, Australia]
---

How many steps does it take to see a kangaroo? I started wondering this soon after moving to the center of the Australian Outback. So, with the assistance of my Fitbit, I started recording the number of steps I took after walking out the front door until I saw a kangaroo.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Steps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3082</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1544</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1861</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1590</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1041</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2351</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1721</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1379</td>
    </tr>
  </tbody>
</table>
</div>


It turns out that it took an average of 1992 steps, with a standard deviation of 734. The minimum was 1041 steps and the maximum was 3333.


```python
# Now let's plot it using seaborn
# Make it large
fig, ax = plt.subplots(figsize=(10,6))

# Set some style preferences
sns.set(style="ticks", palette="deep", color_codes=True, font_scale=1.5)

# Create the boxplot
ax = sns.boxplot(x="Steps", data=df,
                 whis=np.inf, color="c")

# Add in points to show each observation
sns.stripplot(x="Steps", data=df,
              size=10, color=".3", linewidth=0)
ax.set_xticks(ax.get_xticks())
sns.despine(trim=True)
plt.title('How many steps does it take to see a kangaroo?', size = 16)
```








![Box plot of steps needed to see a kangaroo]({{site.baseurl}}/assets/img/2016-07-20-How%20Many%20Steps%20Does%20it%20Take%20to%20See%20a%20Kangaroo_files/2016-07-20-How%20Many%20Steps%20Does%20it%20Take%20to%20See%20a%20Kangaroo_6_1.png)

