---
layout: post
title: "Declining Appetite for Executions in England, Visualized"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/noose.jpg"
tags: [Python, Matplotlib, Data Visualization]
---

I recently came across the numbers of convictions and executions in the latter half of 18<sup>th</sup> century England. The number of executions varied and it wasn't immediately clear if there were any larger trends.


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
      <th>Convictions</th>
      <th>Executions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1749-58</th>
      <td>527</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1759-68</th>
      <td>372</td>
      <td>206</td>
    </tr>
    <tr>
      <th>1769-78</th>
      <td>787</td>
      <td>357</td>
    </tr>
    <tr>
      <th>1779-88</th>
      <td>1152</td>
      <td>531</td>
    </tr>
    <tr>
      <th>1789-98</th>
      <td>770</td>
      <td>191</td>
    </tr>
    <tr>
      <th>1799-1808</th>
      <td>804</td>
      <td>126</td>
    </tr>
  </tbody>
</table>
</div>



But I found that once you look at the number of executions relative to convictions, the growing distaste in executions in Georgian England becomes far more apparent.


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
      <th>Convictions</th>
      <th>Executions</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1749-58</th>
      <td>527</td>
      <td>365</td>
      <td>69.3</td>
    </tr>
    <tr>
      <th>1759-68</th>
      <td>372</td>
      <td>206</td>
      <td>55.4</td>
    </tr>
    <tr>
      <th>1769-78</th>
      <td>787</td>
      <td>357</td>
      <td>45.4</td>
    </tr>
    <tr>
      <th>1779-88</th>
      <td>1152</td>
      <td>531</td>
      <td>46.1</td>
    </tr>
    <tr>
      <th>1789-98</th>
      <td>770</td>
      <td>191</td>
      <td>24.8</td>
    </tr>
    <tr>
      <th>1799-1808</th>
      <td>804</td>
      <td>126</td>
      <td>15.7</td>
    </tr>
  </tbody>
</table>
</div>



I graphed change in the percentage against time but also included the total number of executions by varying the line thickness. I think it's interesting because it shows the decline in the percentage of executions while reminding us that there were still a tremendous number of executions. And while the overall trend is clear and good, there are major setbacks along the way.


```python
x = np.array(range(6))
y = df['Percentage']
widths = df['Executions'].values / 100
```


```python
import matplotlib.patches as patch

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_xlim(-.5,5.5)
ax.set_ylim(10,80)

new_w = np.array(widths)

for i in range(len(x)-1):
    c = [[x[i], y[i]+new_w[i]/2.],
         [x[i+1], y[i+1]+new_w[i+1]/2.],
         [x[i+1], y[i+1]-new_w[i+1]/2.],
         [x[i], y[i]-new_w[i]/2.]
        ]
    p = patch.Polygon(c)
    ax.add_patch(p)
plt.xlabel("Decade", fontsize=16)
plt.ylabel("Execution Percentage", fontsize=16)
plt.title("Declining appetite for executions in 18th century England", fontsize=18)
plt.xticks(x, decade)
plt.show()
```


![png]({{site.baseurl}}/asserts/img/{{site.baseurl}}/assets/img/2016-08-12-Declining-Executions-in-England_files/2016-08-12-Declining-Executions-in-England_10_0.png)

