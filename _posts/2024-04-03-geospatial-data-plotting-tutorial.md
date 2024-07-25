---
layout: post
title: "Geospatial Data Plotting Tutorial"
description: "This is a simple post about how to quickly plot geospatial data"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/costas_hummingbird.jpg"
tags: [Geospatial Analytics, Python]
---

This tutorial shows how to plot geospatial data on a map of the US. There are lots of libraries that do all the hard work for you, so the key is just knowing that they exist and how to use them.

<b>Table of Contents</b>
* TOC
{:toc}


```python
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
```

One of the things you'll have to do is find the right data for mapping. Fortunately, there are datasets built into geopandas that you can use.

The key to plotting geospatial data is in shapefiles. A shapefile is a geospatial vector data format for geographic information system (GIS) software. It is used for storing the location, shape, and attributes of geographic features, such as roads, lakes, or political boundaries. A shapefile is actually a collection of files that work together. The main file (.shp) stores the geometry of the features, the index file (.shx) contains the index of the geometry, and the dBASE table (.dbf) contains attribute information for each shape. Additional files can also be included to store other types of information.

### Download Map from the Internet


```python
# Load a map of the US
us_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).query('continent == "North America" and name == "United States of America"')
us_map
```

    C:\Users\Julius\AppData\Local\Temp\ipykernel_8928\39096665.py:2: FutureWarning: The geopandas.dataset module is deprecated and will be removed in GeoPandas 1.0. You can get the original 'naturalearth_lowres' data from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/.
      us_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).query('continent == "North America" and name == "United States of America"')
    




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
      <th>pop_est</th>
      <th>continent</th>
      <th>name</th>
      <th>iso_a3</th>
      <th>gdp_md_est</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>328239523.0</td>
      <td>North America</td>
      <td>United States of America</td>
      <td>USA</td>
      <td>21433226</td>
      <td>MULTIPOLYGON (((-122.84000 49.00000, -120.0000...</td>
    </tr>
  </tbody>
</table>
</div>



Now we can plot it.


```python
fig, ax = plt.subplots(figsize=(10, 10))
us_map.boundary.plot(ax=ax)
ax.set_title("Map of USA")

plt.show()
```


    
![png](2024-04-03-geospatial-data-plotting-tutorial_files/2024-04-03-geospatial-data-plotting-tutorial_7_0.png)
    


There are other datasets available, although, as you can see, they're deprecated. But, for example, you can also get cities.


```python
# Load the naturalearth cities dataset
cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities')) # note this is worldwide

# Plot the world map as background
fig, ax = plt.subplots(figsize=(15, 10))
us_map.plot(ax=ax, color='lightgray')

# Plot cities on top
cities.plot(ax=ax, marker='o', color='red', markersize=5)

# Focus on the US by setting the limits
ax.set_xlim([-130, -65])
ax.set_ylim([25, 50])

plt.show()
```

    C:\Users\Julius\AppData\Local\Temp\ipykernel_8928\4049436674.py:2: FutureWarning: The geopandas.dataset module is deprecated and will be removed in GeoPandas 1.0. You can get the original 'naturalearth_cities' data from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/.
      cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities')) # note this is worldwide
    


    
![png](2024-04-03-geospatial-data-plotting-tutorial_files/2024-04-03-geospatial-data-plotting-tutorial_9_1.png)
    


### Use Downloaded Shapefile

Because this is getting (unfortunately) deprecated, you might have to download your shapefiles. To use a downloaded shapefile, simply point geopandas to it.


```python
us_states = gpd.read_file('States_shapefile-shp/States_shapefile.shp')
us_states.head()
```




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
      <th>FID</th>
      <th>Program</th>
      <th>State_Code</th>
      <th>State_Name</th>
      <th>Flowing_St</th>
      <th>FID_1</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>PERMIT TRACKING</td>
      <td>AL</td>
      <td>ALABAMA</td>
      <td>F</td>
      <td>919</td>
      <td>POLYGON ((-85.07007 31.98070, -85.11515 31.907...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>None</td>
      <td>AK</td>
      <td>ALASKA</td>
      <td>N</td>
      <td>920</td>
      <td>MULTIPOLYGON (((-161.33379 58.73325, -161.3824...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>AZURITE</td>
      <td>AZ</td>
      <td>ARIZONA</td>
      <td>F</td>
      <td>921</td>
      <td>POLYGON ((-114.52063 33.02771, -114.55909 33.0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>PDS</td>
      <td>AR</td>
      <td>ARKANSAS</td>
      <td>F</td>
      <td>922</td>
      <td>POLYGON ((-94.46169 34.19677, -94.45262 34.508...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>None</td>
      <td>CA</td>
      <td>CALIFORNIA</td>
      <td>N</td>
      <td>923</td>
      <td>MULTIPOLYGON (((-121.66522 38.16929, -121.7823...</td>
    </tr>
  </tbody>
</table>
</div>



We can plot it the same way.


```python
fig, ax = plt.subplots(figsize=(10, 10))
us_states.boundary.plot(ax=ax)
ax.set_title("Map of US States")

plt.show()
```


    
![png](2024-04-03-geospatial-data-plotting-tutorial_files/2024-04-03-geospatial-data-plotting-tutorial_14_0.png)
    


### ArcGIS

You can also get data from ArcGIS.


```python
# URL of the shapefile
url = 'https://opendata.arcgis.com/datasets/1b02c87f62d24508970dc1a6df80c98e_0.zip'

# Read the shapefile directly from the URL
states = gpd.read_file(url)

# Plot it
fig, ax = plt.subplots(figsize=(12, 8))
states.plot(ax=ax, edgecolor='black', facecolor='white', linewidth=0.5)
ax.set_title('Map of US States', fontsize=16)
ax.axis('off')
plt.tight_layout()
plt.show()
```


    
![png](2024-04-03-geospatial-data-plotting-tutorial_files/2024-04-03-geospatial-data-plotting-tutorial_17_0.png)
    


### Folium

You can also use [folium](https://python-visualization.github.io/folium/latest/) and [Nominatim](https://nominatim.org/) with [GeoPy](https://geopy.readthedocs.io/en/stable/).


```python
import folium
from geopy.geocoders import Nominatim
```


```python
# Create a sample DataFrame with addresses
data = {
    "Address": [
        "1600 Pennsylvania Avenue NW, Washington, DC 20500",
        "One Apple Park Way, Cupertino, CA 95014",
        "1 Tesla Road, Austin, TX 78725",
        "1 Microsoft Way, Redmond, WA 98052",
        "1 Amazon Way, Seattle, WA 98109",
    ]
}
df = pd.DataFrame(data)

# Initialize the geocoder
geolocator = Nominatim(user_agent="my_app")


# Function to geocode addresses and return latitude and longitude
def geocode(address):
    location = geolocator.geocode(address)
    if location:
        return [location.latitude, location.longitude]
    return None


# Apply the geocode function to the 'Address' column
df["Coordinates"] = df["Address"].apply(geocode)

# Create a Folium map centered on the United States
map_center = [37.0902, -95.7129]  # Coordinates for the center of the US
map_zoom = 4
usa_map = folium.Map(location=map_center, zoom_start=map_zoom)

# Iterate over the DataFrame rows and add markers to the map
for _, row in df.iterrows():
    if row["Coordinates"]:
        folium.Marker(location=row["Coordinates"], popup=row["Address"]).add_to(usa_map)

# Display the map
usa_map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-3.7.1.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_bbca9871f8989016399e55057840e95c {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_bbca9871f8989016399e55057840e95c&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_bbca9871f8989016399e55057840e95c = L.map(
                &quot;map_bbca9871f8989016399e55057840e95c&quot;,
                {
                    center: [37.0902, -95.7129],
                    crs: L.CRS.EPSG3857,
                    zoom: 4,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_cc4855f2497b4cb70d55ba83c21e5b31 = L.tileLayer(
                &quot;https://tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;\u0026copy; \u003ca href=\&quot;https://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 19, &quot;maxZoom&quot;: 19, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            );


            tile_layer_cc4855f2497b4cb70d55ba83c21e5b31.addTo(map_bbca9871f8989016399e55057840e95c);


            var marker_51be92cdc6234b02a49e734ed01a1815 = L.marker(
                [38.897699700000004, -77.03655315],
                {}
            ).addTo(map_bbca9871f8989016399e55057840e95c);


        var popup_918f53d7e9d408ae4c7002928d4a140b = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_948f72f860e396d2a17b7b207bb48fab = $(`&lt;div id=&quot;html_948f72f860e396d2a17b7b207bb48fab&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;1600 Pennsylvania Avenue NW, Washington, DC 20500&lt;/div&gt;`)[0];
                popup_918f53d7e9d408ae4c7002928d4a140b.setContent(html_948f72f860e396d2a17b7b207bb48fab);



        marker_51be92cdc6234b02a49e734ed01a1815.bindPopup(popup_918f53d7e9d408ae4c7002928d4a140b)
        ;




            var marker_d04f75af40ed8cc743761e3dd4d736ab = L.marker(
                [30.2229058, -97.61874768293302],
                {}
            ).addTo(map_bbca9871f8989016399e55057840e95c);


        var popup_0e8060814106ffd2676ac2d6113f9c23 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_9f5998369c90261ad54ac72b616e1c44 = $(`&lt;div id=&quot;html_9f5998369c90261ad54ac72b616e1c44&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;1 Tesla Road, Austin, TX 78725&lt;/div&gt;`)[0];
                popup_0e8060814106ffd2676ac2d6113f9c23.setContent(html_9f5998369c90261ad54ac72b616e1c44);



        marker_d04f75af40ed8cc743761e3dd4d736ab.bindPopup(popup_0e8060814106ffd2676ac2d6113f9c23)
        ;




            var marker_bb7a5ec29ed6a4ae439ae70f0678ba64 = L.marker(
                [47.6411346, -122.12676761148089],
                {}
            ).addTo(map_bbca9871f8989016399e55057840e95c);


        var popup_534741df466fd44f15ca17842cc0eddc = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_64817f622236557387236e61b710e32a = $(`&lt;div id=&quot;html_64817f622236557387236e61b710e32a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;1 Microsoft Way, Redmond, WA 98052&lt;/div&gt;`)[0];
                popup_534741df466fd44f15ca17842cc0eddc.setContent(html_64817f622236557387236e61b710e32a);



        marker_bb7a5ec29ed6a4ae439ae70f0678ba64.bindPopup(popup_534741df466fd44f15ca17842cc0eddc)
        ;



&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### Contextily

Another option is to use [contextily](https://contextily.readthedocs.io/en/latest/).


```python
import contextily as ctx

# Ensure the CRS is compatible with web tile services
us_states_crs = us_states.to_crs(epsg=3857)

ax = us_states_crs.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
ctx.add_basemap(ax)
plt.show()

```


    
![png](2024-04-03-geospatial-data-plotting-tutorial_files/2024-04-03-geospatial-data-plotting-tutorial_24_0.png)
    

