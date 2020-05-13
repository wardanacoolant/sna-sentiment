# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:34:48 2020

@author: Fathiyarizq Mahendra
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:57:10 2020

@author: Fathiyarizq Mahendra
"""

import multiprocessing as mp
from multiprocessing import Pool
import csv
import requests
import string
import pandas as pd
import numpy as np
import geopy
import matplotlib.pyplot as plt
import folium
from folium.plugins import FastMarkerCluster


from geopy.geocoders import Nominatim
locator = Nominatim(user_agent="myGeocoder")
from geopy.extra.rate_limiter import RateLimiter

#membaca data csv
df = pd.read_csv("training_location_processed_folium.csv")
df.head()
print(df)

#membuat visualisasi peta
map1 = folium.Map(
    #mengatur latitude dan longitude
    location=[-2.548926, 118.0148634],
    #mengatur tampilan peta
    tiles='cartodbpositron',
    #mengatur zoom pada peta
    zoom_start=1,
)
#mengatur titik kordinat peta
df.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]]).add_to(map1), axis=1)
map1
map1.save("mapfix.html")