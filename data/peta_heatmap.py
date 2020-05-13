# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:32:53 2020

@author: Fathiyarizq Mahendra
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:52:39 2020

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
from folium.plugins import MarkerCluster
from folium.plugins import FastMarkerCluster
from folium.plugins import HeatMap


from geopy.geocoders import Nominatim
locator = Nominatim(user_agent="myGeocoder")
from geopy.extra.rate_limiter import RateLimiter

#membaca data csv
df = pd.read_csv("folium_point_check_no jabodetabek.csv",encoding ="latin1")
df.head()

#membuat visualisasi peta
map = folium.Map(
    #mengatur latitude dan longitude
    location=[-2.548926, 118.0148634],
    #mengatur tampilan peta
    tiles='cartodbpositron',
    #mengatur zoom pada peta
    zoom_start=3,
)

heat_df = df[df['sentiment'].str.match('positif')]
heat_df = df[['latitude', 'longitude']]

heat_data = [[row['latitude'],row['longitude']] for index, row in heat_df.iterrows()]

HeatMap(heat_data).add_to(map)

f = 'map_heatmap.html'
map.save(f)
