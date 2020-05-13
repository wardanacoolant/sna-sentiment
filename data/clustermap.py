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

def color(sentiment): 
    if sentiment in ['positif']: 
        col = 'green'
    else: 
        col='red'
    return col 


marker_cluster = MarkerCluster(location=[lat,lan]).add_to(map)

for lat,lan,text,sentiment in zip(df['latitude'],df['longitude'],df['text'],df['sentiment']): 
    # Marker() takes location coordinates  
    # as a list as an argument 
    folium.CircleMarker(location=[lat,lan],
                        radius=9,
                        popup = text,
                        fill_color = color(sentiment),
                        color = "gray",
                        fill_opacity = 0.9).add_to(marker_cluster) 
    
f = 'map_cluster.html'
map.save(f)