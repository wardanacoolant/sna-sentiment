# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 00:04:43 2020

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
df = pd.read_csv("folium_point_check_label_angka.csv",encoding ="latin1")
df.head()

#membuat visualisasi peta
map1 = folium.Map(
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
      
for lat,lan,text,sentiment in zip(df['latitude'],df['longitude'],df['text'],df['sentiment']): 
    # Marker() takes location coordinates  
    # as a list as an argument 
    folium.Marker(location=[lat,lan],popup = text, 
                  icon= folium.Icon(color=color(sentiment))).add_to(map1) 
                    
# Save the file created above 
print(map1.save('point.html')) 



