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
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

#menamai locator atau aplikasi mencari geolokasi 
locator = Nominatim(user_agent="myGeocoder")
df = pd.read_csv("ibukota_new_location_processed.csv",encoding='latin1')
df.head()
print(df)

## 1 - fungsi untuk menunda antar panggilan geocoding
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)

## 2- - membuat kolom lokasi
df['address'] = df['location'].apply(geocode)
print(df['location'])
## 3 - membuat bujur, lintang, dan ketinggian dari kolom lokasi (menghasilkan tupel)
df['point'] = df['address'].apply(lambda loc: tuple(loc.point) if loc else None)
print(df['point'])
#Cetak hasil dalam bentuk csv
df.to_csv('ibukota_new_location_processed.csv',encoding = 'utf-8',index=False)
## 4 - split point column into latitude, longitude and altitude columns

#map1 = folium.Map(
#    location=[40, -1.464582],
#    tiles='cartodbpositron',
#    zoom_start=12,
#)
#df.apply(lambda row:folium.CircleMarker(location=[row['point']]).add_to(map1), axis=1)
#map1