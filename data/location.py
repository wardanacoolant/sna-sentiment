# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:54:32 2020

@author: Fathiyarizq Mahendra
"""

import numpy as np
import pandas as pd
import bs4
from bs4 import BeautifulSoup
import string
import csv
import codecs,json
import requests

#inisialisasi target url yang dituju
base_url = u'https://twitter.com/'

#membaca data csv yang berisi username
df_pd = pd.read_csv("mencari jatidiri.csv",encoding = 'utf-8')

user = df_pd['username']

user_df = pd.DataFrame(user,columns=['username'])

user_unique=user_df['username'].unique()

#melakukan pencarian lokasi bedasarkan username
location = []
for user in user_unique:
    url = str(base_url) + str(user)
    r = requests.get(url)
    soup = BeautifulSoup(r.text,'html.parser')
    loc = soup.find('span',class_='ProfileHeaderCard-locationText')
    if hasattr(loc, 'text') == True:
        location.append(''.join(loc.text.split()))

#mencetak hasil pencarian lokasi         
print(location)
df_pd.to_csv('mencari jatidiri_processed_location.csv', index=False)
	
#df_pd.to_csv('tangerang_processed.csv', index=False)


