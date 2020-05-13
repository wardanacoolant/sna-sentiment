# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from twitterscraper import query_tweets
import pandas as pd
import datetime as dt

limit = 1200
begin_date = dt.date(2019,1,1)
end_date = dt.date(2019,12,31)

tweet = 'ibukota baru OR pemindahan ibu kota'
lang = 'id'

tweets = query_tweets(tweet,begindate = begin_date, enddate = end_date, lang = lang,limit = limit)

df = pd.DataFrame(t.__dict__ for t in tweets) 
export = df.to_csv(r'C:\Users\Fathiyarizq Mahendra\Skripsi\new.csv') 