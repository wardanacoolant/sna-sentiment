# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:52:53 2019

@author: Fathiyarizq Mahendra
"""

from sklearn.model_selection import train_test_split
import nltk
import pandas as pd
import numpy as np
import re
import string
import sklearn as sk
import math
from sklearn import model_selection, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory1 = StopWordRemoverFactory()
stopword = factory1.create_stop_word_remover()

factory2 = StemmerFactory()
stemmer = factory2.create_stemmer()



df_pd = pd.read_csv("ibukota_new.csv",encoding = 'utf-8')
print(list(df_pd))

#Melakukan Case Folding (Penyeragaman Tanda Baca)
case_folding = df_pd['text'].str.lower()
print(case_folding)


#Melakukan Seleksi data bedasarkan nomor baris  
kalimat = df_pd.iloc[0]
print(kalimat['text'])


print (nltk.tokenize.word_tokenize(kalimat['text']))

def identify_tokens(row):
    text = row['text']
    tokens = nltk.tokenize.word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

df_pd['words'] = df_pd.apply(identify_tokens, axis=1)
print(df_pd['words'])


my_list = kalimat
#stop = stopword.remove(my_list['text'])
#print(stop)
#output = stemmer.stem(my_list['text'])
#print(output)

# Using a Python list comprehension method to apply to all words in my_list

print ([stopword.remove(str(word)) for word in my_list])
def remove_stops(row):
    my_list = row['words']
    stop = [stopword.remove(str(word)) for word in my_list]
    return (stop)
df_pd['stop_words'] = df_pd.apply(remove_stops, axis=1)

print ([stemmer.stem(str(word)) for word in my_list])
def stem_list(row):
    my_list = row['stop_words']
    stemmed_list = [stemmer.stem(str(word)) for word in my_list]
    return (stemmed_list)

df_pd['stemmed_words'] = df_pd.apply(stem_list, axis=1)


def rejoin_words(row):
    my_list = row['stemmed_words']
    joined_words = ( " ".join(my_list))
    return joined_words

df_pd['processed'] = df_pd.apply(rejoin_words, axis=1)

#df_pd_train.to_csv('isis_new_train.csv',index=False)
#df_pd_test.to_csv('isis_new_test.csv',index=False)
df_pd.to_csv('ibukota_new_processed.csv', index=False)