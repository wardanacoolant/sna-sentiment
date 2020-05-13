# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 19:54:29 2020

@author: Fathiyarizq Mahendra
"""

from sklearn.model_selection import train_test_split
import nltk
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model
import re
import string
import sklearn as sk
import math
import matplotlib.pyplot as plt
import sys
import operator
from sklearn.linear_model import SGDClassifier
import argparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,RepeatedKFold 
from sklearn.model_selection import cross_val_score
from sklearn import model_selection, svm
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import confusion_matrix,classification_report
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

df_pd = pd.read_csv("folium_point_check_label_angka.csv",encoding="latin1",error_bad_lines=False)

X = np.array(df_pd['processed'])
y = np.array(df_pd['label'])

kf=KFold(n_splits=2, random_state=42, shuffle=False)
print(kf)  #buat tau Kfold dan parameter defaultnya
i=1        #ini gapenting, cuma buat nandain fold nya.
for train_index, test_index in kf.split(X):
    print("Fold ", i)
    print("TRAIN :", train_index, "TEST :", test_index)
    X_train=X[train_index]
    X_test=X[test_index]
    y_train=y[train_index]
    y_test=y[test_index]
    i+=1
print("shape x_train :", X_train.shape)
print("shape x_test :", X_test.shape)

def create_bag_of_words(X):
    from sklearn.feature_extraction.text import CountVectorizer
    
    print ('Creating bag of words...')
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    
    # In this example features may be single words or two consecutive words
    # (as shown by ngram_range = 1,2)
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 ngram_range = (1,2), \
                                 max_features = 10000
                                ) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings. The output is a sparse array
    train_data_features = vectorizer.fit_transform(X)
    
    # Convert to a NumPy array for easy of handling
    train_data_features = train_data_features.toarray()
    
    # tfidf transform
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(train_data_features).toarray()

    # Get words in the vocabulary
    vocab = vectorizer.get_feature_names()
   
    return vectorizer, vocab, train_data_features, tfidf_features, tfidf

vectorizer, vocab, train_data_features, tfidf_features, tfidf  = \
    create_bag_of_words(X_train)
    
bag_dictionary = pd.DataFrame()
bag_dictionary['ngram'] = vocab
bag_dictionary['count'] = train_data_features[0]
bag_dictionary['tfidf_features'] = tfidf_features[0]

# Sort by raw count
bag_dictionary.sort_values(by=['count'], ascending=False, inplace=True)
# Show top 10
print(bag_dictionary.head(10))


svm = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', svm.SVC(C=1.0, kernel='poly', degree=2, gamma='auto')),
               ])
    
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

actual = y_test
predicted = y_pred
results = confusion_matrix(actual, predicted) 

print ('Confusion Matrix :')
print(results) 


print(classification_report(y_test, y_pred))
print('SVM Accuracy : ',accuracy_score(y_pred, y_test)*100)




#SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
#SVM.fit(X_train_Tfidf,y_train)
#
## predict the labels on validation dataset
#predictions_SVM = SVM.predict(X_test_Tfidf)

# Use accuracy_score function to get the accuracy

