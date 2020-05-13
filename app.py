from flask import Flask, render_template, url_for, request, abort, make_response
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
from sklearn.model_selection import cross_val_score
from sklearn import model_selection, svm
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import confusion_matrix, classification_report
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import random
import os
import shutil


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    df_pd = pd.read_csv("data/Pindah Ibukota_complete_processed_ready.csv",
                        encoding="latin1", error_bad_lines=False)

    X = np.array(df_pd['processed'])
    y = np.array(df_pd['sentimen'])

    cv = CountVectorizer()
    X = cv.fit_transform(X)

    from sklearn.model_selection import KFold, RepeatedKFold
    kf = KFold(n_splits=9, random_state=42, shuffle=False)
    print(kf)  # buat tau Kfold dan parameter defaultnya
    i = 1  # ini gapenting, cuma buat nandain fold nya.
    for train_index, test_index in kf.split(X):
        print("Fold ", i)
        print("TRAIN :", train_index, "TEST :", test_index)
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        i += 1
    print("shape x_train :", X_train.shape)
    print("shape x_test :", X_test.shape)

    from sklearn import model_selection, svm
    from sklearn.svm import LinearSVC

    clf = svm.SVC(C=1.0, kernel='linear', degree=1, gamma='auto')
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    content = data  
    y_pred = clf.predict(X_test)
    actual = y_test
    predicted = y_pred
    results = confusion_matrix(actual, predicted) 
    print('Confusion Matrix :')
    print(results)
    akurasi = accuracy_score(y_pred, y_test)*100

    return render_template('prediction.html',content=message,prediction=my_prediction,akurasi=akurasi)

@app.route('/get_map')
def get_map():
    r = int(random.triangular(0,100))
    t = "templates/map_{i}.html"
    for i in range(0,100):
        f = t.format(i=i)
        if os.path.exists(f):
            os.remove(f)
    f = t.format(i=r)
    shutil.copy("templates/map_cluster_complete.html", f)

    r = make_response(render_template(os.path.split(f)[1]))
    r.cache_control.max_age = 0
    r.cache_control.no_cache = True
    r.cache_control.no_store = True
    r.cache_control.must_revalidate = True
    r.cache_control.proxy_revalidate = True
    return r

@app.route('/get_heat')
def get_heat():
    r = int(random.triangular(0,100))
    t = "templates/heatmap{i}.html"
    for i in range(0,100):
        f = t.format(i=i)
        if os.path.exists(f):
            os.remove(f)
    f = t.format(i=r)
    shutil.copy("templates/map_heatmap_3.html", f)

    r = make_response(render_template(os.path.split(f)[1]))
    r.cache_control.max_age = 0
    r.cache_control.no_cache = True
    r.cache_control.no_store = True
    r.cache_control.must_revalidate = True
    r.cache_control.proxy_revalidate = True
    return r

@app.route('/get_positive')
def get_positive():
    r = int(random.triangular(0,100))
    t = "templates/heatmap{i}.html"
    for i in range(0,100):
        f = t.format(i=i)
        if os.path.exists(f):
            os.remove(f)
    f = t.format(i=r)
    shutil.copy("templates/map_heatmap_1.html", f)

    r = make_response(render_template(os.path.split(f)[1]))
    r.cache_control.max_age = 0
    r.cache_control.no_cache = True
    r.cache_control.no_store = True
    r.cache_control.must_revalidate = True
    r.cache_control.proxy_revalidate = True
    return r

@app.route('/get_negative')
def get_negative():
    r = int(random.triangular(0,100))
    t = "templates/heatmap{i}.html"
    for i in range(0,100):
        f = t.format(i=i)
        if os.path.exists(f):
            os.remove(f)
    f = t.format(i=r)
    shutil.copy("templates/map_heatmap_2.html", f)

    r = make_response(render_template(os.path.split(f)[1]))
    r.cache_control.max_age = 0
    r.cache_control.no_cache = True
    r.cache_control.no_store = True
    r.cache_control.must_revalidate = True
    r.cache_control.proxy_revalidate = True
    return r

@app.route('/get_point')
def get_point():
    r = int(random.triangular(0,100))
    t = "templates/point{i}.html"
    for i in range(0,100):
        f = t.format(i=i)
        if os.path.exists(f):
            os.remove(f)
    f = t.format(i=r)
    shutil.copy("templates/point_complete.html", f)

    r = make_response(render_template(os.path.split(f)[1]))
    r.cache_control.max_age = 0
    r.cache_control.no_cache = True
    r.cache_control.no_store = True
    r.cache_control.must_revalidate = True
    r.cache_control.proxy_revalidate = True
    return r

if __name__ == '__main__':
    app.run(debug=True)
