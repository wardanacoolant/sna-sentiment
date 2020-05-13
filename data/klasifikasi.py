import glob
import numpy 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)
from sklearn.calibration import (calibration_curve, CalibratedClassifierCV)


df= pd.read_csv("folium_point_check_no jabodetabek.csv",encoding="latin1",error_bad_lines=False)

df.columns = ['Unnamed','fullname','html','is_retweet','likes','replies','retweet_id','retweeter_userid','retweeter_username','retweets','text','timestamp','timestamp_epochs','tweet_id','tweet_url','user_id','username','words','stemmed_words','stop_words','processed','sentiment','location','address','latitude','longitude','altitude']

data = df.reindex(numpy.random.permutation(df.index))

X = numpy.array(data)

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
    ('tfidf',              TfidfTransformer()),
    ('classifier',         OneVsRestClassifier(LinearSVC()))
])


k_fold = KFold(random_state=len(data), n_splits=6, shuffle=True)

for train_indices, test_indices in enumerate(k_fold(X)):
    train_text = data.iloc[train_indices]['processed'].values
    train_y = data.iloc[train_indices]['sentiment'].values.astype(str)

    test_text = data.iloc[test_indices]['processed'].values
    test_y = data.iloc[test_indices]['sentiment'].values.astype(str)

    #Enter unseen data here
    #files = glob.glob("corpus/*.txt")
    #lines = []
    #for fle in files:
    #    with open(fle) as f:
    #        lines += f.readlines()        
    #test_text = numpy.array(lines)
    #################################

    lb = LabelBinarizer()
    Z = lb.fit_transform(train_y)

    pipeline.fit(train_text, Z)
    predicted = pipeline.predict(test_text)
    predictions = lb.inverse_transform(predicted)

    #Try to add prediction's probability
    #clf = CalibratedClassifierCV(pipeline)
    #clf.fit(train_text, Z)
    #y_proba = clf.predict_proba(test_text)


    df2=pd.DataFrame(predictions)
    df2.index+=1
    df2.index.name='Id'
    df2.columns=['Label']
    df2.to_csv('results.csv',header=True)

    for item, labels in zip(test_text, predictions):
        print('Item: {0} => Label: {1}'.format(item, labels))

    cm = confusion_matrix(test_y, predictions)
    accuracy = accuracy_score(test_y, predictions)

print('The resulting accuracy using Linear SVC is ', (100 * accuracy), '%\n')
#print y_proba

percentage_matrix = 100 * cm / cm.sum(axis=1).astype(float)
plt.figure(figsize=(16, 16))
sns.heatmap(percentage_matrix, annot=True,  fmt='.2f', xticklabels=['positif', 'negatif'], yticklabels=['positif', 'negatif']);
plt.title('Confusion Matrix (Percentage)');
plt.show()
print(classification_report(test_y, predictions,target_names=['positif', 'negatif'], digits=2))