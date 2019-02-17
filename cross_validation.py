import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
import data_scan_pandas
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

path = r'%s' % os.getcwd().replace('\\','/')
filepaths = [path+"/train/pos/", path+"/train/neg/"]
jsonpath = path+"/training_dataframe.json"
english_stop_words = set(stopwords.words('english'))
#print(filepaths)
#print(jsonpath)

def dataframe_from_json(path):
    df = pd.read_json(jsonpath)
    return df

def save_dataframe(df):
    df.to_json(jsonpath)
    return

def dataframe_from_folder(paths):
    text ,sentiment = data_scan_pandas.merge_data(paths)
    processed_text = data_scan_pandas.preprocess(text)
    data = zip(processed_text,sentiment)
    dataframe = pd.DataFrame.from_records(data, columns=['text', 'sentiment'])
    dataframe.to_json(jsonpath)
    return dataframe

def remove_stopwords_from_text(comment):
    comment_without_stopwords = [word for word in comment.split() if not word in english_stop_words]
    return(comment_without_stopwords)

def df_no_stopwords(df):
    df['no stopwords'] = df.apply(lambda row: remove_stopwords_from_text(row['text']),axis =1)
    return(df)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text]
    return lemmatized_text

def df_lemmatize(df):
    df['lemmatized']= df.apply(lambda row: lemmatize_text(row['no stopwords']),axis = 1)
    return(df)

def feature_extraction(df):
    X, y = df.loc[:,df.columns != 'sentiment'], df['sentiment']
    X = df_no_stopwords(X)
    X = df_lemmatize(X)
    return X,y

def kfold(dataframe,fold_number):
    X,y = feature_extraction(dataframe)
    kf = KFold(n_splits=fold_number,shuffle=True)
    for train_index, test_index in kf.split(X):
        #print(train_index, test_index)
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        
    return

#dataframe_from_folder(filepaths)
#kfold(dataframe_from_json(jsonpath), 4)
print(feature_extraction(dataframe_from_json(jsonpath))[0])