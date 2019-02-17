import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
import data_scan_pandas

path = r'%s' % os.getcwd().replace('\\','/')
filepaths = [path+"/train/pos/", path+"/train/neg/"]
jsonpath = path+"/training_dataframe.json"
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

def kfold(dataframe,fold_number):
    X, y = dataframe.loc[:,dataframe.columns != 'sentiment'], dataframe['sentiment']
    kf = KFold(n_splits=fold_number,shuffle=True)
    for train_index, test_index in kf.split(dataframe.shape):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
    return

#dataframe_from_folder(filepaths)
print(dataframe_from_json(jsonpath))