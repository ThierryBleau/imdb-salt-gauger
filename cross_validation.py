import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import data_scan_pandas
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

path = r'%s' % os.getcwd().replace('\\','/')
filepaths = [path+"/train/pos/", path+"/train/neg/"]
jsonpath = path+"/training_dataframe.json"
jsonbigram = path+"/bigram.json"
english_stop_words = set(stopwords.words('english'))
#print(filepaths)
#print(jsonpath)

def dataframe_from_json(path):
    df = pd.read_json(path, orient='records')
    return df

def save_dataframe(df,path):
    df.to_json(path)
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

def ngramize_text(list_of_words):
    string = ' '.join(list_of_words)
    bigram_vectorizer = CountVectorizer(ngram_range=(1,2))
    analyze = bigram_vectorizer.build_analyzer()
    return analyze(string)

def df_ngramize(df):
    df['bigramized'] = df.apply(lambda row: ngramize_text(row['lemmatized']),axis =1)
    return df

def feature_extraction(df):
    X, y = df.loc[:,df.columns != 'sentiment'], df['sentiment']
    X = df_no_stopwords(X)
    X = df_lemmatize(X)
    X = df_ngramize(X)
    return X,y

def kfold(X,y,fold_number):

    kf = KFold(n_splits=fold_number,shuffle=True)
    indices = []
    for train_index, test_index in kf.split(X):
        #print(train_index, test_index)
        index = (X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index])
        
    return indices

def model(X,Y):
    embed_dim = 128
    lstm_out = 196

    model = Sequential()
    model.add(Embedding(2000, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

    print(model.summary())
    batch_size = 32
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
    batch_size = 32
    model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)

    validation_size = 1500

    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]
    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))
    return model.summary()

#dataframe_from_folder(filepaths)
#kfold(dataframe_from_json(jsonpath), 4)
#df,sentiments = feature_extraction(dataframe_from_json(jsonpath))
#df = df['bigramized']
#print(df)
#print(sentiments)
#df = df.to_frame().join(sentiments.to_frame())
#print(df)
#save_dataframe(df,jsonbigram)
bigram = dataframe_from_json(jsonbigram)
model(bigram['bigramized'],bigram['sentiment'])