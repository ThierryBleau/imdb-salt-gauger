import numpy as np
import pandas as pd
import json
import os
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import data_scan_pandas
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import cross_val_score


kaggle_label = []
english_stop_words = set(stopwords.words('english'))
path = r'%s' % os.getcwd().replace('\\','/')
filepaths = [path+"/train/pos/", path+"/train/neg/"]
jsonpath = path+"/training_dataframe.json"
jsonlemmas = path+"/lemmas.json"
bigramjson = path+"/bigram.json"


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
    df['lemmatized']= df.apply(lambda row: ' '.join(lemmatize_text(row['no stopwords'])),axis = 1)
    return(df)

def ngramize_text(list_of_words):
    string = ' '.join(list_of_words)
    bigram_vectorizer = CountVectorizer(ngram_range=(1,2))
    analyze = bigram_vectorizer.build_analyzer()
    return analyze(string)

def df_ngramize(df):
    df['bigramized'] = df.apply(lambda row: ngramize_text(row['lemmatized']),axis =1)
    return df

def alphabetize(lemmas):
    new_lemmas = []
    for lemma in lemmas:
        new_lemma = [word for word in lemma.split() if word.isalpha()]
        new_lemmas.append(new_lemma)
    return new_lemmas

def feature_extraction(df):
    X, y = df.loc[:,df.columns != 'sentiment'], df['sentiment']
    X = df_no_stopwords(X)
    X = df_lemmatize(X)
    X = df_ngramize(X)
    return X,y

def get_data_raw(paths):
    text, sentiment = data_scan_pandas.merge_data_no_test(paths)
    processed_text = data_scan_pandas.preprocess(text)
    data = zip(processed_text,sentiment)
    dataframe = pd.DataFrame.from_records(data, columns=['text', 'sentiment'])
    dataframe.to_json(jsonpath)
    return dataframe

def dataframe_from_json(path):
    df = pd.read_json(path, orient='records')
    return df
 

def main():
    data = dataframe_from_json(bigramjson)
    print(data.head)

    X, y = np.array(alphabetize(data['bigramized'].values.tolist())), np.array(data['sentiment'].values.tolist())

    kf = KFold(n_splits=10)

    accuracy = []
    fold = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        vect = TfidfVectorizer()
        X_train_tfidf = vect.fit_transform(X_train)
        X_test_tfidf = vect.fit_transform(X_test)
        
    
        text_clf = Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True)),
                    ("svc", LinearSVC())])
        text_clf.fit(X_train, y_train)
        text_clf.predict(X_test)
        
        a= text_clf.score(X_test, y_test)
    
        accuracy.append(a)
        print('[INFO]\tFold %d Accuracy: %f' % (fold, a))
        fold += 1
    
    avgAccuracy = sum(accuracy) / fold
    print('[INFO]\tAccuracy: %f' % avgAccuracy)
    return

def main2():
    data = dataframe_from_json(jsonlemmas)

    X, y = data['lemmatized'].values.tolist(), data['sentiment'].values.tolist()

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        #'clf__max_iter': (5,),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__max_iter': (10, 50, 80),
    }
    grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    
    grid_search.fit(X, y)
    
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return

if __name__ == '__main__':
    main2()