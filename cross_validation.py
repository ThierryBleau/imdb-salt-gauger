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
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

path = r'%s' % os.getcwd().replace('\\','/')
filepaths = [path+"/train/pos/", path+"/train/neg/"]
jsonpath = path+"/training_dataframe.json"
jsonlemmas = path+"/lemmas.json"
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

def feature_extraction(df):
    X, y = df.loc[:,df.columns != 'sentiment'], df['sentiment']
    X = df_no_stopwords(X)
    X = df_lemmatize(X)
    #X = df_ngramize(X)
    return X,y

def kfold(X,y,fold_number):

    kf = KFold(n_splits=fold_number,shuffle=True)
    indices = []
    for train_index, test_index in kf.split(X):
        #print(train_index, test_index)
        index = (X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index])
        
    return indices

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])
 
# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

def alphabetize(lemmas):
    new_lemmas = []
    for lemma in lemmas:
        new_lemma = [word for word in lemma.split(' ') if word.isalpha()]
        new_lemma = ' '.join(new_lemma)
        new_lemmas.append(new_lemma)
    return new_lemmas

def model(length, vocab_size):
    # channel 1
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, 100)(inputs1)
	conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)
	# channel 2
	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 100)(inputs2)
	conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)
	# channel 3
	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 100)(inputs3)
	conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
	# merge
	merged = concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = Dense(10, activation='relu')(merged)
	outputs = Dense(1, activation='sigmoid')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize
	print(model.summary())
	#plot_model(model, show_shapes=True, to_file='multichannel.png')
	return model

#dataframe_from_folder(filepaths)
#kfold(dataframe_from_json(jsonpath), 4)
#df,sentiments = feature_extraction(dataframe_from_json(jsonpath))
#df = df['lemmatized']
#print(df)
#print(sentiments)
#df = df.to_frame().join(sentiments.to_frame())
#print(df)
#save_dataframe(df,jsonlemmas)
lemmas = dataframe_from_json(jsonlemmas)
#print(lemmas['lemmatized'].shape)
#model(lemmas['lemmatized'].to_frame(),lemmas['sentiment'].to_frame())
trainLines, trainLabels = lemmas['lemmatized'].values.tolist(), lemmas['sentiment'].values.tolist()
trainLines = alphabetize(trainLines)

tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
print(trainX.shape)
 
# define model
model = model(length, vocab_size)
# fit model
model.fit([trainX,trainX,trainX], trainLabels, epochs=1, batch_size=16)
# save the model
model.save('model.h5')