import data_scan
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas

#NOTE we are doing a logistic regression with TFIDF

#import the polished data
X_train, y_train, kaggle_data, kaggle_files = data_scan.main()
kaggle_label = []


#vetrorizing the vords - bag of words
count_vect = CountVectorizer().fit(X_train)
X_train_counts = count_vect.transform(X_train)
kaggle_data_counts = count_vect.transform(kaggle_data) 

#tfidf
tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
kaggle_data_tfidf = tfidf_transformer.transform(kaggle_data_counts)

#l2 normalisation
normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
kaggle_data_normalized = normalizer_tranformer.transform(kaggle_data_tfidf)



#logistic regression
logreg = LogisticRegression().fit(X_train_normalized, y_train)
#prediction
kaggle_label = logreg.predict(kaggle_data_normalized)
#writing out the data
#np.savetxt("submission.csv", np.column_stack((kaggle_files, kaggle_label)), delimiter=",")

df = pandas.DataFrame(data={"ID": kaggle_files, "Category": kaggle_label})
df.to_csv("./file.csv", sep=',',index=False)