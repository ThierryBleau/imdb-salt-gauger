import data_scan
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
import numpy as np
import pandas

"""using sklern for the bernoulli naive bayes to test my implementation's accuracy"""

#import the polished data
X_train, X_test, y_train, y_tests = data_scan.main_test()
features = 10000
vect = CountVectorizer(max_features=features,binary=True)
X_train_vectorized = vect.fit_transform(X_train)
X_train_vectorized_array = X_train_vectorized.toarray()
X_test_vectorized = vect.fit_transform(X_test)
X_test_vectorized_array = X_test_vectorized.toarray()


#vetrorizing the vords - bag of words


#logistic regression
bernoulli = BernoulliNB().fit(X_train_vectorized,y_train)
#prediction
prediction = bernoulli.predict(X_test_vectorized) # return predicted y

print(metrics.accuracy_score(prediction, y_tests))
#writing out the data
#np.savetxt("submission.csv", np.column_stack((kaggle_files, kaggle_label)), delimiter=",")
