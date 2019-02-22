import data_scan
import vader_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

X_train, X_test, y_train, y_test = data_scan.main_test()

vader_train = vader_score.vader(X_train)
vader_test = vader_score.vader(X_test)

count_vect = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
X_train_counts = count_vect.transform(X_train)
X_test_counts = count_vect.transform(X_test) 

#tfidf
tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

#l2 normalisation
normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)

#logistic regression
sgd = SGDClassifier(loss='log').fit(X_train_normalized, y_train)
y_pred_normal = sgd.predict(X_test_normalized)
y_pred_proba = sgd.predict_proba(X_test_normalized)

for i in range(0,len(y_pred_proba)):
    y_pred_vader = []
    if ((y_pred_proba[1][i] + vader_test[i]) / 2) > 0.5:
        y_pred_vader.append(1)
    else:
        y_pred_vader.append(0)

counter=0
for i in range(0,len(y_pred_vader)):
    if y_test[i]!=y_pred_vader[i]:
        counter = counter + 1
avg = (len(y_test)-counter) / len(y_test)
print(avg)
print(metrics.classification_report(y_test, y_pred_normal))
