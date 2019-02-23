import data_scan
import vader_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import LinearSVC
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas
import time
from yellowbrick.text import FreqDistVisualizer

X_train,X_test, y_train, y_test = data_scan.main_test()
#X_train,y_train, X_test, y_test = data_scan.main()

#vader_train = vader_score.vader(X_train)
vader_test = vader_score.vader(X_test)

vectorizer = CountVectorizer(stop_words="english")
docs = vectorizer.fit_transform(X_train)
features   = vectorizer.get_feature_names()
visualizer = FreqDistVisualizer(features=features)
visualizer.fit(docs)
visualizer.poof()
'''
#X_train_counts = count_vect.transform(X_train)
#X_test_counts = count_vect.transform(X_test) 

#tfidf
tfidf_transformer = TfidfTransformer(sublinear_tf=True).fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

#l2 normalisation
normalizer_tranformer = Normalizer(norm='l2').fit(X=X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)

#classification
start = time.time()
fig, ax = plt.subplots()
#sgd = SGDClassifier(loss='log',class_weight='balanced').fit(X_train_normalized, y_train)
#reg = LogisticRegression(penalty='l2',solver='liblinear').fit(X_train_normalized, y_train)
svc = LinearSVC(loss= "squared_hinge").fit(X_train_normalized, y_train)

'''
plot_decision_regions(X=np.array(count_vect.toarray()), 
                      y=np.array(y_train),
                      clf=svc, 
                      legend=2)

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel('X', size=14)
plt.ylabel('Y', size=14)
plt.title('SVM Decision Region Boundary', size=16)


#sgd_y_pred_normal = sgd.predict(X_test_normalized)
#reg_y_pred_normal = reg.predict(X_test_normalized)
#svc_y_pred_normal = svc.predict(X_test_normalized)

#sgd_y_pred_proba = sgd.predict_proba(X_test_normalized)
#reg_y_pred_proba = reg.predict_proba(X_test_normalized)
'''
#combining
y_pred_vader = []
for i in range(0,len(sgd_y_pred_proba)):
    if ((sgd_y_pred_proba[i][1] + reg_y_pred_proba[i][1])) >= 1:
        y_pred_vader.append(1)
    else:
        y_pred_vader.append(0)

#evaluating
print(y_pred_vader)
counter=0
for i in range(0,len(y_pred_vader)):
    if y_test[i]!=y_pred_vader[i]:
        counter = counter + 1
avg = (len(y_test)-counter) / len(y_test)

print(time.time()-start)
print(y_test[0:9])
print("==")
print(y_pred_vader[0:9])
print(avg)
'''
#print(metrics.classification_report(y_test, y_pred_vader))
#print(metrics.classification_report(y_test, sgd_y_pred_normal))
#print(metrics.classification_report(y_test, reg_y_pred_normal))
#print(metrics.classification_report(y_test, svc_y_pred_normal))

'''
df = pandas.DataFrame(data={"ID": y_test, "Category": y_pred_vader})
df.to_csv("./file.csv", sep=',',index=False)

'''