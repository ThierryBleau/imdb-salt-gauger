import data_scan
import vader_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import decomposition
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas
import time
from nltk.stem import WordNetLemmatizer

X_train,X_test, y_train, y_test = data_scan.main_test()
#X_train,y_train, X_test, y_test = data_scan.main()
'''
lemmatizer = WordNetLemmatizer()

for i in range(len(X_train)):
    X_train[i] = ' '.join([lemmatizer.lemmatize(word) for word in X_train[i].split()])
'''
normalizer_tranformer = Normalizer(norm='l2')
count_vect = CountVectorizer(ngram_range=(1, 2))
tfidf_transformer = TfidfTransformer(sublinear_tf=True)
pca = decomposition.PCA(n_components=40)
svc = LinearSVC( loss = 'hinge')

pipe = Pipeline([('count',count_vect),('tfidf', tfidf_transformer),('normalizer', normalizer_tranformer),('svc', svc)])

start = time.time()

pipe.fit(X_train,y_train)
pred = pipe.predict(X_test)


print(time.time()-start)


print(metrics.classification_report(y_test, pred))

'''
df = pandas.DataFrame(data={"ID": y_test, "Category": pred})
df.to_csv("./file.csv", sep=',',index=False)
'''
