import data_scan
import vader_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas
import time

#X_train,X_test, y_train, y_test = data_scan.main_test()
X_train,y_train, X_test, y_test = data_scan.main()

normalizer_tranformer = Normalizer(norm='l2')
count_vect = CountVectorizer(ngram_range=(1, 2))
tfidf_transformer = TfidfTransformer(sublinear_tf=True)
svc = LinearSVC( loss = 'hinge')

start = time.time()

pipe = Pipeline([('count',count_vect),('tfidf', tfidf_transformer),('norm', normalizer_tranformer),('svc', svc)])
pipe.fit(X_train,y_train)
pred = pipe.predict(X_test)

print(time.time()-start)


#print(metrics.classification_report(y_test, pred))

df = pandas.DataFrame(data={"ID": y_test, "Category": pred})
df.to_csv("./file.csv", sep=',',index=False)

