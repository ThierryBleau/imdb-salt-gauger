import data_scan
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#http://blog.datumbox.com/machine-learning-tutorial-the-naive-bayes-text-classifier/
"""
we use the 10000 most frequent words in the wohole text and encode them by their occurence in the comments (0 or 1)
then we calculate the probabilities for Theta1 and Theta0 - see lecture 9

prob_m_features calculates the conditional probability for a feature
it needs an input which is the number of the feature

all_feture_probaabilities calculates uses prob_m_features to calculate the probabilities for 
all the features. returs a list with tuple contaning the probabilities

classify goes through all the test data points and makes a prediction for them by multiplying the porbabilities
it calls all_feature_probabilites and uses the list to extract the probabilities
it also classifies the unseen data, returns a list with 1s and 0s for pos and neg

predic_evaluate calculates the % of succesfull predictions

predict with 0.5066 accuracy
"""

"""importing nd encoding"""
#we need dense matrix, cant load everything into memory, have to set a max - max_features
X_train, X_test, y_train, y_tests = data_scan.main_test()
features = 10000
vect = CountVectorizer(max_features=features,binary=True)
# max features: If not None, build a vocabulary that only consider 
# the top max_features ordered by term frequency across the corpus.
X_train_vectorized = vect.fit_transform(X_train)
X_train_vectorized_array = X_train_vectorized.toarray()

X_test_vectorized = vect.fit_transform(X_test)
X_test_vectorized_array = X_test_vectorized.toarray()
#now x_treain_v_a is a list of list, every sublist is one comment, and every sublist has 10000 elements with the word occurence

'''calculating probabilities'''
#class probabilities
no_positive = len(list(filter(lambda x: x==1 , y_train)))
no_negative = len(list(filter(lambda x: x==0 , y_train)))
prob_positive = no_positive / len(y_train)
prob_negative = 1-prob_positive

def prob_m_feature(dataset,feature_m):
	""" Calculates the conditional probabilities of feature m with laplace smoothing
		2 cases, where class is 1 and 0 
		goes through all the comments and counts the number of occurences when the word occured
		and it was a pos or neg comment"""
	m_true_1 = 0
	m_true_0 = 0
	feature = []
	for f in range(0,len(dataset)):
		if dataset[f][feature_m]==1:
			if y_train[f]==1:    
				m_true_1 = m_true_1 + 1
			else:
				m_true_0 = m_true_0 + 1
	#conditional probabilitiesfor classes
	theta_m_1 = (m_true_1+1) / (no_positive+2) # +1 and +2 are for laplace smoothing
	theta_m_0 = (m_true_0+1) / (no_negative+2)
	#print(theta_m_0, theta_m_1)
	return theta_m_0, theta_m_1

def all_feature_probabilities(data):
	'''calcualtes the two probabilities for all the features. theta_m_0, theta_m_1 (conditional prob for feature m)
	returns a list of tuples. tuples contain the two probabilities for the features'''
	feature_probabilities = []
	for i in range(0,features):
		theta_m_0, theta_m_1 = prob_m_feature(data,i)
		prob_tuple = (theta_m_0, theta_m_1)
		feature_probabilities.append(prob_tuple)
	return feature_probabilities

def classify(data_train,data_test):
	"""goes through all the features in 1 comment""" 
	classified = []
	f_p = all_feature_probabilities(data_train)
	for i in range(0,len(data_test)):
		#goes throgh all the data points
		prob_feat_sum = 0
		for j in range(0,features):
			#goes through all the features in one data point and calculates the probabilities for a comment
			feature_prob = (data_test[i][j] * math.log(f_p[j][1]/f_p[j][0])) + ( (1-data_test[i][j]) * math.log((1-f_p[j][1])/(1-f_p[j][0])) ) 
			prob_feat_sum = prob_feat_sum + feature_prob
		#probability for 1 comment
		prob_feat_sum = prob_feat_sum + math.log(prob_positive/prob_negative)

		#classification
		if prob_feat_sum>0:
			classified.append(1)
		elif prob_feat_sum<0:
			classified.append(0)
	return classified

def predict_evaluate():
	classified = classify(X_train_vectorized_array,X_test_vectorized_array)
	error = 0
	for i in range(0,len(y_tests)):
		if classified[i]!=y_tests[i]:
			error = error + 1
	precision = (len(y_tests)-error)/len(y_tests)
	print(precision)
	return

predict_evaluate()
		