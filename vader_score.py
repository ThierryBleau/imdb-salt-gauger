import data_scan
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

X_train, X_test, y_train, y_test = data_scan.main_test()

analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    score = 1 + (score['neg'])
    return score

def vader(data):
    vader_score=[]
    for i in range(0,len(data)):
        vader_score.append(sentiment_analyzer_scores(data[i]))
    return vader_score
