from os import listdir
import re
from sklearn.model_selection import train_test_split

#NOTE are going to use this function to scan the train and test data
#NOTE now we have X_train, X_test, kaggle_test
#NOTE X data has no special characters, just words seperated by space
#NOTE Y data only has 0s and 1s for negative and positive

def merge_data():
    #creater lists where each entry is a string with the whole comment
    #and another list with the same index for the postivity or negativity
    #0 is negative ! is positive
    neg = listdir("/home/botond/Boti/uni/mcgill/comp551/a2/neg")
    pos = listdir("/home/botond/Boti/uni/mcgill/comp551/a2/pos")
    data = []
    data_labels = []
    
    for i in neg:
        with open("/home/botond/Boti/uni/mcgill/comp551/a2/neg/"+i,"r") as f:
            data.append(f.read())
            data_labels.append(0)

    for i in pos:
        with open("/home/botond/Boti/uni/mcgill/comp551/a2/pos/"+i,"r") as f:
            data.append(f.read())
            data_labels.append(1)

    return data, data_labels

#removing .!?/
#removing new line characters
#https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess(data):
    data = [REPLACE_NO_SPACE.sub("", line.lower()) for line in data]
    data = [REPLACE_WITH_SPACE.sub(" ", line) for line in data]    
    return data

def kaggle_read():
    kaggle_files = listdir("/home/botond/Boti/uni/mcgill/comp551/a2/comp-551-imbd-sentiment-classification/test")
    kaggle_data = []
    
    for i in kaggle_files:
        with open("/home/botond/Boti/uni/mcgill/comp551/a2/comp-551-imbd-sentiment-classification/test/"+i,"r") as f:
            kaggle_data.append(f.read())        
    return kaggle_data, kaggle_files


def main():
    data, labels = merge_data()
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.8, test_size=0.2)
    # X_train = preprocess(X_train)
    # X_test = preprocess(X_test)
    X_train = preprocess(data)
    y_train = labels
    kaggle_test, kaggle_files = kaggle_read()
    kaggle_test = preprocess(kaggle_test)
    return X_train, y_train, kaggle_test, kaggle_files

