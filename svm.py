import numpy as np
import pandas as pd
import json
import os
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import data_scan_pandas

X_train, y_train, kaggle_data, kaggle_files = data_scan.main()
kaggle_label = []
