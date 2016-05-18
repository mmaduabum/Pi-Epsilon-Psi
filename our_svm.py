#!/usr/bin/env python
import utils
import sys
import random
import time
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


#(should be a way to save trained classifiers so we dont need to do this every time)
#http://scikit-learn.org/stable/modules/model_persistence.html


#train the 5  1 vs all classifiers
#train the 10 binary classifiers
def train_svms(raw_data, size=20):
	#create feature vectors for each review text. Currently just using a random vector
	input_data = np.array([np.array([random.uniform(-0.5, 0.5) for i in range(size)]) for j in range(len(raw_data))]) 
	#based on the classifier being trained, generate a list of labels for each train example
	#e.g. if training 1 vs all labels can be 1, -1
	target_data = [int(pair[1]) % 2 for pair in raw_data]
	state = random.randint(0, int(time.time()))
	#Once data has been translated to feature vector and target classes have been decided, train the model
	#Plan is to keep 15 of these clf's so that any unseen review can be classified
	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(input_data, target_data, test_size=0.1, random_state=state)
	clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)
	print clf.score(X_test, Y_test) 
