#!/usr/bin/env python
import utils
import sys
import random
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


METHOD_NN = 0
METHOD_SVM = 1
BASELINE_VECTOR_SIZE = 20


#Select option: SVM or NN
def get_method():
	print "Which method would you like to test?"
	method = raw_input("(N)eural nets or (S)VM:")
	if method[0] == 'N': return METHOD_NN
	elif method[0] == 'S': return METHOD_SVM
	else:
		print "Invalid option"
		return get_method()


#parse json data into desired form
def get_data():
	# (text, rating) pairs
	raw_data = utils.create_train_data('yelp_data/tiny.json')
	return raw_data

#train the 5  1 vs all classifiers
#train the 10 binary classifiers
def train_svm():
	raw_data = get_data()
	input_data = np.array([random.uniform(-0.5, 0.5) for i in range(BASELINE_VECTOR_SIZE)]) 
	target_data = [int(pair[1]) for pair in raw_data]
	X_train. X_test, Y_train, Y_test = cross_validation.train_test_split(input_data, target_data, test_size=0.1, random_state=0)

def train_nn():
	pass

#to train:
#for example, label in train data:
#	vec = features.build_feature_vec(example)
#	train_examples.append(vec, label)
#PottsNet.fit(train_examples)


#(should be a way to save trained classifiers so we dont need to do this every time)

#predict
#report


def run_baseline(k=5):
	raw_data = get_data()
	target_data = [int(pair[1]) for pair in raw_data]
	predictions = [random.randint(1, 5) for _ in raw_data]
	report_results(target_data, predictions)


def report_results(y_test, y_pred):
	print(f1_score(y_test, y_pred, average="macro"))
	print(precision_score(y_test, y_pred, average="macro"))
	print(recall_score(y_test, y_pred, average="macro")) 
	print(accuracy_score(y_test, y_pred)) 


def main():
	#method = get_method();
	#if method == 0: train_nn()
	#else: train_svm()

	run_baseline()



main()
