#!/usr/bin/env python
import utils
import sys
import random
import time
import our_svm
import our_nn
import features
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


METHOD_NN = 0
METHOD_SVM = 1
BASELINE = 2
BASELINE_VECTOR_SIZE = 20


#Select option: SVM or NN
def get_method():
	print "Which method would you like to test?"
	method = raw_input("(N)eural nets or (S)VM or simple (B)aseline:")
	if method[0] == 'N': return METHOD_NN
	elif method[0] == 'S': return METHOD_SVM
	elif method[0] == 'B': return BASELINE
	else:
		print "Invalid option"
		return get_method()


#parse json data into desired form
def get_data():
	# (text, rating) pairs
	raw_data = utils.create_train_data('yelp_data/tiny.json')
	return raw_data

"""Build an array of SVM binary classifiers to be used inplementing out classification model"""
def run_svm_classifier():
	data = get_data()
	model = our_svm.Our_SVM()
	model.train_submodels(data)
	score = model.score_model()
	print score


"""Build an array of neural net binary classifiers to be used inplementing out classification model"""
def run_nn_classifier():
	pass


"""Randomly guess predictions to get a basline accuracy"""
def run_random_baseline(k=5):
	raw_data = get_data()
	target_data = [int(pair[1]) for pair in raw_data]
	predictions = [random.randint(1, 5) for _ in raw_data]
	report_results(target_data, predictions)


"""Use an SVM natively as a 5-class classifeir to get a legitimate baseline
Currently only uses random vectors for features.
TODO: generate a erawl feature vector to obtain the baseline"""
def run_svm_baseline():
	raw_data = get_data()
	#input_data = np.array([np.array([random.uniform(-0.5, 0.5) for i in range(BASELINE_VECTOR_SIZE)]) for j in range(len(raw_data))])
	input_data = features.generate_feature_vectors(raw_data)
	target_data = np.array([int(pair[1]) for pair in raw_data])
	state = random.randint(0, int(time.time()))
	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(input_data, target_data, test_size=0.1, random_state=state)
	clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)
	x = clf.score(X_test, Y_test)
	print x 


def run_baselines():
	run_svm_baseline();
	#run_nn_baseline()


"""Print evaluation metrics"""
def report_results(y_test, y_pred):
	print(f1_score(y_test, y_pred, average="macro"))
	print(precision_score(y_test, y_pred, average="macro"))
	print(recall_score(y_test, y_pred, average="macro")) 
	print(accuracy_score(y_test, y_pred)) 


def main():
	#set a random seed
	random.seed(int(time.time()))
	#get which prediction method to use
	method = get_method();
	if method == METHOD_NN: run_nn_classifier()
	elif method == METHOD_SVM: run_svm_classifier()
	else: run_baselines()



main()
