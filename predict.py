#!/usr/bin/env python
import utils
import sys
import random
import time
import our_svm
import our_nn
import features
import warnings
import PottsNet
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


METHOD_NN = 0
METHOD_SVM = 1
BASELINE = 2
RANDOM_BASELINE = 3
BASELINE_VECTOR_SIZE = 20
USE_GLOVE = False

#Select option: SVM or NN
def get_method():
	print "Which method would you like to test?"
	method = raw_input("(N)eural nets or (S)VM or simple (B)aseline:")
	if method == "": return get_method()
	if method[0] == 'N': return METHOD_NN
	elif method[0] == 'S': return METHOD_SVM
	elif method[0] == 'B': return BASELINE
	elif method[0] == 'R': return RANDOM_BASELINE
	else:
		print "Invalid option"
		return get_method()


#parse json data into desired form
def get_data():
	# (text, rating) pairs
	raw_data = utils.create_train_data('yelp_data/very_small.json')
	return raw_data

"""Build an array of SVM binary classifiers to be used inplementing out classification model"""
def run_svm_classifier():
	data = get_data()
	model = our_svm.Our_SVM(USE_GLOVE)
	model.train_submodels(data)
	predictions, targets = model.score_model()
	report_results(targets, predictions)


"""Build an array of neural net binary classifiers to be used inplementing out classification model"""
def run_nn_classifier():
	m = our_nn.Our_NN(USE_GLOVE)
	raw_data = get_data()
	m.train_submodels(raw_data)
	preds = m.score_model()
	test_examples = m.test_data
	target = [int(exp[1]) for exp in test_examples]
	report_results(target, preds)
	#my_report(new_name, preds)


"""Randomly guess predictions to get a basline accuracy"""
def run_random_baseline(k=5):
	raw_data = get_data()
	target_data = [int(pair[1]) for pair in raw_data]
	predictions = [random.randint(1, 5) for _ in raw_data]
	report_results(target_data, predictions)


"""Use an SVM natively as a 5-class classifeir to get a legitimate baseline"""
def run_svm_baseline():
	raw_data = get_data()
	#input_data = np.array([np.array([random.uniform(-0.5, 0.5) for i in range(BASELINE_VECTOR_SIZE)]) for j in range(len(raw_data))])
	input_data = features.generate_feature_vectors(raw_data, False)
	target_data = np.array([int(pair[1]) for pair in raw_data])
	state = random.randint(0, int(time.time()))
	#X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(input_data, target_data, test_size=0.1, random_state=state)
	test_data = utils.get_test_data()
	target_test = np.array([int(pair[1]) for pair in test_data])
	input_test = features.generate_feature_vectors(test_data, False)
	clf = svm.SVC(kernel='linear', C=1).fit(input_data, target_data)
	x = clf.score(input_test, target_test)
	print x 


"""Neural Net does not support multiclass classification"""
def nn_baseline_predict(net):
	test_data = utils.get_test_data()
	test_examples = features.generate_feature_vectors(test_data, USE_GLOVE)
	#test_examples = np.array([np.array([random.uniform(-0.5, 0.5) for i in range(BASELINE_VECTOR_SIZE)]) for j in range(len(test_data))])
	test_target = [1 if int(p[1]) > 2 else -1 for p in test_data]
	predictions = []
	for ex in test_examples:
		pred = net.predict(ex)
		if pred > 0: predictions.append(1)
		else: predictions.append(-1)
	report_results(test_target, predictions)

"""Cannot run multiclass baseline."""
def run_nn_baseline():
	raw_data = get_data()
	feature_vectors = features.generate_feature_vectors(raw_data, USE_GLOVE)
	#feature_vectors = np.array([np.array([random.uniform(-0.5, 0.5) for i in range(BASELINE_VECTOR_SIZE)]) for j in range(len(raw_data))])
	target_data = [1 if int(pair[1]) > 2 else -1 for pair in raw_data]
	train_data = [(vec, np.array([star])) for vec, star in zip(feature_vectors, target_data)]
	net = PottsNet.ShallowNeuralNetwork()
	print "Training Neural Net..."
	net.fit(train_data)
	print "Predicting results"
	nn_baseline_predict(net)


def run_baselines():
	run_svm_baseline();
	#run_nn_baseline()


"""Print evaluation metrics"""
def report_results(y_test, y_pred):
	#print(f1_score(y_test, y_pred, average="macro"))
	print "="*80
	print(classification_report(y_test, y_pred, target_names=['1', '2', '3', '4', '5']))
	print "\naccuracy:"
	print(accuracy_score(y_test, y_pred)) 
	print "="*80


def main():
	warnings.filterwarnings("ignore")
	#set a random seed
	random.seed(int(time.time()))
	#get which prediction method to use
	method = get_method();
	if method == METHOD_NN: run_nn_classifier()
	elif method == METHOD_SVM: run_svm_classifier()
	elif method == RANDOM_BASELINE: run_random_baseline()
	else: run_baselines()



main()
