#!/usr/bin/env python
import utils
import sys
import numpy as np
from sklearn import cross_validation
from sklearn import svm


METHOD_NN = 0
METHOD_SVM = 1

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
	data = utils.create_train_data('yelp_data/tiny.json')

#train the 5  1 vs all classifiers
#train the 10 binary classifiers
def train_svm():
	pass

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



def main():
	method = get_method();
	if method == 0: train_nn()
	else: train_svm()



main()
