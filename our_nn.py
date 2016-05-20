#!/usr/bin/env python
import utils
import sys
import random
import time
import features
import operator
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

"""Our multi-class classifier
Uses 15 internal neural nets: 5 using the one vs others method and
10 more for each pair of classes."""
class Our_NN:
	def __init__(self):
		self.submodels = []
		self.test_data = utils.get_test_data()
		self.ONEvALL = 0
		self.TWOvALL = 1
		self.THREEvALL = 2
		self.FOURvALL = 3
		self.FIVEvALL = 4
		self.ONEvTWO = 5
		self.ONEvTHREE = 6
		self.ONEvFOUR = 7
		self.ONEvFIVE = 8
		self.TWOvTHREE = 9
		self.TWOvFOUR = 10
		self.TWOvFIVE = 11
		self.THREEvFOUR = 12
		self.THREEvFIVE = 13
		self.FOURvFIVE = 14

	def train_submodels(self, train_data):
		pass



#to train:
#for example, label in train data:
#	vec = features.build_feature_vec(example)
#	train_examples.append(vec, label)
#PottsNet.fit(train_examples)
