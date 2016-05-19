#!/usr/bin/env python
import utils
import sys
import random
import time
import features
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


class Our_SVM:
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
		print "Building Datasets..."
		input_data = features.generate_feature_vectors(train_data)
		#train the one vs others classifiers
		for i in range(5):
			star = i + 1
			target_data = [1 if int(ex[1]) == star else -1 for ex in train_data] 
			self.submodels.append(self.train_svms(input_data, target_data))

		#train the binary classifiers for the 10 pairs
		#create subsets of the train data that have the relevant ratings
		ones_and_twos = [ex for ex in train_data if int(ex[1]) == 1 or int(ex[1]) == 2]
		ones_and_threes = [ex for ex in train_data if int(ex[1]) == 1 or int(ex[1]) == 3]
		ones_and_fours = [ex for ex in train_data if int(ex[1]) == 1 or int(ex[1]) == 4]
		ones_and_fives = [ex for ex in train_data if int(ex[1]) == 1 or int(ex[1]) == 5]
		twos_and_threes = [ex for ex in train_data if int(ex[1]) == 2 or int(ex[1]) == 3]
		twos_and_fours = [ex for ex in train_data if int(ex[1]) == 2 or int(ex[1]) == 4]
		twos_and_fives = [ex for ex in train_data if int(ex[1]) == 2 or int(ex[1]) == 5]
		threes_and_fours = [ex for ex in train_data if int(ex[1]) == 3 or int(ex[1]) == 4]
		threes_and_fives = [ex for ex in train_data if int(ex[1]) == 3 or int(ex[1]) == 5]
		fours_and_fives = [ex for ex in train_data if int(ex[1]) == 4 or int(ex[1]) == 5]

		#generate feature vectors for each data subset
		input_12 = features.generate_feature_vectors(ones_and_twos)
		input_13 = features.generate_feature_vectors(ones_and_threes)
		input_14 = features.generate_feature_vectors(ones_and_fours)
		input_15 = features.generate_feature_vectors(ones_and_fives)
		input_23 = features.generate_feature_vectors(twos_and_threes)
		input_24 = features.generate_feature_vectors(twos_and_fours)
		input_25 = features.generate_feature_vectors(twos_and_fives)
		input_34 = features.generate_feature_vectors(threes_and_fours)
		input_35 = features.generate_feature_vectors(threes_and_fives)
		input_45 = features.generate_feature_vectors(fours_and_fives)

		#generate the targets for each data subset
		target_12 = [1 if int(ex[1]) == 1 else -1 for ex in ones_and_twos]
		target_13 = [1 if int(ex[1]) == 1 else -1 for ex in ones_and_threes]
		target_14 = [1 if int(ex[1]) == 1 else -1 for ex in ones_and_fours]
		target_15 = [1 if int(ex[1]) == 1 else -1 for ex in ones_and_fives]
		target_23 = [1 if int(ex[1]) == 2 else -1 for ex in twos_and_threes]
		target_24 = [1 if int(ex[1]) == 2 else -1 for ex in twos_and_fours]
		target_25 = [1 if int(ex[1]) == 2 else -1 for ex in twos_and_fives]
		target_34 = [1 if int(ex[1]) == 3 else -1 for ex in threes_and_fours]
		target_35 = [1 if int(ex[1]) == 3 else -1 for ex in threes_and_fives]
		target_45 = [1 if int(ex[1]) == 4 else -1 for ex in fours_and_fives]

		print "Data building complete"

		#train and svm for each pair and save in the class
		self.submodels.append(self.train_svms(input_12, target_12))
		self.submodels.append(self.train_svms(input_13, target_13))
		self.submodels.append(self.train_svms(input_14, target_14))
		self.submodels.append(self.train_svms(input_15, target_15))
		self.submodels.append(self.train_svms(input_23, target_23))
		self.submodels.append(self.train_svms(input_24, target_24))
		self.submodels.append(self.train_svms(input_25, target_25))
		self.submodels.append(self.train_svms(input_34, target_34))
		self.submodels.append(self.train_svms(input_35, target_35))
		self.submodels.append(self.train_svms(input_45, target_45))

		assert(len(self.submodels) == 15)

		
	#(should be a way to save trained classifiers so we dont need to do this every time)
	#http://scikit-learn.org/stable/modules/model_persistence.html


	def train_svms(self, input_data, target_data):
		print "Training next model..."
		state = random.randint(0, int(time.time()))
		#Once data has been translated to feature vector and target classes have been decided, train the model
		clf = svm.SVC(kernel='linear', C=1).fit(input_data, target_data)
		return clf


	def score_model(self):
		print "scoring..."
		answers = [int(ex[1]) for ex in self.test_data]
		vecs = features.generate_feature_vectors(self.test_data)
		predictions = []
		for feature_vector in vecs:
			pass
			

