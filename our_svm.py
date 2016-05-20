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
Uses 15 internal SVMs: 5 using the one vs others method and
10 more for each pair of classes."""
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
		input_data = features.generate_feature_vectors(train_data, True)
		all_targets = [int(ex[1]) for ex in train_data]
		self.baseline_model = self.train_svms(input_data, all_targets)
		#train the one vs others classifiers
		for i in range(5):
			star = i + 1
			target_data = [1 if int(ex[1]) == star else 0 for ex in train_data] 
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
		input_12 = features.generate_feature_vectors(ones_and_twos, True)
		input_13 = features.generate_feature_vectors(ones_and_threes, True)
		input_14 = features.generate_feature_vectors(ones_and_fours, True)
		input_15 = features.generate_feature_vectors(ones_and_fives, True)
		input_23 = features.generate_feature_vectors(twos_and_threes, True)
		input_24 = features.generate_feature_vectors(twos_and_fours, True)
		input_25 = features.generate_feature_vectors(twos_and_fives, True)
		input_34 = features.generate_feature_vectors(threes_and_fours, True)
		input_35 = features.generate_feature_vectors(threes_and_fives, True)
		input_45 = features.generate_feature_vectors(fours_and_fives, True)

		#generate the targets for each data subset
		target_12 = [1 if int(ex[1]) == 1 else 2 for ex in ones_and_twos]
		target_13 = [1 if int(ex[1]) == 1 else 3 for ex in ones_and_threes]
		target_14 = [1 if int(ex[1]) == 1 else 4 for ex in ones_and_fours]
		target_15 = [1 if int(ex[1]) == 1 else 5 for ex in ones_and_fives]
		target_23 = [2 if int(ex[1]) == 2 else 3 for ex in twos_and_threes]
		target_24 = [2 if int(ex[1]) == 2 else 4 for ex in twos_and_fours]
		target_25 = [2 if int(ex[1]) == 2 else 5 for ex in twos_and_fives]
		target_34 = [3 if int(ex[1]) == 3 else 4 for ex in threes_and_fours]
		target_35 = [3 if int(ex[1]) == 3 else 5 for ex in threes_and_fives]
		target_45 = [4 if int(ex[1]) == 4 else 5 for ex in fours_and_fives]

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
		vecs = features.generate_feature_vectors(self.test_data, True)
		predictions = []
		for feature_vector in vecs:
			predictions.append(self.our_predict(feature_vector))

		answers = np.array(answers).reshape(len(answers), 1)
		print str(predictions)	
		predictions = np.array(predictions).reshape(len(predictions), 1)
		return (predictions, answers)

	#sorry this is really shit right now, just trying to get it working
	def our_predict(self, vec):
		first_guesses = []
		#Run each one vs others classifer
		first_guesses.append(self.submodels[self.ONEvALL].predict(vec)[0])
		first_guesses.append(self.submodels[self.TWOvALL].predict(vec)[0])
		first_guesses.append(self.submodels[self.THREEvALL].predict(vec)[0])
		first_guesses.append(self.submodels[self.FOURvALL].predict(vec)[0])
		first_guesses.append(self.submodels[self.FIVEvALL].predict(vec)[0])

		#check if only one class was predicted
		if sum(first_guesses) == 1:
			return first_guesses[first_guesses.index(1)]


		if sum(first_guesses) == 2:
			#otherwise, run the pairwise classifiers
			first_index = first_guesses.index(1)
			class_a = first_index + 1
			class_b = first_guesses.index(1, first_index+1) + 1

			if (class_a, class_b) == (1, 2):
				return self.submodels[self.ONEvTWO].predict(vec)[0]
			elif (class_a, class_b) == (1, 3):
				return self.submodels[self.ONEvTHREE].predict(vec)[0]
			elif (class_a, class_b) == (1, 4):
				return self.submodels[self.ONEvFOUR].predict(vec)[0]
			elif (class_a, class_b) == (1, 5):
				return self.submodels[self.ONEvFIVE].predict(vec)[0]
			elif (class_a, class_b) == (2, 3):
				return self.submodels[self.TWOvTHREE].predict(vec)[0]
			elif (class_a, class_b) == (2, 4):
				return self.submodels[self.TWOvFOUR].predict(vec)[0]
			elif (class_a, class_b) == (2, 5):
				return self.submodels[self.TWOvFIVE].predict(vec)[0]
			elif (class_a, class_b) == (3, 4):
				return self.submodels[self.THREEvFOUR].predict(vec)[0]
			elif (class_a, class_b) == (3, 5):
				return self.submodels[self.THREEvFIVE].predict(vec)[0]
			elif (class_a, class_b) == (4, 5):
				return self.submodels[self.FOURvFIVE].predict(vec)[0]
			else:
				print "ERROR"
	

		#if sum(first_guesses) > 2: print "things could be happening, but aren't"		

		return self.baseline_model.predict(vec)[0]

		#								  |
		#the baseline predictor does this v by default	
		"""#If 0, 3, 4, or 5 classes were positive, run all pairwise calssifiers
		votes = {1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0}
		for i, m in enumerate(self.submodels):
			if i < self.ONEvTWO: continue
			votes[m.predict(vec)[0]] += 1

		
		return max(votes.iteritems(), key=operator.itemgetter(1))[0]"""

