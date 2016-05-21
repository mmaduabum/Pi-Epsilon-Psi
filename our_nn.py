#!/usr/bin/env python
import utils
import sys
import random
import time
import features
import operator
import PottsNet
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

"""Our multi-class classifier
Uses 15 internal neural nets: 5 using the one vs others method and
10 more for each pair of classes."""
class Our_NN:
	def __init__(self, use_glove=False):
		self.use_glove = use_glove
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
		feature_vecs = features.generate_feature_vectors(train_data, self.use_glove)
		all_targets = [int(ex[1]) for ex in train_data]
		#train initial 5 classifiers
		for i in range(5):
			star = i + 1
			target_data = [np.array([1]) if int(ex[1]) == star else np.array([0]) for ex in train_data]
			training_data = [(vec, star) for vec, star in zip(feature_vecs, target_data)]
			self.submodels.append(self.train_net(training_data))


		print "Building datasets..."

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
		input_12 = features.generate_feature_vectors(ones_and_twos, self.use_glove)
		input_13 = features.generate_feature_vectors(ones_and_threes, self.use_glove)
		input_14 = features.generate_feature_vectors(ones_and_fours, self.use_glove)
		input_15 = features.generate_feature_vectors(ones_and_fives, self.use_glove)
		input_23 = features.generate_feature_vectors(twos_and_threes, self.use_glove)
		input_24 = features.generate_feature_vectors(twos_and_fours, self.use_glove)
		input_25 = features.generate_feature_vectors(twos_and_fives, self.use_glove)
		input_34 = features.generate_feature_vectors(threes_and_fours, self.use_glove)
		input_35 = features.generate_feature_vectors(threes_and_fives, self.use_glove)
		input_45 = features.generate_feature_vectors(fours_and_fives, self.use_glove)

		#generate the targets for each data subset
		target_12 = [np.array([1]) if int(ex[1]) == 1 else np.array([-1]) for ex in ones_and_twos]
		target_13 = [np.array([1]) if int(ex[1]) == 1 else np.array([-1]) for ex in ones_and_threes]
		target_14 = [np.array([1]) if int(ex[1]) == 1 else np.array([-1]) for ex in ones_and_fours]
		target_15 = [np.array([1]) if int(ex[1]) == 1 else np.array([-1]) for ex in ones_and_fives]
		target_23 = [np.array([1]) if int(ex[1]) == 2 else np.array([-1]) for ex in twos_and_threes]
		target_24 = [np.array([1]) if int(ex[1]) == 2 else np.array([-1]) for ex in twos_and_fours]
		target_25 = [np.array([1]) if int(ex[1]) == 2 else np.array([-1]) for ex in twos_and_fives]
		target_34 = [np.array([1]) if int(ex[1]) == 3 else np.array([-1]) for ex in threes_and_fours]
		target_35 = [np.array([1]) if int(ex[1]) == 3 else np.array([-1]) for ex in threes_and_fives]
		target_45 = [np.array([1]) if int(ex[1]) == 4 else np.array([-1]) for ex in fours_and_fives]
		
		#generate training data for each subset
		train_12 = [(vec, star) for vec, star in zip(input_12, target_12)]
		train_13 = [(vec, star) for vec, star in zip(input_13, target_13)]
		train_14 = [(vec, star) for vec, star in zip(input_14, target_14)]
		train_15 = [(vec, star) for vec, star in zip(input_15, target_15)]
		train_23 = [(vec, star) for vec, star in zip(input_23, target_23)]
		train_24 = [(vec, star) for vec, star in zip(input_24, target_24)]
		train_25 = [(vec, star) for vec, star in zip(input_25, target_25)]
		train_34 = [(vec, star) for vec, star in zip(input_34, target_34)]
		train_35 = [(vec, star) for vec, star in zip(input_35, target_35)]
		train_45 = [(vec, star) for vec, star in zip(input_45, target_45)]

		print "Data building complete"

		self.submodels.append(self.train_net(train_12))
		self.submodels.append(self.train_net(train_13))
		self.submodels.append(self.train_net(train_14))
		self.submodels.append(self.train_net(train_15))
		self.submodels.append(self.train_net(train_23))
		self.submodels.append(self.train_net(train_24))
		self.submodels.append(self.train_net(train_25))
		self.submodels.append(self.train_net(train_34))
		self.submodels.append(self.train_net(train_35))
		self.submodels.append(self.train_net(train_45))


	def train_net(self, train_data):
		print "Training next NN..."
		net = PottsNet.ShallowNeuralNetwork() 
		net.fit(train_data)
		return net


	def score_model(self):
		print "scoring..."
		test_examples = features.generate_feature_vectors(self.test_data, self.use_glove)
		predictions = []
		for ex in test_examples:
			p = self.our_predict(ex)
			predictions.append(p)
		
		return predictions


	def our_predict(self, ex):
		first_guesses = []
		#Run each one vs others classifer
		first_guesses.append(self.submodels[self.ONEvALL].predict(ex)[0])
		first_guesses.append(self.submodels[self.TWOvALL].predict(ex)[0])
		first_guesses.append(self.submodels[self.THREEvALL].predict(ex)[0])
		first_guesses.append(self.submodels[self.FOURvALL].predict(ex)[0])
		first_guesses.append(self.submodels[self.FIVEvALL].predict(ex)[0])

		
		positives = [1 if x > 0 else 0 for x in first_guesses]
		pos_count = sum(positives)

		if pos_count == 1:
			return positives.index(1) + 1	

		
		if pos_count == 2:
			#otherwise, run the pairwise classifiers
			first_index = first_guesses.index(1)
			class_a = first_index + 1
			class_b = first_guesses.index(1, first_index+1) + 1

			if (class_a, class_b) == (1, 2):
				if self.submodels[self.ONEvTWO].predict(ex)[0] > 0: return 1
				else: return 2
			elif (class_a, class_b) == (1, 3):
				if self.submodels[self.ONEvTHREE].predict(ex)[0] > 0: return 1
				else: return 3
			elif (class_a, class_b) == (1, 4):
				if self.submodels[self.ONEvFOUR].predict(ex)[0] > 0: return 1
				else: return 4
			elif (class_a, class_b) == (1, 5):
				if self.submodels[self.ONEvFIVE].predict(ex)[0] > 0: return 1
				else: return 5
			elif (class_a, class_b) == (2, 3):
				if self.submodels[self.TWOvTHREE].predict(ex)[0] >  0: return 2
				else: return 3
			elif (class_a, class_b) == (2, 4):
				if self.submodels[self.TWOvFOUR].predict(ex)[0] > 0: return 2
				else: return 4
			elif (class_a, class_b) == (2, 5):
				if self.submodels[self.TWOvFIVE].predict(ex)[0] > 0: return 2
				else: return 5
			elif (class_a, class_b) == (3, 4):
				if self.submodels[self.THREEvFOUR].predict(ex)[0] > 0: return 3
				else: return 4
			elif (class_a, class_b) == (3, 5):
				if self.submodels[self.THREEvFIVE].predict(ex)[0] > 0: return 3
				else: return 5
			elif (class_a, class_b) == (4, 5):
				if self.submodels[self.FOURvFIVE].predict(ex)[0] > 0: return 4
				else: return 5
			else:
				print "ERROR"

			
		votes = {1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0}
		onevs = [self.ONEvTWO, self.ONEvTHREE, self.ONEvFOUR, self.ONEvFIVE]
		for i, idx in enumerate(onevs):
			rating = i + 2
			pred = self.submodels[idx].predict(ex)[0] 
			if pred > 0: votes[1] += 1
			else: votes[rating] += 1

		twovs = [self.TWOvTHREE, self.TWOvFOUR, self.TWOvFIVE]
		for i, idx in enumerate(twovs):
			rating = i + 3
			pred = self.submodels[idx].predict(ex)[0]
			if pred > 0: votes[2] += 1
			else: votes[rating] += 1	

		threevs = [self.THREEvFOUR, self.THREEvFIVE]
		for i, idx in enumerate(threevs):
			rating = i + 4
			pred = self.submodels[idx].predict(ex)[0]
			if pred > 0: votes[3] += 1
			else: votes[rating] += 1

		pred = self.submodels[self.FOURvFIVE].predict(ex)[0]
		if pred > 0: votes[4] += 1
		else: votes[5] += 1

		return max(votes.iteritems(), key=operator.itemgetter(1))[0]	

					
