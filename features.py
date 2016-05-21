#!/usr/bin/env python
import PottsUtils
import sys
import random
import time
import our_svm
import our_nn
import nltk
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


MAX_WORD_VECS = 5
MIN_NUM_WORDS = 5

"""Use GloVe word vectors for each word in the review"""
def glove_feature(word, glove_dict):
	if word in glove_dict:
		return (glove_dict[word], True)
	else:
		return (None, False)


"""Feature: the length of the text"""
def size_feature(example):
	return float(len(example))



def count_feature(words, polarity_list):
	count = 0
	for w in words:
		if w in polarity_list:
			count += 1
	return count

"""generates the feature vector for a single train example.
returns a list of values"""
def get_all_features(example, st, glove_dict, word_list, pos_words, neg_words):
	features = []
	features.append(size_feature(example))
	#stemmed_words = [st.stem(word) for word in word_list]
	features.append(count_feature(word_list, pos_words))
	features.append(count_feature(word_list, neg_words))

	return features



def get_glove_features(features, word_list, glove_dict):
	feature_vec_size = 50*MIN_NUM_WORDS +3
	if len(word_list) < MIN_NUM_WORDS:
		print "fucked"
		return [0 for i in range(feature_vec_size)]
	words = 0
	for word in word_list:
		if words >= MAX_WORD_VECS: break
		words += 1
		vec, add = glove_feature(word, glove_dict)
		if add: 
			for val in vec: features.append(val)
		else:
			for i in range(50): features.append(0)
	return features


def nn_features(data, glove_dict, st, ignore_list, pos_words, neg_words):
	vecs = []
	for train_example in data:
		word_list = [w for w in train_example[0].split() if w not in ignore_list]
		feature_vec = get_glove_features(word_list, glove_dict)
		vecs.append(np.array(feature_vec))
	return np.array(vecs)


"""generates feature vectors for each train example and returns
and np matrix with the values"""
def generate_feature_vectors(data, glove=True):
	GLOVE = PottsUtils.glove2dict('glove.6B.50d.txt')
	st = nltk.stem.lancaster.LancasterStemmer()
	stop_words = set(nltk.corpus.stopwords.words('english'))
	corp = nltk.corpus.opinion_lexicon
	pos_words = set(corp.positive())
	neg_words = set(corp.negative())
	vecs = []
	for train_example in data:
		word_list = [w for w in train_example[0].split() if w not in stop_words]
		feature_vec = get_all_features(train_example[0], st, GLOVE, word_list, pos_words, neg_words)
		if glove: feature_vec = get_glove_features(feature_vec, word_list, GLOVE)
		vecs.append(np.array(feature_vec))
	return np.array(vecs)
