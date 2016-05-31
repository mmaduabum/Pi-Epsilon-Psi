#!/usr/bin/env python
import PottsUtils
import sys
import random
import collections
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
unigrams = {}
unigram_init = False

"""Use GloVe word vectors for each word in the review"""
def glove_feature(word, glove_dict):
	if word in glove_dict:
		return (glove_dict[word], True)
	else:
		return (None, False)


"""Feature: the length of the text"""
def size_feature(example):
	return float(len(example))


def contains_1_feature(example):
	if "one" in example or "1" in example:
		return 1
	else:
		return 0

def contains_2_feature(example):
	if "two" in example or "2" in example:
		return 1
	else:
		return 0

def contains_3_feature(example):
	if "three" in example or "3" in example:
		return 1
	else:
		return 0

def contains_4_feature(example):
	if "four" in example or "4" in example:
		return 1
	else:
		return 0

def contains_5_feature(example):
	if "five" in example or "5" in example:
		return 1
	else:
		return 0

def contains_star_feature(example):
	if "star" in example and "stars" not in example:
		return 1
	else:
		return 0

def contains_stars_feature(example):
	if "stars" in example:
		return 1
	else:
		return 0


def num_words_feature(words):
	return len(words)

def count_feature(words, polarity_list):
	count = 0
	for w in words:
		if w in polarity_list:
			count += 1
	return count


def like_feature(words):
	return words["like"]

def great_feature(words):
	return words["great"]

def not_feature(words):
	return words["not"]

def money_feature(words):
	return words["$"]

def exclaim_feature(words):
	return words["!"]

def d_exclaim_feature(words):
	return words["!!"]

def taste_feature(words):
	return words["delicious"] + words["tasty"]

"""generates the feature vector for a single train example.
returns a list of values"""
def get_all_features(example, st, glove_dict, word_list, pos_words, neg_words):
	features = []
	word_dic = collections.Counter(word_list)
	#features.append(size_feature(example))
	#stemmed_words = [st.stem(word) for word in word_list]
	features.append(count_feature(word_list, pos_words))
	features.append(count_feature(word_list, neg_words))
	#features.append(exclaim_feature(word_dic))
	#features.append(d_exclaim_feature(word_dic))
	#features.append(money_feature(word_dic))
	#features.append(num_words_feature(word_dic))
	#features.append(features[1] - features[2])
	features.append(like_feature(word_dic))
	features.append(great_feature(word_dic))
	features.append(not_feature(word_dic))
	#features.append(contains_1_feature(example))
	#features.append(contains_2_feature(example))
	#features.append(contains_3_feature(example))
	#features.append(contains_4_feature(example))
	#features.append(contains_5_feature(example))
	#features.append(contains_star_feature(example))
	#features.append(contains_stars_feature(example))

	sum_ = sum(features)
	if sum_ > 0: norm_features = [float(f)/sum_ for f in features]
	else: norm_features = features

	return norm_features



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


def init_unigram_features(data):
	id_ = 0
	st = nltk.stem.lancaster.LancasterStemmer()
	stop_words = set(nltk.corpus.stopwords.words('english'))
	for example in data:
		words = example[0].split()
		go_words = [st.stem(w) for w in words if w not in stop_words]
		for new_word in go_words:
			if new_word not in unigrams:
				unigrams[new_word] = id_
				id_ += 1
	print len(unigrams)

def get_unigram_features(data):
	vecs = []
	st = nltk.stem.lancaster.LancasterStemmer()
	stop_words = set(nltk.corpus.stopwords.words('english'))
	vec_size = len(unigrams)
	corp = nltk.corpus.opinion_lexicon
	pos_words = set(corp.positive())
	neg_words = set(corp.negative())
	for example in data:
		feature_vec = [0]*vec_size
		words = [st.stem(w) for w in example[0].split() if w not in stop_words]
		word_list = [w for w in example[0].split() if w not in stop_words]
		word_dic = collections.Counter(word_list)
		for word in words:
			if word in unigrams:
				feature_vec[unigrams[word]] = 1
		feature_vec.append(count_feature(word_list, pos_words))
		feature_vec.append(count_feature(word_list, neg_words))
		feature_vec.append(like_feature(word_dic))
		feature_vec.append(great_feature(word_dic))
		feature_vec.append(not_feature(word_dic))
		vecs.append(np.array(feature_vec))
	return vecs



"""generates feature vectors for each train example and returns
and np matrix with the values"""
def generate_feature_vectors(data, glove=True, uni=False):
	if uni:
		return get_unigram_features(data)
	GLOVE = PottsUtils.glove2dict('glove.6B.50d.txt')
	st = nltk.stem.lancaster.LancasterStemmer()
	stop_words = set(nltk.corpus.stopwords.words('english'))
	stop_words.remove("not")
	corp = nltk.corpus.opinion_lexicon
	pos_words = set(corp.positive())
	neg_words = set(corp.negative())
	vecs = []
	for train_example in data:
		word_list = [w for w in train_example[0].split() if w not in stop_words]
		feature_vec = get_all_features(train_example[0].lower(), st, GLOVE, word_list, pos_words, neg_words)
		if glove: feature_vec = get_glove_features(feature_vec, word_list, GLOVE)
		vecs.append(np.array(feature_vec))
	return np.array(vecs)

