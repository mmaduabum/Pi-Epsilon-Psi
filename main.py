#!/usr/bin/env python
import utils

#Select option: SVM or NN

#parse json data into desired form

#train the 5  1 vs all classifiers
#train the 10 binary classifiers


#to train:
#for example, label in train data:
#	vec = features.build_feature_vec(example)
#	train_examples.append(vec, label)
#PottsNet.fit(train_examples)


#(should be a way to save trained classifiers so we dont need to do this every time)

#predict
#report


#testing utils functions
data = utils.create_train_data('yelp_data/tiny.json')
for i in range(89, 92):
	print data[i]
