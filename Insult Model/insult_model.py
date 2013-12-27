#!/usr/bin/env python

import math
import numpy as np
import pandas as pd
import pylab as pl
from dateutil import parser
from operator import itemgetter
from sklearn import cross_validation as cv
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

train_data = 'nb_train.csv'
test_data = 'nb_test.csv'
out_train_data = 'nb_train_formatted.csv'
out_test_data = 'nb_test_formatted.csv'

#in_train = pd.read_csv(train_data, delimiter = ',')
#in_test = pd.read_csv(test_data, delimiter = ',')

f_train = pd.read_csv(out_train_data, delimiter = ',')
f_test = pd.read_csv(out_test_data, delimiter = ',')

num_recs = len(f_train)

#separate out labels
labels = f_train['Insult']

fields = ('Insult' , 'Date', 'Comment')
f_train['tbin'] = 0
f_test['tbin'] = 0


def binary_time():
	"""This bins the time into a binary bin, 0 if the time is between 0 and 7;
		1 if the time is anything else. If the cell doesn't contain a time, it is binned as -1.
		 Leaving any bins empty would make classification on the test set have to exclude 
		 empyt time cells. This is done for both the training and test set"""

	for i, date in enumerate(f_train['Date']):
		try:
			if 0 <= int(date[-5:-3]) < 7:
				f_train['tbin'].ix[i] = 0
			elif int(date[-5:-3]) >= 7:
				f_train['tbin'].ix[i] = 1
		except Exception:
			f_train['tbin'].ix[i] = -1

	for j, dat in enumerate(f_test['Date']):
		try:
			if 0 <= int(dat[-5:-3]) < 7:
				f_test['tbin'].ix[j] = 0
			elif int(dat[-5:-3]) >= 7:
				f_test['tbin'].ix[j] = 1
		except Exception:
			f_test['tbin'].ix[j] = -1



def FourBinTime():
	"""This bins the time into 4 bins by grabbing the hour and dividing by 6. Empty time cells are given -1.
		This is done for both the training and test set"""

	for i, date in enumerate(f_train['Date']):
		try:
			f_train['tbin'].ix[i] = int(date[-5:-3])/6
		except Exception:
			f_train['tbin'].ix[i] = -1

	for j, dat in enumerate(f_test['Date']):
		try:
			f_test['tbin'].ix[j] = int(dat[-5:-3])/6
		except Exception:
			f_test['tbin'].ix[j] = -1


def format_time_train():
	"""format the time column by getting rid of the last character 'Z', and then parsing the time using the dateutil parser
		then write out to a csv"""

	print 'Formatting the training set...', '\n'
	for i in xrange( len( in_train['Date'] )):
		try: 
			in_train['Date'].ix[i] = parser.parse( in_train['Date'].ix[i][:-1] )
		except Exception:
			pass

	in_train.to_csv(out_train_data)

def format_time_test():
	"""this function does the same thing as format_time_train, but does it for the test data"""

	print 'Formatting the test set...', '\n'
	for j in xrange( len( in_test['Date'] )):
		try:
			in_test['Date'].ix[j] = parser.parse( in_test['Date'].ix[j][:-1] )
		except Exception:
			pass

	in_test.to_csv(out_test_data)


def class_insult(mini=2 ,maxi=50, folds=10, nmin=1, nmax=2, bin=4):
	"""This function extracts text features from a series of insults (identified as such), 
		then tries to classify a document that does not have labels. 

		First, this function uses CountVectorizer      
		to extract the features out of only the comments that were classified as insulting. This extracts only features that 
		matter to classifying the insulting comments, thus avoiding overfitting.

		Second, this function uses cross validation to evaluate the quality of the classifier. The vocabulary extracted 
		by count vectorizor is used to contrain the words that are featurized.

		Third, this function uses the extracted features to predict the classes on the test set, then outputs it as a csv"""

	if bin == 'b':
		binary_time()
		print 'binary_time','\n'
	else:
		FourBinTime()
		print 'FourBinTime', '\n'

	print 'Analyzing...', '\n'
	i_train = f_train[f_train['Insult'].copy() ==1]

	stopwords = ['is','Is','was','to','it','It','To','an','An','And','and','a','A','Are','are','What','what','for','For','From','from','When','when','can','Can','This','this']
	
	#vocab = ['fuck','faggot','idiot','stupid','retard','retarded','dick','dickhead','loser','ignorant','prick','troll','pathetic','weirdo','botton','ass','asshole']	
	
	#get cross validation iterator 
	kf = cv.KFold(n=num_recs, n_folds=folds, shuffle=True)

	#initialize vectorizer - this excepts parameters defining how features are extracted from the corpus
	vectorizer = CountVectorizer(min_df=mini, max_df=maxi, decode_error='ignore', ngram_range=(nmin,nmax),stop_words=stopwords)

	#fit the vectorizer with the corpus, i.e extracts features
	XY = vectorizer.fit_transform(i_train['Comment'])
	tvocab = vectorizer.vocabulary_

	
	#Turn feature matrix to a df so it can be concatenated with the dates series
	vectorizer1 = CountVectorizer(min_df=mini, max_df=maxi, decode_error='ignore',stop_words=stopwords, 
		vocabulary=tvocab, ngram_range=(nmin,nmax))

	#Turn the fitted CountVectorizer into a numeric matrix, then into a dataframe, then concatenate with the binned time column
	X = vectorizer1.fit_transform(f_train['Comment'])
	X0 = pd.DataFrame( X.toarray() )
	X1 = pd.concat([pd.DataFrame(f_train['tbin']),X0], axis=1)


	all_acc = np.zeros(folds)

	for i,(traini, testi) in enumerate(kf):

		gnb = GaussianNB()

		#Separate features from labels for training and testing, this is to evaluate the accuracy of the classifier
		train_features = X1.iloc[traini]
		train_labels = labels.iloc[traini]


		test_features = X1.iloc[testi]
		test_labels = labels.iloc[testi]


		gnb.fit(train_features,train_labels)
		predicted_labels = gnb.predict(test_features)
		acc = gnb.score(test_features, test_labels)

		#Print some analysis of the classifier
		print predicted_labels, '\n'
		print "The number of insults predicted =",predicted_labels.sum()

		print "Accuracy =",acc
		print confusion_matrix(test_labels, predicted_labels)
		print classification_report(test_labels,predicted_labels, [0,1])


		all_acc[i] = acc 

	print '\n',"Average accuracy =",np.mean(all_acc)

	#Initialize a new CountVectorizer using the vocabulary learned from the insult comments in the training set
	vectorizer2 = CountVectorizer(min_df=mini, max_df=maxi, decode_error='ignore',
		stop_words=stopwords, vocabulary=tvocab, ngram_range=(nmin,nmax))

	#Fit the CountVectorizer with the comments from the test set, then connect with the binned time column
	result = vectorizer2.fit_transform(f_test['Comment'])
	result_array = pd.DataFrame( result.toarray() )
	result_array1 = pd.concat([pd.DataFrame(f_test['tbin']),result_array], axis=1)

	#Initialize and fit 
	gnb1 = GaussianNB()
	gnb1.fit(X1,labels)

	#Predict labels for test set comments, then connect predictions with test dataframe and rename the column from it's default name
	predicted_results = gnb1.predict(result_array1)
	predicted = pd.concat( [pd.DataFrame(predicted_results),f_test], axis=1 )
	predicted = predicted.rename(columns={0 : 'Insult'})

	#Print some dataframe with predictions to test results
	print predicted.ix[0:15]  

	#Output result dataframe as csv
	predicted.to_csv('predicted_results.csv')


if __name__ == '__main__':
	format_time_train()
	format_time_test()
	class_insult()












