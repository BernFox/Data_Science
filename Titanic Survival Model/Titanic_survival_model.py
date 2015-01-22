#!/usr/bin/env python

import pandas as pd
import pylab as pl

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, make_scorer, zero_one_loss)
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC


input_data = 'train.csv'
t_data = pd.read_csv(input_data, delimiter = ',')

#Grab only the relavant data, columns like name and ticket probably won't help us classify the data. 
#Also, separate out the labels and the data

rel_data = t_data.loc[:,['Pclass','Name','Sex','Age','SibSp','Parch','Fare','Embarked']]
labels = train_data['Survived']

test_data = pd.read_csv('test.csv',delimiter=',')
test = test_data.loc[:,['Pclass','Name','Sex','Age','SibSp','Parch','Fare','Embarked']]

TRAIN_PCT = 0.30


def featurizer(train_data=rel_data, test_data=test, bin_num = 10):

	#Turn gender into a binary so the classifier can understand it, we can do this in place
	train_data['Sex'] = train_data['Sex'].ix[:].map(lambda k: 0 if k == 'female' else 1)
	
	#for some reason the code one line below is not working properly, lets try a different approach
	test_data['Sex'] = test_data['Sex'].ix[:].map(lambda k: 0 if k == 'female' else 1)

	#test_data['Sex'][test_data['Sex'] == 'female'] = 0
	#test_data['Sex'][test_data['Sex'] == 'male'] = 1

	#Turn the embarkment location into binary features
	train_data['Is_S'] = 0
	train_data['Is_S'][train_data['Embarked'] == 'S'] = 1 
	train_data['Is_C'] = 0
	train_data['Is_C'][train_data['Embarked'] == 'C'] = 1 
	train_data['Is_Q'] = 0
	train_data['Is_Q'][train_data['Embarked'] == 'Q'] = 1 

	test_data['Is_S'] = 0
	test_data['Is_S'][test_data['Embarked'] == 'S'] = 1 
	test_data['Is_C'] = 0
	test_data['Is_C'][test_data['Embarked'] == 'C'] = 1 
	test_data['Is_Q'] = 0
	test_data['Is_Q'][test_data['Embarked'] == 'Q'] = 1 

	#Drop the embarkment features so that our classifier doesn't get confused by too much of the same feature
	train_data = train_data.drop('Embarked', axis=1)
	test_data = test_data.drop('Embarked', axis=1)


	#Reduce the names to just the person's title, perhaps some titles mean something for their survival rate. 
	#ie. a 'Master', or 'Countess' may have a higher chance of living than another person with lower social status.
	#We have to turn these into numerical features later
	train_data['Name'] = train_data['Name'].ix[:].map(lambda k: k[k.find(',')+1 : k.find('.')] ) 
	test_data['Name'] = test_data['Name'].ix[:].map(lambda k: k[k.find(',')+1 : k.find('.')] ) 

	#get the sets of the names, this is an easy way to get all the unique names together. 
	#We get the set for each data set, then combine them to get all unique names together. 
	#If we see a new name in the test set, its gonna be weird because the classifier will think its actually a number
	#Should we make binary features out of the names? Overfit???? We'll try it as is for now
	titles, test_titles = set(train_data['Name']), set(test_data['Name'])
	titles |= test_titles

	#Put all the names in a list 
	title_list = [x for x in titles]

	#This for loop turns the titles into numerical labels
	for i,title in enumerate(title_list):
		train_data['Name'][train_data['Name'] == title] = i
		test_data['Name'][test_data['Name'] == title] = i

	#This for loop replaces all empty age values with the average age within in the person's Pclass
	for i in xrange(1,4):
		train_data['Age'][ (train_data['Pclass'] == i) & (train_data['Age'] != train_data['Age']) ] = \
			int(train_data['Age'][ train_data['Pclass'] == i ].mean())

		test_data['Age'][ (test_data['Pclass'] == i) & (test_data['Age'] != test_data['Age']) ] = \
			int(test_data['Age'][ test_data['Pclass'] == i ].mean())

	#This for loop replaces all empty fare values with the average fare within in the person's Pclass
	for i in xrange(1,4):
		train_data['Fare'][ (train_data['Pclass'] == i) & (train_data['Fare'] != train_data['Fare']) ] = \
			int(train_data['Fare'][ train_data['Pclass'] == i ].mean())

		test_data['Fare'][ (test_data['Pclass'] == i) & (test_data['Fare'] != test_data['Fare']) ] = \
			int(test_data['Fare'][ test_data['Pclass'] == i ].mean())

	#bin the ages by dividing by 10
	train_data['Age'] = train_data['Age'].apply(lambda k: int(k/bin_num)) 
	test_data['Age'] = test_data['Age'].apply(lambda k: int(k/bin_num))

	#Get the mean of the three medians
	bin_mean = int((train_data['Fare'][ train_data['Pclass'] == 1 ].median() + train_data['Fare'][ train_data['Pclass'] == 2 ].median() + \
		train_data['Fare'][ train_data['Pclass'] == 3 ].median()) / 3 )

	train_data['Fare'] = train_data['Fare'].apply(lambda k: int(k/bin_mean))
	test_data['Fare'] = test_data['Fare'].apply(lambda k: int(k/bin_mean))


	return train_data, test_data


def grid_search(Y = labels):
	X, test = featurizer()

	X_train, X_test, y_train, y_test = train_test_split(
		X, Y, train_size = int(TRAIN_PCT * len(X)))
	
	gbt = GradientBoostingClassifier(subsample=0.8, min_samples_leaf=50, min_samples_split=20)

	gbt_params = {'max_depth': [1, 2, 3, 4],
		'n_estimators': [10, 20, 50],
		'learning_rate': [0.1, 0.5, 1.0]}

	svc = LinearSVC(dual=False)
	svc_params = {'C': [10 ** -k for k in range(5)],
		'class_weight': [{1: 1}],
		'loss': ['l2'],
		'penalty': ['l1']}
	
	clf = gbt
	param_grid = gbt_params

	#clf = svc
	#param_grid = svc_params

	#score_method = 'roc_auc'
	score_method = 'accuracy'


	print 'thinking...'
	grid_results = GridSearchCV(clf, param_grid,
		scoring='roc_auc',
		cv=StratifiedKFold(y_train, n_folds=10),
		verbose=1)

	grid_results.fit(X_train, y_train)

	print '\ngenlzn errors:'
	for params, mean_score, all_scores in grid_results.grid_scores_:
		print '{}\t{}\t(+/-) {}'.format(
			params,
			round(mean_score, 3),
			round(all_scores.std() / 2, 3))

	print '\nbest model:'
	print '{}\t{}'.format(grid_results.best_params_,
		round(grid_results.best_score_, 3))

	print '\nclassification report:\n'
	print classification_report(y_test, grid_results.predict(X_test))

	print 'confusion matrix ({} total test recs, {} positive)'.format(
		len(y_test), sum(y_test))
	print confusion_matrix(y_test, grid_results.predict(X_test),
	labels=[1, 0])

def classify_survivors(Y = labels, orig_test = test_data):
	X, test = featurizer()

	best_model = {'n_estimators': 20, 'learning_rate': 1.0, 'max_depth': 3}	

	gbt = GradientBoostingClassifier(subsample=0.8, min_samples_leaf=50, min_samples_split=20,
		n_estimators = 20, learning_rate = 1.0, max_depth = 3)

	ID_col = orig_test.loc[:,['PassengerId']]
	print ID_col.ix[0:10]
	gbt.fit(X,Y)
	#print test.ix[0:10]
	predicted_results = gbt.predict(test)
	predicted_results = pd.DataFrame(predicted_results)
	predicted = pd.concat( [ID_col,predicted_results], axis=1 )
	predicted = predicted.rename(columns={0 : 'Survived'})
	#predicted = predicted.drop(' ',axis=1)
	del predicted['']

	#Print some of the dataframe with predictions to test results
	print predicted.ix[0:15],'\n'
	#print X.ix[0:15]

	#Output result dataframe as csv
	predicted.to_csv('predicted_results.csv')



if __name__ == '__main__':
	classify_survivors()


