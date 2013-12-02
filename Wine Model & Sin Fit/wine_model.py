#!/usr/bin/env python
import numpy as np
import pandas as pd
import pylab as pl

from matplotlib.colors import ListedColormap

from sklearn import cross_validation as cv
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import RFECV
from skll.metrics import kappa

input_data = 'wine1.csv'
wdata = pd.read_csv(input_data, delimiter = ',').dropna()

#separate features and labels
features = wdata.iloc[:,1:]
labels = wdata['class']
max_knn = (len(labels))
train_pct = 0.3
num_recs = len(labels)
#n_folds = 10
targets = ['C1','C2','C3']

def find_knn_k( max_k = max_knn):
#this function fits the knn model and finds the highest accuracy value and index 

	#initialize results set
	all_fpr, all_tpr, all_auc, all_acc, all_cm = (np.zeros(max_k), np.zeros(max_k), np.zeros(max_k),np.zeros(max_k), np.zeros(max_k))
	

	#Perfom CV to find best value of k
	for i in xrange(max_k):
		
		#randomize data
		#perm = np.random.permutation(len(labels))
		#features = features.iloc[perm]
		#lables = labels.iloc[perm]

		# perform train/test split
		tts = cv.train_test_split(features,labels, train_size=train_pct)
		train_features, test_features, train_labels, test_labels = tts
		#print test_features, '\n'

    	#initialize model, perform fit
		kclf = knn(n_neighbors=i+1)
		kclf.fit(train_features,train_labels)

		# get conf matrix (requires predicted labels)
		predicted_labels = kclf.predict(test_features)
		cm = confusion_matrix(test_labels, predicted_labels)

		
		#calc ROC, AUC, and accuracy
		fpr, tpr, thresholds = roc_curve(test_labels,predicted_labels, pos_label=1)
		roc_auc = auc(fpr,tpr)
		
		#get model accuracy 
		acc = kclf.score(test_features, test_labels)

    	#Put all stats in arrays
		all_fpr[i] = fpr[1]
		all_tpr[i] = tpr[1]
		all_auc[i] = roc_auc
		all_acc[i] = acc 
		#all_cm[i] = cm
		#all_k[i] = all_acc.argmax(axis=0)
		
		#print i
		#print 'confusion matrix:\n', cm, '\n'
		

	print 'Accuracy Matrix = \n', all_acc
	#print np.mean(all_acc)
	print '\nMax accuracy = {0}'.format(max(all_acc))
	print '\nK = {0}'.format(all_acc.argmax(axis=0) + 1)
	#print len(all_acc)
	#print all_k
	return all_acc, max_k, predicted_labels, test_labels
	#print predicted_labels
	#print test_labels
	#print np.mean(all_cm)

def get_k(iter_num,max_n=max_knn):
#This function finds the optimal k by averaging the index of location of max accuracy in all_acc 

	all_k = np.zeros(iter_num)
	for i in xrange(iter_num):
		this_k,max_k, predicted, tested = find_knn_k(max_n)
		#print this_k
		all_k[i] = this_k.argmax(axis=0) + 1 
		print 'Predicted Labels = \n', predicted, '\n'
		print 'Actual Labels = \n', tested, '\n'
		print '=' * 80
	print all_k, '\n'
	print 'Mean K = {0}'.format(np.mean(all_k))

def model_rank(num_fold=10):
#this function fits a knn and logistic regression model to rank performance for model selection

	#get cv iterators 
	kf = cv.KFold(n=num_recs, n_folds=num_fold, shuffle=True)

	#initialize result set
	LG_fpr, LG_tpr, LG_auc, LG_acc = (np.zeros(num_fold), np.zeros(num_fold), np.zeros(num_fold),np.zeros(num_fold))
	KNN_fpr, KNN_tpr, KNN_auc, KNN_acc = (np.zeros(num_fold), np.zeros(num_fold), np.zeros(num_fold),np.zeros(num_fold))


	for i,(traini, testi) in enumerate(kf):
		
		#initialize model
		model = LR()
		kclf = knn(n_neighbors=15)

		#make sure the records don't have null values
		train_features = features.iloc[traini].dropna()
		train_labels = labels.iloc[traini].dropna()


		test_features = features.iloc[testi].dropna()
		test_labels = labels.iloc[testi].dropna()


		#initialize model, perform fit
		kclf.fit(train_features,train_labels)
		results_LG = model.fit(train_features,train_labels)

		#predict the labels
		predict_LG = results_LG.predict(test_features)
		predict_KNN = kclf.predict(test_features)

		#calc ROC, AUC, and accuracy for LG model
		print 'Logistic Regression Classifier Stats \n'
		print 'True class'
		print test_labels, '\n'
		print 'Predicted Class'
		print '\n'*2

		#NOTE: ROC ANALYSIS ONLY WORKS FOR BINARY CLASSIFICATION PROBLEMS 
		#fpr_LG, tpr_LG, thresholds_LG = roc_curve(test_labels,predict_LG, pos_label=1)
		#roc_auc_LG = auc(fpr_LG,tpr_LG)
		acc_LG = model.score(test_features, test_labels)

		#print 'FPR = {0}'.format(fpr_LG), '\n'
		#print 'TPR = {0}'.format(tpr_LG), '\n'
		#print '\n'

		print 'acc =', acc_LG
		print confusion_matrix(test_labels,predict_LG), '\n'
		print classification_report(test_labels,predict_LG,[1,2,3] ,target_names=targets )
		print 'LG kappa =', kappa(test_labels,predict_LG)
		print '+_' * 40

		print 'KNN Classifier Stats \n'
		print 'True class'
		print test_labels, '\n'
		print 'Predicted Class'
		print predict_KNN
		print '\n'*2

		#LG_fpr[i] = fpr_LG[1]
		#LG_tpr[i] = tpr_LG[1]
		#LG_auc[i] = roc_auc_LG
		LG_acc[i] = acc_LG

		#calc ROC, AUC, and accuracy for KNN model
		#fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(test_labels,predict_KNN, pos_label=1)
		#roc_auc_KNN = auc(fpr_KNN,tpr_KNN)
		acc_KNN = kclf.score(test_features, test_labels)

		print 'acc =', acc_KNN
		print confusion_matrix(test_labels,predict_KNN), '\n'
		print classification_report(test_labels,predict_KNN,[1,2,3] ,target_names=targets )
		print 'KNN kappa =', kappa(test_labels,predict_KNN)
		print '*' * 80
		print '*' * 80
	
		#KNN_fpr[i] = fpr_KNN[1]
		#KNN_tpr[i] = tpr_KNN[1]
		#KNN_auc[i] = roc_auc_KNN
		KNN_acc[i] = acc_KNN
	print '\n', '@_' * 40 
	print 'Logistic Regression Model accuracy on trials =\n', LG_acc, '\n'
	print 'Mean LG accuracy = {0}'.format(np.mean(LG_acc)), '\n'
	print 'KNN Model accuracy on trials =\n' ,KNN_acc, '\n'
	print 'Mean KNN accuracy = {0}'.format(np.mean(KNN_acc))



def model_rank_loo():
#This function uses a LOO cv iterator and fits a knn and logistic regression model to rank performance for model selection

	#get cv iterator 
	kfloo = cv.LeaveOneOut(num_recs)

	#result set for Leave One Out CV
	LG_fpr, LG_tpr, LG_auc, LG_acc = (np.zeros(num_recs), np.zeros(num_recs), np.zeros(num_recs),np.zeros(num_recs))
	KNN_fpr, KNN_tpr, KNN_auc, KNN_acc = (np.zeros(num_recs), np.zeros(num_recs), np.zeros(num_recs),np.zeros(num_recs))

	for i,(traini, testi) in enumerate(kfloo):
		
		#initialize model
		model = LR()
		kclf = knn(n_neighbors=15)

		#make sure the records don't have null values_____________________________________________
		train_features = features.iloc[traini].dropna()

		#train labels for LOO CV
		train_labels = labels.iloc[traini]

		test_features = features.iloc[testi].dropna()

		#test labels for LOO CV
		test_labels = labels.iloc[testi]


		#initialize model, perform fit
		kclf.fit(train_features,train_labels)
		results_LG = model.fit(train_features,train_labels)

		#predict the labels
		predict_LG = results_LG.predict(test_features)
		predict_KNN = kclf.predict(test_features)

		print 'Index =', i, '\n'
		print 'Logistic Regression Classifier Stats \n'
		print 'True class'
		print test_labels, '\n'
		print 'Predicted Class'
		print predict_LG[0]
		print '\n'*2

		print '+' * 80

		print 'KNN Classifier Stats \n'
		print 'True class'
		print test_labels, '\n'
		print 'Predicted Class'
		print predict_KNN[0]
		print '\n'*2

		#Here we update the results arrays with 1 if our classifier was correct, 
		#later we take the average of the arrays - this give us the approximate accuracy of each model 
		if test_labels == predict_LG[0]:
			LG_acc[i] = 1
		else:
			LG_acc[i] = 0 


		if test_labels == predict_KNN[0]:
			KNN_acc[i] = 1
		else:
			KNN_acc[i] = 0 

		print '*' * 80, '\n', '*' * 80
	

	print '\n', '@_' * 40 
	print 'Logistic Regression Model accuracy on trials =\n', LG_acc, '\n'
	print 'Mean LG accuracy = {0}'.format(np.mean(LG_acc)), '\n'
	print 'KNN Model accuracy on trials =\n' ,KNN_acc, '\n'
	print 'Mean KNN accuracy = {0}'.format(np.mean(KNN_acc))
	print '\n' * 2



def find_features(step_num=1,num_cv=5):
#This function finds the best features for each model
#NOTE: This only works for the Logistic Regression model 	

	#initialize model
	model = LR()
	kclf = knn(n_neighbors=15)

	selector_LG = RFECV(model, step=step_num, cv=num_cv)
	selector_LG.fit(features,labels)

	#selector_KNN = RFECV(kclf, step=step_num, cv=num_cv)
	#selector_KNN.fit(features,labels)

	print 'LG features'
	print selector_LG.support_
	print selector_LG.ranking_, '\n'

	#print 'KNN features'
	#print selector_KNN.support_
	#print selector_KNN.ranking_, '\n'










