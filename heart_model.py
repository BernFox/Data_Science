#!/usr/bin/env python

import numpy as np
import pandas as pd
import pylab as pl      # note this is part of matplotlib
from pprint import pprint

from sklearn import cross_validation as cv
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFECV

num_fold = 10
input_data = 'logit-train.csv'

#get data
hdata = pd.read_csv(input_data, delimiter=',').dropna()

#separate features and labels 
#features = hdata[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
features = hdata[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]

#features = hdata[['age','sex','cp','fbs','restecg','exang','oldpeak','slope','ca','thal']]
#features = hdata[['age','sex','cp','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
#features = hdata[['sex','cp','fbs','restecg','exang','oldpeak','slope','ca','thal']]
#features = hdata[['age','sex','cp','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']]
#features = hdata[['cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
#features = hdata[['sex','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
labels = hdata['heartdisease::category|0|1']



def Logmodel():

	#get the cross validation iterator
	num_recs = len(hdata)
	kf = cv.KFold(n=num_recs, n_folds=num_fold, shuffle=True)

	#initialize result set
	all_fpr, all_tpr, all_auc = (np.zeros(num_fold), np.zeros(num_fold), np.zeros(num_fold))


	for i,(traini, testi) in enumerate(kf):


		#initialize model
		model = LR()

		#make sure the records don't have null values
		train_features = features.loc[traini].dropna()
		train_labels = labels.loc[traini].dropna()

		test_features = features.loc[testi].dropna()
		test_labels = labels.loc[testi].dropna()

		results = model.fit(train_features,train_labels)

		#predict the labels
		predicted_labels = results.predict(test_features)

		#calc ROC and AUC
		fpr, tpr, thresholds = roc_curve(test_labels,predicted_labels, pos_label=1)
		roc_auc = auc(fpr,tpr)

		all_fpr[i] = fpr[1]
		all_tpr[i] = tpr[1]
		all_auc[i] = roc_auc

		#print fpr 
		#print tpr 
		#print roc_auc 
		#print ""


		#print '\nfpr = {0}'.format(fpr)
		#print 'tpr = {0}'.format(tpr)
		#print 'auc = {0}'.format(roc_auc)


	#print '\nall_fprs = {0}'.format(all_fpr)
	#print 'all_tprs = {0}'.format(all_tpr)
	#print 'all_aucs = {0}'.format(all_auc)

	fpr_avg = np.average(all_fpr)
	tpr_avg = np.average(all_tpr)
	auc_avg = np.average(all_auc)

	print '\nAverage fpr = {0}'.format(fpr_avg)
	print 'Average tpr = {0}'.format(tpr_avg)
	print 'Average auc = {0}'.format(auc_avg)

	#print '\nModel coefficients = {0}'.format(results.coef_)
	#print results.summary()

	# plot ROC curve
	pl.clf()
	pl.plot([0.0,fpr_avg,1.0], [0.0,tpr_avg,1.0], label='ROC curve (area = %0.2f)' % auc_avg)
	pl.plot([0, 1], [0, 1], 'k--')
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.0])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('Receiver operating characteristic')
	pl.legend(loc="lower right")
	pl.show() 

	#print all_fpr
	#print all_tpr
	#print all_auc



num_feat_fold = 5
feat_labels = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

def Log_feat_model():

	#get the cross validation iterator for records
	num_recs = len(hdata)
	kf = cv.KFold(n=num_recs, n_folds=num_fold, shuffle=True)

	#get the cross validation iterator for features
	num_feats = len(feat_labels)
	kf_feat = cv.KFold(n=num_feats, n_folds=num_feat_fold, shuffle=True)

	#initialize result set
	all_fpr, all_tpr, all_auc = (np.zeros((num_fold,num_feat_fold)), np.zeros((num_fold,num_feat_fold)), np.zeros((num_fold,num_feat_fold)))


	for i,(traini, testi) in enumerate(kf):
		for j, (trainf, testf) in enumerate(kf_feat):
			
			print trainf
			print testf
			#print '\n'

			#initialize model
			model = LR()

			#make sure the records don't have null values
			train_features = features.iloc[traini,trainf].dropna()
			train_labels = labels.iloc[traini].dropna()

			#print train_labels
			#print '\n'

			test_features = features.iloc[traini,trainf].dropna()
			test_labels = labels.iloc[traini].dropna()

			#print test_features
			#print '\n'

			results = model.fit(train_features,train_labels)

			#predict the labels
			predicted_labels = results.predict(test_features)

			#calc ROC and AUC
			fpr, tpr, thresholds = roc_curve(test_labels,predicted_labels, pos_label=1)
			roc_auc = auc(fpr,tpr)

			all_fpr[i][j] = fpr[1]
			all_tpr[i][j] = tpr[1]
			all_auc[i][j] = roc_auc

			#print fpr 
			#print tpr 
			#print 'accuracy = {0}'.format(roc_auc)
			#print ""
			print 'accuracy = {0}'.format(model.score(test_features,test_labels))

			print '\nModel coefficients = {0}'.format(results.coef_)
			print '_' * 30


		#print '\nfpr = {0}'.format(fpr)
		#print 'tpr = {0}'.format(tpr)
		#print 'auc = {0}'.format(roc_auc)


	#print '\nall_fprs = {0}'.format(all_fpr)
	#print 'all_tprs = {0}'.format(all_tpr)
	#print 'all_aucs = {0}'.format(all_auc)

	fpr_avg = np.average(all_fpr)
	tpr_avg = np.average(all_tpr)
	auc_avg = np.average(all_auc)

	#print '\nAverage fpr = {0}'.format(fpr_avg)
	#print 'Average tpr = {0}'.format(tpr_avg)
	#print 'Average auc = {0}'.format(auc_avg)
	#print ""

	#print np.argmax(auc_avg)
	#print 'Index of min fpr is {0}'.format(np.argmin(all_fpr, axis = 1))

	#print '\nModel coefficients = {0}'.format(results.coef_)
	#print results.summary()

	# plot ROC curve
	pl.clf()
	pl.plot([0.0,fpr_avg,1.0], [0.0,tpr_avg,1.0], label='ROC curve (area = %0.2f)' % auc_avg)
	pl.plot([0, 1], [0, 1], 'k--')
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.0])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('Receiver operating characteristic')
	pl.legend(loc="lower right")
	pl.show() 

num_featfold = 5
#feat_labels = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

def Log_feat():

	#get the cross validation iterator for records
	num_recs = len(hdata)
	kf = cv.KFold(n=num_recs, n_folds=num_fold, shuffle=True)

	#get the cross validation iterator for features
	num_feats = len(feat_labels)
	kf_feat = cv.KFold(n=num_feats, n_folds=num_feat_fold, shuffle=True)

	#initialize result set
	all_fpr, all_tpr, all_auc = (np.zeros(num_featfold), np.zeros(num_featfold), np.zeros(num_featfold))



	for j, (trainf, testf) in enumerate(kf_feat):
			
		print trainf
		print testf
		#print '\n'

		#initialize model
		model = LR()

		#make sure the records don't have null values
		train_features = features.iloc[:,trainf].dropna()
		train_labels = labels.iloc[:].dropna()

		#print train_labels
		#print '\n'

		test_features = features.iloc[:,trainf].dropna()
		test_labels = labels.iloc[:].dropna()

		#print test_features
		#print '\n'

		results = model.fit(train_features,train_labels)

		#predict the labels
		predicted_labels = results.predict(test_features)

		#calc ROC and AUC
		fpr, tpr, thresholds = roc_curve(test_labels,predicted_labels, pos_label=1)
		roc_auc = auc(fpr,tpr)

		all_fpr[j] = fpr[1]
		all_tpr[j] = tpr[1]
		all_auc[j] = roc_auc

		#print fpr 
		#print tpr 
		#print 'accuracy = {0}'.format(roc_auc)
		#print ""
		print 'accuracy = {0}'.format(model.score(test_features,test_labels))

		print '\nModel coefficients = {0}'.format(results.coef_)
		print '_' * 30


		#print '\nfpr = {0}'.format(fpr)
		#print 'tpr = {0}'.format(tpr)
		#print 'auc = {0}'.format(roc_auc)


	#print '\nall_fprs = {0}'.format(all_fpr)
	#print 'all_tprs = {0}'.format(all_tpr)
	#print 'all_aucs = {0}'.format(all_auc)

	fpr_avg = np.average(all_fpr)
	tpr_avg = np.average(all_tpr)
	auc_avg = np.average(all_auc)

	#print '\nAverage fpr = {0}'.format(fpr_avg)
	#print 'Average tpr = {0}'.format(tpr_avg)
	#print 'Average auc = {0}'.format(auc_avg)
	#print ""

	#print np.argmax(auc_avg)
	#print 'Index of min fpr is {0}'.format(np.argmin(all_fpr, axis = 1))

	#print '\nModel coefficients = {0}'.format(results.coef_)
	#print results.summary()

	# plot ROC curve
	pl.clf()
	pl.plot([0.0,fpr_avg,1.0], [0.0,tpr_avg,1.0], label='ROC curve (area = %0.2f)' % auc_avg)
	pl.plot([0, 1], [0, 1], 'k--')
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.0])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('Receiver operating characteristic')
	pl.legend(loc="lower right")
	pl.show() 


num_featfold = 5
feat_labels = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

def feat_elim():

	#get the cross validation iterator for records
	num_recs = len(hdata)
	#kf = cv.KFold(n=num_recs, n_folds=num_fold, shuffle=True)

	#get the cross validation iterator for features
	num_feats = len(feat_labels)
	#kf_feat = cv.KFold(n=num_feats, n_folds=num_feat_fold, shuffle=True)

	#initialize result set
	#all_fpr, all_tpr, all_auc = (np.zeros(num_featfold), np.zeros(num_featfold), np.zeros(num_featfold))



	#for j, (trainf, testf) in enumerate(kf_feat):
			
	#print trainf
	#print testf
	#print '\n'

	#initialize model
	model = LR()
	selector = RFECV(model, step=1, cv=5)
	selector = selector.fit(features,labels)

	print selector.n_features_
	#print selector.cv_scores_
	print selector.support_ 
	print selector.ranking_

	#make sure the records don't have null values
	#train_features = features.iloc[:,trainf].dropna()
	#train_labels = labels.iloc[:].dropna()

	#print train_labels
	#print '\n'

	#test_features = features.iloc[:,trainf].dropna()
	#test_labels = labels.iloc[:].dropna()

	#print test_features
	#print '\n'

	#results = model.fit(train_features,train_labels)

	#predict the labels
	#predicted_labels = results.predict(test_features)

	#calc ROC and AUC
	#fpr, tpr, thresholds = roc_curve(test_labels,predicted_labels, pos_label=1)
	#roc_auc = auc(fpr,tpr)





