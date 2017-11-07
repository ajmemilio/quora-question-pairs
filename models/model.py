#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:51:14 2017

@author: joao
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import random as r
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict

RS = 1993
np.random.seed(RS)
K = 10

print "Started"
train = pd.read_csv('../features/train_v62.csv')
train_tagging = pd.read_csv('../features/train_tagging.csv')

class BaseClassifier(object):
    def __init__(self, classifier, params=None):
        self.classifier = classifier(**params)

    def train(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)

    def predict(self, x):
    	return self.classifier.predict(x)

    def getClassifier(self):
    	return self.classifier

class BaseClassifierXGBoost(object):
    def __init__(self, params=None):
        self.param = params

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

    def getParams(self):
    	return self.param

def cross_validation(classifier, X_train, Y_train, get_oversampling=None):
    kf = KFold(X_train.shape[0], n_folds=K, shuffle=False, random_state=RS)
    
    cv_auc = []
    cv_tpr = []
    cv_fpr = []
    
    for i, (train_index, valid_index) in enumerate(kf):
        x_train = X_train[train_index]
        y_train = Y_train[train_index]
        
        x_valid = X_train[valid_index]
        y_valid = Y_train[valid_index]
            
        if get_oversampling is not None:
            print "Get oversampling"
            #UPDownSampling
            pos_train = x_train[y_train == 1]
            neg_train = x_train[y_train == 0]
            x_train = np.concatenate((neg_train, pos_train[:int(get_oversampling*len(pos_train))], neg_train), axis=0)
            y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train[:int(get_oversampling*len(pos_train))].shape[0] + [0] * neg_train.shape[0])
            #print(np.mean(y_train))
            del pos_train, neg_train
            
        classifier.train(x_train, y_train)
        predicts = classifier.predict(x_valid)
        fpr, tpr, thresholds = roc_curve(y_valid, predicts)
        AUC = auc(fpr, tpr)
        cv_auc.append(AUC)
        cv_tpr.append(tpr)
        cv_fpr.append(fpr)
        #AUC, TPR, FPR
        return np.mean(cv_auc), np.mean(cv_tpr), np.mean(cv_fpr)

def main():
    """
    get_leaky = False
    
    print "Create leaky"
    # leaky
    q_dict = defaultdict(set)
    for i in range(train.shape[0]):
            q_dict[train.question1[i]].add(train.question2[i])
            q_dict[train.question2[i]].add(train.question1[i])

    def q1_freq(row):
        return(len(q_dict[row['question1']]))
        
    def q2_freq(row):
        return(len(q_dict[row['question2']]))
        
    def q1_q2_intersect(row):
        return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

    df_train = pd.DataFrame()
    df_train['q1_q2_intersect'] = train.apply(q1_q2_intersect, axis=1, raw=True)
    df_train['q1_freq'] = train.apply(q1_freq, axis=1, raw=True)
    df_train['q2_freq'] = train.apply(q2_freq, axis=1, raw=True)
    """
    #df_train_tagging.
    columns_names = []
    for c in train_tagging.columns:
        columns_names.append(c + "_not_sw")
        
    df_train = pd.DataFrame()
    df_train = train_tagging
    df_train.columns = columns_names
    
    """
    df_train['diff_vb_not_sw'] = df_train['diff_vb_not_sw'].abs()
    df_train['diff_vbd_not_sw'] = df_train['diff_vbd_not_sw'].abs()
    df_train['diff_vbg_not_sw'] = df_train['diff_vbg_not_sw'].abs()
    df_train['diff_vbn_not_sw'] = df_train['diff_vbn_not_sw'].abs()
    df_train['diff_vbp_not_sw'] = df_train['diff_vbp_not_sw'].abs()
    df_train['diff_vbz_not_sw'] = df_train['diff_vbz_not_sw'].abs()
    df_train['diff_nn_not_sw'] = df_train['diff_nn_not_sw'].abs()
    df_train['diff_nnp_not_sw'] = df_train['diff_nnp_not_sw'].abs()
    df_train['diff_nnps_not_sw'] = df_train['diff_nnps_not_sw'].abs()
    df_train['diff_jj_not_sw'] = df_train['diff_jj_not_sw'].abs()
    df_train['diff_jjr_not_sw'] = df_train['diff_jjr_not_sw'].abs()
    df_train['diff_jjs_not_sw'] = df_train['diff_jjs_not_sw'].abs()
    """
    
    X_train = pd.concat((train, df_train), axis=1)
    del df_train
    
    feature_names = ['len_vb_q1_not_sw',
                     'len_vbd_q1_not_sw',
                     'len_vbg_q1_not_sw',
                     'len_vbn_q1_not_sw',
                     'len_vbp_q1_not_sw',
                     'len_vbz_q1_not_sw',
                     'len_nn_q1_not_sw',
                     'len_nnp_q1_not_sw',
                     'len_nnps_q1_not_sw',
                     'len_jj_q1_not_sw',
                     'len_jjr_q1_not_sw',
                     'len_jjs_q1_not_sw',
                     'len_vb_q2_not_sw',
                     'len_vbd_q2_not_sw',
                     'len_vbg_q2_not_sw',
                     'len_vbn_q2_not_sw',
                     'len_vbp_q2_not_sw',
                     'len_vbz_q2_not_sw',
                     'len_nn_q2_not_sw',
                     'len_nnp_q2_not_sw',
                     'len_nnps_q2_not_sw',
                     'len_jj_q2_not_sw',
                     'len_jjr_q2_not_sw',
                     'len_jjs_q2_not_sw',
                     'diff_vb_not_sw',
                     'diff_vbd_not_sw',
                     'diff_vbg_not_sw',
                     'diff_vbn_not_sw',
                     'diff_vbp_not_sw',
                     'diff_vbz_not_sw',
                     'diff_nn_not_sw',
                     'diff_nnp_not_sw',
                     'diff_nnps_not_sw',
                     'diff_jj_not_sw',
                     'diff_jjr_not_sw',
                     'diff_jjs_not_sw',
                     'VB_not_sw',
                     'VBD_not_sw',
                     'VBG_not_sw',
                     'VBN_not_sw',
                     'VBP_not_sw',
                     'VBZ_not_sw',
                     'NN_not_sw',
                     'NNP_not_sw',
                     'NNPS_not_sw',
                     'JJ_not_sw',
                     'JJR_not_sw',
                     'JJS_not_sw',
                     'tfidf_word_match_share',
                     'word_match_share',
                     'len_q1',
                     'len_q2',
                     'diff_len',
                     'words_r2gram',
                     'stemming_shared',
                     'words_hamming']
    """
    if get_leaky:
        print "Get leaky"
        feature_names.append('q1_freq')    
        feature_names.append('q2_freq')    
        feature_names.append('q1_q2_intersect')  
    """
    Y_train = X_train['is_duplicate'].values
    X_train = X_train[feature_names].values
    
    XGB_PARAMETERS={
    	'objective':'binary:logistic',
    	'eta':0.2,
    	'max_depth':7,
    	'silent':1,
    	'eval_metric':"auc",
    	'min_child_weight':1,
    	'subsample':0.7,
    	'colsample_bytree':0.7,
    	'learning_rate':0.01,
    	'seed':r.randint(0, 100),
    	'num_rounds':500,
     'early_stopping_rounds':50
    }
    
    print "To cross validate {}Fold XGBoost".format(K)
    xgbt = BaseClassifierXGBoost(params=list(XGB_PARAMETERS.items()))
    AUC, TPR, FPR = cross_validation(xgbt, X_train, Y_train, 0.8)
    print "XGBoost AUC: {}, TPR: {}, FPR: {}".format(AUC, TPR, FPR)
    
    DTREE_PARAMETERS={
	'max_depth':500, 
	'random_state':r.randint(0, 100)
    }
        
    print "To cross validate {}Fold DTREE ".format(K)
    dtree = BaseClassifier(classifier=DecisionTreeRegressor, params=DTREE_PARAMETERS)
    AUC, TPR, FPR = cross_validation(dtree, X_train, Y_train, 0.8)
    print "DTREE AUC: {}, TPR: {}, FPR: {}".format(AUC, TPR, FPR)
    
    RF_PARAMETERS={
    	'n_estimators':10,
    	'max_depth':50, 
    	'bootstrap':True,
    	'random_state':r.randint(0, 100)
    }
        
    print "To cross validate {}Fold Random Forest".format(K)
    rf = BaseClassifier(classifier=RandomForestRegressor, params=RF_PARAMETERS)
    AUC, TPR, FPR = cross_validation(rf, X_train, Y_train, 0.8)
    print "Random Forest AUC: {}, TPR: {}, FPR: {}".format(AUC, TPR, FPR)
    
main()
print "Done"