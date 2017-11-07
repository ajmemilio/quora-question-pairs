import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import log_loss
import random as r
from sklearn.cross_validation import train_test_split
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import KFold

NFOLDS = 10
N_RF = 5
N_DTREE = 0
N_XGB = 5
N_MLP = 0
SEED = 0

folder = '../../features/'
train = pd.read_csv(folder + 'train_v62.csv')
train_tagging = pd.read_csv(folder + 'train_tagging.csv')

RS = 1993
np.random.seed(RS)

MLP_PARAMETERS={
	'hidden_layer_sizes':[(100, 100, 100, 100), (100, 100, 100), (100, 100), (100,)], 
	'activation':['relu', 'tanh'], 
	'solver':['lbfgs'],
	'alpha':[0.0001, 0.0002, 0.0010, 0.0099],
	'power_t':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
	'max_iter':[200, 100, 50, 150, 55], 
	'shuffle':[True, False], 
	'random_state':[r.randint(0, 100), r.randint(0, 100)], 
	'tol':[0.0001, 0.0010, 0.0009, 0.0008],
	'momentum':[0.9, 0.8, 0.7, 0.6, 0.5], 
	'beta_1':[0.9, 0.8, 0.7, 0.6, 0.5],
	'beta_2':[0.999, 0.888, 0.777, 0.667, 0.555],
}

DTREE_PARAMETERS={
	#'criterion':['gini', 'entropy'], 
	'splitter':['best', 'random'], 
	'max_depth':[None, 100, 300, 200, 250], 
	'min_samples_split':[2, 4, 6], 
	'min_samples_leaf':[1, 2, 4], 
	'random_state':[r.randint(0, 100), r.randint(0, 100)]
}

RF_PARAMETERS={
	'n_estimators':[10, 20, 15, 5],
	#'criterion':['gini', 'entropy'], 
	'max_depth':[None, 20, 30, 50], 
	'bootstrap':[True, False],
	#'oob_score':[False, True],
	'random_state':[r.randint(0, 100), r.randint(0, 100)]
}

XGB_PARAMETERS={
	'objective':['binary:logistic'],
	'eta':[0.1, 0.2, 0.15],
	'max_depth':[5, 6, 7, 6],
	'silent':[1],
	'eval_metric':["auc"],
	'min_child_weight':[1],
	'subsample':[0.7, 0.6, 0.8, 0.6],
	'colsample_bytree':[0.7, 0.5, 0.6, 0.65],
	'learning_rate':[0.075, 0.01, 0.015, 0.0175],
	'seed':[r.randint(0, 100), r.randint(0, 100)],
	'num_rounds':[200, 250, 220]
}

DTREE_PARAMETERS_LIST = list(ParameterSampler(DTREE_PARAMETERS, n_iter=N_DTREE, random_state=r.randint(0, 100)))

RF_PARAMETERS_LIST = list(ParameterSampler(RF_PARAMETERS, n_iter=N_RF, random_state=r.randint(0, 100)))

MLP_PARAMETERS_LIST = list(ParameterSampler(MLP_PARAMETERS, n_iter=N_MLP, random_state=r.randint(0, 100)))

XGB_PARAMETERS_LIST = list(ParameterSampler(XGB_PARAMETERS, n_iter=N_XGB, random_state=r.randint(0, 100)))

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

def get_oof(clf, x_train, y_train, x_test):
	ntrain = x_train.shape[0]
	ntest = x_test.shape[0]

	kfold = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=r.randint(0, 100))

	oof_train = np.zeros((ntrain,))
	oof_test = np.zeros((ntest,))
	oof_test_skf = np.empty((NFOLDS, ntest))

	for i, (train_index, test_index) in enumerate(kfold):
		x_tr = x_train[train_index]
		y_tr = y_train[train_index]
		x_te = x_train[test_index]

		clf.train(x_tr, y_tr)

		oof_train[test_index] = clf.predict(x_te)
		oof_test_skf[i, :] = clf.predict(x_test)

	oof_test[:] = oof_test_skf.mean(axis=0)
	return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def create_artificial_features(x_train, y_train, x_test):
	"""
	Create stacking - fit and predict pool of base classifiers lvl 0
	"""
	base_classifiers = []
	attrs = N_RF + N_DTREE + N_XGB + N_MLP
	stacking_train = np.zeros((x_train.shape[0], attrs))
	stacking_test = np.zeros((x_test.shape[0], attrs))

	stacking_train = np.zeros((x_train.shape[0], attrs))
	stacking_test = np.zeros((x_test.shape[0], attrs))

	for params in RF_PARAMETERS_LIST:
		base_classifiers.append(BaseClassifier(classifier=RandomForestRegressor, params=params))

	for params in XGB_PARAMETERS_LIST:
		base_classifiers.append(BaseClassifierXGBoost(params=params))

	for params in DTREE_PARAMETERS_LIST:
		base_classifiers.append(BaseClassifier(classifier=DecisionTreeRegressor, params=params))

	for params in MLP_PARAMETERS_LIST:
		base_classifiers.append(BaseClassifier(classifier=MLPRegressor, params=params))

	for i, classifier in enumerate(base_classifiers):
		print 'Base classifier {}'.format(i)
		meta_train, meta_test = get_oof(classifier, x_train, y_train, x_test)
		stacking_train[:, i:] = meta_train
		stacking_test[:, i:] = meta_test

	stacking_train = np.concatenate((stacking_train, x_train), axis=1)
	stacking_test = np.concatenate((stacking_test, x_test), axis=1)
	return stacking_train, stacking_test

def run_xgb(train_X, train_y, test_X, test_y=None, num_rounds=500, early_stopping_rounds=25):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.02
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = "auc"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = RS

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

def cross_validation(X_train, Y_train, get_oversampling=None):
    kf = KFold(X_train.shape[0], n_folds=NFOLDS, shuffle=False, random_state=RS)
    
    cv_auc = []
    cv_tpr = []
    cv_fpr = []
    
    for i, (train_index, valid_index) in enumerate(kf):
        print "{}Fold".format(i)
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
        
        stacking_train, stacking_valid = create_artificial_features(x_train, y_train, x_valid)
        predicts, model = run_xgb(stacking_train, y_train, stacking_valid, y_valid)
        fpr, tpr, thresholds = roc_curve(y_valid, predicts)
        AUC = auc(fpr, tpr)
        cv_auc.append(AUC)
        cv_tpr.append(tpr)
        cv_fpr.append(fpr)

        print "Stacking AUC: {}, TPR: {}, FPR: {}".format(np.mean(cv_auc), np.mean(cv_tpr), np.mean(cv_fpr))
    #AUC, TPR, FPR
    return np.mean(cv_auc), np.mean(cv_tpr), np.mean(cv_fpr)

def main():
    #df_train_tagging.
    columns_names = []
    for c in train_tagging.columns:
        columns_names.append(c + "_not_sw")
        
    df_train = pd.DataFrame()
    df_train = train_tagging
    df_train.columns = columns_names

    x_train = pd.concat((train, df_train), axis=1)
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

    y_train = x_train['is_duplicate']
    x_train = x_train[feature_names]

    print "To cross validate {}Fold Stacking".format(NFOLDS)
    AUC, TPR, FPR = cross_validation(x_train.values, y_train.values)
    print "Stacking AUC: {}, TPR: {}, FPR: {}".format(AUC, TPR, FPR, 0.8)
    
    #x_train, x_valid, y_train, y_valid = train_test_split(x_train.values, y_train.values, test_size=0.2, random_state=RS)
    #stacking_train, stacking_test = create_artificial_features(x_train, y_train, x_valid)
    #predicts, model = run_xgb(stacking_train, y_train, stacking_test, y_valid)

main()
print "Done stacking"