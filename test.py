import csv
import math
import numpy as np
import sklearn
import scipy.stats
from pydoc import help
import sklearn.preprocessing as preprocessing

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from sklearn.svm import SVR
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFE    # recursive feature selection using cross validation
from sklearn.datasets import make_classification

from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet

# from sklearn.neural_network import MLPRegressor

from sklearn import cluster

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics.pairwise import euclidean_distances

# to access a data file (i.csv) : data[i]
# to access the jth column of a file i : data[i][:,j]
# stocks[1] = O1
# features[1] = I1
# inputs[0] = input vector at time 0

np.set_printoptions(threshold=np.nan)
# np.set_printoptions(threshold=100)
nb_datasets = 510
var_threshold = 0.07
p_value_threshold = 0.7
nb_stocks = 198


def main():
	# Load Data
	(data, stocks, features, training) = loadData()
	print(len(stocks))
	print(len(stocks[2]))

	#feature pre-processing : normalizing to compare variances
	norm_features = preprocess_features(features, 'normalize')
	#find variance of features
	feat_var = find_variance(norm_features)
	# find the features worth keeping based on variance
	high_var_feats_indices = filter_values_by_threshold(feat_var, var_threshold)
	print("HIIIII")
	print(len(high_var_feats_indices))
	# only keep the needed features
	high_var_features = filter_features_by_values(features,high_var_feats_indices)

	# Now standarize these remaining features
	high_var_std_features = preprocess_features(high_var_features, 'standardize')
	(high_var_feat_corr,high_var_feat_p_corr) = get_correlation_matrix2(high_var_std_features)
	cluster_labels = np.array([label for label in high_var_features.keys()])
	print(len(cluster_labels))
	print((cluster_labels).shape)

	# Cluster the features together using K-means clustering, based on their p-values!
	# either use AffinityPropagation OR K-Means
	ap = cluster.AffinityPropagation()
	print('HELLOOO 1')
	print(len(high_var_feat_p_corr))
	print(np.matrix(high_var_feat_p_corr).shape)
	ap.fit(high_var_feat_p_corr,cluster_labels)
	print('HELLOOO 2')
	print(len(ap.cluster_centers_indices_))
	print(ap.labels_)
	print(len(ap.labels_))
	# for each cluster find 1 feature closest to cluster center
	labels = [label for label in ap.labels_]
	#get 1 index correponding to each cluster / closet feature from cluster
	initial_indices = [index for index in high_var_feats_indices.keys()]
	already_selected = []
	selected_indices = []
	for label in labels:
		if(label not in already_selected):
			selected_indices.append(initial_indices[label])
			already_selected.append(label)

	selected_indices2 = []
	for cluster_index in range(0,len(ap.cluster_centers_)):
		closest = 0
		min_distance = 100000
		for label_index in range(cluster_index,len(ap.labels_)):
			if(ap.labels_[label_index] == cluster_index):
				distance = euclidean_distances(high_var_feat_p_corr[label_index],ap.cluster_centers_[cluster_index])
				if(distance < min_distance):
					closest = label_index
					min_distance = distance
		selected_indices2.append(initial_indices[closest])

	print(len(selected_indices))
	print(len(selected_indices2))
	print(selected_indices2[0])
	print("SELECTED INDICES 1,2")
	print(selected_indices)
	print(selected_indices2)
	#get features from indices
	selected_features = filter_features_by_values(high_var_std_features,selected_indices)
	X = [feature for feature in selected_features.values()]
	X = [feature for feature in X]
	X = np.transpose(np.array(X))
	print(np.array(X).shape)
	print(np.array(stocks[1]).shape)

	selected_features2 = filter_features_by_values(high_var_std_features,selected_indices2)
	stock_features = {}
	#use these clusters in feature elimination
	# for stock in range(1, nb_stocks+1):
	# 	# PERFORM FEAUTRE SELECTION ALGORITHM

	# 	# SELECT K-BEST on f-test 
	# 	kb = SelectKBest(f_regression, k=2)
	# 	# here we also modify the stock values... not sure if we should!
	# 	stock_features[stock] = kb.fit_transform(X, stocks[stock])

	# 	X2 = [feature for feature in selected_features2.values()]
	# 	X2 = [feature for feature in X2]
	# 	X2 = np.transpose(np.array(X2))
	# 	print(np.array(X2).shape)
	# 	print(np.array(stocks[1]).shape)

	# 	#Recursive feature elimination
	# 	svr = SVR(kernel="linear")
	# 	rfe = RFE(svr, step=1)
	# 	rfe = rfe.fit(X2,stocks[stock])
	# 	print(rfe.support_)
	# 	print(rfe.ranking_)

	# IT MIGHT BE BETTER TO USE ONLY THE LAST FEATURES OF EACH DAY IN OUR RFE VS PREDICTING VALUE IN OUR TESTING DATA...
	closing_inputs = {}
	for stock in range(1, nb_stocks+1):
		closing_inputs[stock] = {}
		for dataset in range(1,nb_datasets+1):
			closing_inputs[stock][dataset] = data[dataset][54,stock-1]
	inputs = {}
	for stock in range(1, nb_stocks+1):
		inputt = []
		for dataset in range(1, nb_datasets+1):
			print(len(selected_features2))
			inputt.append(np.append((closing_inputs[stock][dataset]),[selected_features2[index][dataset] for index in selected_indices2]))
		inputs[stock] = inputt
		print(len(inputs[stock]))

	selected_feature_keys = {}
	for stock in range(1, nb_stocks+1):

		X3 = [inputt for inputt in inputs[stock]][:200]
		outputs = [output for output in training[stock]][:200]
		
		# kb = SelectKBest(f_regression, k=2)
		# # here we also modify the stock values... not sure if we should!
		# kb.fit(X3, outputs)
		# print(kb.scores_)
		svr = SVR(kernel="linear")
		rfe = RFE(svr, step=1)
		rfe = rfe.fit(X3,outputs)
		print(rfe.support_)
		print(rfe.ranking_)
		selected_feature_keys[stock] = []
		count = 0
		for key in selected_features2.keys():
			if (rfe.support_[count] == True):
				selected_feature_keys[stock].append(key)
			count = count + 1


	feature_inputs = {}
	for stock in range(1, nb_stocks+1):
		feature_inputs[stock] = {}
		count = 0
		for dataset in range(1,nb_datasets+1):
			feature_inputs[stock][dataset] = []
			for key in selected_feature_keys[stock]:
				feature_inputs[stock][dataset].append(data[dataset][54,198:][key-1])

	closing_inputs2 = {}
	for stock in range(1, nb_stocks+1):
		closing_inputs2[stock] = {}
		for dataset in range(1,nb_datasets+1):
			closing_inputs2[stock][dataset] = data[dataset][54,stock-1]

	inputs2 = {}
	for stock in range(1, nb_stocks+1):
		inputt = []
		for dataset in range(1, nb_datasets+1): 
			inputt.append(np.append((closing_inputs2[stock][dataset]),(feature_inputs[stock][dataset])))
		inputs2[stock] = inputt
		print(len(inputs2[stock]))

	return inputs2

# Load data
def loadData():
    trainLabels = np.genfromtxt(fname = 'trainLabels.csv', skip_header = 1, delimiter = ',', dtype=float )

    # Get the training data
    training = {}
    for x in range(1,nb_stocks+1):
        training[x] = np.array(trainLabels[:,x], dtype=float)

    # data : array containing all the csv data up to the "nb_datasets"'s csv file.
    data = {}
    for x in range(1,nb_datasets+1):
        data[x] = np.genfromtxt(fname = 'data/%d.csv' %x,skip_header = 1, delimiter = ',', dtype=float )
        
    # Each output reprensents a security : stocks[5] = stock prices of O5 over all datasets
    stocks = {}
    for x in range(1,nb_stocks+1):  # there are 198 securities 
        stocks[x] = np.array(data[1][:,x-1])  # gets the colomn of the first datafile
        for y in range(2,nb_datasets+1):
            stocks[x] = np.append(stocks[x],data[y][:,x-1]) 

    # Evolution of the features (I5 = features[5]) over all datasets 
    features = {}
    for x in range(1,245):    # 244 features total
        features[x] = np.array(data[1][:,x+197])
        for y in range(2,nb_datasets+1):
            features[x] = np.append(features[x],data[y][:,x+197])

    # Vector of features used to predict the stocks
    inputs = {}
    dataset = 1
    y = 1
    total_datapoints = nb_datasets*55   # 55 time entries per datafile
    x = 0
    inputs[x] = np.array(data[dataset][y-1][198:])
    while( x < total_datapoints+1 and dataset <= nb_datasets):
        if(y < 55):       # we haven't yet reached the end of the datafile yet
            x = x + 1
            inputs[x] = np.array(data[dataset][y][198:]) # start taking entries from 198th column
            y = y + 1
        else:
            dataset = dataset+1   # we've reached the end of the datafile
            y = 0

    return (data, stocks, features, training)

def preprocess_features(features, type = 'standardize', indices = None):  # default is standarize, unless normalized specified
    if(type == 'normalize'):
        return normalize_features(features, indices)
    else:
        return standarize_features(features, indices)

# if we choose to standarize the features
def standarize_features(features, indices= None):
    std_features = {}
    if(indices):  # indices are specified
        for x in indices:
            std_scale = preprocessing.StandardScaler().fit(features[x])
            std_features[x] = np.array(std_scale.transform(features[x]))
    else:         # default: go through all features
        for x in features:
        	print(x)
        	std_scale = preprocessing.StandardScaler().fit(features[x])
        	std_features[x] = np.array(std_scale.transform(features[x]))
    return std_features

# if we choose to normalize the features... Need to be very careful with outliers...
def normalize_features(features, indices = None):
    norm_features = {}
    if(indices):
        for x in indices:
        	feat_min = min(features[x])
        	feat_max = max(features[x])
        	norm_features[x] = np.array([(feat_val-feat_min)/(feat_max-feat_min) for feat_val in features[x]])
    else:     # default
        for x in features:
        	print(x)
        	feat_min = min(features[x])
        	feat_max = max(features[x])
        	norm_features[x] = np.array([(feat_val-feat_min)/(feat_max-feat_min) for feat_val in features[x]])
    return norm_features
        
# Getting the correlation matrix for the features, used in feature selection
def get_correlation_matrix(features, indices = None): 
	corr = {}   # matrix containing Pearson's correclaiton coefficients of features
	p_corr = {}  # matrix containing 2-tail p-value for uncorrelation.. high p_value => uncorrelated
	if(indices):
		for x in indices:
			for y in indices:
				corr[(x,y)] = scipy.stats.pearsonr(features[x],features[y])[0]   #requires normal distribution of features..
				p_corr[(x,y)] = scipy.stats.pearsonr(features[x],features[y])[1]
	else:
		for x in features:
			for y in features:
				corr[(x,y)] = scipy.stats.pearsonr(features[x],features[y])[0]   #requires normal distribution of features..
				p_corr[(x,y)] = scipy.stats.pearsonr(features[x],features[y])[1]
	return (corr, p_corr)

def get_correlation_matrix2(features, indices = None): 
    corr = []   # matrix containing Pearson's correclaiton coefficients of features
    p_corr = []  # matrix containing 2-tail p-value for uncorrelation.. high p_value => uncorrelated
    if(indices):
        for x in indices:
            corr.append([scipy.stats.pearsonr(features[x],features[y])[0] for y in indices])   #requires normal distribution of features..
            p_corr.append([scipy.stats.pearsonr(features[x],features[y])[1] for y in indices])
    else:
        for x in features:
            corr.append([scipy.stats.pearsonr(features[x],features[y])[0] for y in features])   #requires normal distribution of features..
            p_corr.append([scipy.stats.pearsonr(features[x],features[y])[1] for y in features])
    return (corr, p_corr)


# find the features that are not correlated    => idea is to remove the correlated ones...
def filter_matrix_by_threshold(values, threshold, keep = 'greater'):
    filtered = []
    for x in range(0,values.shape[0]):
        for y in range(0,values.shape[1]):
            if(keep == 'greater'):
                if(values[x][y] >= threshold):
                    filtered.append((x,y))
            else:
                if(values[x][y] <= threshold):
                    filtered.append((x,y))
    return filtered
    
# find the features that are not correlated    => idea is to remove the correlated ones...
def filter_tuples_by_threshold(tuples, threshold, keep = 'greater'):
    filtered = {}
    for tuple in tuples:
        if(keep == 'greater'):
            if(tuples[tuple] >= threshold):
                filtered[tuple] = tuples[tuple]
        else:
            if(tuples[tuple] <= threshold):
                filtered[tuple] = tuples[tuple]
    return filtered


def filter_values_by_threshold(values, threshold, keep = 'greater'):
    filtered = {}
    for key in values:
        if(keep == 'greater'):
            if(values[key] >= threshold):
                filtered[key] = values[key]
        else:
            if(values[key] <= threshold):
                filtered[key] = values[key]
    return filtered

def filter_features_by_values(features, values):
    filtered_features = {}
    for key in values:
        if not (key in filtered_features):
            filtered_features[key] = features[key]
    return filtered_features

# find the variance of the features   => idea is to remove low variance features... need to normalize first!
def find_variance(features):
    var = {}
    for x in features:
    	print(x)
        var[x] = np.var(features[x])
    return var

# used as a dummy x_axis for features
def generate_axis(start, end, step):
    axis = np.zeros(int(end+1/step))
    for i in range(start, end+1):
        axis[i] = i
    axis = axis[1:]
    return axis

def plot_key_value(d):
	x_axis = np.fromiter(d.keys(), dtype = int)
	y_axis = [float(value) for value in d.values()]
	fig = plt.figure()
	plt.plot(y_axis)
	fig.suptitle('Variance of normalized features', fontsize=20)
	plt.xlabel('Features', fontsize=18)
	plt.ylabel('Variance', fontsize=16)
	fig.savefig('normed_feat_var.jpg')

main()

