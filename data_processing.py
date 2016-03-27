import csv
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
# from sklearn.neural_network import MLPRegressor


# to access a data file (i.csv) : data[i]
# to access the jth column of a file i : data[i][:,j]
# stocks[1] = O1
# features[1] = I1
# inputs[0] = input vector at time 0

np.set_printoptions(threshold=np.nan)
#np.set_printoptions(threshold=100)
nb_datasets = 510
var_threshold = 0.1
p_value_threshold = 0.7
nb_stocks = 198


def main():
	# Load Data
	(data, stocks, features, training) = loadData()

	#feature pre-processing : normalizing to compare variances
	norm_features = preprocess_features(features, 'normalize')

	#find variance of features
	feat_var = find_variance(norm_features)

	# find the features worth keeping based on variance:
	high_var_feats_indices = filter_values_by_threshold(feat_var, var_threshold)

	# only keep the needed features
	high_var_features = filter_features_by_values(features,high_var_feats_indices)

	# Now standarize these remaining features
	high_var_std_features = preprocess_features(high_var_features, 'standardize')

	# get correlation matrix with standarized remaining features
	(high_var_feat_corr,high_var_feat_p_corr) = get_correlation_matrix(high_var_std_features)

	# filter by high p-value
	final_tuples = filter_tuples_by_threshold(high_var_feat_p_corr,p_value_threshold)

	# get final feature names
	final_feature_ids = [value[0] for value in final_tuples.keys()]
	final_feats = filter_features_by_values(features,final_feature_ids)

	# get input features for selection (as vectors)
	input_feats = {}
	for x in range(0,len(stocks[1])):
	    input_feats[x] = np.zeros(len(final_feats))
	    count = 0
	    for key in final_feats:
	        input_feats[x][count] = final_feats[key][x]
	        count = count+1

	final_features = [final_feature for final_feature in input_feats.values()]

	# now select the best features for each stocks
	feature_inputs = {}
	selected_feature_keys = {}
	for stock in range(1, nb_stocks+1):
		# Recursive feature elimination
		svr = SVR(kernel="linear")
		rfe = RFE(svr, step=1)
		rfe = rfe.fit(final_features,stocks[stock])
		rfe.support_
		rfe.ranking_

		# selected features by RFE
		selected_feature_keys[stock] = []
		count = 0
		for key in final_feats.keys():
		    if (rfe.support_[count] == True):
		        selected_feature_keys[stock].append(key)
		    count = count + 1

	for stock in range(1, nb_stocks+1):
		feature_inputs[stock] = {}
		count = 0
		for dataset in range(1,nb_datasets+1):
			feature_inputs[stock][dataset] = []
			for key in selected_feature_keys[stock]:
				feature_inputs[stock][dataset].append(data[dataset][54,198:][key-1])
		print(len(feature_inputs[stock]))

	# get the closing values to be used in our inputs 
	# closing_inputs[1] = vector of closing values for security 1
	closing_inputs = {}
	for stock in range(1, nb_stocks+1):
		closing_inputs[stock] = {}
		for dataset in range(1,nb_datasets+1):
			closing_inputs[stock][dataset] = data[dataset][54,stock-1]

	inputs = {}
	for stock in range(1, nb_stocks+1):
		inputt = []
		for dataset in range(1, nb_datasets+1):	
			inputt.append(np.append((closing_inputs[stock][dataset]),(feature_inputs[stock][dataset])))
		inputs[stock] = inputt
		print(len(inputs[stock]))

	predictions = []
	predictions.append([x for x in range(201,nb_datasets+1)])
	for stock in range(1,nb_stocks+1):
		outputs = [output for output in training[stock]] 
		final_inputs = [inputt for inputt in inputs[stock]]

		training_inputs = final_inputs[:200]
		prediction_inputs = final_inputs[200:]
		print(np.array(training_inputs).shape)
		print(np.array(outputs).shape)

		# perceptron = Perceptron(penalty = 'l1')
		# perceptron = perceptron.fit(np.array(inputs, dtype=float), np.array(outputs, dtype=float))
		# perceptron = perceptron.fit(np.array(inputs), np.array(outputs))

		# sgd = SGDClassifier(loss = 'perceptron', penalty = 'l1')
		# sgd.fit(np.array(inputs), np.array(outputs))

		# mlp = MLPRegressor()
		# mlp.fit(np.array(inputs), np.array(outputs, dtype='float')[0])

		sgd = SGDRegressor(loss="huber", penalty='l1')
		sgd.fit(np.array(training_inputs), np.array(outputs))

		# now predict...
		sgd_predict = sgd.predict(np.array(prediction_inputs))
		predictions.append([prediction for prediction in sgd_predict])

	predictions = np.array(predictions)
	with open('sampleSubmission1.csv', 'w') as f:
		# np.savetxt(f, predictions, delimiter=",")
		writer = csv.writer(f, delimiter=',')
		writer.writerows(predictions)


# Load data
def loadData():
	trainLabels = np.genfromtxt(fname = '/Users/Clovis/Documents/My Courses/Machine Learning/Project/trainLabels.csv', skip_header = 1, delimiter = ',', dtype=float )

	# Get the training data
	training = {}
	for x in range(1,nb_stocks+1):
		training[x] = np.array(trainLabels[:,x], dtype=float)

	# data : array containing all the csv data up to the "nb_datasets"'s csv file.
	data = {}
	for x in range(1,nb_datasets+1):
	    data[x] = np.genfromtxt(fname = '/Users/Clovis/Documents/My Courses/Machine Learning/Project/data/%d.csv' %x,skip_header = 1, delimiter = ',', dtype=float )
	    
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
            
    plt.plot(y_axis)
    plt.show()


main()

