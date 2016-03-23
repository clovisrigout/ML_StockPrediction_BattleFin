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


# to access a data file (i.csv) : data[i]
# to access the jth column of a file i : data[i][:,j]
# outputs[1] = O1
# features[1] = I1
# inputs[0] = input vector at time 0

# np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=100)
nb_datasets = 30

# training labels
trainLabels = np.genfromtxt(
    fname = '/Users/Clovis/Documents/My Courses/Machine Learning/Project/trainLabels.csv',
    skip_header = 1,
    delimiter = ',',
    )

# data : array containing all the csv data up to the "nb_datasets"'s csv file.
data = {}
for x in range(1,nb_datasets+1):
    data[x] = np.genfromtxt(
    fname = '/Users/Clovis/Documents/My Courses/Machine Learning/Project/data/%d.csv' %x,
    skip_header = 1,
    delimiter = ',',
    )
    
# Each output reprensents a security : outputs[5] = stock prices of O5 over all datasets
outputs = {}
for x in range(1,199):  # there are 198 securities 
    outputs[x] = np.array(data[1][:,x-1])  # gets the colomn of the first datafile
    for y in range(2,nb_datasets):
        outputs[x] = np.append(outputs[x],data[y][:,x-1]) 

# Evolution of the features (I5 = features[5]) over all datasets 
features = {}
for x in range(1,245):    # 244 features total
    features[x] = np.array(data[1][:,x+197])
    for y in range(2,nb_datasets):
        features[x] = np.append(features[x],data[y][:,x+197])

# Vector of features used to predict the outputs
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

def static_preprocess_features(features, type = 'standardize', indices = None):  # default is standarize, unless normalized specified
    if(type == 'normalize'):
        static_normalize_features(features, indices)
    else:
        static_standarize_features(features, indices)

def preprocess_features(features, type = 'standardize', indices = None):  # default is standarize, unless normalized specified
    if(type == 'normalize'):
        return normalize_features(features, indices)
    else:
        return standarize_features(features, indices)

# if we choose to standarize the features
def static_standarize_features(features, indices= None):
    if(indices):  # indices are specified
        for x in indices:
            std_scale = preprocessing.StandardScaler().fit(features[x])
            features[x] = std_scale.transform(features[x])
    else:         # default: go through all features
        for x in range(0,len(features)):
            std_scale = preprocessing.StandardScaler().fit(features[x])
            features[x] = std_scale.transform(features[x])

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
    
features = preprocess_features(features, 'normalize')

    
# Getting the correlation matrix for the features, used in feature selection
def get_correlation_matrix2(features, indices = None):     
    if(indices):
        corr = np.zeros([len(indices),len(indices)])   # matrix containing Pearson’s correlation coefficients of features
        p_corr = np.zeros([len(indices),len(indices)]) # matrix containing 2-tail p-value for uncorrelation (high p-value => uncorrelated)
        for x in indices:
            for y in indices:
                corr[x][y] = scipy.stats.pearsonr(features[x],features[y])[0]   #requires normal distribution of features..
                p_corr[x][y] = scipy.stats.pearsonr(features[x],features[y])[1]
    else:
        corr = np.zeros([len(features),len(features)])   # matrix containing Pearson’s correlation coefficients of features
        p_corr = np.zeros([len(features),len(features)]) # matrix containing 2-tail p-value for uncorrelation (high p-value => uncorrelated)
        for x in range(1,len(features)):
            for y in range(1,len(features)):
                corr[x][y] = scipy.stats.pearsonr(features[x],features[y])[0]   #requires normal distribution of features..
                p_corr[x][y] = scipy.stats.pearsonr(features[x],features[y])[1]
    return (corr, p_corr)
    
# Getting the correlation matrix for the features, used in feature selection
def get_correlation_matrix(features, indices = None): 
    corr = {}    # matrix containing Pearson’s correlation coefficients of features
    p_corr = {}  # matrix containing 2-tail p-value for uncorrelation (high p-value => uncorrelated)
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


(feat_corr, feat_p_corr) = get_correlation_matrix(features)


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

# find the variance of the features   => idea is to remove low variance features... NEED TO STANDARIZE FIRST!
def find_variance(features):
    var = {}
    for x in features:
        var[x] = np.var(features[x])
    return var
    
feat_var = find_variance(features)

# used as a dummy x_axis for features
def generate_axis(start, end, step):
    axis = np.zeros(int(end+1/step))
    for i in range(start, end+1):
        axis[i] = i
    axis = axis[1:]
    return axis
    
feat_x_axis = generate_axis(0,len(feat_var),1)

def plot_key_value(d: dict):
    x_axis = np.fromiter(d.keys(), dtype = int)
    y_axis = [float(value) for value in d.values()]
            
    plt.plot(y_axis)
    plt.show()
    
plot_key_value(feat_var)


# find the features worth keeping based on variance:

high_var_feats_indices = filter_values_by_threshold(feat_var, 0.05)

# only keep the needed features
high_var_features = filter_features_by_values(features,high_var_feats_indices)

# Now standarize these remaining features
high_var_std_features = preprocess_features(high_var_features, 'standardize')

# get correlation matrix with standarized remaining features
(high_var_feat_corr,high_var_feat_p_corr) = get_correlation_matrix(high_var_std_features)

# filter by high p-value
final_tuples = filter_tuples_by_threshold(high_var_feat_p_corr,0.6)

# get final feature names
final_feature_ids = [value[0] for value in final_tuples.keys()]
final_feats = filter_features_by_values(features,final_feature_ids)

# get final inputs (as vectors)
final_inputs = {}
for x in range(0,len(outputs[1])):
    final_inputs[x] = np.zeros(len(final_feats))
    count = 0
    for key in final_feats:
        final_inputs[x][count] = final_feats[key][x]
        count = count+1

inputs = [input for input in final_inputs.values()]

# Recursive feature elimination

svr = SVR(kernel="linear")
rfe = RFE(svr, step=1)
rfe = rfe.fit(inputs,outputs[1])
rfe.support_
rfe.ranking_


# selected features by RFE
selected_features = []
count = 0
for key in final_feats.keys():
    if (rfe.support_[count] == True):
        selected_features.append(key)
    count = count + 1
    
    
# Randomized Lasso for feature selection
rlasso = RandomizedLasso(alpha=1)
rlasso.fit(inputs, outputs[2])
rlasso.scores_


