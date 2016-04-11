# take the last value of the stock and add/substract from it a constant
# 0.42667 => 228 (factor = 150)
import csv
import math
import numpy as np
import sklearn
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error as mse

#np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=100)
nb_datasets = 510
var_threshold = 0.1
mse_threshold = 2
p_value_threshold = 0.7
nb_stocks = 198

def main():
	# Load Data
	actual = {}
	prices = {}
	train_prediction = {}
	use_perceptron = {}
	prediction = {}
	predictions = []

	for stock in range(1, nb_stocks+1):
		print(stock)
		actual[stock] = {}   # dictionary where key is the day and value is stock value
		prices[stock] = {}   # dictionary where key is day and value is array of prices
		train_prediction[stock] = {}
		prediction[stock] = {}
		# for dataset in range(1, 200):
		# 	actual_temp, prices_temp = getData(stock,dataset)
		# 	actual[stock][dataset] = actual_temp
		# 	prices[stock][dataset] = prices_temp
		# 	train_prediction[stock][dataset], weights[stock][dataset] = perceptron(prices[stock][dataset][-1:],[1.0]*1, 0.0, 0.05, actual[stock][dataset])

		# 	# prediction[stock][dataset] = 0.0
		# weight[stock] = [0]*1
		# for i in range(1, len(weights[stock])):
		# 	weight[stock] = [x + y for x,y in zip(weight[stock],weights[stock][i])]
		# weight[stock] = [float(x)/len(weights[stock]) for x in weight[stock]]
		
		for dataset in range(200, nb_datasets+1):
			prices[stock][dataset] = getPrices(stock,dataset)
		# for dataset in range(1, nb_datasets+1):
		# 	prices[stock][dataset] = prices[stock][dataset][-len(weight[stock]):]

		stock_std = np.std(np.array(prices[stock].values()))
		stock_mean = np.mean(np.array(prices[stock].values()))
		if(True): #actual predictions	
			for dataset in range(201, nb_datasets+1):
				prediction[stock][dataset] = 0
				print("Taking last value")
				if(prices[stock][dataset][-1] <= stock_mean):
					prediction[stock][dataset] = prices[stock][dataset][-1]
				else:
					prediction[stock][dataset] = prices[stock][dataset][-1]
				print(prediction[stock][dataset])
		else:
			for dataset in range(1, 200):
				prediction[stock][dataset] = 0
				for i in range(0,len(weight[stock])):
					prediction[stock][dataset] += prices[stock][dataset][i] * weight[stock][i]

	for stock in range(1, nb_stocks+1):
		predictions.append([predict for predict in prediction[stock].values()])

	with open('sampleSubmission2.csv', 'w') as f:
		# np.savetxt(f, predictions, delimiter=",")
		writer = csv.writer(f, delimiter=',')
		writer.writerows(predictions)


def getData(n, d):
# Fetch prices vector for 'n'th stock from 'd'.csv
	prices = []
	with open("data/%d.csv" %d, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) # skipping column names
		for row in csvFileReader:
			prices.append(float(row[n-1]))

	# Fetch actual price of stock 2 hours later on that day from trainLabels.csv
	with open("trainLabels.csv", 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
	  	for skip in range(d):
	  		next(csvFileReader)
	  	actual = float(next(csvFileReader)[n])

	trainLabels = np.genfromtxt(fname = 'trainLabels.csv', skip_header = 1, delimiter = ',', dtype=float )
	actual = trainLabels[d,n]

	return actual, prices

def getPrices(n, d):
# Fetch prices vector for 'n'th stock from 'd'.csv
	prices = []
	with open("data/%d.csv" %d, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) # skipping column names
		for row in csvFileReader:
			prices.append(float(row[n-1]))

	return prices


###############################################

def perceptron(inputNodes, weight, bias_factor, learn_rate, actual):
# Not exactly a by-the-books perceptron, but the idea is to use
# an input of 5 prices to learn the weight associated with each price
# and then use this weight vector to make predictions on new inputs

	# Training it 10 times, idk why 10.
	for rep in range(10):
		# Weighted Sum function
		prediction = -bias_factor
		for i in range(0,len(inputNodes)):
			prediction += inputNodes[i] * weight[i]

		# Weight update rule
		for i in range(len(inputNodes)):
			# Multiplied the weight update by ((i+1)*1.0/len(inputNodes))
			# to factor the fact that the 5th input is more
			# important than the 1st one
			weight[i] = weight[i] + learn_rate * (actual - prediction) 
			#              * inputNodes[i] 
			#              * ((i+1)*1.0/len(inputNodes))
		total_weight = 0
		print("INITIAL WEIGHT")
		print(weight)
		# for x in range(0,len(weight)):
		# 	total_weight += weight[x]
		# print("TOTAL WEIGHT")
		# print(total_weight)
		# for x in range(0, len(weight)):
		# 	weight[x] = float(weight[x])/float(total_weight)
		# print("WEIGHT")
		# print(weight)
		prediction = -bias_factor
		for i in range(len(inputNodes)):										
			prediction += inputNodes[i] * weight[i]

		return float(prediction), weight

def calculateMSE(inputNodes, actual, avg_weights):
	predictions = [0]*len(inputNodes)
	for node in range(0, len(inputNodes)):
		predictions[node] = 0
		for i in range(0,len(avg_weights)):
			predictions[node] += inputNodes[node][i] * avg_weights[i]
	y_predicted = [prediction for prediction in predictions]
	loss = mse(actual, y_predicted)
	return loss

def perceptronModel(inputNodes, actuals, stock_mean, stock_std):
	#get the corresponding weight for each dataset
	weights = {}
	train_prediction = {}
	for node in range(0, len(inputNodes)):
		#find bias for model
		bias = 0.0
		print("STD DEV")
		print(stock_std)
		if(stock_mean > inputNodes[node]):
			bias = -float(stock_std)/float(50)
		elif(stock_mean < inputNodes[node]):
			bias = float(stock_std)/float(50)
		else:
			bias = 0.0
		weights[node] = []
		print("actuals")
		print(actuals[node])
		print("inputNodes")
		print(inputNodes[node])
		train_prediction[node], weights[node] = perceptron(inputNodes[node],[1.0]*len(inputNodes[0]), bias, 0.1, actuals[node])
		print("NODE WEIGHT")
		print(weights[node])
	#return average of weights
	avg_weights = [0]*len(weights[1])
	for i in range(0, len(weights[1])):
		weights = [x + y for x,y in zip(weights,weights[i])]
	avg_weights = [float(x)/float(len(weights)) for x in weights]

	return avg_weights

# find the variance of the features   => idea is to remove low variance features... need to normalize first!
def find_variance(features):
	var = {}
	for x in features:
		print(x)
		var[x] = np.var(features[x])
	return var

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

main()

