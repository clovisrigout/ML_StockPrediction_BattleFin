# current : 	0.43480 => 375
import os
import csv
import math
import numpy as np
import sklearn
from sklearn import cross_validation

#np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=100)
nb_datasets = 510
var_threshold = 0.1
p_value_threshold = 0.7
nb_stocks = 198

def main():
	# get which features to use in perceptron per stock...
	import test
	inputsss = test.main()
	print(len(inputsss))
	# Load Data
	(data, stocks, features, training) = loadData()
	actual = {}
	prices = {}
	train_prediction = {}
	prediction = {}
	weight = {}
	weights = {}
	predictions = []
	for stock in range(1, nb_stocks+1):
		print("CLOVIS")
		print(stock)
		actual[stock] = {}   # dictionary where key is the day and value is stock value
		prices[stock] = {}   # dictionary where key is day and value is array of prices
		weights[stock] = {}
		train_prediction[stock] = {}
		prediction[stock] = {}
		for dataset in range(1, 200):
			actual_temp, prices_temp = getData(stock,dataset)
			actual[stock][dataset] = actual_temp
			prices[stock][dataset] = prices_temp
			last_values = prices[stock][dataset][-1:]
			print(last_values)
			print(len(inputsss[stock][dataset-1]))
			inputt = last_values
			print(inputt)
			inputt.append(data[dataset][54,10+197])
			inputt.append(data[dataset][54,129+197])
			print(inputt)
			print("HELLO")
			print(len(inputt))
			inital_weights = []
			inital_weights.append(0.9)
			inital_weights.extend([1/(len(inputsss[stock][dataset-1])-1)]*(len(inputsss[stock][dataset-1])-1))
			print(len(inital_weights))
			print(inital_weights)
			train_prediction[stock][dataset], weights[stock][dataset] = perceptron(inputsss[stock][dataset-1],inital_weights, 0.0, 0.05, actual[stock][dataset])

			# prediction[stock][dataset] = 0.0

		weight[stock] = [0]*len(inputsss[stock][0])
		for i in range(1, len(weights[stock])):
			weight[stock] = [x + y for x,y in zip(weight[stock],weights[stock][i])]
		weight[stock] = [float(x)/len(weights[stock]) for x in weight[stock]]
		
		for dataset in range(200, nb_datasets+1):
			prices[stock][dataset] = getPrices(stock,dataset)
		for dataset in range(1, nb_datasets+1):
			prices[stock][dataset] = prices[stock][dataset][-len(weight[stock]):]
		if(True): #actual predictions	
			for dataset in range(201, nb_datasets+1):
				prediction[stock][dataset] = 0
				for i in range(len(weight[stock])):
					prediction[stock][dataset] += prices[stock][dataset][i] * weight[stock][i]
		else:
			for dataset in range(1, 200):
				prediction[stock][dataset] = 0
				for i in range(len(weight[stock])):
					prediction[stock][dataset] += prices[stock][dataset][i] * weight[stock][i]
	for stock in range(1, nb_stocks+1):
		predictions.append([predict for predict in prediction[stock].values()])

	predictions = np.array(predictions)
	with open('sampleSubmission1.csv', 'w') as f:
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
		prediction = float(-bias_factor)
		for i in range(len(inputNodes)):
			prediction += inputNodes[i] * weight[i]

		# Weight update rule
		for i in range(len(inputNodes)):
			# Multiplied the weight update by ((i+1)*1.0/len(inputNodes))
			# to factor the fact that the 5th input is more
			# important than the 1st one
			weight[i] += learn_rate * (actual - prediction) * inputNodes[i] 
			#              * ((i+1)*1.0/len(inputNodes))
		total_weight = 0
		for x in range(0,len(weight)):
			total_weight += weight[x]
		for x in range(0, len(weight)):
			weight[x] = float(weight[x])/float(total_weight)

		prediction = -bias_factor
		for i in range(len(inputNodes)):										
			prediction += inputNodes[i] * weight[i]


		return float(prediction), weight

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


main()

