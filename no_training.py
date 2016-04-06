# 4.2889 => 329
# 4.3203 => 354
import csv
import math
import numpy as np
import sklearn

#np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=100)
nb_datasets = 510
var_threshold = 0.1
p_value_threshold = 0.7
nb_stocks = 198

def main():
	# Load Data
	actual = {}
	prices = {}
	train_prediction = {}
	prediction = {}
	weight = {}
	weights = {}
	predictions = []
	for stock in range(1, nb_stocks+1):
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

			train_prediction[stock][dataset], weights[stock][dataset] = perceptron(prices[stock][dataset][-5:],[1.0/5]*5, 0.0, 0.05, actual[stock][dataset])

			# prediction[stock][dataset] = 0.0

		weight[stock] = [0]*5
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
		prediction = -bias_factor
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

main()

