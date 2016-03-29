import csv
import numpy as np
import matplotlib.pyplot as plt

################################################

def getPrice(n, d):
# Fetch prices vector for 'n'th stock from 'd'.csv
	prices = []
	with open("data/%d.csv" %d, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) # skipping column names
		for row in csvFileReader:
			prices.append(float(row[n-1]))
	return prices

def getActualPrice(n, d):
# Fetch actual price of 'n'th stock 2 hours later the 'd'th day from trainLabels.csv
	with open("trainLabels.csv", 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
	  	for skip in range(d):
	  		next(csvFileReader)
	  	actual = float(next(csvFileReader)[n])
	return actual

###############################################

def perceptronTrain(stockPrices, actualPrices, weights, bias_factor,learn_rate):
# Train the weights vector using prices and final price 2 hours later
	for day in stock:
		prediction = -bias_factor
		for time in range(len(day)):	#weighted sum
			prediction += stockPrices[day][time] * weights[time]

		for index in range(len(weights)):	#weight update rule
			weights[index] += learn_rate * (actual - prediction) * stockPrices[day][time] * ((index + 1.0)/len(weights))

	return weights

################################################

def predict(stockPrices_day, weights, bias_factor):
# Predict the final price 2 hours later for given prices
	prediction = -bias_factor
	for index in range(len(weights)):
		prediction += stockPrices_day[index] * weights[index]
	return prediction

################################################

# for each stock:
#	get prices list for each training day -> append this price list to a list of lists
#	so we have a list of lists containing prices list on each training day
#	also get the actual price 2 hours later on each training day
#
#	train the weights vector for each stock using the last 5 (?) prices 
#				+ their actual 2 hours later prices from each training day
#	
#	use this weights vector to predict the the actual 2 hour later prices for the testing days 
#				(will need to get prices vectors for the selected stock from the training data files)

