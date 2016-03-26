import csv
import numpy as np
import matplotlib.pyplot as plt

########################################################

def getData(n):
	# Fetch prices vector for n'th stock from 1.csv
	prices = []
	with open("data/1.csv", 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) # skipping column names
		for row in csvFileReader:
			prices.append(float(row[n-1]))

	# Fetch actual price of stock 2 hours later from trainLabels.csv
	with open("trainLabels.csv", 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
	  	next(csvFileReader) # skipping column names
	  	actual = float(next(csvFileReader)[n])

	return actual, prices

###############################################

def perceptron(inputNodes, weight, bias_factor, learn_rate):

	for rep in range(5):
		# Weighted Sum function
		prediction = -bias_factor
		for i in range(len(inputNodes)):
			prediction += inputNodes[i] * weight[i]

		# Weight update rule
		# global variable actual used in loss function
		for i in range(len(inputNodes)):
			weight[i] += learn_rate * (actual - prediction) * inputNodes[i] * ((i+1)*1.0/len(inputNodes))

		print rep+1, prediction
		print weight
		print

	prediction = -bias_factor
	for i in range(len(inputNodes)):										
		prediction += inputNodes[i] * weight[i]

	return prediction, weight

#############################################################

actual, prices = getData(1)
time = [i for i in range(1,len(prices)+1)]

plt.scatter(time, prices)
plt.xlabel('Time Interval')
plt.ylabel('Stock Price')
plt.show()

prediction, weight = perceptron(prices[-5:], [1.0/5]*5, 0.0, 0.05)

print prediction, actual

##################################################################

# Now use the weights vector obtained on training the 
# perceptron for the 1st stock on another stock

actual, prices = getData(2)

plt.scatter(time, prices)
plt.xlabel('Time Interval')
plt.ylabel('Stock Price')
plt.show()

prices = prices[-5:]
prediction = 0.0
for i in range(len(weight)):
	prediction += prices[i] * weight[i]

print prediction, actual
