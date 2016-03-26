import csv
import numpy as np
import matplotlib.pyplot as plt

########################################################

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

	return actual, prices

###############################################

def perceptron(inputNodes, weight, bias_factor, learn_rate):
# Not exactly a by-the-books perceptron, but the idea is to use
# an input of 5 prices to learn the weight associated with each price
# and then use this weight vector to make predictions on new inputs

	# Training it 5 times, idk why 5.
	for rep in range(5):
		# Weighted Sum function
		prediction = -bias_factor
		for i in range(len(inputNodes)):
			prediction += inputNodes[i] * weight[i]

		# Weight update rule
		# global variable actual used in loss function
		for i in range(len(inputNodes)):
			# Multiplied the weight update by ((i+1)*1.0/len(inputNodes))
			# to factor the fact that the 5th input is more
			# important than the 1st one
			weight[i] += learn_rate * (actual - prediction) * inputNodes[i] * ((i+1)*1.0/len(inputNodes))

		print rep+1, prediction
		print weight
		print

	prediction = -bias_factor
	for i in range(len(inputNodes)):										
		prediction += inputNodes[i] * weight[i]

	return prediction, weight

#############################################################

actual, prices = getData(1,1)
time = [i for i in range(1,len(prices)+1)]

plt.scatter(time, prices)
plt.xlabel('Time Interval')
plt.ylabel('Stock Price')
plt.show()

prediction, weight = perceptron(prices[-5:], [1.0/5]*5, 0.0, 0.05)

print prediction, actual

##################################################################

# Use the weights vector obtained on training the perceptron 
# for the 1st stock for day 1 on the 2nd stock for day 1

actual, prices = getData(2,1)

plt.scatter(time, prices)
plt.xlabel('Time Interval')
plt.ylabel('Stock Price')
plt.show()

prices = prices[-5:]
prediction = 0.0
for i in range(len(weight)):
	prediction += prices[i] * weight[i]

print prediction, actual

##############################################################

# Use the weights vector obtained on training the perceptron 
# for the 1st stock for day 1 on the 1st stock for day 2

actual, prices = getData(1,2)

plt.scatter(time, prices)
plt.xlabel('Time Interval')
plt.ylabel('Stock Price')
plt.show()

prices = prices[-5:]
prediction = 0.0
for i in range(len(weight)):
	prediction += prices[i] * weight[i]

print prediction, actual
