import csv
# import numpy as np

# take day 1
	# then take last 5 prices of each stock
	# and take actual 2hr later prices for each stock also
	# get a weights vector for each stock from that
# use this weights vector to predict for rest

trainDay = 2

def getPrices(d):
	# get last 5 prices for day 'd'
	prices = []
	with open("data/%d.csv" %d, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		for skip in range(51):
			next(csvFileReader)
		for priceRow in csvFileReader:
			prices.append((priceRow[:198]))
	return prices 	# this is a weird list of lists btw- each element is a list of prices of all 198 stocks at that time.


prices = getPrices(trainDay)
actualPrices = []
with open("trainLabels.csv", 'r') as csvfile:	# get actual prices for each stock for day 1
	csvFileReader = csv.reader(csvfile)
	for skip in range(trainDay):
		next(csvFileReader)
	actualPrices = next(csvFileReader)[1:]

weights = [ [1.0/5]*5 ] * 198	# random initial values

for stock in range(198):
	prediction = 0.0
	prediction_prev = 0.0
	count = 0

	#training the weights for each stock
	while abs(float(actualPrices[stock]) - prediction) > 0.01:
		for i in range(5):
			prediction += weights[stock][i] * float(prices[i][stock])

		if abs(float(actualPrices[stock]) - prediction) > abs(float(actualPrices[stock]) - prediction_prev):
			break
		else: 
			prediction_prev = prediction
		
		for i in range(5):
			weights[stock][i] += 0.05 * (float(actualPrices[stock]) - prediction) * float(prices[i][stock]) * ((i+1.0)/5)

with open("day%d_last5_perceptron_weights.csv" %trainDay, 'a') as output:
	output.write('FileID')
	for i in range(198):
		output.write(',O'+str(i+1))
	output.write('\n')

	for day in range(201,511):
		prices = getPrices(day)

		predictedPrices = [day]
		for stock in range(198):
			prediction = 0.0
			for i in range(5):
				prediction += weights[stock][i] * float(prices[i][stock])
			predictedPrices.append(prediction)

		output.write(str(predictedPrices[0]))
		for item in predictedPrices[1:]:
			output.write(',' + str(item))
		output.write('\n')
