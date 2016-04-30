import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
trainDay = 1

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


# prices = getPrices(trainDay)
actualPrices = []
with open("trainLabels.csv", 'r') as csvfile:	# get actual prices for each stock for day 1
	csvFileReader = csv.reader(csvfile)
	for skip in range(trainDay):
		next(csvFileReader)
	actualPrices = next(csvFileReader)[1:]

# weights = [ [1.0/5]*5 ] * 198	# random initial values

# for stock in range(198):
# 	prediction = 0.0
# 	prediction_prev = 0.0
# 	count = 0

# 	#training the weights for each stock
# 	while abs(float(actualPrices[stock]) - prediction) > 0.01:
# 		for i in range(5):
# 			prediction += weights[stock][i] * float(prices[i][stock])

# 		if abs(float(actualPrices[stock]) - prediction) > abs(float(actualPrices[stock]) - prediction_prev):
# 			break
# 		else: 
# 			prediction_prev = prediction
		
# 		for i in range(5):
# 			weights[stock][i] += 0.05 * (float(actualPrices[stock]) - prediction) * float(prices[i][stock]) * ((i+1.0)/5)

with open("last_price_linearmodel.csv", 'w') as output:
	output.write('FileID')
	for i in range(198):
		output.write(',O'+str(i+1))
	output.write('\n')

	for day in range(1,2):
		prices = getPrices(day)
		days = [i for i in range(5)]

		predictedPrices = [day]
		for stock in range(1):
			# prediction = 0.0
			# for i in range(5):
			# 	prediction += weights[stock][i] * float(prices[i][stock])
			# prediction = float(prices[4][stock])
			days = np.reshape(days, (len(days),1)) # converting to matrix of n X 1
			
			stockPrices = []
			for i in range(5):
				stockPrices.append(float(prices[i][stock]))
			stockPrices = np.reshape(stockPrices, (len(stockPrices),1))

			linear_mod = linear_model.LinearRegression() # defining the linear regression model
			linear_mod.fit(days, stockPrices) # fitting the data points in the model

			prediction = linear_mod.predict(5)[0][0]
			print prediction
			print actualPrices[stock]
			
			predictedPrices.append(prediction)

			plt.scatter(days, stockPrices, color= 'black', label= 'Data') # plotting the initial datapoints 
			plt.plot(days, linear_mod.predict(days), color= 'red', label= 'Linear model') # plotting the line made by linear regression
			plt.xlabel('Day')
			plt.ylabel('Price')
			plt.title('Linear Regression - %d' %stock)
			plt.legend()
			plt.show()

		output.write(str(predictedPrices[0]))
		for item in predictedPrices[1:]:
			output.write(',' + str(item))
		output.write('\n')
