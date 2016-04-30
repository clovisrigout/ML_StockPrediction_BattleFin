import csv
import numpy as np
from sklearn.svm import SVR

# trainDay = 1

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
# actualPrices = []
# with open("trainLabels.csv", 'r') as csvfile:	# get actual prices for each stock for day 1
# 	csvFileReader = csv.reader(csvfile)
# 	for skip in range(trainDay):
# 		next(csvFileReader)
# 	actualPrices = next(csvFileReader)[1:]

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

output1 = open("last_price_svr_rbf.csv", 'a')
output1.write('FileID')

output2 = open("last_price_svr_lin.csv", 'a')
output2.write('FileID')

output3 = open("last_price_svr_poly.csv", 'a')
output3.write('FileID')

for i in range(198):
	output1.write(',O'+str(i+1))
	output2.write(',O'+str(i+1))
	output3.write(',O'+str(i+1))
output1.write('\n')
output2.write('\n')
output3.write('\n')

for day in range(201,511):
	print day
	prices = getPrices(day)
	days = [i for i in range(5)]

	predictedPrices1 = [day]
	predictedPrices2 = [day]
	predictedPrices3 = [day]
	for stock in range(198):
		# prediction = 0.0
		# for i in range(5):
		# 	prediction += weights[stock][i] * float(prices[i][stock])
		# prediction = float(prices[4][stock])
		days = np.reshape(days, (len(days),1)) # converting to matrix of n X 1
		
		stockPrices = []
		for i in range(5):
			if(float(prices[i][stock]) < 40.0):
				stockPrices.append(float(prices[i][stock]))
			else: stockPrices.append(float(prices[i][stock])*1.0/100)
		#stockPrices = np.reshape(stockPrices, (len(stockPrices),1))

		svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
		svr_lin = SVR(kernel= 'linear', C= 1e3)
		svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)

		svr_rbf.fit(days, stockPrices) # fitting the data points in the models
		svr_lin.fit(days, stockPrices)
		svr_poly.fit(days, stockPrices)
		
		prediction = svr_rbf.predict(5)[0], svr_lin.predict(5)[0], svr_poly.predict(5)[0]
		
		predictedPrices1.append(prediction[0])
		predictedPrices2.append(prediction[1])
		predictedPrices3.append(prediction[2])

	output1.write(str(predictedPrices1[0]))
	for item in predictedPrices1[1:]:
		output1.write(',' + str(item))
	output1.write('\n')

	output2.write(str(predictedPrices1[0]))
	for item in predictedPrices1[1:]:
		output2.write(',' + str(item))
	output2.write('\n')

	output3.write(str(predictedPrices1[0]))
	for item in predictedPrices1[1:]:
		output3.write(',' + str(item))
	output3.write('\n')

output1.close()
output2.close()
output3.close()