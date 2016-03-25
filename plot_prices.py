import csv
import numpy as np
import matplotlib.pyplot as plt

n = 2	# the specific stock we want

# Fetch prices vector for the stock for a particular day
prices = []
with open("data/1.csv", 'r') as csvfile:
	csvFileReader = csv.reader(csvfile)
	next(csvFileReader) # skipping column names
	for row in csvFileReader:
		prices.append(float(row[n-1]))

# Fetch actual price of stock 2 hours later
with open("trainLabels.csv", 'r') as csvfile:
	csvFileReader = csv.reader(csvfile)
  	next(csvFileReader) # skipping column names
  	actual = next(csvFileReader)[n]

time = [i for i in range(1,len(prices)+1)]

print prices, actual

plt.scatter(time, prices)
plt.xlabel('Time Interval')
plt.ylabel('Stock Price')
plt.show()
