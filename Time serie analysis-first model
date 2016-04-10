import csv
import numpy as np
import matplotlib.pyplot as plt



def getData(n,d):
#Fetch prices vector for nth stock the day d    
    prices=[]
    with open("%d.csv" %d, 'r')as csvfile:
        csvFileReader=csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            prices.append(float(row[n-1]))
#Fetch actual price of stock 2 hours later on that day
    with open("trainLabels.csv",'r') as csvfile:
        csvFileReader=csv.reader(csvfile)
        for skip in range(d):
            next(csvFileReader)
        actual=float(next(csvFileReader)[n])

    return prices,actual

#Same function but without the actual price : used for the forecasting day
def getData2(n,d):
#Fetch prices vector for nth stock the day d    
    prices=[]
    with open("%d.csv" %d, 'r')as csvfile:
        csvFileReader=csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            prices.append(float(row[n-1]))

    return prices

#Using time serie analysis to forecast new data using Single Exponential Smoothing
#We change the value of each stock price in order to approach a time serie;
def forecasting(prices,alpha):
    forecasted=[1]*(len(prices)-1)
    forecasted[0]=prices[1]
    for i in range(1,len(prices)-1):
        forecasted[i]=alpha*prices[i+1]+(1-alpha)*forecasted[i-1]

    return forecasted


#Predicting the next 2 hours from the last given price
#We predict each stock price using the precedent stock price by using a fixed origin
def bootstraping(origin,alpha,lastForecasted):
    bootstrapped=[1]*24
    bootstrapped[0]=alpha*origin+(1-alpha)*lastForecasted
    for i in range(1,24):
        bootstrapped[i]=alpha*origin+(1-alpha)*bootstrapped[i-1]

    return bootstrapped


#Algorithm for finding the best alpha
#We learn the best value of alpha by comparing their different score on the training labels
def findAlpha(prices,actualValue):
    historicalScore=[1]*100
    historicalAlpha=[1]*100
    alpha=0.0
    predictedValue=bootstraping(prices[54],alpha,forecasting(prices,alpha)[53])[23]
    deltaAlpha=0.01
    i=0
    while alpha<=0.95:
        alpha+=deltaAlpha
        predictedValue=bootstraping(prices[54],alpha,forecasting(prices,alpha)[53])[23]
        historicalScore[i]=abs(predictedValue-actualValue)
        historicalAlpha[i]=alpha
        i+=1
        
    indexBestScore =historicalScore.index(min(historicalScore))
    alpha=historicalAlpha[indexBestScore]
           
    return alpha
#Learning the best alpha for nth stock 

def bestAlpha(n):
    alphaTable=[1]*201
    scoreTable=[1]*201
    for i in range(1,201):
        prices,actual=getData(n,i)
        alpha=findAlpha(prices,actual)
        alphaTable[i]=alpha
        scoreTable[i]=abs(bootstraping(prices[54],alpha,forecasting(prices,alpha)[53])[23]-actual)
    indexBestAlpha=scoreTable.index(min(scoreTable))
    return alphaTable[indexBestAlpha]



# Best alpha for the 198 stocks
bestAlphaStock=[1]*198
for i in range(198):
    bestAlphaStock[i]=bestAlpha(i)
#print(bestAlphaStock)
#table=range(198)
#plt.plot(table,bestAlphaStock)
#plt.show()
#The best alphas are stocked here used it instead of executing the code again because it take a lot of time
bestAlpha=[0.01, 0.01, 0.01, 0.7400000000000004, 0.7500000000000004, 0.7400000000000004, 0.01, 0.7400000000000004, 0.01, 0.01, 0.7500000000000004, 0.01, 0.7500000000000004, 0.01, 0.01, 0.7600000000000005, 0.7600000000000005, 0.7500000000000004, 0.01, 0.01, 0.01, 0.7600000000000005, 0.7400000000000004, 0.01, 0.01, 0.7400000000000004, 0.01, 0.01, 0.7500000000000004, 0.7300000000000004, 0.01, 0.01, 0.7100000000000004, 0.9400000000000006, 0.01, 0.7500000000000004, 0.01, 0.01, 0.7600000000000005, 0.7500000000000004, 0.7600000000000005, 0.7600000000000005, 0.7200000000000004, 0.7300000000000004, 0.01, 0.01, 0.01, 0.01, 0.7500000000000004, 0.7600000000000005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.7400000000000004, 0.7700000000000005, 0.01, 0.7500000000000004, 0.7200000000000004, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.7700000000000005, 0.8000000000000005, 0.01, 0.7600000000000005, 0.01, 0.01, 0.7600000000000005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.7400000000000004, 0.01, 0.7700000000000005, 0.7600000000000005, 0.01, 0.7900000000000005, 0.7400000000000004, 0.01, 0.7700000000000005, 0.7600000000000005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.7600000000000005, 0.01, 0.01, 0.7400000000000004, 0.7600000000000005, 0.01, 0.7600000000000005, 0.7300000000000004, 0.01, 0.01, 0.01, 0.7800000000000005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.7700000000000005, 0.01, 0.01, 0.01, 0.7100000000000004, 0.01, 0.7600000000000005, 0.7500000000000004, 0.01, 0.7700000000000005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.7600000000000005, 0.7600000000000005, 0.7700000000000005, 0.01, 0.7300000000000004, 0.01, 0.7500000000000004, 0.01, 0.01, 0.7700000000000005, 0.01, 0.7500000000000004, 0.01, 0.7500000000000004, 0.7700000000000005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.7300000000000004, 0.01, 0.7500000000000004, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.7600000000000005, 0.01, 0.01, 0.01, 0.7100000000000004, 0.7000000000000004, 0.01, 0.01, 0.01, 0.7700000000000005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.7500000000000004, 0.01, 0.01, 0.7300000000000004, 0.01, 0.01, 0.01]



#Predicting the nex 311 days stocks price
for day in range(201,511):
    predictedPrices=[day]
    for stock in range(198):
        prices=getData2(stock,day)
        prediction=bootstraping(prices[54],bestAlpha[stock],forecasting(prices,bestAlpha[stock])[53])[23]
        predictedPrices.append(prediction)
    with open("submission.csv",'a') as output:
        output.write(str(predictedPrices[0]))
        for item in predictedPrices[1:]:
            output.write(','+str(item))
        output.write('\n')
        
        



