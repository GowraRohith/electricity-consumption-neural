# Electricity Consumption Using Neural Networks

In this example, neural networks are used to forecast energy consumption of the Dublin City Council Civic Offices using data between April 2011 – February 2013. The original dataset is available from [data.gov.ie](https://data.gov.ie/dataset/energy-consumption-gas-and-electricity-civic-offices-2009-2012/resource/6091c604-8c94-4b44-ac52-c1694e83d746), and daily data was created by summing up the consumption for each day across the 15 minute intervals provided.

## Introduction to LSTM

LSTMs (or long-short term memory networks) allow for analysis of **sequential** or ordered data with long-term dependencies present. Traditional neural networks fall short when it comes to this task, and in this regard an LSTM will be used to predict electricity consumption patterns in this instance. One particular advantage of LSTMs compared to models such as ARIMA, is that the data does not necessarily need to be stationary (constant mean, variance, and autocorrelation), in order for LSTM to analyse the same - even if doing so might result in an increase in performance.

## Autocorrelation Plots, Dickey-Fuller test and Log-Transformation

In order to determine whether **stationarity** is present in our model:

1.  Autocorrelation and partial autocorrelation plots are generated
2.  A Dickey-Fuller test is conducted
3.  The time series is log-transformed and the above two procedures are run once again in order to determine the change (if any) in stationarity

Firstly, here is a plot of the time series: 
[![lstm kilowatts consumed per day](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/kilowatts-consumed-per-day.png)](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/kilowatts-consumed-per-day.png)

It is observed that the volatility (or change in consumption from one day to the next) is quite high. In this regard, a logarithmic transformation could be of use in attempting to smooth this data somewhat. Before doing so, the ACF and PACF plots are generated, and a Dickey-Fuller test is conducted. 

**Autocorrelation Plot** 

[![autocorrelation without log](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/autocorrelation-without-log.png)](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/autocorrelation-without-log.png)

**Partial Autocorrelation Plot**

[![partial autocorrelation function](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/partial-autocorrelation-function.png)](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/partial-autocorrelation-function.png)

Both the autocorrelation and partial autocorrelation plots exhibit significant volatility, implying that correlations exist across several intervals in the time series. When a Dickey-Fuller test is run, the following results are yielded:

**Input [1]:**

```
result = adfuller(data1)
print('ADF Statistic: %f' % result[0])
ADF Statistic: -2.703927
print('p-value: %f' % result[1])
p-value: 0.073361
print('Critical Values:')
Critical Values:
for key, value in result[4].items():
     print('\t%s: %.3f' % (key, value))
```

**Output [1]:**

```
Output
	1%: -3.440
	5%: -2.866
	10%: -2.569
```

With a p-value above 0.05, the null hypothesis of non-stationarity cannot be rejected.

```>>> std1=np.std(dataset)
>>> mean1=np.mean(dataset)
>>> cv1=std1/mean1 #Coefficient of Variation
>>> std1
954.7248
>>> mean1
4043.4302
>>> cv1
0.23611754
```

The coefficient of variation (or mean divided by standard deviation) is 0.236, demonstrating significant volatility in the series. Now, the data is transformed into logarithmic format.

```from numpy import log
dataset = log(dataset)
```

While the time series remains volatile, the size of the deviations have decreased slightly when expressed in logarithmic format: 

[![kilowatts consumed per day logarithmic format](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/kilowatts-consumed-per-day-logarithmic-format-1.png)](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/kilowatts-consumed-per-day-logarithmic-format-1.png)

Moreover, the coefficient of variation has decreased significantly to 0.0319, implying that the variability of the trend in relation to the mean is significantly lower than previously.

```>>> std2=np.std(dataset)
>>> mean2=np.mean(dataset)
>>> cv2=std2/mean2 #Coefficient of Variation
>>> std2
0.26462445
>>> mean2
8.272395
>>> cv2
0.031988855
```

Again, ACF and PACF plots are generated on the logarithmic data, and a Dickey-Fuller test is conducted once again. 

**Autocorrelation Plot** 

[![autocorrelation with log](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/autocorrelation-with-log.png)](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/autocorrelation-with-log.png) 

**Partial Autocorrelation Plot** 

[![partial autocorrelation function log](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/partial-autocorrelation-function-log.png)](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/partial-autocorrelation-function-log.png) 

**Dickey-Fuller Test**

```>>> # Dickey-Fuller Test
... result = adfuller(logdataset)
>>> print('ADF Statistic: %f' % result[0])
ADF Statistic: -2.804265
>>> print('p-value: %f' % result[1])
p-value: 0.057667
>>> print('Critical Values:')
Critical Values:
>>> for key, value in result[4].items():
...     print('\t%s: %.3f' % (key, value))
... 
	1%: -3.440
	5%: -2.866
	10%: -2.569
```

The p-value for the Dickey-Fuller test has decreased to 0.0576\. While this technically does not enter the 5% level of significance threshold necessary to reject the null hypothesis, the logarithmic time series has shown lower volatility based on the CV metric, and therefore this time series is used for forecasting purposes with LSTM.

## Time Series Analysis with LSTM

Now, the LSTM model itself is used for forecasting purposes.

### Data Processing

Firstly, the relevant libraries are imported and data processing is carried out:

```import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import pylab
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os;
path="filepath"
os.chdir(path)
os.getcwd()

# Form dataset matrix
def create_dataset(dataset, previous=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-previous-1):
		a = dataset[i:(i+previous), 0]
		dataX.append(a)
		dataY.append(dataset[i + previous, 0])
	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(7)

# load dataset
dataframe = read_csv('data.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

from numpy import log
dataset = log(dataset)

meankwh=np.mean(dataset)

# normalize dataset with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Training and Test data partition
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t-1 and Y=t
previous = 1
X_train, Y_train = create_dataset(train, previous)
X_test, Y_test = create_dataset(test, previous)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
```

### LSTM Generation and Predictions

The model is trained over **100** epochs, and the predictions are generated.

```# Generate LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, previous)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

# Generate predictions
trainpred = model.predict(X_train)
testpred = model.predict(X_test)

# Convert predictions back to normal values
trainpred = scaler.inverse_transform(trainpred)
Y_train = scaler.inverse_transform([Y_train])
testpred = scaler.inverse_transform(testpred)
Y_test = scaler.inverse_transform([Y_test])

# calculate RMSE
trainScore = math.sqrt(mean_squared_error(Y_train[0], trainpred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Y_test[0], testpred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Train predictions
trainpredPlot = np.empty_like(dataset)
trainpredPlot[:, :] = np.nan
trainpredPlot[previous:len(trainpred)+previous, :] = trainpred

# Test predictions
testpredPlot = np.empty_like(dataset)
testpredPlot[:, :] = np.nan
testpredPlot[len(trainpred)+(previous*2)+1:len(dataset)-1, :] = testpred

# Plot all predictions
inversetransform, =plt.plot(scaler.inverse_transform(dataset))
trainpred, =plt.plot(trainpredPlot)
testpred, =plt.plot(testpredPlot)
plt.title("Predicted vs. Actual Consumption")
plt.show()
```

### Accuracy

Here is the output when 100 epochs are generated:

```Epoch 94/100
 - 2s - loss: 0.0406
Epoch 95/100
 - 2s - loss: 0.0406
Epoch 96/100
 - 2s - loss: 0.0404
Epoch 97/100
 - 2s - loss: 0.0406
Epoch 98/100
 - 2s - loss: 0.0406
Epoch 99/100
 - 2s - loss: 0.0403
Epoch 100/100
 - 2s - loss: 0.0406

>>> # Generate predictions
... trainpred = model.predict(X_train)
>>> testpred = model.predict(X_test)
>>> 
>>> # Convert predictions back to normal values
... trainpred = scaler.inverse_transform(trainpred)
>>> Y_train = scaler.inverse_transform([Y_train])
>>> testpred = scaler.inverse_transform(testpred)
>>> Y_test = scaler.inverse_transform([Y_test])
>>> 
>>> # calculate RMSE
... trainScore = math.sqrt(mean_squared_error(Y_train[0], trainpred[:,0]))
>>> print('Train Score: %.2f RMSE' % (trainScore))
Train Score: 0.24 RMSE
>>> testScore = math.sqrt(mean_squared_error(Y_test[0], testpred[:,0]))
>>> print('Test Score: %.2f RMSE' % (testScore))
Test Score: 0.23 RMSE
```

The model shows a root mean squared error of **0.24** on the training dataset, and **0.23** on the test dataset. The mean kilowatt consumption (expressed in logarithmic format) is **8.27**, which means that the error of 0.23 represents less than 3% of the mean consumption. Here is the plot of predicted versus actual consumption: 

[![predicted vs actual consumption 1 day](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/predicted-vs-actual-consumption-1-day.png)](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/predicted-vs-actual-consumption-1-day.png) 

Interestingly, when the predictions are generated on the raw data (not converted into logarithmic format), the following training and test errors are yielded:

```>>> # calculate RMSE
... trainScore = math.sqrt(mean_squared_error(Y_train[0], trainpred[:,0]))
>>> print('Train Score: %.2f RMSE' % (trainScore))
Train Score: 840.95 RMSE
>>> testScore = math.sqrt(mean_squared_error(Y_test[0], testpred[:,0]))
>>> print('Test Score: %.2f RMSE' % (testScore))
Test Score: 802.62 RMSE
```

In the context of a mean consumption of 4043 kilowatts per day, the mean squared error for the test score represents nearly 20% of the total mean daily consumption, and is quite high in comparison to that generated on the logarithmic data. That said, it is important to bear in mind that the prediction was made using 1-day of previous data, i.e. Y represents consumption at time t, while X represents consumption at time t-1, as set by the **previous** variable in the code previously. Let's see what happens if this is increased to **10** and **50** days. 

**10 days** 

[![10 days](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/over-10-days.png)](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/over-10-days.png)

```>>> # calculate RMSE
... trainScore = math.sqrt(mean_squared_error(Y_train[0], trainpred[:,0]))
>>> print('Train Score: %.2f RMSE' % (trainScore))
Train Score: 0.08 RMSE
>>> testScore = math.sqrt(mean_squared_error(Y_test[0], testpred[:,0]))
>>> print('Test Score: %.2f RMSE' % (testScore))
Test Score: 0.10 RMSE
```

**50 days** 

[![50 days](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/over-50-days.png)](http://www.michaeljgrogan.com/wp-content/uploads/2018/12/over-50-days.png)

```>>> print('Train Score: %.2f RMSE' % (trainScore))
Train Score: 0.07 RMSE
>>> testScore = math.sqrt(mean_squared_error(Y_test[0], testpred[:,0]))
>>> print('Test Score: %.2f RMSE' % (testScore))
Test Score: 0.10 RMSE
```

We can see that the test error was significantly lower over the 10 and 50-day periods, and the volatility in consumption was much better captured given that the LSTM model took more historical data into account when forecasting. Given the data is in logarithmic format, it is now possible to obtain the true values of the predictions by obtaining the exponent of the data. For instance, the **testpred** variable is reshaped with (1, -1):

```>>> testpred.reshape(1,-1)
array([[7.7722197, 8.277015 , 8.458941 , 8.455311 , 8.447589 , 8.445035, 
 ......
8.425287 , 8.404881 , 8.457063 , 8.423954 , 7.98714 , 7.9003944,
8.240862 , 8.41654 , 8.423854 , 8.437414 , 8.397851 , 7.9047146]],
dtype=float32)```

Using numpy, the exponent is then calculated:

```>>> np.exp(testpred)
array([[2373.7344],
       [3932.4375],
       [4717.062 ],
......
       [4616.6016],
       [4437.52  ],
       [2710.0288]], dtype=float32)
```

# Conclusion

For this example, LSTM proved to be quite accurate at predicting fluctuations in electricity consumption. Moreover, expressing the time series in logarithmic format allowed for a smoothing of the volatility in the data and improved the prediction accuracy of the LSTM.
