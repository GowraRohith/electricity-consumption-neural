[Home](https://mgcodesandstats.github.io/) |
[Portfolio](https://mgcodesandstats.github.io/portfolio/) |
[Terms and Conditions](https://mgcodesandstats.github.io/terms/) |
[E-mail me](mailto:contact@michaeljgrogan.com) |
[LinkedIn](https://www.linkedin.com/in/michaeljgrogan/)

# Predicting Electricity Consumption Using LSTM Neural Networks

In this example, an LSTM neural network is used to forecast energy consumption of the Dublin City Council Civic Offices using data between April 2011 – February 2013. The original dataset is available from [data.gov.ie](https://data.gov.ie/dataset/energy-consumption-gas-and-electricity-civic-offices-2009-2012/resource/6091c604-8c94-4b44-ac52-c1694e83d746).

## Data Processing

Given the data is presented in 15 minute intervals, daily data was created by summing up the consumption for each day across the 15 minute intervals contained in the original dataset, along with the dataset matrix being formed to allow for a specified number of lags to be regressed against electricity consumption at time *t*. NA values were dropped from the dataset, along with irrelevant columns.

Here is a sample of the original data:

![sample-data](sample-data.png)

First, the relevant dataset matrix is formed for processing with the LSTM model:

*Dataset Matrix*

```
# Form dataset matrix
def create_dataset(dataset, previous=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-previous-1):
		a = dataset[i:(i+previous), 0]
		dataX.append(a)
		dataY.append(dataset[i + previous, 0])
	return np.array(dataX), np.array(dataY)
```
Then, the data in its original format is cleaned accordingly:

*Data Cleaning*

```
# fix random seed for reproducibility
np.random.seed(7)

# load dataset
df = read_csv('dccelectricitycivicsblocks34p20130221-1840.csv', engine='python', skipfooter=3)
df2=df.rename(columns=df.iloc[0])
df3=df2.drop(df.index[0])
df3
df3.drop(df3.index[0])
df4=df3.drop('Date', axis=1)
df5=df4.drop('Values', axis=1)
df5
df6=df5.dropna()
df7=df6.values
df7
dataset=np.sum(df7, axis=1, dtype=float)
dataset
```
Now, here is a modified numpy array showing the total electricity consumption for each day (with NAs dropped from the model):

![array](array.png)

## Introduction to LSTM

LSTMs (or long-short term memory networks) allow for analysis of **sequential** or ordered data with long-term dependencies present. Traditional neural networks fall short when it comes to this task, and in this regard an LSTM will be used to predict electricity consumption patterns in this instance. One particular advantage of LSTMs compared to models such as ARIMA, is that the data does not necessarily need to be stationary (constant mean, variance, and autocorrelation), in order for LSTM to analyse the same - even if doing so might result in an increase in performance.

## Autocorrelation Plots, Dickey-Fuller test and Log-Transformation

Firstly, here is a plot of the time series:

![lstm kilowatts consumed per day](kilowatts-consumed-per-day.png)

It is observed that the volatility (or change in consumption from one day to the next) is quite high. In this regard, a logarithmic transformation could be of use in attempting to smooth this data somewhat.

```
from numpy import log
dataset = log(dataset)
```

While the time series remains volatile, the size of the deviations have decreased slightly when expressed in logarithmic format:

![kilowatts consumed per day logarithmic format](kilowatts-consumed-per-day-logarithmic-format-1.png)

The coefficient of variation (standard deviation/mean) has decreased to 0.031 from a prior 0.232, implying that the variability of the trend in relation to the mean is significantly lower than previously.

ACF and PACF plots are generated on the logarithmic data, and a Dickey-Fuller test is conducted. 

**Autocorrelation Plot** 

![autocorrelation with log](autocorrelation-with-log.png) 

**Partial Autocorrelation Plot** 

![partial autocorrelation function log](partial-autocorrelation-function-log.png) 

**Dickey-Fuller Test**

```
>>> # Dickey-Fuller Test
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

The p-value for the Dickey-Fuller test has decreased to 0.0576. While this technically does not enter the 5% level of significance threshold necessary to reject the null hypothesis, the logarithmic time series has shown lower volatility based on the CV metric, and therefore this time series is used for forecasting purposes with LSTM.

## Time Series Analysis with LSTM

Now, the LSTM model itself is used for forecasting purposes.

### Data Processing

Firstly, the relevant libraries are imported and data processing is carried out:

```
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import pylab
import tensorflow as tf
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import statsmodels
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
```
The data is now normalised for analysis with the LSTM model, and the time series of *t* is regressed against the series at *t-50* (a 50 day lag was chosen after experimentation with numerous time intervals):

```
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# normalize dataset with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Training and Test data partition
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t-50 and Y=t
previous = 50
X_train, Y_train = create_dataset(train, previous)
X_test, Y_test = create_dataset(test, previous)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
```

### LSTM Generation and Predictions

The model is trained over **150** epochs, and the predictions are generated.

```
model = tf.keras.Sequential()
model.add(LSTM(4, input_shape=(1, previous)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=150, batch_size=1, verbose=2)
```

### Accuracy

Here is the output when 150 epochs are generated:

```
Epoch 145/150
493/493 - 2s - loss: 0.0058
Epoch 146/150
493/493 - 2s - loss: 0.0061
Epoch 147/150
493/493 - 2s - loss: 0.0059
Epoch 148/150
493/493 - 2s - loss: 0.0062
Epoch 149/150
493/493 - 2s - loss: 0.0058
Epoch 150/150
493/493 - 2s - loss: 0.0056
```

The model shows a root mean squared error of **0.07** on the training dataset, and **0.13** on the test dataset. 

```
>>> # Generate predictions
>>> trainpred = model.predict(X_train)
>>> testpred = model.predict(X_test)

>>> # Convert predictions back to normal values
... trainpred = scaler.inverse_transform(trainpred)
>>> Y_train = scaler.inverse_transform([Y_train])
>>> testpred = scaler.inverse_transform(testpred)
>>> Y_test = scaler.inverse_transform([Y_test])

>>> # calculate RMSE
... trainScore = math.sqrt(mean_squared_error(Y_train[0], trainpred[:,0]))
>>> print('Train Score: %.2f RMSE' % (trainScore))
Train Score: 0.07 RMSE
>>> testScore = math.sqrt(mean_squared_error(Y_test[0], testpred[:,0]))
>>> print('Test Score: %.2f RMSE' % (testScore))
Test Score: 0.13 RMSE
```

**50 days** 

![50 days](over-50-days.png)

Upon transforming the predictions back to the original format through calculating the exponent, we are now in a position to calculate the percentage error between the predicted and actual consumption.

```
>>> percentage_error=((predictions-Y_test)/Y_test)
>>> percentage_error=abs(percentage_error)
>>> mean=np.mean(percentage_error)
>>> mean
0.0850737316467645
>>> percentage_error=pd.DataFrame(percentage_error)
>>> below10=percentage_error[percentage_error < 0.10].count()
>>> all=percentage_error.count()
>>> np.sum(below10)
63
>>> np.sum(all)
85
```

**63** of the **85** predictions showed a deviation of less than 10%. Moreover, the mean percentage error was 8.5%, indicating that the model did quite a good job at forecasting electricity consumption.

## Comparison with ARIMA

In this example, an LSTM (which is a reasonably complex machine learning model) was employed for forecasting purposes.

Is LSTM increasing the forecast accuracy significantly so as to justify using this model over a simpler one? To determine this, an ARIMA model was used to generate forecasts on the test set for comparison purposes.

Given that the autocorrelation function appears to be showing weekly seasonality (i.e. a spike in ACF is observed every seven lags), the seasonal parameter **m** was set to 7, and a SARIMAX(3, 1, 3)x(0, 1, 2, 7) was identified as the best configuration by *auto_arima* from the pyramid library.

```
 Statespace Model Results Dep. Variable: 	y 	No. Observations: 	593
Model: 	SARIMAX(3, 1, 3)x(0, 1, 2, 7) 	Log Likelihood 	394.515
Date: 	Sat, 24 Aug 2019 	AIC 	-769.030
Time: 	16:36:19 	BIC 	-725.313
Sample: 	0 	HQIC 	-751.993
	- 593 		
Covariance Type: 	opg 		
	coef 	std err 	z 	P>|z| 	[0.025 	0.975]
intercept 	-1.72e-05 	0.000 	-0.080 	0.936 	-0.000 	0.000
ar.L1 	-1.2755 	0.046 	-27.595 	0.000 	-1.366 	-1.185
ar.L2 	-0.4351 	0.049 	-8.919 	0.000 	-0.531 	-0.339
ar.L3 	0.0928 	0.033 	2.833 	0.005 	0.029 	0.157
ma.L1 	0.7089 	0.036 	19.840 	0.000 	0.639 	0.779
ma.L2 	-0.7842 	0.029 	-27.179 	0.000 	-0.841 	-0.728
ma.L3 	-0.8883 	0.038 	-23.548 	0.000 	-0.962 	-0.814
ma.S.L7 	-0.1809 	0.035 	-5.119 	0.000 	-0.250 	-0.112
ma.S.L14 	-0.1345 	0.049 	-2.745 	0.006 	-0.231 	-0.038
sigma2 	0.0148 	0.001 	24.751 	0.000 	0.014 	0.016
Ljung-Box (Q): 	67.61 	Jarque-Bera (JB): 	884.78
Prob(Q): 	0.00 	Prob(JB): 	0.00
Heteroskedasticity (H): 	0.61 	Skew: 	-0.09
Prob(H) (two-sided): 	0.00 	Kurtosis: 	9.02


Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
```

# Conclusion

For this example, LSTM proved to be quite accurate at predicting fluctuations in electricity consumption. Moreover, expressing the time series in logarithmic format allowed for a smoothing of the volatility in the data and improved the prediction accuracy of the LSTM.
