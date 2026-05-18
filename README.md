# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 18-05-2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load data
data = pd.read_csv('market.csv')

# Use Close column
X = data['Close']

# Make stationary
X = X.diff().dropna()

N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

# Plot original data
plt.plot(X)
plt.title('Stationary Data')
plt.show()

# ACF and PACF
plt.subplot(2,1,1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('ACF')

plt.subplot(2,1,2)
plot_pacf(X, lags=40, ax=plt.gca())
plt.title('PACF')

plt.tight_layout()
plt.show()

# ARMA(1,1)
arma11_model = ARIMA(X, order=(1,0,1)).fit()

phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1)
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1)')
plt.show()

plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()

# ARMA(2,2)
arma22_model = ARIMA(X, order=(2,0,2)).fit()

phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']

theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2)
ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2)')
plt.show()

plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()
```
### OUTPUT:
### DATA:
<img width="989" height="505" alt="image" src="https://github.com/user-attachments/assets/8f3e12a9-6da9-4127-aba5-55a1e047b9dc" />

### ACF:

<img width="997" height="244" alt="image" src="https://github.com/user-attachments/assets/74173bec-ac97-401a-81ca-5b2649c57081" />
### PACF:

<img width="1002" height="245" alt="image" src="https://github.com/user-attachments/assets/d017ef85-857b-44b8-8f70-8a745fb219e7" />


#### SIMULATED ARMA(1,1) PROCESS:

<img width="988" height="519" alt="image" src="https://github.com/user-attachments/assets/34b760b2-a98a-471d-966a-6a27ba7051bc" />


#### Partial Autocorrelation

<img width="1004" height="516" alt="image" src="https://github.com/user-attachments/assets/1126edcd-90f4-4303-8d0b-468a5689a6fa" />

#### Autocorrelation


<img width="983" height="512" alt="image" src="https://github.com/user-attachments/assets/d2cf523b-c5fd-4789-a7f4-57df5b9682ca" />


#### SIMULATED ARMA(2,2) PROCESS:

<img width="976" height="524" alt="image" src="https://github.com/user-attachments/assets/58f2c92e-3128-4f00-b92c-934edfcedc3e" />

#### Partial Autocorrelation

<img width="983" height="529" alt="image" src="https://github.com/user-attachments/assets/723642eb-acd9-4e1e-818c-8a5fa8968d55" />



#### Autocorrelation

<img width="997" height="526" alt="image" src="https://github.com/user-attachments/assets/e8f13996-543b-4656-8046-6f4158fc5405" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
