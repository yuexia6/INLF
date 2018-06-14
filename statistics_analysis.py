import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
rcParams['figure.figsize'] = 15, 6
data = pd.read_csv('AirPassengers.csv')
print (data.head())
print ('\n Data Types:')
print (data.dtypes)

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
data = pd.read_csv('test18.csv', parse_dates=['time'], index_col=['time'],date_parser=dateparse)
print (data.head())
ts = data
ts_log = np.log(ts)
#plt.plot(ts_log)
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)


#plt.plot(ts_log_diff)
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
'''
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
'''
model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)
model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
predictions_ARIMA_diff = pd.Series(results_MA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)
ts2=pd.DataFrame(predictions_ARIMA)
ts2.rename(columns={ ts2.columns[0]: "load" }, inplace=True)
ts1=pd.read_csv('test18.csv')
ts1=ts1.drop([0])
ts1=ts1.to_csv('medium.csv',index=None)
ts1=pd.read_csv('medium.csv', parse_dates=['time'], index_col=['time'],date_parser=dateparse)
ts2=ts2.dropna(axis=0,how='any')
mse=mean_squared_error(ts1, ts2)