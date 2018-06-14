import pandas as pd
import numpy as np
    
for node in range(1,21):
    weatherdata=pd.read_csv('apartment2016.csv')
    rawdata2=pd.read_csv('2016years_timestamps_level2/test'+str(node)+'.csv')
    weatherdata = pd.merge(weatherdata, rawdata2, how='left', on='Timestamp')  
    processdata=weatherdata.dropna(axis=0,how='any')
    processdata.to_csv('2016years_timestamps_level2/addingweather'+str(node)+'.csv')
    print('finish:'+str(node))

