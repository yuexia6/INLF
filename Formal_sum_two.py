import pandas as pd
import numpy as np
conbine=[]
node1=2
node2=3
print('start with:'+str(node1))
rawdata1=pd.read_csv('level2/2016years_level2/test'+str(node1)+'.csv')
rawdata1.columns=['time','load']
rawdata2=pd.read_csv('level1/2016years_level1/test'+str(node2)+'.csv')
rawdata2.columns=['time','load'+str(node2)]
rawdata1 = pd.merge(rawdata1, rawdata2, how='left', on='time')  
print('adding:'+str(node2))
       
processdata=rawdata1.dropna(axis=0,how='any')
processdata1=np.array(processdata)
average=np.mean(processdata1[:,1::],axis=1)
conbine.append(processdata1[:,0])
conbine.append(average)
processdata2=np.array(conbine)
processdata2=np.transpose(processdata2)
y=pd.DataFrame(processdata2)
setnode=7
y.to_csv('level3/2016years_level3/test'+str(setnode)+'.csv',header=['time','load'],index=False)
print('formulation for node:'+str(setnode))
reader=pd.read_csv('level3/2016years_level3/test'+str(setnode)+'.csv',header=None)
reader = reader.drop(0)
reader.to_csv('level3/2016years_timestamps_level3/node'+str(setnode)+'.csv',index=False,header=None)
          