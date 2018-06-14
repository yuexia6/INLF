'''
timeStamp = 1451631600
timeArray = time.localtime(timeStamp)
otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
print(timeArray)
'''
import pandas as pd
import numpy as np
span=100
conbine=[]
for node in range(1,2):
            apt=1+(node-1)*span
            print('start with:'+str(apt))
            rawdata1=pd.read_csv('level1/2016years_level1/test'+str(apt)+'.csv')
            rawdata1.columns=['time','load']
            for x in range(apt+1, apt+span):
                    if  (x!=54):                       
                        rawdata2=pd.read_csv('level1/2016years_level1/test'+str(x)+'.csv')
                        rawdata2.columns=['time','load'+str(x)]
                        rawdata1 = pd.merge(rawdata1, rawdata2, how='left', on='time')  
                        print('adding:'+str(x))
                    else:
                        continue
            processdata=rawdata1.dropna(axis=0,how='any')
            processdata1=np.array(processdata)
            average=np.sum(processdata1[:,1::],axis=1)
            conbine.append(processdata1[:,0])
            conbine.append(average)
            processdata2=np.array(conbine)
            processdata2=np.transpose(processdata2)
            y=pd.DataFrame(processdata2)
            y.to_csv('level3/2016years_level3/test'+str(node)+'.csv',header=['time','load'],index=False)
            print('formulation for node:'+str(node))
            reader=pd.read_csv('level3/2016years_level3/test'+str(node)+'.csv',header=None)
            reader = reader.drop(0)
            reader.to_csv('level3/2016years_timestamps_level3/node'+str(node)+'.csv',index=False,header=None)
            conbine=[]
            processdata=[]
            processdata1=[]
            processdata2=[]
            y=[]
            reader=[]
