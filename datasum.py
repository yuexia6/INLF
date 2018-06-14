'''
timeStamp = 1451631600
timeArray = time.localtime(timeStamp)
otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
print(timeArray)
'''
import pandas as pd
import numpy as np
span=30
conbine=[]
for node in range(1,2):
            apt=1+(node-1)*span
            print('start with:'+str(apt))
            rawdata1=pd.read_csv('2016years/test'+str(apt)+'.csv')
            rawdata1.columns=['time','load']
            for x in range(apt+1, apt+span):
                    rawdata2=pd.read_csv('2016years/test'+str(x)+'.csv')
                    rawdata2.columns=['time','load'+str(x)]
                    rawdata1 = pd.merge(rawdata1, rawdata2, how='left', on='time')  
                    print('adding:'+str(x))
          
            processdata=rawdata1.dropna(axis=0,how='any')
            processdata1=np.array(processdata)
            average=np.mean(processdata1[:,1::],axis=1)
            conbine.append(processdata1[:,0])
            conbine.append(average)
            processdata2=np.array(conbine)
            processdata2=np.transpose(processdata2)
            y=pd.DataFrame(processdata2)
            y.to_csv('level2/'+str(node)+'.csv',header=['time','load'],index=False)
            print('formulation for node:'+str(node))
            reader=pd.read_csv('level2/'+str(node)+'.csv')
            reader.drop_duplicates(subset=['time'], keep='first', inplace=True)
            reader.to_csv('level2/'+str(node)+'.csv',index=False)
            conbine=[]
            processdata=[]
            processdata1=[]
            processdata2=[]
            y=[]
            reader=[]