import time
import csv
import pandas as pd
data=[]
load=[]
close=[]
times=[]
month=[]
months=[]
day=[]
days=[]
#os.mkdir('2016years')

for apt in range(1,2):
                    
                   
                    print(apt)
                    with open('level3/2016years_timestamps_level3/node'+str(apt)+'.csv','r') as csvfile:
                        reader = csv.reader(csvfile)
                        loadd = [row[1] for row in reader]
                        for items in loadd:
                            if reader.line_num == 0:  
                                        continue  
                            load.append(items)

                    with open('level3/2016years_timestamps_level3/node'+str(apt)+'.csv','r') as csvfile:
                        reader = csv.reader(csvfile)  
                        timee = [row[0] for row in reader]
                        for items in timee:
                              a=items
                              timeArray = time.strptime(a, "%Y-%m-%d %H:%M:%S") 
                              timeStamp = int(time.mktime(timeArray))
                              data.append(timeStamp) 
                 
                   
                   
               
                    with open('level3/2016years_timestamps_level3/nodee'+str(apt)+'.csv','w') as file:
                        file.write('Close,Timestamp\n')
                        for n,p in zip(load,data):
                            file.write("{},{}\n".format(n,p))
                    data=[]
                    load=[]
                    close=[]
                    times=[]
                    month=[]
                    months=[]
                    day=[]
                    days=[]
                    