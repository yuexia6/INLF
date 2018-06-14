import pandas as pd
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
pi2 = cmath.pi * 2.0
def DFT(fnList):
    N = len(fnList)
    FmList = []
    for m in range(N):
        Fm = 0.0
        for n in range(N):
            Fm += fnList[n] * cmath.exp(- 1j * pi2 * m * n / N)
        FmList.append(Fm / N)
        print(m)
    return FmList

def inverseDft(df2):
    FmList=np.array(df2)   
    N = len(FmList)
    fnList = []
    for n in range(N):
        fn = 0.0
        for m in range(N):
            fn += FmList[m] * cmath.exp(1j * pi2 * m * n / N)
        fnList.append(fn)
        print(n)
    return fnList 

w=[]
index =[]
wlow=[]
reader=pd.read_csv('level2/2016years_level2/test1.csv')
load_number=5
df = pd.DataFrame(reader)
df['time'] = pd.to_datetime(df['time'])
df_show1 = df.loc[(df["time"].dt.month ==10 )&(df["time"].dt.day >=15 ),:]
sample=df_show1.load
sample=sample/load_number

FT=DFT(np.array(sample))
df11= np.zeros(len(FT))
df22= np.zeros(len(FT))
df33= np.zeros(len(FT))
df44= np.zeros(len(FT))
df1= pd.DataFrame(FT)
df2= pd.DataFrame(FT)
df3= pd.DataFrame(FT)
df4= pd.DataFrame(FT)
df2.to_csv('DFT1.csv')
wday=[0]
wweek=[]
wlow=[]
whigh=[]
T=1440


for i in range(1,len(sample)):
    if T%(len(sample)/i)==0:
        wday.append(i) 
    if ((T*7)%(len(sample)/i)==0):
        wweek.append(i)
    if ((T*7)%(len(sample)/i)!=0)&(T%(len(sample)/i)!=0)&((len(sample)/i)<=T):
        wlow.append(i)  
    if ((T*7)%(len(sample)/i)!=0)&(T%(len(sample)/i)!=0)&((len(sample)/i)>T):
        whigh.append(i)

for i in range(len(wday)):
        df11[ wday[i] ]=df1.loc[ wday[i] ]
for i in range(len(wweek)):
        df22[ wweek[i] ]=df2.loc[ wweek[i] ]
for i in range(len(wlow)):
        df33[ wlow[i] ]=df3.loc[ wlow[i] ]
for i in range(len(whigh)):
        df44[ whigh[i] ]=df4.loc[ whigh[i] ]

plt.subplot(2,2,1)
plt.plot(inverseDft(df11),label = 'day')
plt.legend(loc='upper left')
plt.xlabel('samples')
plt.ylabel('Load Distribution')
plt.subplot(2,2,2)
plt.plot(inverseDft(df22),label = 'week')
plt.legend(loc='upper left')
plt.xlabel('samples')
plt.ylabel('Load Distribution')
plt.subplot(2,2,3)
plt.plot(inverseDft(df33),label = 'low')
plt.legend(loc='upper left')
plt.xlabel('samples')
plt.ylabel('Load Distribution')
plt.subplot(2,2,4)
plt.plot(inverseDft(df44),label = 'high')
plt.legend(loc='upper left')
plt.xlabel('samples')
plt.ylabel('Load Distribution')

     

       


'''
# TEST
print ("Input Sine Wave Signal:")
N = 360 # degrees (Number of samples)
a = float(random.randint(1, 100))
f = float(random.randint(1, 100))
p = float(random.randint(0, 360))
print ("frequency = " + str(f))
print ("amplitude = " + str(a))
print ("phase ang = " + str(p))
fnList = []
for n in range(N):
    t = float(n) / N * pi2
    fn = a * math.sin(f * t + p / 360 * pi2)
    fnList.append(fn)

print ("DFT Calculation Results:")
FmList = DFT(fnList)
threshold = 0.001
for (i, Fm) in enumerate(FmList):
    if abs(Fm) > threshold:
        print ("frequency = " + str(i))
        print ("amplitude = " + str(abs(Fm) * 2.0))
        p = int(((cmath.phase(Fm) + pi2 + pi2 / 4.0) % pi2) / pi2 * 360 + 0.5)
        print ("phase ang = " + str(p))
        print
        '''