import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
reader=pd.read_csv('level2/2016years_level2/sumall1.csv')
timeindex=pd.read_csv('level2/2016years_level2/sumall1.csv',index_col=['time'])
load=reader['load']
load_arr=np.array(load)
time=reader['time']
plt.plot(load)
span=5
spanarray=[]
for k, g in groupby(sorted(load_arr), key=lambda x: x//span):
    print('{}-{}: {}'.format(k*span, (k+1)*span-1, len(list(g))))#the g only can be used once
for k, g in groupby(sorted(load_arr), key=lambda x: x//span):
    spanarray.append(len(list(g)))

N_points = 100000
n_bins = 1
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
#axs[0].hist(load_arr, bins=n_bins)
#axs[1].hist(load_arr, bins=n_bins)