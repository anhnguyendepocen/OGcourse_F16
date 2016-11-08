#import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd


def zeta(bq1, bq2, bq3, yr1, yr2, yr3, age2013, wgt):
    # only use bequests from 2011 - 2013
    for y in range(len(yr1)):
        bq1[y] = bq1[y] * (yr1[y] >= 2011 and yr1[y] <= 2013)
        if yr1[y] == 2011:
            bq1[y] = bq1[y] * 0.9652
        if yr1[y] == 2012:
            bq1[y] = bq1[y] * 0.9854
    
    for y in range(len(yr2)):
        bq2[y] = bq2[y] * (yr2[y] >= 2011 and yr2[y] <= 2013)
        if yr2[y] == 2011:
            bq2[y] = bq2[y] * 0.9652
        if yr2[y] == 2012:
            bq2[y] = bq2[y] * 0.9854
    
    for y in range(len(yr3)):
        bq3[y] = bq3[y] * (yr3[y] >= 2011 and yr3[y] <= 2013)
        if yr3[y] == 2011:
            bq3[y] = bq3[y] * 0.9652
        if yr3[y] == 2012:
            bq3[y] = bq3[y] * 0.9854
    
    agebq1 = age2013 - (2013 - yr1)
    agebq2 = age2013 - (2013 - yr2)
    agebq3 = age2013 - (2013 - yr3)
    
    
    bqage1 = np.zeros(100 - 20)
    bqage2 = np.zeros(100 - 20)
    bqage3 = np.zeros(100 - 20)
    for age in range(20,100):
        bqage1[age - 20] = sum(bq1[np.where(agebq1 == age)])
        bqage2[age - 20] = sum(bq2[np.where(agebq2 == age)])
        bqage3[age - 20] = sum(bq3[np.where(agebq3 == age)])
    
    BQage = bqage1 + bqage2 + bqage3
    
    zeta_s = BQage / sum(BQage)
    
    return zeta_s

## a
df_main = pd.read_stata('p13i6.dta')
df_summ = pd.read_stata('rscfp2013.dta')

networth = np.array(df_summ.networth)
wgt = np.array(df_summ.wgt)

bq1 = np.array(df_main.X5804) * wgt
yr1 = np.array(df_main.X5805)
bq2 = np.array(df_main.X5809) * wgt
yr2 = np.array(df_main.X5810)
bq3 = np.array(df_main.X5814) * wgt
yr3 = np.array(df_main.X5815)
age2013 = np.array(df_main.X8022)

zeta_s = zeta(bq1, bq2, bq3, yr1, yr2, yr3, age2013, wgt)

age = np.arange(len(zeta_s)) + 21

cur_path = os.path.split(os.path.abspath("__file__"))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

fig = plt.figure()
plt.hist(zeta_s, bins='auto')
plt.title("Histogram of $\zeta_s$")
plt.xlabel("Value")
plt.ylabel("Frequency") 
output_path = os.path.join(output_dir, "zeta_s_hist")
plt.savefig(output_path)
plt.show()

fig = plt.figure()
plt.bar(age, zeta_s)  
plt.title("$\zeta_s$ as a function of age $s$")
plt.xlabel("Age")
plt.ylabel("$\zeta_s$") 
output_path = os.path.join(output_dir, "zeta_func")
plt.savefig(output_path)
plt.show()   

##b 
   
bq11 = bq1 * ( networth <= np.percentile(networth, 25))
bq12 = bq1 * np.logical_and( networth > np.percentile(networth, 25) ,
                   networth <= np.percentile(networth, 50))
bq13 = bq1 * np.logical_and( networth > np.percentile(networth, 50) ,
                   networth <= np.percentile(networth, 75))
bq14 = bq1 * ( networth > np.percentile(networth, 75))

bq21 = bq2 * ( networth <= np.percentile(networth, 25))
bq22 = bq2 * np.logical_and( networth > np.percentile(networth, 25) ,
                   networth <= np.percentile(networth, 50))
bq23 = bq2 * np.logical_and( networth > np.percentile(networth, 50) ,
                   networth <= np.percentile(networth, 75))
bq24 = bq2 * ( networth > np.percentile(networth, 75))

bq31 = bq3 * ( networth <= np.percentile(networth, 25))
bq32 = bq3 * np.logical_and( networth > np.percentile(networth, 25) ,
                   networth <= np.percentile(networth, 50))
bq33 = bq3 * np.logical_and( networth > np.percentile(networth, 50) ,
                   networth <= np.percentile(networth, 75))
bq34 = bq3 * ( networth > np.percentile(networth, 75))

## compute marginal distribution for each quartile separately
zeta1 = zeta(bq11, bq21, bq31, yr1, yr2, yr3, age2013, wgt)
zeta2 = zeta(bq12, bq22, bq32, yr1, yr2, yr3, age2013, wgt)
zeta3 = zeta(bq13, bq23, bq33, yr1, yr2, yr3, age2013, wgt)
zeta4 = zeta(bq14, bq24, bq34, yr1, yr2, yr3, age2013, wgt)

