
import matplotlib.pyplot as plt
import numpy as np


def get_zeta(bq_1, bq_2, bq_3, year_1, year_2, year_3, age_2013, wgt):
    # only use bequests from 2011 - 2013
    for y in range(len(year_1)):
        bq_1[y] = bq_1[y] * (year_1[y] >= 2011 and year_1[y] <= 2013)
        if year_1[y] == 2011:
            bq_1[y] = bq_1[y] * 0.9652
        if year_1[y] == 2012:
            bq_1[y] = bq_1[y] * 0.9854
    
    for y in range(len(year_2)):
        bq_2[y] = bq_2[y] * (year_2[y] >= 2011 and year_2[y] <= 2013)
        if year_2[y] == 2011:
            bq_2[y] = bq_2[y] * 0.9652
        if year_2[y] == 2012:
            bq_2[y] = bq_2[y] * 0.9854
    
    for y in range(len(year_3)):
        bq_3[y] = bq_3[y] * (year_3[y] >= 2011 and year_3[y] <= 2013)
        if year_3[y] == 2011:
            bq_3[y] = bq_3[y] * 0.9652
        if year_3[y] == 2012:
            bq_3[y] = bq_3[y] * 0.9854
    
    agebq_1 = age_2013 - (2013 - year_1)
    agebq_2 = age_2013 - (2013 - year_2)
    agebq_3 = age_2013 - (2013 - year_3)
    
    
    bqage_1 = np.zeros(100 - 20)
    bqage_2 = np.zeros(100 - 20)
    bqage_3 = np.zeros(100 - 20)
    for age in range(20,100):
        bqage_1[age - 20] = sum(bq_1[np.where(agebq_1 == age)])
        bqage_2[age - 20] = sum(bq_2[np.where(agebq_2 == age)])
        bqage_3[age - 20] = sum(bq_3[np.where(agebq_3 == age)])
    
    BQage = bqage_1 + bqage_2 + bqage_2
    
    zeta_s = BQage / sum(BQage)
    
    return zeta_s