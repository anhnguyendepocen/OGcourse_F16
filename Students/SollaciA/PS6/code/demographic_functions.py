# -*- coding: utf-8 -*-
"""
Economic Policy Analysis with Overlapping Genration Models
Demogrphic Functions for Problem Set #6
Alexandre B. Sollaci
The Univeristy of Chicago
Fall 2016
"""
import numpy as np
import scipy.interpolate as interp
from math import floor , ceil
#import scipy.optimize as opt
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
#from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

def get_fert(totpers, graph=True):  
    ''' 
    Inputs:
        totpers = number of periods an agent lives
        graph = whether or not to display a graph with the fertility rates aby age (Boolean).
            This will produce two graphs: a graph with data points and the interpolated
            fertiliy rates (for 100 year-lived agents) and a graph with the fertility rates
            rescaled for totpers periods.
                
    This function does 2 things:
        1) Interpolates from the data th fertility rates in each year for a 100
        year-lived agent.
        2) Rescales those fertility rates to fit a model where agents live
        totpers periods, where totpers <= 100.
 
    The difficulty comes from the fact that 100 might not be divisible by totpers.
    To overcome this, I first divide 100 into totpers intervals, represented in bins.
    If either the upper or lower bouns of bins is not an integer, I use weighted averages:
        Ex: bins[2] = [2.5, 3.75] --> fert_totpers[2] = 
                        0.5*fert_100[2] + 0.75*fert_100[3]
    If the bins are larger, I sum over the integers cointained in bin:
        Ex: bins[2] = [2.5, 5.75] --> fert_totpers[2] = 
                    0.5*fert_100[2] + fert_100[3] + fert_100[4] + 0.75*fert_100[5]
    '''
    # data
    birth_rates = np.array([0.0, 0.0, 0.3, 12.3, 47.1, 80.7, 105.5, 98.0,\
                            49.3, 10.4, 0.8, 0.0, 0.0])
    age = np.array([1, 9, 11, 15, 18, 21, 26, 31, 36, 41, 46, 51, 100])
    # interpolation
    fert_rates = birth_rates / 2000
    int_fert = interp.interp1d(age, fert_rates)
    # fertility rates for 100-year lived agents
    age_100 = np.linspace(1, 100, 100)
    fert_100 = int_fert(age_100)
    
    ##### Rescale to totpers-year lived agents #####
    
    age_totpers = np.linspace(1, totpers, totpers)
    bin_size = 100 / (totpers)
    
    bins = np.zeros([totpers,2])
    fert_totpers = np.ones(totpers)
    for i in range(totpers):
        bins[i,:] = np.array([i*bin_size, (i+1)*bin_size])
        if i < totpers - 1:
            fert_totpers[i] = (ceil(bins[i,0]) - bins[i,0]) * fert_100[ floor(bins[i,0]) ] \
                    + sum(fert_100[ ceil(bins[i,0]) : floor(bins[i,1])]) \
                    + (bins[i,1] - floor(bins[i,1])) * fert_100[ ceil(bins[i,1]) ]
        else:
            fert_totpers[i] = (ceil(bins[i,0]) - bins[i,0]) * fert_100[ floor(bins[i,0]) ] \
                    + sum(fert_100[ ceil(bins[i,0]) : ceil(bins[i,1])]) 

    if graph == True:
        
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath("__file__"))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)
        
        # Create plots
        fig, ax = plt.subplots()
        plt.plot(age_100, fert_100, label='Interpolated')
        plt.plot(age, fert_rates, 'o', label='Data')
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Fertility Rates', fontsize=20)
        plt.xlabel(r'Age')
        plt.ylabel(r'Fertility Rate')
        #plt.xlim((0, S + 1))
        #plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "fert_rate")
        plt.savefig(output_path)
        plt.show()
        
        fig2, ax2 = plt.subplots()
        plt.plot(age_100, fert_100, label='Interpolated from data')
        plt.plot(age_totpers, fert_totpers, 'o', label='Rescaled to totpers')
        minorLocator = MultipleLocator(1)
        ax2.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')        
        plt.title("Fertility Rate Rescaled to 'totpers' Years")
        plt.xlabel("Age")
        plt.ylabel("Ferility Rate")
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "fert_totpers")
        plt.savefig(output_path)
        plt.show() 

    return fert_totpers 
    
    
def get_mort(totpers, graph=True):
      
    var_names = ('age', 'male_mort', 'male_live',	'male_lifeexp', 'fem_mort',\
                 'fem_live', 'fem_lifeexp')
    df_mort = pd.read_csv('mort_rates2011.csv', thousands=',', header=0, names=var_names)
    mort = df_mort.as_matrix()
    
    # find out when no-one is still alive
    d_m = np.array(np.where(mort[:,2] == 0)).min()
    d_f = np.array(np.where(mort[:,5] == 0)).min()
    d = max(d_m, d_f)
    
    male_death = mort[:d,1]*mort[:d,2]
    fem_death = mort[:d,4]*mort[:d,5]
    data_mort_rate = (male_death + fem_death) / (mort[:d,2] + mort[:d,5])
    #data_age = mort[:d,0]
    
    # get infant mortality rate in USA (2015)
    infmort_rate = 6/1000
    
    # separate from infant mortality rate and cap life at 100 years
    mort_100 = data_mort_rate[:100]
    mort_100[-1] = 1
    #age_100 = np.linspace(1, 100, 100)
    
    ##### Rescale to totpers-year lived agents #####
    
    age_totpers = np.linspace(1, totpers, totpers)
    bin_size = 100 / totpers
    
    bins = np.zeros([totpers,2])
    mort_totpers = np.ones(totpers)
    for i in range(totpers):
        bins[i,:] = np.array([i*bin_size, (i+1)*bin_size])
        
        if i < totpers - 1:
            mort_totpers[i] = 1 - (1 - (ceil(bins[i,0]) - bins[i,0])*mort_100[floor(bins[i,0])]) \
                    * np.prod(1 - mort_100[ ceil(bins[i,0]) : floor(bins[i,1])]) \
                    * (1 - (bins[i,1] - floor(bins[i,1])) * mort_100[ceil(bins[i,1])])
        else:
            mort_totpers[i] = 1 - (1 - (ceil(bins[i,0]) - bins[i,0])*mort_100[floor(bins[i,0])]) \
                    * np.prod(1 - mort_100[ ceil(bins[i,0]) : ceil(bins[i,1])]) 
    
    if graph == True:
        
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath("__file__"))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)
        
        # Create plots
        age_100 = np.linspace(1, 100, 100)
        fig, ax = plt.subplots()
        plt.plot(age_100, mort_100, label='Data')
        plt.plot(np.append([0], age_totpers), np.append(infmort_rate, mort_totpers), \
                           'o', label='Rescaled to totpers')
        #minorLocator = MultipleLocator(1)
        #ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')        
        plt.title("Mortality Rate Rescaled to 'totpers' Years")
        plt.xlabel("Age")
        plt.ylabel("Mortality Rate") 
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, "mort_totpers")
        plt.savefig(output_path)
        plt.show() 
    
    return mort_totpers, infmort_rate

def get_imm_resid(totpers, graph=True):
    
    var_names = ('age', 'year1', 'year2')   
    df_pop = pd.read_csv('pop_data.csv', thousands=',', header=0, names=var_names)
    pop_data = df_pop.as_matrix()
    
    pop = pop_data[:, 1:]
    
    ##### Rescale population to totpers-year lived agents #####
    
    age_totpers = np.linspace(1, totpers, totpers)
    bin_size = 100 / totpers
    
    bins = np.zeros([totpers,2])
    pop_totpers = np.ones([totpers, pop.shape[1]])
    for i in range(totpers):
        bins[i,:] = np.array([i*bin_size, (i+1)*bin_size])
        if i < totpers - 1:
            pop_totpers[i, :] = (ceil(bins[i,0]) - bins[i,0]) * pop[ floor(bins[i,0]) , : ] \
                    + sum(pop[ ceil(bins[i,0]) : floor(bins[i,1]) , :]) \
                    + (bins[i,1] - floor(bins[i,1])) * pop[ ceil(bins[i,1]) , : ]
        else:
            pop_totpers[i, :] = (ceil(bins[i,0]) - bins[i,0]) * pop[ floor(bins[i,0]), : ] \
                    + sum(pop[ ceil(bins[i,0]) : ceil(bins[i,1]) , :])
    
    ## Get fertility and mortality
    fert_totpers = get_fert(totpers, False)
    mort_totpers, infmort_rate = get_mort(totpers, False)
    
    ## Compute immigration
    imm_totpers = np.zeros(totpers)
    imm_totpers[0] = ( pop_totpers[0,1] - \
        (1-infmort_rate)*sum(fert_totpers*pop_totpers[:,0]) ) / pop_totpers[0,0]
    
    for j in range(1,totpers):
        imm_totpers[j] = ( pop_totpers[j,1] - (1-mort_totpers[j-1])*pop_totpers[j-1,0] ) \
            / pop_totpers[j-1,1]
    
    if graph == True:
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath("__file__"))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)
        
        # Create plots
        plt.figure()
        plt.plot(age_totpers, imm_totpers)
        #minorLocator = MultipleLocator(1)
        #ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')        
        plt.title("Immigration Rates Rescaled to 'totpers' Years")
        plt.xlabel("Age")
        plt.ylabel("Immigration Rate") 
        output_path = os.path.join(output_dir, "imm_totpers")
        plt.savefig(output_path)
        plt.show() 
    
    return imm_totpers
    
