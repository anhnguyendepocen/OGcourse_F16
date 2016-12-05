import pandas as pd
from scipy import interpolate
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

def get_fert(totpers,graph=False):
	d=dict({"09":0,"10":0,"10-14":0.3, "15-17":12.3, "18-19":47.1, "20-24":80.7, "25-29":105.5, "30-34":98.0, "35-39":49.3, "40-44":10.4, "45-49":0.8,"55":0,"56":0})
	x=np.array([[9,9],[10,10],[10,14],[15,17],[18,19],[20,24],[25,29],[30,34],[35,39],[40,44],[45,49],[55,55],[56,56]])
	frates=pd.Series(d)
	intf=interpolate.interp1d(np.median(x,axis=1),frates/2000,kind='cubic',bounds_error=False,fill_value=0)
	new_y=intf(np.linspace(0,100,totpers))	
	if graph:
#		cur_path = os.path.split(os.path.abspath(__file__))[0]
#		output_fldr = "images"
#		output_dir = os.path.join(cur_path, output_fldr)
#		if not os.access(output_dir, os.F_OK):
#              os.makedirs(output_dir)
		plt.figure()
		ax = plt.gca()
		ax.set_xlim([0,totpers])
		ax.set_ylim([0,0.08])
		ax.plot(np.linspace(0,100,100),intf((np.linspace(0,100,100))),'g-')
		ax.scatter(x=np.median(x,axis=1), y=frates/2000, marker = 'o', color='b')
		ax.scatter(x=np.linspace(0,100,totpers), y=new_y, marker = 'o', color='r')
		plt.grid(b=True, which='major', color='0.65', linestyle='-')
		plt.legend(['Cubic spline','Data','Model Data'])
		plt.title('Fertility Rate', fontsize=20)
		plt.xlabel(r'Year $s$')
		plt.ylabel('Fertility Rate')
		plt.legend(loc='upper right')
		plt.show()
		
	return new_y
 
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
    
get_fert(100,graph=True)
output_path = os.path.join(output_dir, "fert_rate1")
plt.savefig(output_path)	

get_fert(80, True)
output_path = os.path.join(output_dir, "fert_rate2")
plt.savefig(output_path)

get_fert(20, True)
output_path = os.path.join(output_dir, "fert_rate3")
plt.savefig(output_path)



def get_mort(totpers,graph=False):
	mrates=pd.read_csv('mort_rates2011.csv')
	mrates['Total Pop']=mrates['Num. Male Lives'].str.replace(',','').astype(int)+mrates['Num. Female Lives'].str.replace(',','').astype(int)
	intf=interpolate.interp1d(mrates.index,mrates['Total Pop'],kind='cubic',bounds_error=False,fill_value=0)
	vper=np.linspace(0,100,totpers)	
	pop=intf(vper)
	y=np.divide(np.diff(-mrates['Total Pop'],1),mrates['Total Pop'][:119])
	new_y=np.divide(np.diff(-pop,1),pop[0:totpers-1])
	#add rate 1 for yr =100
	new_y=np.append(new_y,1)
	if graph:
		plt.figure()
		ax = plt.gca()
		ax.set_xlim([0,totpers])
		ax.set_ylim([-0.01,1.02])
		ax.scatter(np.linspace(0,119,119), y, color='g')
		ax.scatter(range(totpers),new_y,color='r')
		ax.grid()
		plt.legend(['Data', 'Model Data'])
		plt.xlabel('Age')
		plt.ylabel('Mortility Rate')
		plt.show()
	return new_y,new_y[0]

cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
    
get_mort(100,graph=True)
output_path = os.path.join(output_dir, "mort1")
plt.savefig(output_path)

get_mort(80,graph=True)
output_path = os.path.join(output_dir, "mort2")
plt.savefig(output_path)

get_mort(20,graph=True)
output_path = os.path.join(output_dir, "mort3")
plt.savefig(output_path)


def get_imm_resid(totpers, graph):
    pop = pd.read_csv("pop_data.csv", thousands = ",")
    pop = pop.iloc[ : -1, :]
#    year_input = df_pop.Age + 1
    pop_12 = pop["2012"]
    pop_13 = pop["2013"]
    step = 100 / totpers
    year = [(i + 1) * step for i in range(totpers)]
    start = 0
    end = 0
    pop_t = []
    pop_tplus1 = []
    for y in year:
        start= end
        end = y
        startind = int(start // 1)
        endind = int(end // 1)
        pop_1 = 0
        pop_2 = 0
        for i in range(startind, endind + 1):
            if i == startind:
                pop_1 += pop_12[startind] * (startind + 1 - start)
                pop_2 += pop_13[startind] * (startind + 1 - start)
            elif i == endind and end > endind:
                pop_1 += pop_12[endind] * (end - endind)
                pop_2 += pop_13[endind] * (end - endind)
            elif startind < i < endind:
                pop_1 += pop_12[i]
                pop_2 += pop_13[i]
        pop_t.append(pop_1)
        pop_tplus1.append(pop_2)
    pop_t = np.array(pop_t)
    pop_tplus1 = np.array(pop_tplus1)
    mort_rate, inf_mortrate = get_mort(totpers, False)
    fert_rate = get_fert(totpers, False)
    immi_rate = []
    for y in range(totpers):
        if y == 0:
            t_fert = (pop_t * fert_rate).sum()
            immi_rate_t = (pop_tplus1[y] - (1 - inf_mortrate) * t_fert) / pop_t[y]
            immi_rate.append(immi_rate_t)
        else:
            immi_rate_t = (pop_tplus1[y] - (1 - mort_rate[y - 1]) * pop_t[y - 1]) / pop_t[y]
            immi_rate.append(immi_rate_t)
    if graph:
        plt.plot(year, immi_rate, marker = "D")
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        plt.plot(year, immi_rate, marker = "o",label= "{} period".format(totpers))
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Immigration Rate', fontsize=20)
        plt.xlabel(r'Year $s$')
        plt.ylabel(r'Rate')
        plt.xlim((0, 101))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, "im{}".format(totpers))
        plt.savefig(output_path)
        plt.show()
    return np.array(immi_rate)

get_imm_resid(100, True)
get_imm_resid(80, True)
get_imm_resid(20, True)



def pop_next(pop_t, fert_rate, mort_rate, inf_mortrate, immi_rate):
    pop_tplus1 = np.zeros(len(pop_t))
    pop_tplus1[1 :] = (1 - mort_rate[: -1]) * pop_t[: -1] + immi_rate[1 :] * pop_t[1 :]
    pop_tplus1[0] = (1- inf_mortrate) * np.dot(fert_rate, pop_t) + immi_rate[0] * pop_t[0]
    pop_tplus1_norm =pop_tplus1 / pop_tplus1.sum()
    g_rate = (pop_tplus1.sum() / pop_tplus1_norm.sum()) - 1
    return (pop_tplus1_norm, g_rate)

    
def get_pop_ss(graph):
    mort_rate, inf_mortrate = get_mort(100, False)
    immi = get_imm_resid(100, False)
    fert_rate = get_fert(100, False)
    omega = np.zeros([100, 100])
    for i in range(100):
        if i == 0:
            omega[0, :] = (1- inf_mortrate) * fert_rate
            omega[0, 0] += immi[0]
        else:
            omega[i, i - 1] = 1 - mort_rate[i - 1]
            omega[i, i] = immi[i]

    eig = np.linalg.eig(omega)
    eigval = np.real(eig[0]).max()
    eigvec = eig[1][:, 0]
    eigvec /= eigvec.sum()

    if graph:
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        plt.plot(np.linspace(1, 100, 100), eigvec)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Stationary Population Distribution', fontsize=20)
        plt.xlabel(r'Year $s$')
        plt.ylabel(r'Population')
        plt.xlim((0, 101))
        plt.ylim((0, 0.05))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "pop_ss")
        plt.savefig(output_path)
        plt.show()
    return (omega, eigval - 1, eigvec)

#get_pop_ss(True)

def pop_timepath(T, graph):
    omega, g_rate_ss, pop_ss = get_pop_ss(False)
    
    mort_rate, inf_mortrate = get_mort(100, False)
    immi = get_imm_resid(100, False)
    fert_rate = get_fert(100, False)
    pop = pd.read_csv("pop_data.csv", thousands = ",")
    pop = pop.iloc[ : -1, :]
    pop13 = np.array(pop["2013"])
    pop0 = pop13 / pop13.sum()

    pop_tpath = np.zeros([100, T + 100])
    gpath = np.zeros(T + 100 -1)
    for i in range(T - 1):
        if i == 0:
            pop_tpath[:, i] = pop0
        else:
            pop_t = pop_tpath[:, i - 1]
            pop_tplus1, g = pop_next(pop_t, fert_rate, mort_rate, inf_mortrate, immi)
            gpath[i - 1] += g
            pop_tpath[:, i] += pop_tplus1
    for i in range(100):
        pop_tpath[:, T - 1 + i] = pop_ss
    gpath[T  - 2:] = g_rate_ss

    if graph:
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        plt.plot(np.linspace(1, 100, 100), pop_tpath[:, 0], label= "period{}".format(1))
        plt.plot(np.linspace(1, 100, 100), pop_tpath[:, 9], label= "period{}".format(10))
        plt.plot(np.linspace(1, 100, 100), pop_tpath[:, 29], label= "period{}".format(30))
        plt.plot(np.linspace(1, 100, 100), pop_tpath[:, 199], label= "period{}".format(200))
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Population Distribution', fontsize=20)
        plt.xlabel(r'Year $s$')
        plt.ylabel(r'Population')
        plt.xlim((0, 101))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "pop_dist")
        plt.savefig(output_path)
        plt.show()
        
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        plt.plot(np.linspace(2, T + 100, T + 100 - 1), gpath, label = "Growth Rate")
        plt.plot(np.linspace(2, T + 100, T + 100 - 1), np.array([g_rate_ss] * len(np.linspace(2, T + 100, T + 100 - 1))), '--')
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Population Growth Rate Path', fontsize=20)
        plt.xlabel(r'Year $s$')
        plt.ylabel(r'Growth Rate')
        plt.xlim((0, 100))
        plt.ylim((0, 0.02))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "pop_growth")
        plt.savefig(output_path)
        plt.show()
    return pop_tpath, gpath

pop_timepath(200, True)