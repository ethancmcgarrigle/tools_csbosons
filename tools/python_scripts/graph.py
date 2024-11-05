import csv
#from mpmath import *
import subprocess
import os
import re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import pdb
import yaml
import math
## This function runs statistics on the runs accessed (i.e. parameter sweep). Then it collects the relevant data and plots it at the end

def sech(x):
  return 1/(np.cosh(x))



 
input_file = './operators0.dat'

 #with open(input_file, "r") as infi: 
 #  reader = csv.reader(infi, delimiter=' ')
 #  CL_time = list(zip(*reader))[2]
 #  lines = infi.readlines()

#CL_time = np.float_(CL_time[1:])

 #N_vals = [] 
 #N2_vals = [] 
 #
 #N_up_vals = []
 #N_dwn_vals = []
#for f in input_file:
  # unpack all the data
cols = np.loadtxt(input_file, unpack=True)
CL_time = cols[2]
N_up = cols[3]
N_dwn = cols[5]
N_imag_up = cols[4]
N_imag_dwn = cols[6]

N = N_up + N_dwn



plt.style.use('~/csbosonscpp/tools/python_scripts/plot_style.txt')

plt.figure(2)
plt.plot(CL_time, N_up, 'g-', linewidth=1.2, markersize = 3, label = 'Up Real')
plt.plot(CL_time, N_dwn, 'r-', linewidth=1.2, markersize = 3, label = 'Down Real')
plt.plot(CL_time, N_imag_up, 'g--', linewidth=1.2, markersize = 3, label = 'Up Imaginary ')
plt.plot(CL_time, N_imag_dwn, 'r--', linewidth=1.2, markersize = 3, label = 'Down Imaginary')
plt.title('2D Bosefluid, Random Seed $\kappa = 0$', fontsize = 20, fontweight = 'bold')
plt.xlabel('Langevin Time', fontsize = 20, fontweight = 'bold')
plt.ylabel('$N$', fontsize = 20, fontweight = 'bold')
#plt.xlim(0, 35) # CL time 
#plt.ylim(-10,20)
#plt.yscale('log')
plt.legend()
plt.savefig('zero_kapp_hmg_1species.eps', dpi = 300)
 ##plt.savefig("plt/"+args.title[0]+'_'+str(i)+'.png', dpi=300) #dpi=72
plt.show()



 #plt.figure(4)
 ##plt.title('1 Spin, T = 1K', fontweight = 'bold')
 #plt.plot(CL_time, N_vals, 'k-', linewidth=1.0, markersize = 2, label = 'CL Sampling')
 #plt.plot(CL_time, np.ones(len(N_vals)), 'r--', linewidth=3.0, markersize = 2, label = 'Constraint')
 ##plt.title('ETD-$\psi$ Single Spin', fontsize = 14)
 #plt.xlabel('$\\bf{Simulation \hspace{5px} Time}$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('$\\bf{N}$', fontsize = 20, fontweight = 'bold')
 ##plt.xlim(0, 200) # CL time 
 #plt.ylim(-10,30)
 ##plt.yscale('log')
 #plt.legend(loc = 'best', bbox_to_anchor=(0.45,0.9), prop = {'weight' : 'bold'})
 #plt.savefig('N_chaotic.eps', format='eps', dpi = 1200)
 #plt.savefig('N_chaotic.svg', format='svg', dpi = 1200)
 #plt.show()
