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
 
input_file = './operators0.dat'

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



