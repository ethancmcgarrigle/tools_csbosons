import csv
from mpmath import *
import subprocess
import os
import re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb
import yaml
import math
## This function runs statistics on the runs accessed (i.e. parameter sweep). Then it collects the relevant data and plots it at the end

def sech(x):
  return 1/(np.cosh(x))


#U=[0.01, 0.1, 0.3, 0.5, 0.55, 0.57, 0.6, 0.62, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.25, 1.5, 2.0]
U=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25]

with open('input.yml') as infile:
  master_params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
hz = master_params['system']['hz']
dt = master_params['simulation']['dt']  
N_tau = master_params['system']['ntau']
#dt = master_params['simulation']['dt']
psi_lambda = master_params['simulation']['lambdapsi']
stepper = master_params['simulation']['DriverType']
#stepper = 'psi-PEC,ADT'
lamb = [psi_lambda]
#U_val = master_params['system']['U']
num_steps = master_params['simulation']['numtsteps']
#hz = master_params['system']['hz']
N_spins = master_params['system']['NSpinsPerDim']
dimension = master_params['system']['ndim']
T = master_params['system']['beta']
T = 1/T
Jnn = master_params['system']['Jnn']
Jnnn = master_params['system']['Jnnn'] 
noise_pcnt = master_params['simulation']['psinoisefactor']
apply_ADT = master_params['simulation']['apply_ADT']
total_CL_time = master_params['simulation']['Total_CL_Time']
runlength_max = num_steps * dt
apply_ADT = False
# read the dat files 

no_cap_file = 'no_cap.dat'
cap_file = 'cap.dat'
stabilized_file = 'cap_stabilized.dat'
files = [no_cap_file, cap_file, stabilized_file]
U_vals = []
M_vals = []
M_err_vals = []
M2_vals = []
M2_err_vals = []

for f in files:
  # unpack all the data
  cols = np.loadtxt(f, unpack=True)
  U_vals.append(cols[0])
  M_vals.append(cols[1])
  M_err_vals.append(cols[2])
  M2_vals.append(cols[3])
  M2_err_vals.append(cols[4])

# Plots! 


plt.figure(1)
for i in range(0, len(files)):
  plt.errorbar(U_vals[i], M_vals[i], M_err_vals[i], linewidth=1, markersize=6, marker = '*', label = files[i])
plt.plot(U_vals[i], np.ones(len(U_vals[i]))*np.tanh(0.2), linewidth=1, label = 'exact', color = 'k')
plt.title('Single Spin, Convergence of $M$ with $U_1$ Strength , $\lambda_{\psi}$ = ' + str(psi_lambda), fontsize = 11)
plt.xlabel('$U$', fontsize = 20, fontweight = 'bold')
plt.ylabel('$M$', fontsize = 20, fontweight = 'bold')
plt.ylim(0,0.5)
plt.legend()
#plt.savefig("plt/"+args.title[0]+'_'+str(i)+'.png', dpi=300) #dpi=72
plt.show()

plt.figure(2)
for i in range(0, len(files)):
  plt.errorbar(U_vals[i], M2_vals[i], M2_err_vals[i], linewidth=1, markersize=6, marker = '*', label = files[i])
plt.plot(U_vals[i], np.ones(len(U_vals[i]))*1.00, linewidth=1, label = 'exact', color = 'k')
plt.title('Single Spin, Convergence of $M^2$ with $U_1$ Strength , $\lambda_{\psi}$ = ' + str(psi_lambda), fontsize = 11)
plt.xlabel('$U$', fontsize = 20, fontweight = 'bold')
plt.ylabel('$M^2$', fontsize = 20, fontweight = 'bold')
plt.ylim(-1,3)
plt.legend()
#plt.savefig("plt/"+args.title[0]+'_'+str(i)+'.png', dpi=300) #dpi=72
plt.show()

