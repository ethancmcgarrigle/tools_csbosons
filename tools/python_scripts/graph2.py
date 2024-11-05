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
CL_time = cols[2] # n - 1
N_total = cols[11] # n - 1
rho_NF_xx = cols[41]
rho_NF_xy = cols[43]
rho_NF_yy = cols[45]
Px = cols[47]
Py = cols[49]
Vol = 31.1415**2
rho_SF = 0.5 * (rho_NF_xx + rho_NF_yy)
rho = N_total/Vol
rho_SF = rho - rho_SF

rho_SF_xx = rho - rho_NF_xx
rho_SF_yy = rho - rho_NF_yy
rho_SF_xy = -rho_NF_xy

#plt.style.use('~/csbosonscpp/tools/python_scripts/plot_style.txt')
plt.style.use('./plot_style.txt')

plt.figure(2)
plt.plot(CL_time, rho_SF_xx/rho, color = 'b', linestyle = 'solid', linewidth=1.6, markersize = 3, label = r'$\rho^{xx}_{SF} / \rho $')
plt.plot(CL_time, rho_SF_yy/rho, color = 'g', linestyle = 'dashdot', linewidth=2.4, markersize = 3, label = r'$\rho^{yy}_{SF} / \rho $')
plt.plot(CL_time, rho_SF_xy/rho, color = 'r', linestyle = 'dashed', linewidth=2.4, markersize = 3, label = r'$\rho^{xy}_{SF} / \rho $')
#plt.plot(CL_time, rho_SF_imag/rho, 'r-', linewidth=1.2, markersize = 3, label = 'Down Real')
plt.title('Superfluid Fractions, ' + r'$\tilde T = 1$, Stripe Phase, Isotropic SOC', fontsize = 26, fontweight = 'bold')
plt.xlabel('Simulation Time', fontsize = 24, fontweight = 'bold')
plt.ylabel(r'$\rho_{SF} / \rho $', fontsize = 40, fontweight = 'bold')
plt.axhline(y = 1.0, color = 'k', linestyle = '--', linewidth = 2, label='100% Superfluid')
plt.xlim(0, 18000) # CL time 
#plt.ylim(-10,20)
#plt.yscale('log')
#plt.legend(fontsize='small')
plt.legend(loc='best', ncol = 4)
#plt.legend(ncol = 4, fontsize='medium')
#plt.legend(loc=(0.2, 0.65),ncol = 4)
#plt.legend(loc=(0.10, 0.65), bbox_transform=fig.transFigure,ncol = 4)
#plt.savefig('zero_kapp_hmg_1species.eps', dpi = 300)
 ##plt.savefig("plt/"+args.title[0]+'_'+str(i)+'.png', dpi=300) #dpi=72
plt.show()

print('Mean SF_xx : ' + str(np.mean(rho_SF_xx/rho)))
print('Mean SF_yy : ' + str(np.mean(rho_SF_yy/rho)))


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
