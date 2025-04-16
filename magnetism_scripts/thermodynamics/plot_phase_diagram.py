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
# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_phase_diagram.txt') 

J1 = -2.0
# x-axis for verticals  
J2 = np.array([-0.0, -0.33, -0.45, -0.66, -0.8, -0.92, -1.04, -1.2, -1.4, -1.8]) 
J2_J1_ratio = J2/J1
# Verticals, y-axis: Critical temperature (KT transition)
Tc = np.array([4.4, 3.2, 2.75, 1.93, 1.4, 0.85, 1.0, 1.7, 2.67, 3.55]) 
#Tc_errs = np.ones_like(Tc)*0.10
Tc_errs = np.array([0.1, 0.12, 0.09, 0.12, 0.11, 0.12, 0.14, 0.15, 0.11, 0.09])



# horizontals: 
T_values = np.array([0.5])
J2_J1_c = np.array([0.5])

 
plt.style.use(style_path)

#plt.figure(figsize=(3.38583, 3.38583))
plt.figure(figsize=(5.648, 5.648))
plt.errorbar(J2_J1_ratio, Tc/np.abs(J1), yerr = Tc_errs, marker='s', xerr=None, linewidth=0., elinewidth = 2.5, color = 'b',  markersize = 9)
plt.errorbar(J2_J1_c, T_values/np.abs(J1), xerr = np.zeros_like(J2_J1_c), marker='o', elinewidth = 2.5, linewidth=0.,color = 'k',  markersize = 9)
plt.xlabel(r'$J_{2} / J_{1}$', fontsize = 36, fontweight = 'bold') # fontsize 28 previously 
plt.ylabel(r'$T / J_{1} $', fontsize = 36, fontweight = 'bold', rotation=0, verticalalignment='center', labelpad=20)
#plt.ylim(1.0, 17)   # 20 prev
plt.xlim(-0.01, 1.0) # 1.03 prev
plt.savefig('S_5halves_classical_J1J2_phase_diagram.pdf', dpi = 300)
plt.show()

 #plt.figure(figsize=(5.648, 5.648))
 #plt.errorbar(eta_SOC_Tc, T_c, yerr = T_c_errs, marker='s', xerr=None, linewidth=0., elinewidth = 2.5, color = 'b',  markersize = 9)
 #plt.errorbar(eta_SOC_crossover, T_crossover, yerr = T_crossover_errs, marker='o', fillstyle= 'none', elinewidth = 2.5, linewidth=0.,color = 'k',  markersize = 9)
 #plt.errorbar(eta_SOC_horizontal_crossover, T_crossover_horizontal, xerr = eta_SOC_horizontal_crossover_err, marker='o', fillstyle= 'none', elinewidth = 2.5, linewidth=0.,color = 'k',  markersize = 9)
 #plt.xlabel(r'$\eta_{\kappa}$', fontsize = 36, fontweight = 'bold') # fontsize 28 previously 
 #plt.ylabel(r'$\bar T $', fontsize = 36, fontweight = 'bold', rotation=0, verticalalignment='center', labelpad=15)
 ##plt.ylabel(r'$\tilde T $', fontsize = 28, fontweight = 'bold', rotation=0, verticalalignment='center', labelpad=10)
 ##plt.xscale('log')
 #plt.ylim(1.0, 17)   # 20 prev
 #plt.xlim(0.0, 1.02) # 1.03 prev
 #plt.xscale('log')
 ##plt.savefig('Tneq0_Phase_diagram_SOC_anisotropy.eps', dpi = 300)
 #plt.show()

