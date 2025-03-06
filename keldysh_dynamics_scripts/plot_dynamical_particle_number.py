import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
if 'Linux' in platform.platform():
  matplotlib.use('TkAgg')
else:
  matplotlib.rcParams['text.usetex'] = True
import pandas as pd 
from scipy.stats import sem 
from matplotlib.colors import LogNorm
import glob 
from scipy import stats 

# Import our custom package for Csbosons data analysis
#from csbosons_data_analysis.field_analysis import *
from csbosons_data_analysis.import_parserinfo import *
from csbosons_data_analysis.error_propagation import *

# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path_data = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_data.txt') 

#### Begin script #### 

# Script to load and plot correlation data 
params = import_parser('input.yml')

try: 
  dimensionless = params['system']['dimensionless']
except:
  print('Dimensionless keyword not found, setting to false.')
  dimensionless = False 

if(dimensionless):
  tmax_key = 'tmax'
else:
  tmax_key = 'tmax_ns'

tmax = params['system'][tmax_key] 
Nt = params['simulation']['nt'] 
dt = tmax/Nt


# import data 
ops_file = './operators0.dat'
cols = np.loadtxt(ops_file, unpack=True)
CL_time = cols[2]
eqb_partnum = cols[3]
eqb_partnum_im = cols[4]


# Averaging CL sampling data:
pcnt_to_remove = 0.25
warmup_samples = int(pcnt_to_remove * len(cols[0])) # num samples to be thrown out!  

print('Averaging ' + str(len(cols[0]) - warmup_samples) + ' CL samples' )
N_avg_eqb = np.mean(eqb_partnum[warmup_samples:])
N_avg_eqb_err = stats.sem(eqb_partnum[warmup_samples:])

# Get the t range; the first 5 columns are non-dynamical particle number info. 
first_indx = 5
num_tpoints = (len(cols) - first_indx)//2

assert(num_tpoints == Nt)
real_time = np.linspace(0, tmax, num_tpoints) 
dynamical_partnum = np.zeros(num_tpoints, dtype=np.complex128)
dynamical_partnum_err = np.zeros(num_tpoints, dtype=np.complex128)

for i in range(num_tpoints): 
  dynamical_partnum[i] = np.mean(cols[first_indx + 2*i][warmup_samples:] + 1j*cols[first_indx + 2*i + 1][warmup_samples:] ) 
  dynamical_partnum_err[i] = stats.sem(cols[first_indx + 2*i][warmup_samples:] + 1j*cols[first_indx + 2*i + 1][warmup_samples:] ) 


savefig = True

plt.style.use('~/CSBosonsCpp/tools/python_plot_styles_examples/plot_style_data.txt')

if dimensionless:
  title_str = r'$\bar{t}_{max} = ' + str(tmax) + '$' 
else:
  title_str = r'$t_{max} = ' + str(tmax) + '$ [ns]'

plt.figure(figsize = (4,4))
plt.plot(CL_time, eqb_partnum, color = 'b', linewidth=1.2, markersize = 2, label = r'Re[$N$]')
plt.plot(CL_time, eqb_partnum_im, color = 'k', linewidth=1.2, markersize = 2, label = r'Im[$N$]')
plt.title(title_str, fontsize = 16, fontweight = 'bold')
plt.xlabel('Langevin Time', fontsize = 22)
plt.ylabel(r'Eqb. Particle Number', fontsize = 20)
plt.legend()
if(savefig):
  plt.savefig('N_eqb_plot.pdf', dpi=300)
plt.show()


# Get index near t = 0.01 [ns] 
 #t_0 = 0.01 # [ns] 
 #N_t0 = int(t_0 / dt)
 #plt.figure(figsize = (4,4))
 #plt.plot(CL_time, cols[N_t0 + 5], color = 'b', linewidth=1.2, markersize = 2, label = r'Re[$N$]')
 #plt.plot(CL_time, cols[N_t0 + 5+1], color = 'k', linewidth=1.2, markersize = 2, label = r'Im[$N$]')
 #plt.title('Ideal gas sampling at $t = 0.01$ [ns]', fontsize = 20, fontweight = 'bold')
 #plt.xlabel('Langevin Time', fontsize = 24)
 #plt.ylabel(r'Particle Number', fontsize = 20)
 #plt.legend()
 #plt.show()


# plot limits: 
if(N_avg_eqb < 150.):
  p_width = 2. * N_avg_eqb 
else:
  p_width = 0.1 * N_avg_eqb

showImPart = True
plt.figure(figsize = (4,4))
plt.errorbar(real_time, dynamical_partnum.real, dynamical_partnum_err.real, color = 'b', linewidth=1.2, elinewidth = 1.2, markersize = 1, label = r'Re[$N$]')
plt.axhline(y = N_avg_eqb, color = 'k', linestyle = 'solid', linewidth=1.2, label = r'Equilibrium')
if(showImPart):
  plt.errorbar(real_time, dynamical_partnum.imag, np.abs(dynamical_partnum_err), color = 'k', linewidth=1.2, elinewidth = 1.2, markersize = 1, label = r'Im[$N$]')
axes = plt.gca()
axes.fill_between(real_time, N_avg_eqb - N_avg_eqb_err, N_avg_eqb + N_avg_eqb_err, color = 'k', alpha = 0.5) 
plt.title(title_str, fontsize = 20, fontweight = 'bold')
plt.xlabel('Real time', fontsize = 24)
plt.ylabel(r'Dynamical Particle Number', fontsize = 20)
plt.legend()
plt.ylim(N_avg_eqb - p_width, N_avg_eqb + p_width) 
if(savefig):
  plt.savefig('N_t_plot.pdf', dpi=300)
plt.show()

# Sample and plot every N points 
N = 100
plt.figure(figsize = (4,4))
plt.errorbar(real_time[::N], dynamical_partnum[::N].real, dynamical_partnum_err[::N].real, color = 'b', linewidth=1.2, elinewidth = 1.2, markersize = 1, label = r'Re[$N$]')
plt.axhline(y = N_avg_eqb, color = 'k', linestyle = 'solid', linewidth=1.2, label = r'Equilibrium')
axes = plt.gca()
axes.fill_between(real_time, N_avg_eqb - N_avg_eqb_err, N_avg_eqb + N_avg_eqb_err, color = 'k', alpha = 0.5) 
plt.title('Dynamical particle number', fontsize = 20, fontweight = 'bold')
plt.xlabel('Real time', fontsize = 24)
plt.ylabel(r'$N$', fontsize = 24)
plt.ylim(N_avg_eqb - p_width, N_avg_eqb + p_width) 
plt.legend()
plt.show()


