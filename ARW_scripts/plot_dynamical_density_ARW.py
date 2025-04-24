import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
if 'Linux' in platform.platform():
  matplotlib.use('TkAgg')
else:
  matplotlib.rcParams['text.usetex'] = True
from scipy.stats import sem 
import glob 
from scipy import stats 

# Import our custom package for Csbosons data analysis
#from csbosons_data_analysis.field_analysis import *
from csbosons_data_analysis.import_parserinfo import *
from csbosons_data_analysis.error_propagation import *

# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path_data = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_data.txt') 



def visualize_rho_t(warmup_percentage, savefig):
  #### Begin script #### 
  
  # Script to load and plot correlation data 
  params = import_parser('input.yml')
  
  tmax_key = 'tmax'
  tmax = params['system'][tmax_key] 
  Nt = params['simulation']['nt'] 
  dt = tmax/Nt
  
  # System parameters 
  D = params['system']['Diffusivity'] 
  k = params['system']['Reaction_constant'] 
  
  # import data 
  ops_file = './operators0.dat'
  cols = np.loadtxt(ops_file, unpack=True)
  CL_time = cols[2]
  
  # Averaging CL sampling data:
  warmup_samples = int(warmup_percentage * len(cols[0])) # num samples to be thrown out!  
  
  print('Averaging ' + str(len(cols[0]) - warmup_samples) + ' CL samples' )
  
  # Get the t range; the first 3 columns are non-dynamical particle number info. 
  first_indx = 3
  num_tpoints = (len(cols) - first_indx)//2
  
  assert(num_tpoints == Nt)
  real_time = np.linspace(0, tmax, num_tpoints) + dt
  dynamical_partnum = np.zeros(num_tpoints, dtype=np.complex128)
  dynamical_partnum_err = np.zeros(num_tpoints, dtype=np.complex128)
  
  for i in range(num_tpoints): 
    dynamical_partnum[i] = np.mean(cols[first_indx + 2*i][warmup_samples:] + 1j*cols[first_indx + 2*i + 1][warmup_samples:] ) 
    dynamical_partnum_err[i] = stats.sem(cols[first_indx + 2*i][warmup_samples:] + 1j*cols[first_indx + 2*i + 1][warmup_samples:] ) 
  
  
  plt.style.use(style_path_data) 
  
  title_str = r'$t_{max} = ' + str(tmax) + '$, $D = ' + str(D) + '$, $\mu = ' + str(k) + '$'  
  
  showImPart = True
  plt.figure(figsize = (5,5))
  plt.errorbar(real_time, dynamical_partnum.real, dynamical_partnum_err.real, color = 'b', linewidth=1.2, elinewidth = 1.2, markersize = 1, label = r'Re[$\rho$]')
  if(showImPart):
    plt.errorbar(real_time, dynamical_partnum.imag, np.abs(dynamical_partnum_err), color = 'k', linewidth=1.2, elinewidth = 1.2, markersize = 1, label = r'Im[$\rho$]')
  plt.title(title_str, fontsize = 20) 
  plt.xlabel('Real time', fontsize = 22)
  plt.ylabel(r'$\rho(t)$', fontsize = 20, rotation=0, labelpad=22)
  plt.legend()
  if(savefig):
    plt.savefig('rho_t_plot.pdf', dpi=300)
  plt.show()



if __name__ == "__main__":
  ''' Script to plot the bulk density over (real) time '''
  ''' Script requires the following parameters:
      - the warmup percentage (what percentage of samples we throw out before averaging) 
      - whether you want to save the plot ''' 
  warmup_pcnt = 0.25
  savePlots = True
  visualize_rho_t(warmup_pcnt, savePlots)

