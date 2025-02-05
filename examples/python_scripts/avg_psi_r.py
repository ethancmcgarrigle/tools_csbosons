import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
#matplotlib.rcParams['text.usetex'] = True
matplotlib.use('TkAgg')
import pdb
import pandas as pd 
from scipy.stats import sem


def calculate_field_average(field_data, Nx, dim, N_samples_to_avg): # assumes cubic/square mesh 
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data

    # Get number of samples 
    N_samples = len(field_data)/(Nx**dim)
    assert(N_samples.is_integer())
    N_samples = int(N_samples)

    # Use split (np) to get arrays that represent each sample (1 array per sample) Throw out the first sample (not warmed up properly) 
    sample_arrays = np.split(field_data, N_samples) 
    sample_arrays = sample_arrays[len(sample_arrays) - N_samples_to_avg:len(sample_arrays)]

    # Final array, initialized to zeros. 
    averaged_data = np.zeros(len(sample_arrays[0]), dtype=np.complex_)
    averaged_data += np.mean(sample_arrays, axis=0) # axis=0 calculates element-by-element mean
    # Calculate the standard error 
    std_errs = np.zeros(len(sample_arrays[0]))
    std_errs += sem(sample_arrays, axis=0)
    return averaged_data, std_errs



def integrate_intensive(field_data):
    result = 0. + 1j*0.
    result = np.sum(field_data) # consider nan-sum 
    result /= len(field_data) # divide by num elements 
    return result



def calc_err_division(x, y, x_err, y_err):
    result = 0.
    # assumes x and y are real 

    # Calculate error using standard error formula 
    result = np.sqrt( ((-x * y_err / (y**2))**2 ) + (x_err/y)**2) 
    return result    






with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['simulation']['Nx']
d = params['system']['Dim']
L = params['system']['CellLength'] # Need box-size 

Vol = L**d

# import the data
psi = np.loadtxt('psi_r_a.dat', unpack=True)

# Need to average the real and imaginary parts of each operator across the simulation  
# for 1 dataset 
psi_data = psi[d] + 1j*psi[d+1]

psi_avg, psi_errs = calculate_field_average(psi_data, Nx, d, int(0.50 * len(psi_data))) # take final 80% of samples  

mod_psi = np.zeros(len(psi_avg))
mod_psi = psi_avg * psi_avg

# integrate psi_avg
N_cond = 0. + 1j*0.
N_cond = integrate_intensive(mod_psi) * Vol # multiply by volume  


# Retrieve condensate fraction CL average from the run 
data_file = 'data0.dat'
try:
  print('Parsing ' + data_file)
except:
  print('please run process.sh first, script cannot find data0.dat')
else:
  noise_averaged_data = np.loadtxt(data_file, unpack=True) # parses into 2 columns, 1st column is means; 2nd is errorbars 


# Pull out the condensate fraction 
# TODO: change to not hardcode!!!!!
#<<<<<<< Updated upstream
N = noise_averaged_data[0][8]  # in data0.dat for SOC, the total particle number is the 8th entry  
#N_kO = noise_averaged_data[0][16] # in data0.dat for SOC, the Nk0 particle number (total) is the 16th entry 
#N_err = noise_averaged_data[1][8] # 2nd column is error, 1st element is real(density)
#=======
#N = noise_averaged_data[0][8]
#N_kO = noise_averaged_data[0][16]
N_err = noise_averaged_data[1][8] # 2nd column is error, 1st element is real(density)
#>>>>>>> Stashed changes
#cond_frac = N_kO/(density * (Nx**d))




print('CL averaged condensate number: ' + str(round(N_cond.real, 2)))
print('CL averaged condensate fraction: ' + str(round(N_cond.real/N, 2)))


# Get condensate trace 
N_samples = int(len(psi_data)/(Nx**d))
psi_samples = np.split(psi_data, N_samples)

# integrate each sample 
N_cond_samples = np.zeros(N_samples, dtype = np.complex_)

for i in range(0, N_samples):
    N_cond_samples[i] = integrate_intensive(psi_samples[i]) * Vol 

N_cond_err = sem(N_cond_samples.real)

# Pull out time data
ops_file = 'operators0.dat'
column_ops = np.loadtxt(ops_file, unpack=True)

CL_time_data = column_ops[2]
CL_iteration_data = column_ops[0]
#N_kzero = column_ops[7]
N_kzero = column_ops[19]


#<<<<<<< Updated upstream
#=======
_isPlotting = False
# Plot the time trace 
if(_isPlotting):
  plt.figure(1)
  #plt.plot(list_x[0:midway_indx], corr_sorted_y_avg[0:midway_indx].real, linewidth=0.5, markersize=6, marker = '*', label = '$C(r)$')
  plt.plot(CL_time_data, N_cond_samples.real/N, '-r', linewidth = 0.5, label = '$N_{0}/N$') 
  plt.plot(CL_time_data, N_kzero/N, '-b', linewidth = 0.5, label = '$N_{k = 0}/N$') 
  plt.title('$n(k_{x} = \kappa_x , k_{y} = 0)$, CL Simulation ', fontsize = 16)
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold') # actually "x"
  plt.ylabel('N', fontsize = 20, fontweight = 'bold')
  # plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
  #plt.colorbar()
  #plt.xlim(-1, 1)
  plt.ylim(-0.5, 1.5)
  plt.legend()
  plt.show()
  
  

#>>>>>>> Stashed changes
# propagate error 
N_cond_frac_err = calc_err_division(N_cond.real, N, N_cond_err, N_err)

# Print the N_SOC and its error to a data file 
outfile = 'N_psi.dat' 
with open(outfile, 'w') as filehandle:
  filehandle.write("# N_psi N_psi_err psi_frac psi_frac_err\n") 
  filehandle.write("{} {} {} {}\n".format(N_cond.real, N_cond_err.real, N_cond.real/N, N_cond_frac_err))



