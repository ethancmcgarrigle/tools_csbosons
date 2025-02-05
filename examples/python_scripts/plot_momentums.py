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
cols_up = np.loadtxt('gx_0.dat', unpack=True)
cols_dwn = np.loadtxt('gx_1.dat', unpack=True) 

# list of x's and y's 
x = cols_up[0][0:Nx**d]
y = cols_up[1][0:Nx**d]

assert(d == 2)

gx_up = cols_up[d] + (1j * cols_up[d+1])
gx_dwn = cols_dwn[d] + (1j * cols_dwn[d+1])

N_samples = int(len(gx_up)/(Nx**d))

# ----- noise average data ----- 
# average the fields 
gx_up_avg, gx_up_errs = calculate_field_average(gx_up, Nx, d, int(0.90 * N_samples)) # take final 80% of samples  
gx_dwn_avg, gx_dwn_errs = calculate_field_average(gx_dwn, Nx, d, int(0.90 * N_samples)) # take final 80% of samples  

# average the sum of the fields 
# species momentum is NOT conserved 
gx_avg, gx_errs = calculate_field_average(gx_up + gx_dwn, Nx, d, int(0.90 * N_samples))

# Integrate x-fields 
momentum_x_up = integrate_intensive(gx_up_avg) * Vol
momentum_x_dwn = integrate_intensive(gx_dwn_avg) * Vol
momentum_x = integrate_intensive(gx_avg) * Vol

#print('integrated average x-momentum (up-species): ' + str(round(momentum_x_up, 3)))
#print('integrated average x-momentum (dwn-species): ' + str(round(momentum_x_dwn, 3)))
print()
print('total x-momentum: ' + str(round(momentum_x, 3)))


# ----- CL Time trace data ----- 
# get time trace data
gx_up_samples = np.split(gx_up, N_samples)
gx_dwn_samples = np.split(gx_dwn, N_samples)

# integrate each sample 
mom_x_up_samples = np.zeros(N_samples, dtype = np.complex_)
mom_x_dwn_samples = np.zeros(N_samples, dtype = np.complex_)
mom_x_total_samples = np.zeros(N_samples, dtype = np.complex_)

for i in range(0, N_samples):
    mom_x_up_samples[i] = integrate_intensive(gx_up_samples[i]) * Vol 
    mom_x_dwn_samples[i] = integrate_intensive(gx_dwn_samples[i]) * Vol 
    mom_x_total_samples[i] = integrate_intensive(gx_up_samples[i] + gx_dwn_samples[i]) * Vol 

mom_x_up_error = sem(mom_x_up_samples.real)
mom_x_dwn_error = sem(mom_x_dwn_samples.real)
mom_x_total_error = sem(mom_x_total_samples.real)


# Pull out time data
ops_file = 'operators0.dat'
column_ops = np.loadtxt(ops_file, unpack=True)

CL_time_data = column_ops[2]
CL_iteration_data = column_ops[0]

# signatures of the 2 goldstone modes? 
N_data = column_ops[11] 
N_kzero = column_ops[19] 
Mx_data = column_ops[33]


_isPlotting = True
# Plot the time trace 
if(_isPlotting):
  plt.figure(1)
  #plt.plot(list_x[0:midway_indx], corr_sorted_y_avg[0:midway_indx].real, linewidth=0.5, markersize=6, marker = '*', label = '$C(r)$')
  plt.plot(CL_time_data, mom_x_up_samples.real, '-r', linewidth = 0.5, label = '$p_{x}$ up') 
  plt.plot(CL_time_data, mom_x_dwn_samples.real, '-b', linewidth = 0.5, label = '$p_{x}$ down ') 
  #plt.plot(CL_time_data, mom_x_up_samples.real + mom_x_dwn_samples.real, '-k', linewidth = 0.5, label = '$p_{x}$') 
  plt.title('x-momentum, CL Simulation ', fontsize = 16)
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold') # actually "x"
  plt.ylabel('$p_{x}$', fontsize = 20, fontweight = 'bold')
  # plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
  #plt.colorbar()
  #plt.xlim(-1, 1)
  #plt.ylim(-0.5, 1.5)
  plt.legend()
  plt.show()
  
  plt.figure(2)
  #plt.plot(list_x[0:midway_indx], corr_sorted_y_avg[0:midway_indx].real, linewidth=0.5, markersize=6, marker = '*', label = '$C(r)$')
  plt.plot(CL_time_data, Mx_data, '-b', linewidth = 0.5, label = '$M_{x}$') 
  plt.plot(CL_time_data, N_kzero/N_data, '-k', linewidth = 0.5, label = '$N_{0}/N$') 
  #plt.plot(CL_time_data, mom_x_dwn_samples.real, '-b', linewidth = 0.5, label = '$N_{k = 0}/N$') 
  plt.title('CL Simulation ', fontsize = 16)
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold') # actually "x"
  plt.ylabel('$O(t_{CL})$', fontsize = 20, fontweight = 'bold')
  # plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
  #plt.colorbar()
  #plt.xlim(-1, 1)
  plt.ylim(-0.1, 1.25)
  plt.legend()
  plt.show()

  plt.figure(3)
  #plt.plot(list_x[0:midway_indx], corr_sorted_y_avg[0:midway_indx].real, linewidth=0.5, markersize=6, marker = '*', label = '$C(r)$')
  plt.plot(CL_time_data, mom_x_total_samples.real, '-k', linewidth = 0.5, label = '$p_{x}$ total (real)') 
  plt.plot(CL_time_data, mom_x_total_samples.imag, '-p', linewidth = 0.5, label = '$p_{x}$ total (imag)') 
  #plt.plot(CL_time_data, mom_x_dwn_samples.real, '-b', linewidth = 0.5, label = '$p_{x}$ down ') 
  #plt.plot(CL_time_data, mom_x_up_samples.real + mom_x_dwn_samples.real, '-k', linewidth = 0.5, label = '$p_{x}$') 
  plt.title('x-momentum, CL Simulation ', fontsize = 16)
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold') # actually "x"
  plt.ylabel('$p_{x}$', fontsize = 20, fontweight = 'bold')
  # plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
  #plt.colorbar()
  #plt.xlim(-1, 1)
  #plt.ylim(-0.5, 1.5)
  plt.legend()
  plt.show()
 


 

#>>>>>>> Stashed changes
# propagate error 
#N_cond_frac_err = calc_err_division(N_cond.real, N, N_cond_err, N_err)

# Print the N_SOC and its error to a data file 
 #outfile = 'N_psi.dat' 
 #with open(outfile, 'w') as filehandle:
 #  filehandle.write("# N_psi N_psi_err psi_frac psi_frac_err\n") 
 #  filehandle.write("{} {} {} {}\n".format(N_cond.real, N_cond_err.real, N_cond.real/N, N_cond_frac_err))
 #


