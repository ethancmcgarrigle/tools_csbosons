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


def calculate_field_average(field_data, Nx, Ny, dim, N_samples_to_avg): # assumes cubic/square mesh 
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data

    # Get number of samples 
    N_samples = len(field_data)/(Nx*Ny)
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



def calc_err_division(x, y, x_err, y_err):
    # x/y 
    # assumes x and y are real 
    z = x/y
    # Calculate error using standard error formula 
    #result = np.sqrt( ((-x * y_err / (y**2))**2 ) + (x_err/y)**2)
    #result = z * np.sqrt( ((x_err/x)**2) + ((y_err/y)**2) ) 
    result =  z * np.sqrt( ((x_err/x)**2) + ((y_err/y)**2) ) 
    return result


def calc_err_average(vector):
   # error propagation for summing over a whole vector of numbers. The input vector is the 1D list of errors to be propagated  
   # returns the resulting error
   err = 0. + 1j*0. 
   err += (1./len(vector)) * np.sqrt( np.sum( vector**2  ) )
   return err 

# Script to load and plot correlation data 

# import the data
ops_file = 'n_k.dat'
cols = np.loadtxt(ops_file, unpack=True)

k_x = cols[0]
k_y = cols[1]
n_x = cols[2]
n_y = cols[3]


n_k_real = cols[4]
n_k_imag = cols[5]

# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['simulation']['Nx'] 
Ny = params['simulation']['Ny'] 
_kappa = params['system']['kappa'] 
_T = 1./float(params['system']['beta']) 
d = params['system']['Dim']
assert(d == 2)
N_samples = int(len(n_k_real)/(Nx*Ny))

kx = cols[0][0:Nx*Ny]
ky = cols[1][0:Nx*Ny]
kx_unique = np.unique(kx)
ky_unique = np.unique(ky)


CL_time = np.loadtxt('operators0.dat', unpack=True)
CL_time = CL_time[2] # third column is CL time 
#CL_time = CL_time[0:-1] # cut off last one 
#n_k = np.zeros(len(np.unique(k_x)), dtype=np.complex_)
n_k = np.zeros(len(n_k_real), dtype=np.complex_) 
n_k = n_k_real + 1j*n_k_imag

n_k_split = np.split(n_k, N_samples)

#n_k_split = n_k_split[0:-1]
plt.style.use('~/csbosonscpp/tools/python_scripts/plot_style.txt')

#n_k_sorted = np.transpose(n_k_sorted)
# Plot the N_k distribution in k-space  
indx = int(Nx * 2) 
#indx = int(Nx * 2) + 1
#indx = int(Nx*Ny) - int(Nx * 2) 

n_q_t = np.array([i[indx] for i in n_k_split]).real
plt.figure(1)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.plot(CL_time, n_q_t[0:len(CL_time)], color = 'r', linewidth = 1.5) 
#plt.plot(CL_time, np.array([i[4] for i in n_k]).real, color = 'r', linewidth = 1.5) 
plt.title('$n(k)$ Isotropic SOC, ' + r'$\tilde \kappa = ' + str(_kappa) + '$ , ' + r'$\tilde T = ' + str(np.round(_T, 2)) + '$', fontsize = 22)
plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
plt.ylabel('$n(k)$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
#plt.colorbar()
#plt.legend()
#plt.savefig('n_k_mixed_1K.eps')
plt.show()

 #plt.figure(2)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.plot(kx_list, S_k_sorted[:,0].real, 'r', label = '$S(k_{x})$')
 ##n_k_sorted = np.transpose(n_k_sorted)
 #plt.errorbar(np.unique(kx), n_kx.real, n_kx_errs, marker='o', elinewidth=0.25, linewidth = 0.25, color = 'blue', label='$CL$')
 ##plt.errorbar(np.unique(ky), n_ky.real, n_ky_errs, marker='o', elinewidth=0.25, linewidth = 0.25, color = 'red', label='$n(k_{y})$')
 #plt.title('Stripe Phase Momentum Distribution, ' + r'$\tilde T = 1$', fontsize = 16)
 #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('$n(k_x)$', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 ##plt.xlim(-1, 1)
 ##plt.savefig('n_k_mixed_1K.eps')
 #plt.legend()
 #plt.show()

