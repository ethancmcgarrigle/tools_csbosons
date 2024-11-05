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
d = params['system']['Dim']
N_samples = int(len(n_k_real)/(Nx*Ny))
#N_samples = int(len(n_k_real)/(Nx*Ny))

n_k = n_k_real + 1j*n_k_imag 
#n_k, n_k_errs = calculate_field_average(n_k_real + 1j * n_k_imag, Nx, d, int(0.80 * N_samples))
#n_k, n_k_errs = calculate_field_average(n_k_real + 1j * n_k_imag, Nx, d, int(1.00 * N_samples))

_data = {'kx': k_x, 'ky': k_y, 'n_k': n_k}
d_frame = pd.DataFrame.from_dict(_data)


print('Sorting the data frame into ascending order')

d_frame.sort_values(by=['kx', 'ky'], ascending = True, inplace=True) 


# Redefine numpy array post sorting
n_k_sorted = np.array(d_frame['n_k']) 
N_total = np.sum(n_k_sorted.real)
n_k_sorted.resize(Nx, Ny)
n_k_sorted = np.transpose(n_k_sorted)

# import the input parameters, specifically the i and j indices 

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
 #r_i = params['system']['operators']['i'] 
 #r_j = params['system']['operators']['j']



# Open the resulting data file 
 #in_file = open("./" + C_ij_data_filename, "r")
 #tmp = in_file.read()
 #tmp = re.split(r"\s+", tmp)
 #tmp = tmp[0:-1]
 #tmp = tuple(map(float, tmp))


plt.style.use('~/CSBosonsCpp/tools/python_scripts/plot_style.txt')

print('N total: ' + str(N_total))
# Plot the N_k distribution in k-space  
plt.figure(1)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(n_k_sorted.real/N_total, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.title('$n(k)$', fontsize = 28, fontweight = 'bold')
plt.xlabel('$k_x$', fontsize = 24, fontweight = 'bold')
plt.ylabel('$k_y$', fontsize = 24, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
plt.colorbar()
#plt.xlim(-1.25,1.25)
#plt.ylim(-1.25,1.25)
# plt.legend()
#plt.savefig('n_k_Stripe_T0.eps')
plt.show()

