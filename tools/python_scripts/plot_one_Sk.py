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



with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['simulation']['Nx'] 
Ny = params['simulation']['Ny'] 
dim = params['system']['Dim']

#N_samples = int(len(n_k_real)/(Nx*Ny))
# Script to load and plot correlation data 
Sk = np.loadtxt('S_k_00_1sample.dat', unpack=True)
rho_k_0 = np.loadtxt('rho_k_0_one.dat', unpack=True)
rho_negk_0 = np.loadtxt('rho_-k_0_one.dat', unpack=True)


# Need to average the real and imaginary parts of each operator across the simulation  
# for 1 dataset
Sk_data = Sk[2*(dim)] + 1j*Sk[2*(dim) + 1]
rho_k_data_0 = rho_k_0[2*(dim)] + 1j*rho_k_0[2*(dim) + 1]
rho_negk_data_0 = rho_negk_0[2*(dim)] + 1j*rho_negk_0[2*(dim) + 1]

Structure_factor = np.zeros(len(Sk_data), dtype=np.complex_)
Structure_factor += Sk_data 
Structure_factor -= (rho_k_data_0 * rho_negk_data_0)

print('Max structure factor value: ' + str(np.max(Structure_factor.real)))
k_x = Sk[0]
k_y = Sk[1]
if(dim > 2):
  k_z = Sk[2]
  N_samples = int(len(k_x)/(Nx*Ny*Nz))
elif(dim == 2):
  N_samples = int(len(k_x)/(Nx*Ny))


k_x = np.split(k_x, N_samples)
k_y = np.split(k_y, N_samples)
kx = k_x[0] 
ky = k_y[0] 
if(dim > 2):
  k_z = np.split(k_z, N_samples)
  kz = k_z[0]

if(dim > 2):
  data = {'kx': kx, 'ky': ky, 'kz': kz, 'S_k': Structure_factor} 
else:
  data = {'kx': kx, 'ky': ky, 'S_k': Structure_factor} 

d_frame = pd.DataFrame.from_dict(data)

#d_frame.sort_values(by=['kx', 'ky','kz'], ascending = True, inplace=True)
d_frame.sort_values(by=['kx', 'ky'], ascending = True, inplace=True)

kx_unique = np.unique(kx)
ky_unique = np.unique(ky)

# Redefine numpy array post sorting
ctr = 1

# Redefine numpy array post sorting
S_k_sorted = np.array(d_frame['S_k']) 
S_k_sorted.resize(Nx, Ny)
S_k_sorted = np.transpose(S_k_sorted)
S_k_sorted = np.flip(S_k_sorted, 0)

#plt.style.use('~/CSBosonsCpp/tools/python_scripts/plot_style.txt')
#plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style.txt')
plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_spins.txt')

print('Max S(k) value: ' + str(np.round(np.max(S_k_sorted), 4)))



plt.figure(figsize=(6.77166, 6.77166))
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
# normalize it by k=0 peak or max?
plt.imshow(S_k_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
#plt.title(r'$S_{\alpha \alpha} (k)$', fontsize = 30)
plt.title(r'$S_{\alpha \alpha} (\mathbf{k})$', fontsize = 30)
plt.xlabel('$k_x$', fontsize = 32)
plt.ylabel('$k_y$', fontsize = 32)
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.xlim(-24,24)
#plt.ylim(-24,24)
#plt.colorbar()
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
# plt.legend()



