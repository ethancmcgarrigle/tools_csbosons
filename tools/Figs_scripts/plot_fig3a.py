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




x = np.loadtxt('X_data.dat') 
y = np.loadtxt('Y_data.dat') 
Sx = np.loadtxt('stripe_fig3a_spinX_Data.dat') 
Sy = np.loadtxt('stripe_fig3a_spinY_Data.dat')
Sz = np.loadtxt('stripe_fig3a_spinZ_Data.dat')


# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)



plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_spins.txt')
#K Plot the S(kx) distribution in k-space  
#plt.figure(1)
plt.figure(figsize=(3.38583, 3.38583))
#plt.quiver(x,y, Sx, Sy, Sz, units = 'xy', scale = 1.2, cmap='jet') looks good 
plt.quiver(x,y, Sx, Sy, Sz, units = 'xy', scale = 1.5, cmap='jet') # looks better, let's stick with scale = 1.5 
#plt.quiver(x,y, Sx, Sy, Sz, units = 'xy', scale = 1.2, cmap='plasma')
#plt.quiver(list_x, list_y, Sx_sorted.real, Sy_sorted.real, Sz_sorted.real/np.max(Sz_sorted.real), units = 'xy', scale = 1.2, cmap='jet')
#plt.title('Spin field : $<S_{x} , S_{y}>$ ', fontsize = 20)
 #plt.xlabel(r'$\tilde x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel(r'$\tilde y$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar(label = r'$\langle S_{z} \rangle$') 
plt.colorbar().set_label(label = r'$\langle M_{z} \rangle$', size = 20)
plt.clim(-1.0,1.0)
#plt.xlim(0, 16)
#plt.ylim(0, 16)
#plt.legend()
plt.show()

