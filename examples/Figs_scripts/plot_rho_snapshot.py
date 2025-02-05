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

def calculate_field_average(field_data, Nx, Ny, N_samples_to_avg): # assumes cubic/square mesh 
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
    print('Averaging ' + str(int(len(sample_arrays))) + ' samples')
    averaged_data += np.mean(sample_arrays, axis=0) # axis=0 calculates element-by-element mean
    # Calculate the standard error 
    std_errs = np.zeros(len(sample_arrays[0]))
    std_errs += sem(sample_arrays, axis=0)
    return averaged_data, std_errs


# Script to load and plot correlation data 
# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['simulation']['Nx'] 
Ny = params['simulation']['Ny'] 
d = params['system']['Dim']
T = 1./np.float(params['system']['beta']) 
kappa = params['system']['kappa']

# import the data
Sx_file = 'density1.dat'
Sz_file = 'density2.dat'
cols_x = np.loadtxt(Sx_file, unpack=True)
cols_y = np.loadtxt(Sz_file, unpack=True)

#x = cols_x[0]
#y = cols_x[1]
# Extract 1 set of x and y column data 
x = cols_x[0][0:Nx*Ny]
y = cols_x[1][0:Nx*Ny]

S_x_real = cols_x[2]
S_x_imag = cols_x[3]

S_z_real = cols_y[2]
S_z_imag = cols_y[3]

list_x = np.unique(x)
list_y = np.unique(y)

N_samples = int(len(S_x_real)/(Nx*Ny))

print('Total number of samples: ' + str(int(N_samples)))

Sx_vector = np.zeros(len(x), dtype=np.complex_)
Sz_vector = np.zeros(len(x), dtype=np.complex_)

# Calculate N = 2 samples 
# Average the data
if(N_samples == 1):
  Sx_vector += S_x_real + 1j*S_x_imag 
  Sz_vector += S_z_real + 1j*S_z_imag 
else:
  pcnt_avg = 3./N_samples 
  Sx_vector, Sx_errs = calculate_field_average(S_x_real + 1j*S_x_imag, Nx, Ny, int(pcnt_avg * N_samples))   
  Sz_vector, Sz_errs = calculate_field_average(S_z_real + 1j*S_z_imag, Nx, Ny, int(pcnt_avg * N_samples))   


# Store a 2D array of <Sx, Sy> data
rho_data = {'x': x, 'y': y, 'rho_up': Sx_vector, 'rho_dwn': Sz_vector}
d_frame_rho = pd.DataFrame.from_dict(rho_data)

# Sort the data  
d_frame_rho.sort_values(by=['x', 'y'], ascending = True, inplace=True) 
assert(len(list_x) == Nx)


# Redefine numpy array post sorting
Sx_sorted = np.array(d_frame_rho['rho_up']) 
Sx_sorted.resize(Nx, Ny)
Sx_sorted = np.transpose(Sx_sorted)
Sx_sorted = np.flip(Sx_sorted, 0)

Sz_sorted = np.array(d_frame_rho['rho_dwn']) 
Sz_sorted.resize(Nx, Ny)
Sz_sorted = np.transpose(Sz_sorted)
Sz_sorted = np.flip(Sz_sorted, 0)


density_up = Sx_sorted
density_dwn = Sz_sorted

plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style.txt')

# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

avg_rho = np.mean(density_up.real)

print('average density: ' + str(np.round(avg_rho,3)))

plt.figure(figsize=(3.38583, 3.38583))
plt.imshow(density_up.real/avg_rho, cmap = 'magma', interpolation='none', extent=[np.min(x) ,np.max(x) ,np.min(y),np.max(y)]) 
#plt.colorbar()
plt.clim(0, np.max(density_up.real/avg_rho))
#plt.savefig('Fig1_mu_86mm.eps', dpi = 300)
plt.show()

plt.figure(figsize=(3.38583, 3.38583))
plt.imshow(density_dwn.real/np.mean(density_dwn.real), cmap = 'magma', interpolation='none', extent=[np.min(x) ,np.max(x) ,np.min(y),np.max(y)]) 
#plt.colorbar()
plt.clim(0, np.max(density_dwn.real/np.mean(density_dwn.real)))
#plt.savefig('Fig1_mu_86mm.eps', dpi = 300)
plt.show()


total_density = density_up.real + density_dwn.real

plt.figure(figsize=(3.38583, 3.38583))
plt.imshow(total_density/np.mean(total_density), cmap = 'magma', interpolation='none', extent=[np.min(x) ,np.max(x) ,np.min(y),np.max(y)]) 
#plt.colorbar()
plt.clim(0, np.max(total_density/np.mean(total_density)))
#plt.clim(0, 4.4)
#plt.savefig('Fig1_mu_86mm.eps', dpi = 300)
plt.show()



