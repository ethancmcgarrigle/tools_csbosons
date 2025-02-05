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



# Script to load and plot correlation data 
# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['simulation']['Nx'] 
Ny = params['simulation']['Ny'] 
d = params['system']['Dim']

# import the data
Sx_file = 'phase_0.dat'
Sz_file = 'phase_1.dat'
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


Sx_vector = np.zeros(len(x), dtype=np.complex_)
Sz_vector = np.zeros(len(x), dtype=np.complex_)

# Average the data 
pcnt_avg = 0.1
Sx_vector, Sx_errs = calculate_field_average(S_x_real + 1j*S_x_imag, Nx, Ny, d, int(pcnt_avg * N_samples))   
Sz_vector, Sz_errs = calculate_field_average(S_z_real + 1j*S_z_imag, Nx, Ny, d, int(pcnt_avg * N_samples))   

print('N samples averaging ' + str(int(pcnt_avg * N_samples)) + ' ')

 #Sx_vector += S_x_real + 1j * S_x_imag 
 #Sz_vector += S_z_real + 1j * S_z_imag 

# Store a 2D array of <Sx, Sy> data
 #Sx_data = {'x': x, 'y': y, 'Sx': Sx_vector}
 #Sz_data = {'x': x, 'y': y, 'Sz': Sz_vector}
Sx_data = {'x': x, 'y': y, 'Sx': Sx_vector}
Sz_data = {'x': x, 'y': y, 'Sz': Sz_vector}
d_frame_Sx = pd.DataFrame.from_dict(Sx_data)
d_frame_Sz = pd.DataFrame.from_dict(Sz_data)


print('Sorting the data frame into ascending order')


# Sort the data  
d_frame_Sx.sort_values(by=['x', 'y'], ascending = True, inplace=True) 
d_frame_Sz.sort_values(by=['x', 'y'], ascending = True, inplace=True) 



# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
#Nx = params['simulation']['Nx'] 

assert(len(list_x) == Nx)


# Redefine numpy array post sorting
Sx_sorted = np.array(d_frame_Sx['Sx']) 
Sx_sorted.resize(Nx, Ny)
Sx_sorted = np.transpose(Sx_sorted)


Sz_sorted = np.array(d_frame_Sz['Sz']) 
Sz_sorted.resize(Nx, Ny)
Sz_sorted = np.transpose(Sz_sorted)

# Phase \theta(x,y) data: want to plot 1 * exp(i * theta)  

# Vector components 
# x = R cos(theta)
# y = R sin(theta) 
# Up-phase 
x_phase_up = 1. * np.cos(Sx_sorted)
y_phase_up = 1. * np.sin(Sx_sorted)
# Down-phase 
x_phase_dwn = 1. * np.cos(Sz_sorted)
y_phase_dwn = 1. * np.sin(Sz_sorted)
# total-phase 
x_phase_tot = 1. * np.cos(Sx_sorted + Sz_sorted)
y_phase_tot = 1. * np.sin(Sx_sorted + Sz_sorted)
# Phase difference (up - down) 
x_phase_diff = 1. * np.cos(Sx_sorted - Sz_sorted)
y_phase_diff = 1. * np.sin(Sx_sorted - Sz_sorted)


# Plot the structure factor 

ctr = 1
#K Plot the S(kx) distribution in k-space  
plt.figure(ctr)
#plt.quiver(list_x, list_y, Sx_sorted.real, Sz_sorted.real, color = 'b', units = 'xy', scale = 1.)
plt.quiver(list_x, list_y, x_phase_up.real, y_phase_up.real, color = 'b', units = 'xy', scale = 1.25)
plt.title('Up-Phase vector field : ' + r'$<cos(\theta (x, y)) , sin( \theta (x, y) ) >$ ', fontsize = 16)
plt.xlabel(r'$\tilde x$', fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$\tilde y$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
#plt.xlim(-1, 1)
#plt.legend()
plt.show()
ctr += 1


plt.figure(ctr)
#plt.quiver(list_x, list_y, Sx_sorted.real, Sz_sorted.real, color = 'b', units = 'xy', scale = 1.)
plt.quiver(list_x, list_y, x_phase_dwn.real, y_phase_dwn.real, color = 'b', units = 'xy', scale = 1.25)
plt.title('Down-Phase vector field : ' + r'$<cos(\theta (x, y)) , sin( \theta (x, y) ) >$ ', fontsize = 16)
plt.xlabel(r'$\tilde x$', fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$\tilde y$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
#plt.xlim(-1, 1)
#plt.legend()
plt.show()
ctr += 1


plt.figure(ctr)
#plt.quiver(list_x, list_y, Sx_sorted.real, Sz_sorted.real, color = 'b', units = 'xy', scale = 1.)
plt.quiver(list_x, list_y, x_phase_tot.real, y_phase_tot.real, color = 'b', units = 'xy', scale = 1.25)
plt.title('Total-Phase vector field : ' + r'$<cos(\theta (x,y)) , sin( \theta (x, y) ) >$ ', fontsize = 16)
plt.xlabel(r'$\tilde x$', fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$\tilde y$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
#plt.xlim(-1, 1)
#plt.legend()
plt.show()
ctr += 1

plt.figure(ctr)
#plt.quiver(list_x, list_y, Sx_sorted.real, Sz_sorted.real, color = 'b', units = 'xy', scale = 1.)
plt.quiver(list_x, list_y, x_phase_diff.real, y_phase_diff.real, color = 'b', units = 'xy', scale = 1.25)
plt.title('Phase-Difference vector field : ' + r'$<cos(\theta (x,y)) , sin( \theta (x, y) ) >$ ', fontsize = 16)
plt.xlabel(r'$\tilde x$', fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$\tilde y$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
#plt.xlim(-1, 1)
#plt.legend()
plt.show()
ctr += 1
