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
Sx_file = 'Sx.dat'
Sy_file = 'Sy.dat'
Sz_file = 'Sz.dat'
cols_x = np.loadtxt(Sx_file, unpack=True)
cols_y = np.loadtxt(Sy_file, unpack=True)
cols_z = np.loadtxt(Sz_file, unpack=True)

#x = cols_x[0]
#y = cols_x[1]
# Extract 1 set of x and y column data 
x = cols_x[0][0:Nx*Ny]
y = cols_x[1][0:Nx*Ny]

S_x_real = cols_x[2]
S_x_imag = cols_x[3]

S_y_real = cols_y[2]
S_y_imag = cols_y[3]

S_z_real = cols_z[2]
S_z_imag = cols_z[3]

list_x = np.unique(x)
list_y = np.unique(y)

N_samples = int(len(S_x_real)/(Nx*Ny))


Sx_vector = np.zeros(len(x), dtype=np.complex_)
Sy_vector = np.zeros(len(x), dtype=np.complex_)
Sz_vector = np.zeros(len(x), dtype=np.complex_)

# Average the data 
pcnt = 1./N_samples
Sx_vector, Sx_errs = calculate_field_average(S_x_real + 1j*S_x_imag, Nx, Ny, d, int(pcnt * N_samples))   
Sy_vector, Sy_errs = calculate_field_average(S_y_real + 1j*S_y_imag, Nx, Ny, d, int(pcnt * N_samples))   
Sz_vector, Sz_errs = calculate_field_average(S_z_real + 1j*S_z_imag, Nx, Ny, d, int(pcnt * N_samples))   


Sx_data = {'x': x, 'y': y, 'Sx': Sx_vector}
Sy_data = {'x': x, 'y': y, 'Sy': Sy_vector}
Sz_data = {'x': x, 'y': y, 'Sz': Sz_vector}
d_frame_Sx = pd.DataFrame.from_dict(Sx_data)
d_frame_Sy = pd.DataFrame.from_dict(Sy_data)
d_frame_Sz = pd.DataFrame.from_dict(Sz_data)


print('Sorting the data frame into ascending order')


# Sort the data  
d_frame_Sx.sort_values(by=['x', 'y'], ascending = True, inplace=True) 
d_frame_Sy.sort_values(by=['x', 'y'], ascending = True, inplace=True) 
d_frame_Sz.sort_values(by=['x', 'y'], ascending = True, inplace=True) 



# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
#Nx = params['simulation']['Nx'] 

assert(len(list_x) == Nx)


# Redefine numpy array post sorting
Sx_sorted = np.array(d_frame_Sx['Sx']) 
Sx_sorted.resize(Nx, Ny)
Sx_sorted = np.transpose(Sx_sorted)

Sy_sorted = np.array(d_frame_Sy['Sy']) 
Sy_sorted.resize(Nx, Ny)
Sy_sorted = np.transpose(Sx_sorted)

Sz_sorted = np.array(d_frame_Sz['Sz']) 
Sz_sorted.resize(Nx, Ny)
Sz_sorted = np.transpose(Sz_sorted)

# Create 2D array that contains 2 doubles at each grid point to host the spin vector 
# Likely unnecessary: 
#Spin_vector_data = np.zeros((len(list_x),len(list_y),2), dtype=np.complex_) 

# Fill with Sx and Sy data 
#Spin_vector_data[:][:][0] += Sx_sorted
#Spin_vector_data[:][:][1] += Sy_sorted
#Sx_sorted /= np.sqrt(Sx_sorted**2 + Sy_sorted**2 + Sz_sorted**2)
#Sy_sorted /= np.sqrt(Sx_sorted**2 + Sy_sorted**2 + Sz_sorted**2)
Sx_sorted /= np.sqrt(Sx_sorted**2 + Sy_sorted**2) 
Sy_sorted /= np.sqrt(Sx_sorted**2 + Sy_sorted**2)
Sz_sorted /= np.sqrt(Sx_sorted**2 + Sy_sorted**2 + Sz_sorted**2)


# Normalize vector components by Sqrt(Sx^2 + Sy^2)
 #Spin_vector_data[:][:][0] /= np.sqrt(Sx_sorted + Sy_sorted)
 #Spin_vector_data[:][:][1] /= np.sqrt(Sx_sorted + Sy_sorted)




# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Plot the structure factor 

plt.style.use('~/csbosonscpp/tools/python_scripts/plot_style.txt')
#K Plot the S(kx) distribution in k-space  
plt.figure(1)
#plt.quiver(list_x, list_y, Sx_sorted.real, Sz_sorted.real, color = 'b', units = 'xy', scale = 1.)
#plt.imshow(Sz_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(list_x) ,np.max(list_x) ,np.min(list_y),np.max(list_y)]) 
#plt.quiver(list_x, list_y, Sx_sorted.real, Sy_sorted.real, Sz_sorted.real, units = 'xy', scale = 2.50)
#plt.quiver(list_x, list_y, Sx_sorted.real, Sy_sorted.real, Sz_sorted.real/np.max(Sz_sorted.real), width=.01,linewidth=0.5) 
plt.quiver(list_x, list_y, Sx_sorted.real, Sy_sorted.real, Sz_sorted.real/np.max(Sz_sorted.real), units = 'xy', scale = 1.2, cmap='jet')
#plt.imshow(Sz_sorted.real/np.max(Sz_sorted.real))
plt.title('Spin vector field : ' + r'$<S_{x} , S_{y}>$ ', fontsize = 20)
plt.xlabel(r'$\tilde x$', fontsize = 26, fontweight = 'bold')
plt.ylabel(r'$\tilde y$', fontsize = 26, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
plt.colorbar(label = r'$\langle S_{z} \rangle$')
#plt.clim(-0.5, 0.5)
#plt.xlim(-1, 1)
#plt.legend()
plt.show()

 #plt.figure(2)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #plt.plot(kx_list, S_k_yavgd.imag, 'r', label = '$S(k_{x})$')
 #plt.title('Structure Factor: $S(k_{x})$ (y-averaged)', fontsize = 16)
 #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Im($S(k_{x})$)', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 #
 #

#Plot a y-slice of the S(k_x) structure factor 
 #plt.figure(3)
 #plt.plot(kx_list, S_k_sorted[:][1].real, 'r', label = '$S(k_{x})$')
 #plt.title('Structure Factor: $S(k_{x})$ (y-averaged)', fontsize = 16)
 #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Re($S(k_{x})$)', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
