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
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import seaborn as sns

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

print('Averaging: ' + str(int(pcnt * N_samples)) + ' samples')

Sx_data = {'x': x, 'y': y, 'Sx': Sx_vector}
Sy_data = {'x': x, 'y': y, 'Sy': Sy_vector}
Sz_data = {'x': x, 'y': y, 'Sz': Sz_vector}
S_data = {'x': x, 'y': y, 'Sx': Sx_vector, 'Sy': Sy_vector, 'Sz': Sz_vector}

d_frame_S = pd.DataFrame.from_dict(S_data)
 #d_frame_Sx = pd.DataFrame.from_dict(Sx_data)
 #d_frame_Sy = pd.DataFrame.from_dict(Sy_data)
 #d_frame_Sz = pd.DataFrame.from_dict(Sz_data)


print('Sorting the data frame into ascending order')


# Sort the data  
d_frame_S.sort_values(by=['x', 'y'], ascending = True, inplace=True) 
 #d_frame_Sx.sort_values(by=['x', 'y'], ascending = True, inplace=True) 
 #d_frame_Sy.sort_values(by=['x', 'y'], ascending = True, inplace=True) 
 #d_frame_Sz.sort_values(by=['x', 'y'], ascending = True, inplace=True) 



# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
#Nx = params['simulation']['Nx'] 

assert(len(list_x) == Nx)

# Redefine numpy array post sorting
Sx_sorted = np.array(d_frame_S['Sx']) 
Sx_sorted.resize(Nx, Ny)
Sx_sorted = np.transpose(Sx_sorted)
Sx_sorted = np.flip(Sx_sorted, 0)

Sy_sorted = np.array(d_frame_S['Sy']) 
Sy_sorted.resize(Nx, Ny)
Sy_sorted = np.transpose(Sy_sorted)
Sy_sorted = np.flip(Sy_sorted, 0)

Sz_sorted = np.array(d_frame_S['Sz']) 
Sz_sorted.resize(Nx, Ny)
Sz_sorted = np.transpose(Sz_sorted)
Sz_sorted = np.flip(Sz_sorted, 0)

# Create 2D array that contains 2 doubles at each grid point to host the spin vector 
# Likely unnecessary: 
#Spin_vector_data = np.zeros((len(list_x),len(list_y),2), dtype=np.complex_) 

# Fill with Sx and Sy data 
#Spin_vector_data[:][:][0] += Sx_sorted
#Spin_vector_data[:][:][1] += Sy_sorted
#Sx_sorted /= np.sqrt(Sx_sorted**2 + Sy_sorted**2 + Sz_sorted**2)
#Sy_sorted /= np.sqrt(Sx_sorted**2 + Sy_sorted**2 + Sz_sorted**2)
#Sx_sorted /= np.sqrt(Sx_sorted**2 + Sy_sorted**2) 
#Sy_sorted /= np.sqrt(Sx_sorted**2 + Sy_sorted**2)
#Sz_sorted /= np.sqrt(Sx_sorted**2 + Sy_sorted**2 + Sz_sorted**2)

planar_norm = np.sqrt(Sx_sorted.real**2 + Sy_sorted.real**2)
total_norm = np.sqrt(Sx_sorted.real**2 + Sy_sorted.real**2 + Sz_sorted.real**2)

# Normalize vector components by Sqrt(Sx^2 + Sy^2)
 #Spin_vector_data[:][:][0] /= np.sqrt(Sx_sorted + Sy_sorted)
 #Spin_vector_data[:][:][1] /= np.sqrt(Sx_sorted + Sy_sorted)

np.savetxt('X_data.dat', list_x)
np.savetxt('Y_data.dat', list_y)
np.savetxt('spinX_Data.dat', Sx_sorted.real / total_norm)
np.savetxt('spinY_Data.dat', Sy_sorted.real / total_norm)
np.savetxt('spinZ_Data.dat', Sz_sorted.real / total_norm)


# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Plot the structure factor 

 #plt.style.use('~/csbosonscpp/tools/python_scripts/plot_style.txt')
 ##K Plot the S(kx) distribution in k-space  
 #plt.figure(1)
 ##plt.quiver(list_x, list_y, Sx_sorted.real, Sz_sorted.real, color = 'b', units = 'xy', scale = 1.)
 ##plt.imshow(Sz_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(list_x) ,np.max(list_x) ,np.min(list_y),np.max(list_y)]) 
 ##plt.quiver(list_x, list_y, Sx_sorted.real, Sy_sorted.real, Sz_sorted.real, units = 'xy', scale = 2.50)
 ##plt.quiver(list_x, list_y, Sx_sorted.real, Sy_sorted.real, Sz_sorted.real/np.max(Sz_sorted.real), width=.01,linewidth=0.5) 
 #plt.quiver(list_x, list_y, Sx_sorted.real, Sy_sorted.real, Sz_sorted.real/np.max(Sz_sorted.real), units = 'xy', scale = 1.2, cmap='jet')
 ##plt.imshow(Sz_sorted.real/np.max(Sz_sorted.real))
 #plt.title('Spin vector field : ' + r'$<S_{x} , S_{y}>$ ', fontsize = 20)
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 #plt.colorbar(label = r'$\langle S_{z} \rangle$')
 ##plt.xlim(-1, 1)
 ##plt.legend()
 #plt.show()

sns_cmap = ListedColormap(sns.color_palette("RdBu", 256)) 
plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style.txt')
#plt.figure(1)
plt.figure(figsize=(3.38583*2, 3.38583*2))
#plt.quiver(list_x, list_y, Sx_sorted.real, Sy_sorted.real, Sz_sorted.real, units = 'xy', scale = 1.0, cmap='jet')
#plt.quiver(Sx_sorted.real/total_norm, Sy_sorted.real/total_norm, Sz_sorted.real/total_norm, units = 'xy', scale = 1.0, cmap=sns_cmap)
plt.quiver(Sx_sorted.real/total_norm, Sy_sorted.real/total_norm, Sz_sorted.real/total_norm, units = 'xy', cmap=sns_cmap)
#plt.quiver(list_x, list_y, Sx_sorted.real/total_norm, Sy_sorted.real/total_norm, Sz_sorted.real/total_norm, units='xy', scale = dx, cmap=sns_cmap)
#plt.title('Spin field : $<S_{x} , S_{y}>$ ', fontsize = 20)
 #plt.xlabel(r'$\tilde x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel(r'$\tilde y$', fontsize = 20, fontweight = 'bold')
plt.colorbar(label = r'$\langle S_{z} \rangle$')
plt.clim(-1.0, 1.0)
#plt.xlim(-1, 1)
#plt.legend()
plt.show()
