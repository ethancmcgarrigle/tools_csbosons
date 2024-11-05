import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
# matplotlib.use('TkAgg')
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


# import the data
 #ops_file = 'S_k_tot_cp.dat'
 #Sk = np.loadtxt('S_k_tot_cp.dat', unpack=True)
 #rho_k = np.loadtxt('rho_k_cp.dat', unpack=True)
 #rho_negk = np.loadtxt('rho_-k_cp.dat', unpack=True)
Sk = np.loadtxt('S_k_tot.dat', unpack=True)
rho_k = np.loadtxt('rho_k.dat', unpack=True)
rho_negk = np.loadtxt('rho_-k.dat', unpack=True)

# Need to average the real and imaginary parts of each operator across the simulation  
# for 1 dataset 
Sk_data = Sk[4] + 1j*Sk[5]
rho_k_data = rho_k[4] + 1j*rho_k[5]
rho_negk_data = rho_negk[4] + 1j*rho_negk[5]

corr_avg, corr_err = calculate_field_average(Sk_data, Nx, Ny, 2, int(len(Sk_data) * 0.80))
rho_k_avg, rho_k_err = calculate_field_average(rho_k_data, Nx, Ny, 2, int(len(Sk_data) * 0.80))
rho_negk_avg, rho_negk_err = calculate_field_average(rho_negk_data, Nx, Ny, 2, int(len(Sk_data) * 0.80))

Structure_factor = np.zeros(len(corr_avg), dtype=np.complex_)
Structure_factor += corr_avg 
Structure_factor -= (rho_k_avg * rho_negk_avg)


#comp = Sk[4] + Sk[5]*1j - ((rho_k[4] + 1j*rho_k[5]) * (rho_negk[4] + 1j*rho_negk[5]))
#reals = Sk[4]  - (rho_k[4] * rho_negk[4])

k_x = Sk[0]
k_y = Sk[1]

# Need only 1 of the kx and ky arrays (There are N_sample copies of them)
N_samples = int(len(k_x)/(Nx*Ny))
print('assuming 2D geometry')
k_x = np.split(k_x, N_samples)
k_y = np.split(k_y, N_samples)
kx = k_x[0] 
ky = k_y[0] 

#kx_unique = np.unique(kx)
#ky_unique = np.unique(ky)

data = {'kx': kx, 'ky': ky, 'S_k': Structure_factor}
d_frame = pd.DataFrame.from_dict(data)


d_frame.sort_values(by=['kx', 'ky'], ascending = True, inplace=True)

print(d_frame['kx'])
print(d_frame['ky'])

# Redefine numpy array post sorting
S_k_sorted = np.array(d_frame['S_k'])
S_k_sorted.resize(Nx, Ny)
S_k_sorted = np.transpose(S_k_sorted)


kx_list = np.unique(k_x)
ky_list = np.unique(k_y)
assert(len(kx_list) == Nx)

plt.style.use('~/csbosonscpp/tools/python_scripts/plot_style.txt')


plt.figure(1)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)])
plt.title('$S(k)$', fontsize = 11)
plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
plt.ylabel('$k_y$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
plt.colorbar()
# plt.legend()
plt.show()





# Take y-average of the (S(k_x))
#for i in ky:
#  S_k_sorted[:, i]


 ## Plot the structure factor 
 #plt.figure(10)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #plt.imshow(S_k_final.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #plt.title('$S(k)$', fontsize = 11)
 #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('$k_y$', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 #plt.colorbar()
 ## plt.legend()
#plt.show()



# Average over the y-coordinate to plot only S(k_x) 
#K Plot the S(kx) distribution in k-space  
plt.figure(2)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
#plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
#plt.plot(kx_list, S_k_sorted[:,0].real, 'r', label = '$S(k_{x})$')
S_k_sorted = np.transpose(S_k_sorted)
plt.plot(np.unique(kx), S_k_sorted[:,1]/np.max(S_k_sorted), '-o', linewidth = 0.25, color = 'red', label='$S(k_{x}, k_{y} = 0)$')
plt.title('Normalized Structure Factor: $S(k_{x}, k_{y} = 0)$', fontsize = 16)
plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
plt.ylabel('$S(k_{x})$', fontsize = 20, fontweight = 'bold')
#plt.savefig('S_k_Stripe_1D.eps')
plt.legend()
plt.show()
plt.xlim(-1, 1)
plt.legend()
plt.show()



#K Plot the S(kx) distribution in k-space  
plt.figure(3)
plt.plot(np.unique(kx), S_k_sorted[:,1], '-o', linewidth = 0.25, color = 'red', label='$S(k_{x}, k_{y} = 0)$')
plt.title('Stripe Phase Structure Factor: $S(k_{x}, k_{y} = 0)$', fontsize = 16)
plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
plt.ylabel('$S(k_{x})$', fontsize = 20, fontweight = 'bold')
#plt.savefig('S_k_Stripe_1D.eps')
plt.legend()
plt.show()
plt.xlim(-1, 1)
plt.legend()
plt.show()


 #plt.figure(2)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #plt.plot(kx_list, S_k_sorted[:,0].imag, 'r', label = '$S(k_{x})$')
 #plt.title('Structure Factor: $S(k_{x})$, $k_{y} = 0$', fontsize = 16)
 #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Im($S(k_{x})$)', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 # 
 #plt.figure(3)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #plt.plot(ky_list, S_k_sorted[0,:].real, 'r', label = '$S(k_{y})$')
 #plt.title('Structure Factor: $S(k_{y})$, $k_{x} = 0$', fontsize = 16)
 #plt.xlabel('$k_y$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Re($S(k_{y})$)', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 #
 #plt.figure(4)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #plt.plot(ky_list, S_k_sorted[0,:].imag, 'r', label = '$S(k_{y})$')
 #plt.title('Structure Factor: $S(k_{y})$ , $k_{x} = 0$', fontsize = 16)
 #plt.xlabel('$k_y$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Im($S(k_{y})$)', fontsize = 20, fontweight = 'bold')
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 #
 #
 #
#K Plot the S(kx) distribution in k-space  
 #plt.figure(1)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #plt.plot(kx_list, S_k_yavgd.real, 'r*', label = '$S(k_{x})$')
 #plt.title('Structure Factor: $S(k_{x})$ (y-averaged)', fontsize = 16)
 #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Re($S(k_{x})$)', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 #
 #plt.figure(2)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #plt.plot(kx_list, S_k_yavgd.imag, 'r*', label = '$S(k_{x})$')
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
 #
 ##Plot a vertical slice: of the S(k_y) structure factor 
 #plt.figure(3)
 #plt.plot(ky_list, S_k_sorted[0][:].real, 'r*', label = '$S(k_{x} = 0, k_{y})$')
 #plt.title('Structure Factor cut: $S(k_{y})$' , fontsize = 16)
 #plt.xlabel('$k_y$', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Re($S(k_{y})$)', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 #
 ##Plot S(k_y), averaged over x  
 #plt.figure(4)
 #plt.plot(ky_list, S_k_xavgd.real, 'r*', label = '$S(k_{y})$')
 #plt.title('Structure Factor: $S(k_{y})$' , fontsize = 16)
 #plt.xlabel('$k_y$', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Re($S(k_{y})$)', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 #
