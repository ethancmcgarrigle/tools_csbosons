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
    print('Averaging ' + str(int(len(sample_arrays))) + ' samples')
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
T_ = 1./float(params['system']['beta']) 
d = params['system']['Dim']
N_samples = int(len(n_k_real)/(Nx*Ny))

kx = cols[0][0:Nx*Ny]
ky = cols[1][0:Nx*Ny]
kx_unique = np.unique(kx)
ky_unique = np.unique(ky)

#n_k = np.zeros(len(np.unique(k_x)), dtype=np.complex_)
pcnt_avg = 0.70
n_k = np.zeros(len(kx), dtype=np.complex_)
n_k, n_k_errs = calculate_field_average(n_k_real + 1j * n_k_imag, Nx, Ny, d, int(pcnt_avg * N_samples))

_data = {'kx': kx, 'ky': ky, 'n_k': n_k, 'n_k_errs': n_k_errs}
d_frame = pd.DataFrame.from_dict(_data)


print('Sorting the data frame into ascending order')

d_frame.sort_values(by=['kx', 'ky'], ascending = True, inplace=True) 


# Find ky = 0 element to get n(kx) distribution
n_kx = np.zeros(len(kx_unique), dtype=np.complex_)
n_kx_errs = np.zeros(len(kx_unique), dtype=np.complex_)
n_ky = np.zeros(len(kx_unique), dtype=np.complex_)
n_ky_errs = np.zeros(len(kx_unique), dtype=np.complex_)

 #print('Averaging over the remaining dimensions for n(kx)')
 #
 #for i, uniq_kx in enumerate(kx_unique):
 #  # for each unique kx value,  Loop over other dimensions and average them 
 #  # 1. Get the list of indices of the unique kx value 
 #  tmp_frame = (d_frame['kx'] == uniq_kx)
 #  #indices = np.array(np.where(tmp_frame == True)) 
 #  indices = (np.where(tmp_frame == True)) 
 #  indices = indices[0] # 0th element is the list of true indices 
 #  #print(indices)
 #
 #  # pick a random index and check to make sure it is yielding the correct kx 
 #  #print(d_frame['kx'].iloc[indices[0]]) 
 #  #print(uniq_kx)
 #  assert(d_frame['kx'].iloc[indices[0]] == uniq_kx)
 #  # 2. Average the structure factor with all of those kx values 
 #  n_kx[i] = np.mean(d_frame['n_k'].iloc[indices])
 #  #n_kx_errs[i] = np.mean(d_frame['n_k_errs'].iloc[indices]) # incorrect, but a good proxy 
 #  n_kx_errs[i] = calc_err_average(d_frame['n_k_errs'].iloc[indices]) 
 #  # repeat 
 #
 ## repeat for ky 
 #for i, uniq_ky in enumerate(ky_unique):
 #  # for each unique ky value,  Loop over other dimensions and average them 
 #  # 1. Get the list of indices of the unique ky value 
 #  tmp_frame = (d_frame['ky'] == uniq_ky)
 #  #indices = np.array(np.where(tmp_frame == True)) 
 #  indices = (np.where(tmp_frame == True)) 
 #  indices = indices[0] # 0th element is the list of true indices 
 #  #print(indices)
 #
 #  # pick a random indey and check to make sure it is yielding the correct ky 
 #  #print(d_frame['ky'].iloc[indices[0]]) 
 #  #print(uniq_ky)
 #  assert(d_frame['ky'].iloc[indices[0]] == uniq_ky)
 #  # 2. Average the structure factor with all of those ky values 
 #  n_ky[i] = np.mean(d_frame['n_k'].iloc[indices])
 #  #n_ky_errs[i] = np.mean(d_frame['n_k_errs'].iloc[indices]) # incorrect, good proxy/placeholder   
 #  n_ky_errs[i] = calc_err_average(d_frame['n_k_errs'].iloc[indices]) 
 #  # repeat 


# Extract n(kx)
tmp_frame = (d_frame['ky'] == 0.)
#indices = np.array(np.where(tmp_frame == True)) 
indices = (np.where(tmp_frame == True)) 
indices = indices[0] # 0th element is the list of true indices 
#print(uniq_ky)
assert(d_frame['ky'].iloc[indices[0]] == 0.)
# 2. Extract S or n(kx) evalutaed at ky == 0 
n_kx = d_frame['n_k'].iloc[indices]
  #n_ky_errs[i] = np.mean(d_frame['n_k_errs'].iloc[indices]) # incorrect, good proxy/placeholder   
#n_kx_errs = calc_err_average(d_frame['n_k_errs'].iloc[indices]) 
n_kx_errs = d_frame['n_k_errs'].iloc[indices] 
  # repeat 
n_kx = n_kx.values
n_kx_errs = n_kx_errs.values



# Extract n(ky)
tmp_frame = (d_frame['kx'] == 0.)
#indices = np.array(np.where(tmp_frame == True)) 
indices = (np.where(tmp_frame == True)) 
indices = indices[0] # 0th element is the list of true indices 
#print(uniq_ky)
assert(d_frame['kx'].iloc[indices[0]] == 0.)
# 2. Extract S or n(kx) evalutaed at ky == 0 
n_ky = d_frame['n_k'].iloc[indices]
n_ky_errs = d_frame['n_k_errs'].iloc[indices] 
  # repeat 
n_ky = n_ky.values
n_ky_errs = n_ky_errs.values



# Redefine numpy array post sorting
n_k_sorted = np.array(d_frame['n_k']) 
n_k_sorted.resize(Nx, Ny)
n_k_sorted = np.transpose(n_k_sorted)

# import the input parameters, specifically the i and j indices 



plt.style.use('~/CSBosonsCpp/tools/python_scripts/plot_style.txt')

# HARD CODE, artifically set ky coords to zero 

#n_k_sorted = np.transpose(n_k_sorted)
# Plot the N_k distribution in k-space  
plt.figure(1)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(n_k_sorted.real/np.sum(n_k_sorted.real), cmap = 'hot', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
plt.title('$n(k)/N$ Isotropic SOC, ' + r'$\tilde \kappa = ' + str(_kappa) + '$ , ' + r'$\tilde T = ' + str(np.round(T_, 2)) + '$', fontsize = 22)
plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
plt.ylabel('$k_y$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
plt.xlim(-18,18)
plt.ylim(-18,18)
plt.colorbar()
# plt.legend()
#plt.savefig('n_k_mixed_1K.eps')
plt.show()



plt.figure(2)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
#plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
#plt.plot(kx_list, S_k_sorted[:,0].real, 'r', label = '$S(k_{x})$')
#n_k_sorted = np.transpose(n_k_sorted)
plt.errorbar(np.unique(kx), n_kx.real/np.sum(n_k_sorted.real), n_kx_errs.real/np.sum(n_k_sorted.real), marker='o', markersize = 6, elinewidth=0.25, linewidth = 0.25, color = 'blue', label='$n(k_{x}, k_{y}=0)$')
plt.errorbar(np.unique(ky), n_ky.real/np.sum(n_k_sorted.real), n_ky_errs.real/np.sum(n_k_sorted.real), marker='o', markersize = 6, elinewidth=0.25, linewidth = 0.25, color = 'red', label='$n(k_{x}=0, k_{y})$')
plt.title('Momentum Distribution, ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 16)
plt.xlabel('$k_{\mu}$', fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$n(k_{\mu})/N $', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
#plt.xlim(-1, 1)
#plt.savefig('n_k_plane_wave_1D.eps')
plt.legend()
plt.show()


# Calculate angular average 
kr = np.sqrt(kx**2 + ky**2)
theta = np.arctan(ky/kx) # rads 

kr_uniq = np.unique(kr)

n_kr = np.zeros(len(kr_uniq), dtype=np.complex_)
n_kr_errs = np.zeros(len(kr_uniq), dtype=np.complex_)

_polar_data = {'kr': kr, 'theta': theta, 'n_k': n_k, 'n_k_errs': n_k_errs}
polar_d_frame = pd.DataFrame.from_dict(_polar_data)
polar_d_frame.sort_values(by=['kr'], ascending = True, inplace=True) 


n_kr[0] += polar_d_frame['n_k'].iloc[0]
n_kr_errs[0] += polar_d_frame['n_k_errs'].iloc[0]
i = 0
print(kr[0])
for kr_ in kr_uniq[1:len(kr_uniq)]:
  i += 1
  tmp_frame = (polar_d_frame['kr'] == kr_)
  indices = np.where(tmp_frame == True)[0] 
  #indices = indices[0] # 0th element is the list of true indices 
  assert(polar_d_frame['kr'].iloc[indices[0]] == kr_)
  # 2. Extract 
  n_kr[i] += polar_d_frame['n_k'].iloc[indices].mean()
  # propagate error across the average 
  n_kr_errs[i] += calc_err_average(polar_d_frame['n_k_errs'].iloc[indices].values) 



# Plot angular average 
 #plt.figure(3)
 #plt.errorbar(kr_uniq, n_kr.real/np.sum(n_k_sorted.real), n_kr_errs.real/np.sum(n_k_sorted.real), marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='Langevin')
 #plt.title('Polar Averaged Momentum Distribution, ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 22)
 #plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
 #plt.ylabel(r'$n(k_{r})/N $', fontsize = 24, fontweight = 'bold')
 #plt.axvline(x = _kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$\tilde{\kappa} = ' + str(_kappa) + '$')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 ##plt.xlim(-1, 1)
 ##plt.savefig('n_k_plane_wave_1D.eps')
 #plt.legend()
 #plt.show()
 #
 #
 #
 ## Plot angular average 
 #plt.figure(4)
 #plt.errorbar(kr_uniq, n_kr.real, n_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='Langevin')
 #plt.title('Polar Averaged Momentum Distribution, ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 22)
 #plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
 #plt.ylabel(r'$N(k_{r})$', fontsize = 24, fontweight = 'bold')
 #plt.axvline(x = _kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$\tilde{\kappa} = ' + str(_kappa) + '$')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 ##plt.xlim(-1, 1)
 ##plt.savefig('n_k_plane_wave_1D.eps')
 #plt.legend()
 #plt.show()



