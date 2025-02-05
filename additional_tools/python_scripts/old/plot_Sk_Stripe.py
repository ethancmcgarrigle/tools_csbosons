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

def calculate_field_average(field_data, Nx, Ny, Nz, dim, N_samples_to_avg): # assumes cubic/square mesh 
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data

    # Get number of samples
    if(dim == 2): 
      N_samples = len(field_data)/(Nx*Ny)
    elif(dim == 3):
      N_samples = len(field_data)/(Nx*Ny*Nz)

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


def calc_err_multiplication(x, y, x_err, y_err):
    # z = x * y
    z = x*y
    result = z * np.sqrt( ((x_err/x)**2)  + ((y_err/y)**2) ) 
    return result



def calc_err_addition(x_err, y_err):
    # Error propagation function for x + y 
    #result = 0.
    # assumes x and y are real 

    # Calculate error using standard error formula 
    result = np.sqrt( (x_err**2) + (y_err**2) )
    return result


def calc_err_average(vector):
   # error propagation for summing over a whole vector of numbers. The input vector is the 1D list of errors to be propagated  
   # returns the resulting error
   err = 0. + 1j*0. 
   err += (1./len(vector)) * np.sqrt( np.sum( vector**2  ) )
   return err 



# Script to load and plot correlation data 

# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['simulation']['Nx']
Ny = params['simulation']['Ny']
Nz = params['simulation']['Nz']
dim = params['system']['Dim']

# import the data
 #ops_file = 'S_k_tot_cp.dat'
 #Sk = np.loadtxt('S_k_tot_cp.dat', unpack=True)
 #rho_k = np.loadtxt('rho_k_cp.dat', unpack=True)
 #rho_negk = np.loadtxt('rho_-k_cp.dat', unpack=True)
Sk = np.loadtxt('S_k_diag.dat', unpack=True)
rho_k = np.loadtxt('rho_k_0.dat', unpack=True)
rho_negk = np.loadtxt('rho_-k_0.dat', unpack=True)



# Need to average the real and imaginary parts of each operator across the simulation  
# for 1 dataset
Sk_data = Sk[2*(dim)] + 1j*Sk[2*(dim) + 1]
rho_k_data = rho_k[2*(dim)] + 1j*rho_k[2*(dim) + 1]
rho_negk_data = rho_negk[2*(dim)] + 1j*rho_negk[2*(dim) + 1]

pcnt_averaging = 0.80
corr_avg, corr_err = calculate_field_average(Sk_data, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
rho_k_avg, rho_k_err = calculate_field_average(rho_k_data, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
rho_negk_avg, rho_negk_err = calculate_field_average(rho_negk_data, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))

Structure_factor = np.zeros(len(corr_avg), dtype=np.complex_)
Structure_factor += corr_avg 
Structure_factor -= (rho_k_avg * rho_negk_avg)

print('Max structure factor value: ' + str(np.max(Structure_factor)))


S_k_errs = np.zeros(len(corr_avg), dtype=np.complex_)

# 1. calc error multiplication for rho(k) and rho(-k)
# 2. calc error addition for 1) and then <rho(k) rho(-k)> 

S_k_errs += calc_err_multiplication(rho_k_avg, rho_negk_avg, rho_k_err,  rho_negk_err) 
S_k_errs = calc_err_addition(S_k_errs, corr_err) 


#comp = Sk[4] + Sk[5]*1j - ((rho_k[4] + 1j*rho_k[5]) * (rho_negk[4] + 1j*rho_negk[5]))
#reals = Sk[4]  - (rho_k[4] * rho_negk[4])

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





# Need only 1 of the kx and ky arrays (There are N_sample copies of them)

#kx_unique = np.unique(kx)
#ky_unique = np.unique(ky)

if(dim > 2):
  data = {'kx': kx, 'ky': ky, 'kz': kz, 'S_k': Structure_factor, 'S_k_errs': S_k_errs}
else:
  data = {'kx': kx, 'ky': ky, 'S_k': Structure_factor, 'S_k_errs': S_k_errs}

d_frame = pd.DataFrame.from_dict(data)

#d_frame.sort_values(by=['kx', 'ky','kz'], ascending = True, inplace=True)
d_frame.sort_values(by=['kx', 'ky'], ascending = True, inplace=True)

kx_unique = np.unique(kx)
ky_unique = np.unique(ky)

S_kx = np.zeros(len(kx_unique), dtype=np.complex_)
S_kx_errs = np.zeros(len(kx_unique), dtype=np.complex_)
S_ky = np.zeros(len(ky_unique), dtype=np.complex_)
S_ky_errs = np.zeros(len(ky_unique), dtype=np.complex_)

#print('Averaging over the remaining dimensions for S(kx)')

for i, uniq_kx in enumerate(kx_unique):
  # for each unique kx value,  Loop over other dimensions and average them 
  # 1. Get the list of indices of the unique kx value 
  tmp_frame = (d_frame['kx'] == uniq_kx)
  #indices = np.array(np.where(tmp_frame == True)) 
  indices = (np.where(tmp_frame == True)) 
  indices = indices[0] # 0th element is the list of true indices 
  #print(indices)

  # pick a random index and check to make sure it is yielding the correct kx 
  #print(d_frame['kx'].iloc[indices[0]]) 
  #print(uniq_kx)
  assert(d_frame['kx'].iloc[indices[0]] == uniq_kx)
  # 2. Average the structure factor with all of those kx values 
  S_kx[i] = np.mean(d_frame['S_k'].iloc[indices])
  #S_kx_errs[i] = np.mean(d_frame['S_k_errs'].iloc[indices]) # incorrect, but a good proxy 
  S_kx_errs[i] = calc_err_average(d_frame['S_k_errs'].iloc[indices]) 
  # repeat 

# repeat for ky 
for i, uniq_ky in enumerate(ky_unique):
  # for each unique ky value,  Loop over other dimensions and average them 
  # 1. Get the list of indices of the unique ky value 
  tmp_frame = (d_frame['ky'] == uniq_ky)
  #indices = np.array(np.where(tmp_frame == True)) 
  indices = (np.where(tmp_frame == True)) 
  indices = indices[0] # 0th element is the list of true indices 
  #print(indices)

  # pick a random indey and check to make sure it is yielding the correct ky 
  #print(d_frame['ky'].iloc[indices[0]]) 
  #print(uniq_ky)
  assert(d_frame['ky'].iloc[indices[0]] == uniq_ky)
  # 2. Average the structure factor with all of those ky values 
  S_ky[i] = np.mean(d_frame['S_k'].iloc[indices])
  #S_ky_errs[i] = np.mean(d_frame['S_k_errs'].iloc[indices]) # incorrect, good proxy/placeholder   
  S_ky_errs[i] = calc_err_average(d_frame['S_k_errs'].iloc[indices]) 
  # repeat 


# propagate error for normalizing
max_index_kx = np.where(S_kx == np.max(S_kx))[0][0] 
max_index_ky = np.where(S_ky == np.max(S_ky))[0][0]

normalized_Sk_x_errs = calc_err_division(S_kx.real, np.max(S_kx.real), S_kx_errs.real, S_kx_errs.real[max_index_kx]) 
normalized_Sk_y_errs = calc_err_division(S_ky.real, np.max(S_ky.real), S_ky_errs.real, S_ky_errs.real[max_index_ky]) 

print('Max structure factor S(kx) value: ' + str(np.max(S_kx)))
# Redefine numpy array post sorting
#S_k_sorted = np.array(d_frame['S_k'])

 #
ctr = 1
# Average over the y-coordinate to plot only S(k_x) 
#K Plot the S(kx) distribution in k-space  
 #plt.figure(ctr)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #
 #plt.errorbar(kx_unique, S_kx.real/np.max(S_kx.real), normalized_Sk_x_errs, marker='o', markersize = 6, linewidth = 0.5, elinewidth = 0.5, color = 'red', label='$S(k_{x})$')
 #plt.errorbar(ky_unique, S_ky.real/np.max(S_ky.real), normalized_Sk_y_errs, marker='o', markersize = 6, linewidth = 0.5, elinewidth = 0.5, color = 'blue', label='$S(k_{y})$')
 ##plt.errorbar(kx_unique, S_kx.real, S_kx_errs.real, marker='o', markersize = 6, linewidth = 0.5, elinewidth = 0.5, color = 'red', label='CL')
 #plt.title('Normalized Structure Factor: $S(k_{x}, k_{y})$', fontsize = 16)
 #plt.xlabel('$k$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('$S(k)$', fontsize = 20, fontweight = 'bold')
 #plt.xlim(0, np.max(kx_unique))
 #plt.legend()
 ##plt.savefig('S_k_Stripe_many_periods.eps')
 #plt.show()
 #
#ctr += 1



# Redefine numpy array post sorting
S_k_sorted = np.array(d_frame['S_k']) 
S_k_sorted.resize(Nx, Ny)
S_k_sorted = np.transpose(S_k_sorted)

plt.style.use('~/csbosonscpp/tools/python_scripts/plot_style.txt')

print('Max S(k) value: ' + str(np.round(np.max(S_k_sorted), 4)))

plt.figure(ctr)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
plt.title(r'$S_{\alpha \alpha} (k)$', fontsize = 24)
plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
plt.ylabel('$k_y$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.colorbar()
# plt.legend()
#plt.savefig('n_k_Stripe_1K_many_periods.eps')
plt.show()





# Off-diagonal structure factor too 
Sk = np.loadtxt('S_k_offdiag.dat', unpack=True)
rho_k = np.loadtxt('rho_k_0.dat', unpack=True)
rho_negk = np.loadtxt('rho_-k_1.dat', unpack=True)


# Need to average the real and imaginary parts of each operator across the simulation  
# for 1 dataset
Sk_data = Sk[2*(dim)] + 1j*Sk[2*(dim) + 1]
rho_k_data = rho_k[2*(dim)] + 1j*rho_k[2*(dim) + 1]
rho_negk_data = rho_negk[2*(dim)] + 1j*rho_negk[2*(dim) + 1]

pcnt_averaging = 0.80
corr_avg, corr_err = calculate_field_average(Sk_data, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
rho_k_avg, rho_k_err = calculate_field_average(rho_k_data, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
rho_negk_avg, rho_negk_err = calculate_field_average(rho_negk_data, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))

Structure_factor = np.zeros(len(corr_avg), dtype=np.complex_)
Structure_factor += corr_avg 
Structure_factor -= (rho_k_avg * rho_negk_avg)

#print('Max structure factor value: ' + str(np.max(Structure_factor)))


S_k_errs = np.zeros(len(corr_avg), dtype=np.complex_)

# 1. calc error multiplication for rho(k) and rho(-k)
# 2. calc error addition for 1) and then <rho(k) rho(-k)> 

S_k_errs += calc_err_multiplication(rho_k_avg, rho_negk_avg, rho_k_err,  rho_negk_err) 
S_k_errs = calc_err_addition(S_k_errs, corr_err) 


#comp = Sk[4] + Sk[5]*1j - ((rho_k[4] + 1j*rho_k[5]) * (rho_negk[4] + 1j*rho_negk[5]))
#reals = Sk[4]  - (rho_k[4] * rho_negk[4])

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

# Need only 1 of the kx and ky arrays (There are N_sample copies of them)

#kx_unique = np.unique(kx)
#ky_unique = np.unique(ky)

if(dim > 2):
  data = {'kx': kx, 'ky': ky, 'kz': kz, 'S_k': Structure_factor, 'S_k_errs': S_k_errs}
else:
  data = {'kx': kx, 'ky': ky, 'S_k': Structure_factor, 'S_k_errs': S_k_errs}

d_frame = pd.DataFrame.from_dict(data)

#d_frame.sort_values(by=['kx', 'ky','kz'], ascending = True, inplace=True)
d_frame.sort_values(by=['kx', 'ky'], ascending = True, inplace=True)

kx_unique = np.unique(kx)
ky_unique = np.unique(ky)


# Redefine numpy array post sorting
S_k_sorted = np.array(d_frame['S_k']) 
S_k_sorted.resize(Nx, Ny)
S_k_sorted = np.transpose(S_k_sorted)


plt.style.use('~/csbosonscpp/tools/python_scripts/plot_style.txt')

print('Max S(k) offdigonal value: ' + str(np.round(np.max(S_k_sorted), 4)))

plt.figure(ctr)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
#plt.title('$S(k)_{\uparrow \downarrow}$', fontsize = 22)
plt.title(r'$S_{\alpha \beta} (k)$', fontsize = 24)
plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
plt.ylabel('$k_y$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.colorbar()
# plt.legend()
#plt.savefig('n_k_Stripe_1K_many_periods.eps')
plt.show()


plt.figure(ctr)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(-S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
#plt.title('$S(k)_{\uparrow \downarrow}$', fontsize = 22)
plt.title(r'$S_{\alpha \beta} (k)$', fontsize = 24)
plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
plt.ylabel('$k_y$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.colorbar()
# plt.legend()
#plt.savefig('n_k_Stripe_1K_many_periods.eps')
plt.show()

 #
 #plt.style.use('~/csbosonscpp/tools/python_scripts/plot_style.txt')
 #
 #
 #plt.figure(1)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)])
 #plt.title('$S(k)$', fontsize = 11)
 #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('$k_y$', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 #plt.colorbar()
 ## plt.legend()
 #plt.show()
 #
 #
 #
 #
 #
 ## Take y-average of the (S(k_x))
 ##for i in ky:
 ##  S_k_sorted[:, i]
 #
 #
 # ## Plot the structure factor 
 # #plt.figure(10)
 # ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 # #plt.imshow(S_k_final.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 # #plt.title('$S(k)$', fontsize = 11)
 # #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 # #plt.ylabel('$k_y$', fontsize = 20, fontweight = 'bold')
 # ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 # #plt.colorbar()
 # ## plt.legend()
 ##plt.show()
 #
 #
 #
 ## Average over the y-coordinate to plot only S(k_x) 
 ##K Plot the S(kx) distribution in k-space  
 #plt.figure(2)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.plot(kx_list, S_k_sorted[:,0].real, 'r', label = '$S(k_{x})$')
 #S_k_sorted = np.transpose(S_k_sorted)
 #plt.plot(np.unique(kx), S_k_sorted[:,1]/np.max(S_k_sorted), '-o', linewidth = 0.25, color = 'red', label='$S(k_{x}, k_{y} = 0)$')
 #plt.title('Normalized Structure Factor: $S(k_{x}, k_{y} = 0)$', fontsize = 16)
 #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('$S(k_{x})$', fontsize = 20, fontweight = 'bold')
 ##plt.savefig('S_k_Stripe_1D.eps')
 #plt.legend()
 #plt.show()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 #
 #
 #
 ##K Plot the S(kx) distribution in k-space  
 #plt.figure(3)
 #plt.plot(np.unique(kx), S_k_sorted[:,1], '-o', linewidth = 0.25, color = 'red', label='$S(k_{x}, k_{y} = 0)$')
 #plt.title('Stripe Phase Structure Factor: $S(k_{x}, k_{y} = 0)$', fontsize = 16)
 #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('$S(k_{x})$', fontsize = 20, fontweight = 'bold')
 ##plt.savefig('S_k_Stripe_1D.eps')
 #plt.legend()
 #plt.show()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()


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
