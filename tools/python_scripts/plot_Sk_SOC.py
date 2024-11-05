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
T_ = 1./float(params['system']['beta'])
_kappa = params['system']['kappa'] 

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

pcnt_averaging = 0.50
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
 #  S_kx[i] = np.mean(d_frame['S_k'].iloc[indices])
 #  #S_kx_errs[i] = np.mean(d_frame['S_k_errs'].iloc[indices]) # incorrect, but a good proxy 
 #  S_kx_errs[i] = calc_err_average(d_frame['S_k_errs'].iloc[indices]) 
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
 #  S_ky[i] = np.mean(d_frame['S_k'].iloc[indices])
 #  #S_ky_errs[i] = np.mean(d_frame['S_k_errs'].iloc[indices]) # incorrect, good proxy/placeholder   
 #  S_ky_errs[i] = calc_err_average(d_frame['S_k_errs'].iloc[indices]) 
 #  # repeat 


# propagate error for normalizing
max_index_kx = np.where(S_kx == np.max(S_kx))[0][0] 
max_index_ky = np.where(S_ky == np.max(S_ky))[0][0]

 #normalized_Sk_x_errs = calc_err_division(S_kx.real, np.max(S_kx.real), S_kx_errs.real, S_kx_errs.real[max_index_kx]) 
 #normalized_Sk_y_errs = calc_err_division(S_ky.real, np.max(S_ky.real), S_ky_errs.real, S_ky_errs.real[max_index_ky]) 

print('Max structure factor S(kx) value: ' + str(np.max(S_kx)))
# Redefine numpy array post sorting
#S_k_sorted = np.array(d_frame['S_k'])

 #


# Extract n(kx)
tmp_frame = (d_frame['ky'] == 0.)
#indices = np.array(np.where(tmp_frame == True)) 
indices = (np.where(tmp_frame == True)) 
indices = indices[0] # 0th element is the list of true indices 
#print(uniq_ky)
assert(d_frame['ky'].iloc[indices[0]] == 0.)
# 2. Extract S or n(kx) evalutaed at ky == 0 
S_kx = d_frame['S_k'].iloc[indices]
  #S_ky_errs[i] = np.mean(d_frame['S_k_errs'].iloc[indices]) # incorrect, good proxy/placeholder   
#S_kx_errs = calc_err_average(d_frame['S_k_errs'].iloc[indices]) 
S_kx_errs = d_frame['S_k_errs'].iloc[indices] 
  # repeat 
S_kx = S_kx.values
S_kx_errs = S_kx_errs.values



# Extract n(ky)
tmp_frame = (d_frame['kx'] == 0.)
#indices = np.array(np.where(tmp_frame == True)) 
indices = (np.where(tmp_frame == True)) 
indices = indices[0] # 0th element is the list of true indices 
#print(uniq_ky)
assert(d_frame['kx'].iloc[indices[0]] == 0.)
# 2. Extract S or n(kx) evalutaed at ky == 0 
S_ky = d_frame['S_k'].iloc[indices]
S_ky_errs = d_frame['S_k_errs'].iloc[indices] 
  # repeat 
S_ky = S_ky.values
S_ky_errs = S_ky_errs.values

print('min S_ky: ' + str(np.min(S_ky_errs)))
print('max S_ky: ' + str(np.max(S_ky_errs)))
print('min S_kx: ' + str(np.min(S_kx_errs)))
print('max S_kx: ' + str(np.max(S_kx_errs)))



ctr = 1
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
ctr += 1


plt.figure(ctr)
plt.errorbar(np.unique(kx), S_kx.real, S_kx_errs.real, marker='o', markersize = 6, elinewidth=0.25, linewidth = 0.25, color = 'blue', label='$S(k_{x}, k_{y}=0)$')
plt.errorbar(np.unique(ky), S_ky.real, S_ky_errs.real, marker='o', markersize = 6, elinewidth=0.25, linewidth = 0.25, color = 'red', label='$S(k_{x}=0, k_{y})$')
plt.title('Structure Factor, ' + r'$\tilde T = ' + str(T_) + '$', fontsize = 16)
plt.xlabel('$k_{\mu}$', fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$S_{\alpha \alpha} (k_{\mu})$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
#plt.xlim(-1, 1)
#plt.savefig('n_k_plane_wave_1D.eps')
plt.legend()
plt.show()
ctr += 1




# Calculate angular average 
kr = np.sqrt(kx**2 + ky**2)
theta = np.arctan(ky/kx) # rads 

kr_uniq = np.unique(kr)

S_kr = np.zeros(len(kr_uniq), dtype=np.complex_)
S_kr_errs = np.zeros(len(kr_uniq), dtype=np.complex_)

_polar_data = {'kr': kr, 'theta': theta, 'S_k': Structure_factor, 'S_k_errs': S_k_errs}
polar_d_frame = pd.DataFrame.from_dict(_polar_data)
polar_d_frame.sort_values(by=['kr'], ascending = True, inplace=True) 


S_kr[0] += polar_d_frame['S_k'].iloc[0]
S_kr_errs[0] += polar_d_frame['S_k_errs'].iloc[0]
i = 0
print(kr[0])
for kr_ in kr_uniq[1:len(kr_uniq)]:
  i += 1
  tmp_frame = (polar_d_frame['kr'] == kr_)
  indices = np.where(tmp_frame == True)[0] 
  #indices = indices[0] # 0th element is the list of true indices 
  assert(polar_d_frame['kr'].iloc[indices[0]] == kr_)
  # 2. Extract 
  S_kr[i] += polar_d_frame['S_k'].iloc[indices].mean()
  # propagate error across the average 
  S_kr_errs[i] += calc_err_average(polar_d_frame['S_k_errs'].iloc[indices].values) 



# Plot angular average 
plt.figure(ctr)
#plt.errorbar(kr_uniq, n_kr.real/np.sum(n_k_sorted.real), n_kr_errs.real/np.sum(n_k_sorted.real), marker='o', markersize = 6, elinewidth=0.25, linewidth = 0.25, color = 'black', label='Langevin')
plt.errorbar(kr_uniq, S_kr.real, S_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='Langevin')
plt.title('Polar Averaged Diagonal Structure Factor, ' + r'$\tilde T = ' + str(T_) + '$', fontsize = 18)
plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
plt.ylabel(r'$S_{\alpha \alpha} (k_{r})$', fontsize = 20, fontweight = 'bold')
plt.axvline(x = 2 * _kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2 \tilde{\kappa} = ' + str(np.round(2.*_kappa, 2)) + '$')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
#plt.xlim(-1, 1)
#plt.savefig('n_k_plane_wave_1D.eps')
plt.legend()
plt.show()
ctr += 1



# Default to false 
_offDiag = False
if(_offDiag):
  # Off-diagonal structure factor too 
  Sk = np.loadtxt('S_k_offdiag.dat', unpack=True)
  rho_k = np.loadtxt('rho_k_0.dat', unpack=True)
  rho_negk = np.loadtxt('rho_-k_1.dat', unpack=True)
  
  
  # Need to average the real and imaginary parts of each operator across the simulation  
  # for 1 dataset
  Sk_data = Sk[2*(dim)] + 1j*Sk[2*(dim) + 1]
  rho_k_data = rho_k[2*(dim)] + 1j*rho_k[2*(dim) + 1]
  rho_negk_data = rho_negk[2*(dim)] + 1j*rho_negk[2*(dim) + 1]
  
  pcnt_averaging = 0.40
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
  ctr = 1
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
  ctr += 1
  
  
  
  
  
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
  ctr += 1
  
  
  
  
