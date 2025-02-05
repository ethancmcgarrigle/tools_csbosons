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
from matplotlib.colors import LogNorm

## TODO factor and clean up, make a class and functions 


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
_kappa = params['system']['kappa']
T_ = float(params['system']['beta'])
T_ = 1./T_
dim = params['system']['Dim']

# import the data
 #ops_file = 'S_k_tot_cp.dat'
 #Sk = np.loadtxt('S_k_tot_cp.dat', unpack=True)
 #rho_k = np.loadtxt('rho_k_cp.dat', unpack=True)
 #rho_negk = np.loadtxt('rho_-k_cp.dat', unpack=True)
#Sk = np.loadtxt('S_k_00.dat', unpack=True)
#rho_k = np.loadtxt('rho_k_0.dat', unpack=True)
#rho_negk = np.loadtxt('rho_-k_0.dat', unpack=True)

Sk = np.loadtxt('S_k_00.dat', unpack=True)
rho_k_0 = np.loadtxt('rho_k_0.dat', unpack=True)
rho_negk_0 = np.loadtxt('rho_-k_0.dat', unpack=True)


# Need to average the real and imaginary parts of each operator across the simulation  
# for 1 dataset
Sk_data = Sk[2*(dim)] + 1j*Sk[2*(dim) + 1]
rho_k_data_0 = rho_k_0[2*(dim)] + 1j*rho_k_0[2*(dim) + 1]
rho_negk_data_0 = rho_negk_0[2*(dim)] + 1j*rho_negk_0[2*(dim) + 1]

pcnt_averaging = 0.825
corr_avg, corr_err = calculate_field_average(Sk_data, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
rho_k_avg_0, rho_k_err_0 = calculate_field_average(rho_k_data_0, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
rho_negk_avg_0, rho_negk_err_0 = calculate_field_average(rho_negk_data_0, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))

Structure_factor = np.zeros(len(corr_avg), dtype=np.complex_)
Structure_factor += corr_avg 
Structure_factor -= (rho_k_avg_0 * rho_negk_avg_0)

print('Max structure factor value: ' + str(np.max(Structure_factor)))


S_k_errs = np.zeros(len(corr_avg), dtype=np.complex_)

# 1. calc error multiplication for rho(k) and rho(-k)
# 2. calc error addition for 1) and then <rho(k) rho(-k)> 
S_k_errs += calc_err_multiplication(rho_k_avg_0, rho_negk_avg_0, rho_k_err_0,  rho_negk_err_0) 
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
ctr = 1

# Redefine numpy array post sorting
S_k_sorted = np.array(d_frame['S_k']) 
S_k_sorted.resize(Nx, Ny)
S_k_sorted = np.transpose(S_k_sorted)
S_k_sorted = np.flip(S_k_sorted,0)

plt.style.use('~/CSBosonsCpp/tools/python_plot_styles_examples/plot_style_spins.txt')

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
# plt.legend()
plt.savefig('Sk_00.eps')
plt.show()

np.savetxt('S_k_00_figure.dat', S_k_sorted.real)

#plt.figure(figsize=(8.0, 8.0))
plt.figure(figsize=(6.77166, 6.77166))
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
# normalize it by k=0 peak or max?
plt.imshow(S_k_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)], norm=LogNorm()) 
#plt.title(r'$S_{\alpha \alpha} (k)$', fontsize = 30)
plt.title(r'$S_{\alpha \alpha} (\mathbf{k})$', fontsize = 30)
plt.xlabel('$k_x$', fontsize = 32)
plt.ylabel('$k_y$', fontsize = 32)
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.xlim(-24,24)
#plt.ylim(-24,24)
#plt.colorbar()
plt.colorbar(fraction=0.046, pad=0.04)
# plt.legend()
plt.savefig('Sk_00_log.eps')
plt.show()



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
plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_orderparams.txt')
plt.figure(3)
plt.errorbar(kr_uniq, S_kr.real, S_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='CL')
plt.title('Angular Averaged S(k), ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 22)
plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
plt.ylabel(r'$S(k_{r}) $', fontsize = 24, fontweight = 'bold')
plt.axvline(x = 2*_kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2 \tilde{\kappa} = ' + str(2*_kappa) + '$')
plt.legend()
plt.savefig('S_k_angular_avg.eps')
plt.show()
 
#
# Export list_x[1:stop_indx], corr_sorted[0][1:stop_indx]
np.savetxt('S_k_00_angularAvg_data.dat', np.column_stack( [kr_uniq, S_kr.real, S_kr_errs.real] ))
 #
 #plt.figure(4)
 #plt.errorbar(kr_uniq, np.log10(S_kr.real), S_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='CL')
 #plt.title('Angular Averaged S(k), ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 22)
 #plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
 #plt.ylabel(r'log($S(k_{r}) $)', fontsize = 24, fontweight = 'bold')
 #plt.axvline(x = 2*_kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2 \tilde{\kappa} = ' + str(_kappa) + '$')
 #plt.legend()
 #plt.savefig('S_k_angular_avg_log.eps')
 #plt.show()
 #

# Off-diagonal structure factor too 
Sk = np.loadtxt('S_k_12.dat', unpack=True)
rho_k_1 = np.loadtxt('rho_k_1.dat', unpack=True)
rho_negk_1 = np.loadtxt('rho_-k_1.dat', unpack=True)


# Need to average the real and imaginary parts of each operator across the simulation  
# for 1 dataset
Sk_data = Sk[2*(dim)] + 1j*Sk[2*(dim) + 1]
rho_k_data_1 = rho_k_1[2*(dim)] + 1j*rho_k_1[2*(dim) + 1]
rho_negk_data_1 = rho_negk_1[2*(dim)] + 1j*rho_negk_1[2*(dim) + 1]

pcnt_averaging = 0.82
corr_avg, corr_err = calculate_field_average(Sk_data, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
rho_k_avg_1, rho_k_err_1 = calculate_field_average(rho_k_data_1, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
rho_negk_avg_1, rho_negk_err_1 = calculate_field_average(rho_negk_data_1, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))

Structure_factor = np.zeros(len(corr_avg), dtype=np.complex_)
Structure_factor += corr_avg 
Structure_factor -= (rho_k_avg_0 * rho_negk_avg_1)

S_k_errs = np.zeros(len(corr_avg), dtype=np.complex_)

# 1. calc error multiplication for rho(k) and rho(-k)
# 2. calc error addition for 1) and then <rho(k) rho(-k)> 

S_k_errs += calc_err_multiplication(rho_k_avg_0, rho_negk_avg_1, rho_k_err_0,  rho_negk_err_1) 
S_k_errs = calc_err_addition(S_k_errs, corr_err) 

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
S_k_sorted = np.flip(S_k_sorted,0)


#plt.style.use('~/CSBosonsCpp/tools/python_scripts/plot_style.txt')
print('Max S(k) offdigonal value: ' + str(np.round(np.max(S_k_sorted), 4)))

plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_spins.txt')
plt.figure(figsize=(6.77166, 6.77166))
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(S_k_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)], norm=LogNorm())
#plt.title('$S(k)_{\uparrow \downarrow}$', fontsize = 22)
plt.title(r'$S_{\alpha \beta} (\mathbf{k})$', fontsize = 30)
plt.xlabel('$k_x$', fontsize = 32)
plt.ylabel('$k_y$', fontsize = 32)
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
plt.savefig('S_k_alpha_beta_inferno.eps')
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.colorbar()
# plt.legend()
#plt.savefig('S_k_alpha_beta_angular_avg.eps')
plt.show()

np.savetxt('S_k_01_figure.dat', S_k_sorted.real)

plt.figure(figsize=(6.77166, 6.77166))
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(S_k_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)])
#plt.title('$S(k)_{\uparrow \downarrow}$', fontsize = 22)
plt.title(r'$S_{\alpha \beta} (\mathbf{k})$', fontsize = 30)
plt.xlabel('$k_x$', fontsize = 32)
plt.ylabel('$k_y$', fontsize = 32)
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
plt.savefig('S_k_alpha_beta_inferno_noLog.eps')
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.colorbar()
# plt.legend()
#plt.savefig('S_k_alpha_beta_angular_avg.eps')
plt.show()


# Off-diagonal structure factor too 
Sk = np.loadtxt('S_k_tot.dat', unpack=True)
Sk_11 = np.loadtxt('S_k_11.dat', unpack=True)
#rho_k = np.loadtxt('rho_k_0.dat', unpack=True)
#rho_negk = np.loadtxt('rho_-k_1.dat', unpack=True)


# Need to average the real and imaginary parts of each operator across the simulation  
# for 1 dataset
Sk_data = Sk[2*(dim)] + 1j*Sk[2*(dim) + 1]
Sk_11_data = Sk_11[2*(dim)] + 1j*Sk_11[2*(dim) + 1]
#rho_k_data = rho_k[2*(dim)] + 1j*rho_k[2*(dim) + 1]
#rho_negk_data = rho_negk[2*(dim)] + 1j*rho_negk[2*(dim) + 1]

pcnt_averaging = 0.825
corr_avg, corr_err = calculate_field_average(Sk_data, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
Sk_11_avg, Sk_11_err = calculate_field_average(Sk_11_data, Nx, Ny, Nz, dim, int(len(Sk_11_data) * pcnt_averaging))

# Sum rho_k and rho_-k data 
rho_k_tot = rho_k_avg_0 + rho_k_avg_1 
rho_negk_tot = rho_negk_avg_0 + rho_negk_avg_1

Structure_factor = np.zeros(len(corr_avg), dtype=np.complex_)
Structure_factor += corr_avg 
Structure_factor -= (rho_k_tot * rho_negk_tot)

Structure_factor_11 = np.zeros(len(Sk_11_avg), dtype=np.complex_)
Structure_factor_11 += Sk_11_avg
Structure_factor_11 -= (rho_k_avg_1 * rho_negk_avg_1)
#print('Max structure factor value: ' + str(np.max(Structure_factor)))
S_k_11_errs = np.zeros(len(Sk_11_avg), dtype=np.complex_)
S_k_11_errs += calc_err_multiplication(rho_k_avg_1, rho_negk_avg_1, rho_k_err_1,  rho_negk_err_1) 
S_k_11_errs = calc_err_addition(S_k_11_errs, Sk_11_err)


S_k_errs = np.zeros(len(corr_avg), dtype=np.complex_)

# 1. calc error multiplication for rho(k) and rho(-k)
# 2. calc error addition for 1) and then <rho(k) rho(-k)> 
rho_k_tot_err = calc_err_addition(rho_k_err_0, rho_k_err_1)
rho_negk_tot_err = calc_err_addition(rho_negk_err_0, rho_negk_err_1)
S_k_errs += calc_err_multiplication(rho_k_tot, rho_negk_tot, rho_k_tot_err,  rho_negk_tot_err) 
S_k_errs = calc_err_addition(S_k_errs, corr_err) 

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
  data = {'kx': kx, 'ky': ky, 'S_k': Structure_factor, 'S_k_errs': S_k_errs, 'S_k_11' : Structure_factor_11, 'S_k_11_errs' : S_k_11_errs}

d_frame = pd.DataFrame.from_dict(data)

#d_frame.sort_values(by=['kx', 'ky','kz'], ascending = True, inplace=True)
d_frame.sort_values(by=['kx', 'ky'], ascending = True, inplace=True)

kx_unique = np.unique(kx)
ky_unique = np.unique(ky)


# Redefine numpy array post sorting
S_k_sorted = np.array(d_frame['S_k']) 
S_k_sorted.resize(Nx, Ny)
S_k_sorted = np.transpose(S_k_sorted)
S_k_sorted = np.flip(S_k_sorted,0)

S_k_11sorted = np.array(d_frame['S_k_11']) 
S_k_11sorted.resize(Nx, Ny)
S_k_11sorted = np.transpose(S_k_11sorted)
S_k_11sorted = np.flip(S_k_11sorted,0)


#plt.style.use('~/CSBosonsCpp/tools/python_scripts/plot_style.txt')
print('Max S(k) total value: ' + str(np.round(np.max(S_k_sorted), 4)))

plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_spins.txt')
plt.figure(figsize=(6.77166, 6.77166))
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(S_k_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)], norm=LogNorm()) 
#plt.title('$S(k)_{\uparrow \downarrow}$', fontsize = 22)
plt.title(r'$S_{total} (\mathbf{k})$', fontsize = 30)
plt.xlabel('$k_x$', fontsize = 32)
plt.ylabel('$k_y$', fontsize = 32)
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.colorbar()
# plt.legend()
plt.savefig('S_k_tot_inferno.eps')
plt.show()

plt.figure(figsize=(6.77166, 6.77166))
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(S_k_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
#plt.title('$S(k)_{\uparrow \downarrow}$', fontsize = 22)
plt.title(r'$S_{total} (\mathbf{k})$', fontsize = 30)
plt.xlabel('$k_x$', fontsize = 32)
plt.ylabel('$k_y$', fontsize = 32)
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.colorbar()
# plt.legend()
plt.savefig('S_k_tot_inferno_noLog.eps')
plt.show()

np.savetxt('S_k_tot_figure.dat', S_k_sorted.real)

# get angular average 
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
plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_orderparams.txt')
plt.figure(5)
plt.errorbar(kr_uniq, S_kr.real, S_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='CL')
plt.title('Angular Averaged S(k), ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 22)
plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
plt.ylabel(r'$S(k_{r}) $', fontsize = 24, fontweight = 'bold')
plt.axvline(x = 2*_kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2 \tilde{\kappa} = ' + str(2*_kappa) + '$')
plt.legend()
plt.savefig('S_k_angular_avg_tot.eps')
plt.show()

np.savetxt('S_k_tot_angularAvg_data.dat', np.column_stack( [kr_uniq, S_kr.real, S_kr_errs.real] ))


print('Max S(k) 11 value: ' + str(np.round(np.max(S_k_sorted), 4)))

plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_spins.txt')
plt.figure(figsize=(6.77166, 6.77166))
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(S_k_11sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)], norm=LogNorm()) 
#plt.title('$S(k)_{\uparrow \downarrow}$', fontsize = 22)
plt.title(r'$S_{\beta \beta} (\mathbf{k})$', fontsize = 30)
plt.xlabel('$k_x$', fontsize = 32)
plt.ylabel('$k_y$', fontsize = 32)
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.colorbar()
# plt.legend()
plt.savefig('S_k_11_inferno.eps')
plt.show()

np.savetxt('S_k_11_figure.dat', S_k_11sorted.real)

plt.figure(figsize=(6.77166, 6.77166))
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.imshow(S_k_11sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
#plt.title('$S(k)_{\uparrow \downarrow}$', fontsize = 22)
plt.title(r'$S_{\beta \beta} (\mathbf{k})$', fontsize = 30)
plt.xlabel('$k_x$', fontsize = 32)
plt.ylabel('$k_y$', fontsize = 32)
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.colorbar()
# plt.legend()
plt.savefig('S_k_11_inferno_noLog.eps')
plt.show()
