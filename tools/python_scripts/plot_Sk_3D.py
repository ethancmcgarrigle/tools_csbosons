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

def calculate_field_average(field_data, N_spatial, N_samples_to_avg): # assumes cubic/square mesh 
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data
    print('Calculating the average of the field \n')

    N_samples = int(len(field_data)/(N_spatial))

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



def import_file_data(fname, dim, getkgrid):
    # import and unpack the data 
    print('Importing the data in ' + fname + ' \n')
    raw_data_cols = np.loadtxt(fname, unpack=True)  
    data = raw_data_cols[2*(dim)] + 1j*raw_data_cols[2*(dim)+1]
    kx = raw_data_cols[0]
    if(getkgrid):
      ky = raw_data_cols[1]
      if(dim == 2):
        return data, [kx, ky]
      elif(dim == 3):
        kz = raw_data_cols[2]
        return data, [kx, ky, kz]
      else:
        return data, [kx]
    else:
      return data 



def calculate_Sk(rho_k_kprime, rho_k, rho_negk, k_list, pcnt_averaging, N_spatial):
    # Average all the samples over the Langevin time 
    corr_avg, corr_err = calculate_field_average(rho_k_kprime, N_spatial, int(len(rho_k_kprime) * pcnt_averaging))
    rho_k_avg_0, rho_k_err_0 = calculate_field_average(rho_k, N_spatial, int(len(rho_k_kprime) * pcnt_averaging))
    rho_negk_avg_0, rho_negk_err_0 = calculate_field_average(rho_negk, N_spatial, int(len(rho_k_kprime) * pcnt_averaging))

    print('Calculating structure factor: \n')
    Structure_factor = np.zeros(len(corr_avg), dtype=np.complex_)
    Structure_factor += corr_avg 
    Structure_factor -= (rho_k_avg_0 * rho_negk_avg_0)
    
    #print('Max structure factor value: ' + str(np.max(Structure_factor)))
    S_k_errs = np.zeros(len(corr_avg), dtype=np.complex_)
    
    # 1. calc error multiplication for rho(k) and rho(-k)
    # 2. calc error addition for 1) and then <rho(k) rho(-k)> 
    S_k_errs += calc_err_multiplication(rho_k_avg_0, rho_negk_avg_0, rho_k_err_0,  rho_negk_err_0) 
    S_k_errs = calc_err_addition(S_k_errs, corr_err) 

    N_samples = int(len(rho_k_kprime)/(N_spatial))
    kx = np.split(k[0], N_samples)[0]
    #kx = kx[0]
    ky = np.split(k[1], N_samples)[0]
    #ky = ky[0] 
    if(dim > 2):
      kz = np.split(k[2], N_samples)[0]
      #kz = kz[0] 
      data = {'kx': kx, 'ky': ky, 'kz': kz, 'S_k': Structure_factor, 'S_k_errs': S_k_errs}
    else:
      data = {'kx': kx, 'ky': ky, 'S_k': Structure_factor, 'S_k_errs': S_k_errs}

    d_frame = pd.DataFrame.from_dict(data)
    
    if(dim > 2):
      d_frame.sort_values(by=['kx', 'ky','kz'], ascending = True, inplace=True)
    else:
      d_frame.sort_values(by=['kx', 'ky'], ascending = True, inplace=True)
    
    return d_frame


def angular_avg(S_k_df, dim):
    # Calculate angular average
    kx = np.array(S_k_df['kx']) 
    ky = np.array(S_k_df['ky'])
    if(dim == 3): 
      kz = np.array(S_k_df['kz'])
      kr = np.sqrt(kx**2 + ky**2 + kz**2)
      # for 3D, do a spherical average 
    elif(dim == 2):
      kr = np.sqrt(kx**2 + ky**2)

    kr_uniq = np.unique(kr)

    S_kr = np.zeros(len(kr_uniq), dtype=np.complex_)
    S_kr_errs = np.zeros(len(kr_uniq), dtype=np.complex_)
    
    radial_data = {'kr': kr, 'S_k': np.array(S_k_df['S_k']), 'S_k_errs': np.array(S_k_df['S_k_errs'])}
    radial_d_frame = pd.DataFrame.from_dict(radial_data)
    radial_d_frame.sort_values(by=['kr'], ascending = True, inplace=True) 

    # get k_r == 0 entry     
    S_kr[0] += radial_d_frame['S_k'].iloc[0]
    S_kr_errs[0] += radial_d_frame['S_k_errs'].iloc[0]
    i = 0
    # loop over non-zero k_r 
    #for kr_ in kr_uniq[0:len(kr_uniq)]:
    for kr_ in kr_uniq[1:len(kr_uniq)]:
      i += 1
      tmp_frame = (radial_d_frame['kr'] == kr_)
      indices = np.where(tmp_frame == True)[0] 
      assert(radial_d_frame['kr'].iloc[indices[0]] == kr_)
      # 2. Average over the remaining degrees of freedom at fixed k_r 
      S_kr[i] += radial_d_frame['S_k'].iloc[indices].mean()
      # 3. propagate error across the average 
      S_kr_errs[i] += calc_err_average(radial_d_frame['S_k_errs'].iloc[indices].values) 

    return kr_uniq, S_kr, S_kr_errs



def visualize_2D_spectrum(S_k_sorted, kx, ky, title_str):

    plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_spins.txt')
    plt.figure(figsize=(6.77166, 6.77166))
    plt.imshow(S_k_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
    plt.title(title_str, fontsize = 30)
    plt.xlabel('$k_x$', fontsize = 32)
    plt.ylabel('$k_y$', fontsize = 32)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig('Sk_00.eps')
    plt.show()
    
    np.savetxt('S_k_00_figure.dat', S_k_sorted.real)
    
    #plt.figure(figsize=(8.0, 8.0))
    plt.figure(figsize=(6.77166, 6.77166))
    # plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
    # normalize it by k=0 peak or max?
    plt.imshow(S_k_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)], norm=LogNorm()) 
    plt.title(title_str, fontsize = 30)
    plt.xlabel('$k_x$', fontsize = 32)
    plt.ylabel('$k_y$', fontsize = 32)
    plt.colorbar(fraction=0.046, pad=0.04)
    # plt.legend()
    plt.savefig('Sk_00_log.eps')
    plt.show()

def plot_radial_avg_Sk(kr_uniq, S_kr, S_kr_errs, title_str):
    # Plot angular average 
    plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_orderparams.txt')
    plt.figure(figsize=(4,4))
    plt.errorbar(kr_uniq, S_kr.real, S_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='CL')
    #plt.title('Angular Averaged ' + title_str + ', ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 20)
    plt.title(title_str, fontsize = 20)
    plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
    plt.ylabel(r'$S(k_{r}) $', fontsize = 24, fontweight = 'bold')
    #plt.axvline(x = 2*_kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2 \tilde{\kappa} = ' + str(2*_kappa) + '$')
    plt.legend()
    plt.savefig('S_k_angular_avg.eps')
    plt.show()
 
    np.savetxt('S_k_00_angularAvg_data.dat', np.column_stack( [kr_uniq, S_kr.real, S_kr_errs.real] ))


  
# Script to load and plot correlation data 

# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['simulation']['Nx']
Ny = params['simulation']['Ny']
Nz = params['simulation']['Nz']
#_kappa = params['system']['kappa']
T_ = float(params['system']['beta'])
T_ = 1./T_
dim = params['system']['Dim']

N_spatial = Nx
if(dim > 1):
  N_spatial *= Ny
  if(dim > 2):
    N_spatial *= Nz

### 1. Calculate diagonal S_k for upspins
Sk_data_00, k = import_file_data('S_k_00.dat', dim, True) 
rho_k_data_0 = import_file_data('rho_k_0.dat', dim, False) 
rho_negk_data_0 = import_file_data('rho_-k_0.dat', dim, False) 

# Calculate S_k, return the data frame with the info 
pcnt_averaging = 0.825
Sk_dataframe = calculate_Sk(Sk_data_00, rho_k_data_0, rho_negk_data_0, k, pcnt_averaging, N_spatial)

# For 3D, we need to angular average! 

### IF DIM == 2 only 
title_str = r'$S_{\alpha \alpha} (k)$'
if(dim == 2):
  # Redefine numpy array post sorting
  S_k_sorted = np.array(Sk_dataframe['S_k'])
  kx = np.array(Sk_dataframe['kx'])
  ky = np.array(Sk_dataframe['ky'])
  S_k_sorted.resize(Nx, Ny)
  S_k_sorted = np.transpose(S_k_sorted)
  S_k_sorted = np.flip(S_k_sorted,0)
  visualize_2D_spectrum(S_k_sorted, kx, ky, title_str)

kr, S_kr, S_kr_errs = angular_avg(Sk_dataframe, dim)

plot_radial_avg_Sk(kr, S_kr, S_kr_errs, 'Angular Averaged ' + title_str + ', ' + r'$\bar{T} = ' + str(np.round(T_, 2)) + '$') 




### 2. Calculate diagonal S_k for down spin species  
Sk_data_11, k = import_file_data('S_k_11.dat', dim, True) 
rho_k_data_1 = import_file_data('rho_k_1.dat', dim, False) 
rho_negk_data_1 = import_file_data('rho_-k_1.dat', dim, False) 

# Calculate S_k, return the data frame with the info 
pcnt_averaging = 0.825
Sk_dataframe = calculate_Sk(Sk_data_11, rho_k_data_1, rho_negk_data_1, k, pcnt_averaging, N_spatial)

# For 3D, we need to angular average! 

### IF DIM == 2 only 
title_str = r'$S_{\gamma \gamma} (k)$'
if(dim == 2):
  # Redefine numpy array post sorting
  S_k_sorted = np.array(Sk_dataframe['S_k'])
  kx = np.array(Sk_dataframe['kx'])
  ky = np.array(Sk_dataframe['ky'])
  S_k_sorted.resize(Nx, Ny)
  S_k_sorted = np.transpose(S_k_sorted)
  S_k_sorted = np.flip(S_k_sorted,0)
  visualize_2D_spectrum(S_k_sorted, kx, ky, title_str)

kr, S_kr, S_kr_errs = angular_avg(Sk_dataframe, dim)

plot_radial_avg_Sk(kr, S_kr, S_kr_errs, 'Angular Averaged ' + title_str + ', ' + r'$\bar{T} = ' + str(np.round(T_, 2)) + '$') 


## 3. Calculate the cross-structure factor S(k)_alpha, gamma
Sk_data_12, k = import_file_data('S_k_12.dat', dim, True) 

# Calculate S_k, return the data frame with the info 
pcnt_averaging = 0.825
Sk_dataframe = calculate_Sk(Sk_data_12, rho_k_data_0, rho_negk_data_1, k, pcnt_averaging, N_spatial)

# For 3D, we need to angular average! 
title_str = r'$S_{\alpha \gamma} (k)$'
### IF DIM == 2 only 
if(dim == 2):
  # Redefine numpy array post sorting
  S_k_sorted = np.array(Sk_dataframe['S_k'])
  kx = np.array(Sk_dataframe['kx'])
  ky = np.array(Sk_dataframe['ky'])
  S_k_sorted.resize(Nx, Ny)
  S_k_sorted = np.transpose(S_k_sorted)
  S_k_sorted = np.flip(S_k_sorted,0)
  visualize_2D_spectrum(S_k_sorted, kx, ky, title_str)

kr, S_kr, S_kr_errs = angular_avg(Sk_dataframe, dim)

plot_radial_avg_Sk(kr, S_kr, S_kr_errs, 'Angular Averaged ' + title_str + ', ' + r'$\bar{T} = ' + str(np.round(T_, 2)) + '$') 


## 4. Calculate the total structure factor S(k)
rho_k_total = rho_k_data_0 + rho_k_data_1
rho_negk_total = rho_negk_data_0 + rho_negk_data_1
# Calculate S_k, return the data frame with the info 
pcnt_averaging = 0.825
Sk_dataframe = calculate_Sk(Sk_data_00 + Sk_data_11, rho_k_total, rho_negk_total, k, pcnt_averaging, N_spatial)

# For 3D, we need to angular average! 
title_str = r'$S(k)$'
# IF DIM == 2 only 
if(dim == 2):
  # Redefine numpy array post sorting
  S_k_sorted = np.array(Sk_dataframe['S_k'])
  kx = np.array(Sk_dataframe['kx'])
  ky = np.array(Sk_dataframe['ky'])
  S_k_sorted.resize(Nx, Ny)
  S_k_sorted = np.transpose(S_k_sorted)
  S_k_sorted = np.flip(S_k_sorted,0)
  visualize_2D_spectrum(S_k_sorted, kx, ky, title_str)

kr, S_kr, S_kr_errs = angular_avg(Sk_dataframe, dim)

plot_radial_avg_Sk(kr, S_kr, S_kr_errs, 'Angular Averaged ' + title_str + ', ' + r'$\bar{T} = ' + str(np.round(T_, 2)) + '$') 




