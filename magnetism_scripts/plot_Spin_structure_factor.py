import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
import matplotlib.tri as tri
#matplotlib.rcParams['text.usetex'] = True
matplotlib.use('TkAgg')
import pdb
import pandas as pd 
from scipy.stats import sem 
from matplotlib.colors import LogNorm


def calculate_field_average(field_data, N_spatial, averaging_pcnt): 
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data

    # Get number of samples
    N_samples = len(field_data)/(N_spatial)

    assert(N_samples.is_integer())
    N_samples = int(N_samples)

    # Use split (np) to get arrays that represent each sample (1 array per sample) Throw out the first sample (not warmed up properly) 
    sample_arrays = np.split(field_data, N_samples)

    N_samples_to_avg = int(averaging_pcnt * N_samples)
    sample_arrays = sample_arrays[len(sample_arrays) - N_samples_to_avg:len(sample_arrays)]
    print('Averaging over ' + str(N_samples_to_avg) + ' samples') 

    # Final array, initialized to zeros. 
    averaged_data = np.zeros(len(sample_arrays[0]), dtype=np.complex_)
    averaged_data += np.mean(sample_arrays, axis=0) # axis=0 calculates element-by-element mean
    # Calculate the standard error 
    std_errs = np.zeros(len(sample_arrays[0]))
    std_errs += sem(sample_arrays, axis=0)
    return averaged_data, std_errs



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



def process_data(spin_file, N_gridpoints, dim, _Langevin):
    # Load the data 
    print('Processing data in file ' + spin_file) 
    Sk_raw_data = np.loadtxt(spin_file, unpack=True)
    Sk_data = Sk_raw_data[2*(dim)] + 1j*Sk_raw_data[2*(dim) + 1]

    if(_Langevin):
      # Average the data 
      pcnt_averaging = 0.60
      Sk_avg, Sk_errs = calculate_field_average(Sk_data, N_gridpoints, pcnt_averaging)
    else:
      Sk_avg = Sk_data
      Sk_errs = np.zeros_like(Sk_avg)

    #rho_k_avg_0, rho_k_err_0 = calculate_field_average(rho_k_data_0, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
    #rho_negk_avg_0, rho_negk_err_0 = calculate_field_average(rho_negk_data_0, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
    #Structure_factor -= (rho_k_avg_0 * rho_negk_avg_0)

    # 1. calc error multiplication for rho(k) and rho(-k)
    # 2. calc error addition for 1) and then <rho(k) rho(-k)> 
    #S_k_errs += calc_err_multiplication(rho_k_avg_0, rho_negk_avg_0, rho_k_err_0,  rho_negk_err_0) 
    #S_k_errs = calc_err_addition(S_k_errs, corr_err) 
    #S_k_errs = corr_err 

    # Extract the reciprocal (k) grid 
    kx = Sk_raw_data[0][0:N_gridpoints]
    if(dim > 1):
      ky = Sk_raw_data[1][0:N_gridpoints]
      if(dim > 2):
        kz = Sk_raw_data[2][0:N_gridpoints]
      else:
        kz = np.zeros_like(ky)
    else:
      ky = np.zeros_like(kx)
      kz = np.zeros_like(kx)

    processed_data = {'kx': kx, 'ky': ky, 'kz': kz, 'S(k)': Sk_avg, 'S(k)_errs': Sk_errs} 

    d_frame_Sk = pd.DataFrame.from_dict(processed_data)

    if(lattice == 'square'): # process for later usage of imshow()
      d_frame_Sk.sort_values(by=['kx', 'ky'], ascending = True, inplace=True)
      # Redefine numpy array post sorting
      Sk_processed = np.array(d_frame_Sk['S(k)']) 
      Sk_processed.resize(Nx, Ny)
      Sk_processed = np.transpose(Sk_processed)
      Sk_processed = np.flip(Sk_processed, 0)
    else:
      Sk_processed = np.array(d_frame_Sk['S(k)']) 


    return [np.array(d_frame_Sk['kx']), np.array(d_frame_Sk['ky']), np.array(d_frame_Sk['kz']), Sk_processed]
  

def plot_structure_factor(Sk_alpha_tmp, save_data, save_plot, basis_site_indx=1, basis_sites = 1):
    ''' Takes in a list Sk which contains Sk_xx, Sk_yy, Sk_zz (structure factor for each spin direction). 
        - Sk_nu,nu is a list of form [kx, ky, kz, Sk_data] 
        - e.g. Sk_list[0] is the list: [kx, ky, kz, Sk_xx] 
        - e.g. Sk_list[1] is the list: [kx, ky, kz, Sk_yy] '''
    plt.style.use('~/tools_csbosons/python_plot_styles/plot_style_spins.txt')
    for nu in range(0, 3):
      Sk = Sk_alpha_tmp[nu][3]
      kx = Sk_alpha_tmp[nu][0]
      ky = Sk_alpha_tmp[nu][1]
      kz = Sk_alpha_tmp[nu][2]

      file_str = 'Sk_' + dirs[nu] + dirs[nu] + '_' + str(basis_site_indx) 

      plt.figure(figsize=(6.77166, 6.77166))
      if(lattice == 'square'):
        plt.imshow(Sk.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
      else:
        triangles = tri.Triangulation(kx, ky)
        plt.tricontourf(triangles, Sk.real, cmap = 'inferno', levels = 50) 

      if(basis_sites > 1):
        plt.title(r'$S^{' + basis_site_labels[basis_site_indx] + '}_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)
      else:
        plt.title(r'$S_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)
      plt.xlabel('$k_{x}$', fontsize = 32)
      plt.ylabel('$k_{y}$', fontsize = 32)
      # plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
      plt.colorbar(fraction=0.046, pad=0.04)
      if(save_plot):
        plt.savefig(file_str + '.eps', dpi=300)
        plt.savefig(file_str + '.pdf', dpi=300)
      plt.show()

      if(save_data): 
        np.savetxt(file_str + '_figure.dat', Sk.real)
      
      plt.figure(figsize=(6.77166, 6.77166))
      if(basis_sites > 1):
        plt.title(r'$S^{' + basis_site_labels[basis_site_indx] + '}_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)
      else:
        plt.title(r'$S_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)

      if(lattice == 'square'):
        plt.imshow(Sk.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)], norm=LogNorm()) 
      else:
        triangles = tri.Triangulation(kx, ky)
        plt.tricontourf(triangles, Sk.real, cmap = 'inferno', norm=LogNorm(), levels = 50) 
      plt.xlabel('$k_{x}$', fontsize = 32)
      plt.ylabel('$k_{y}$', fontsize = 32)
      # plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
      plt.colorbar(fraction=0.046, pad=0.04)
      if(save_plot):
        plt.savefig(file_str + '_log.eps', dpi=300)
        plt.savefig(file_str + '_log.pdf', dpi=300)
      plt.show()




''' Script to load and visualize spin-spin correlation data''' 

# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['system']['NSitesPer-x']
Ny = params['system']['NSitesPer-y']
Nz = params['system']['NSitesPer-z']
T = float(params['system']['beta'])
T = 1./T
dim = params['system']['Dim']
system = params['system']['ModelType'] 
lattice = params['system']['lattice'] 
_CL = params['simulation']['CLnoise']

N_spatial = Nx
_isPlotting = True

if(dim > 1):
  N_spatial *= Ny
  if( dim > 2):
    N_spatial *= Nz
  else:
    Nz = 1
else:
  Ny = 1
  Nz = 1

dirs = {0 : 'x', 1 : 'y', 2 : 'z'}


if(system == 'HONEYCOMB'):
  # Retrieve the spin textures for each sublattice, and then combine into a single plot 
  num_basis_sites = 2
  basis_site_labels = {0: 'A', 1: 'B'}
else:
  num_basis_sites = 1
  basis_site_labels = {0: 'A'}

Sk_list = [] # list of length sublattice basis sites (2 for honeycomb) 
# Main loop to process the correlation data for each sublattice and each spin
for K in range(0, num_basis_sites):
  Sk_alpha = []
  # loop over each spin direction 
  for nu in range(0, 3):
    S_file = 'S' + str(dirs[nu]) + '_k_S' + str(dirs[nu]) + '_-k_' + str(K) + '.dat' 
    Sk_alpha.append(process_data(S_file, N_spatial, dim, _CL))
  Sk_list.append(Sk_alpha)

for K in range(0, num_basis_sites):
  plot_structure_factor(Sk_list[K], False, False, K, num_basis_sites) 


# Calculate angular average 
#kr = np.sqrt(kx**2 + ky**2)
#theta = np.arctan(ky/kx) # rads 
#
#kr_uniq = np.unique(kr)
#
#S_kr = np.zeros(len(kr_uniq), dtype=np.complex_)
#S_kr_errs = np.zeros(len(kr_uniq), dtype=np.complex_)
#
#_polar_data = {'kr': kr, 'theta': theta, 'S_k': Structure_factor, 'S_k_errs': S_k_errs}
#polar_d_frame = pd.DataFrame.from_dict(_polar_data)
#polar_d_frame.sort_values(by=['kr'], ascending = True, inplace=True) 
#
#
#S_kr[0] += polar_d_frame['S_k'].iloc[0]
#S_kr_errs[0] += polar_d_frame['S_k_errs'].iloc[0]
#i = 0
#print(kr[0])
#for kr_ in kr_uniq[1:len(kr_uniq)]:
#  i += 1
#  tmp_frame = (polar_d_frame['kr'] == kr_)
#  indices = np.where(tmp_frame == True)[0] 
#  #indices = indices[0] # 0th element is the list of true indices 
#  assert(polar_d_frame['kr'].iloc[indices[0]] == kr_)
#  # 2. Extract 
#  S_kr[i] += polar_d_frame['S_k'].iloc[indices].mean()
#  # propagate error across the average 
#  S_kr_errs[i] += calc_err_average(polar_d_frame['S_k_errs'].iloc[indices].values) 
#
#
#
## Plot angular average 
#plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_orderparams.txt')
#plt.figure(3)
#plt.errorbar(kr_uniq, S_kr.real, S_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='CL')
#plt.title('Angular Averaged S(k), ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 22)
#plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
#plt.ylabel(r'$S(k_{r}) $', fontsize = 24, fontweight = 'bold')
#plt.axvline(x = 2*_kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2 \tilde{\kappa} = ' + str(2*_kappa) + '$')
#plt.legend()
#plt.savefig('S_k_angular_avg.eps')
#plt.show()
# 
##
## Export list_x[1:stop_indx], corr_sorted[0][1:stop_indx]
#np.savetxt('S_k_00_angularAvg_data.dat', np.column_stack( [kr_uniq, S_kr.real, S_kr_errs.real] ))
# #
# #plt.figure(4)
# #plt.errorbar(kr_uniq, np.log10(S_kr.real), S_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='CL')
# #plt.title('Angular Averaged S(k), ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 22)
# #plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
# #plt.ylabel(r'log($S(k_{r}) $)', fontsize = 24, fontweight = 'bold')
# #plt.axvline(x = 2*_kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2 \tilde{\kappa} = ' + str(_kappa) + '$')
# #plt.legend()
# #plt.savefig('S_k_angular_avg_log.eps')
# #plt.show()
