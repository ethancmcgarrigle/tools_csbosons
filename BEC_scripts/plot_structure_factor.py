import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
if 'Linux' in platform.platform():
  matplotlib.use('TkAgg')
else:
  matplotlib.rcParams['text.usetex'] = True
import pandas as pd 
from scipy.stats import sem 
from matplotlib.colors import LogNorm
import glob 

# Import our custom package for Csbosons data analysis
from csbosons_data_analysis.field_analysis import *
from csbosons_data_analysis.import_parserinfo import *
from csbosons_data_analysis.error_propagation import *


#### Begin script #### 

# Script to load and plot correlation data 
params = import_parser('input.yml')

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
lattice = False
realspace = False
grid_pts, d = extract_grid_details(params, lattice) 

Nx = grid_pts[0] 
Ny = grid_pts[1] 
Nz = grid_pts[2] 

system = params['system']['ModelType'] 
d = params['system']['Dim']
_CL = params['simulation']['CLnoise']

N_spatial = calculate_Nspatial(grid_pts, d)
T = 1./float(params['system']['beta']) 

if('SOC' in system):
  kappa = params['system']['kappa']
  print( 'Retrieving SOC strength' )

# Get the number of density files (species) in the system 
files = glob.glob("S_k*")
files.sort()
print(files)
kgrid, Sk, Sk_errs = process_data(files, N_spatial, _CL, realspace)

files = glob.glob("rho_k*")
files.sort()
print(files)
kgrid, rho_k, rho_k_errs = process_data(files, N_spatial, _CL, realspace)

files = glob.glob("rho_-k*")
files.sort()
print(files)
kgrid, rho_negk, rho_negk_errs = process_data(files, N_spatial, _CL, realspace)


N_species = len(files)

kx = kgrid[0]
ky = kgrid[1]
kz = kgrid[2]

plt.style.use('~/CSBosonsCpp/tools/python_plot_styles_examples/plot_style_spins.txt')

# Species loop to plot the structure factors 
for i, data in enumerate(Sk[0:N_species]):
  # Create a dictionary for each file, store the grid and necessary data 
  _data = {'kx': kx, 'ky': ky, 'kz' : kz, 'S_k': Sk[i], 'S_k_errs': Sk_errs[i], 
          'rho_k' : rho_k[i], 'rho_k_errs' : rho_k_errs[i], 'rho_-k' : rho_negk[i], 'rho_-k_errs' : rho_negk_errs[i]}
  # Create a data frame 
  d_frame = pd.DataFrame.from_dict(_data)
  d_frame.sort_values(by=['kx', 'ky', 'kz'], ascending = True, inplace=True) 

  # TODO: Handle different dimensionalities 
  # Redefine numpy array post sorting
  Sk_sorted = np.array(d_frame['S_k']) - (np.array(d_frame['rho_k'])*np.array(d_frame['rho_-k'])) 

  # Make an unsorted copy of the flattened array for angular averaging 
  Sk_unsorted = np.zeros_like(Sk[i])
  Sk_unsorted += (Sk[i] - (rho_k[i]*rho_negk[i])) 
  # Compute errors  
  structure_factor_errs = np.zeros_like(Sk_unsorted)
  structure_factor_errs += calc_err_multiplication(rho_k[i], rho_negk[i], rho_k_errs[i],  rho_negk_errs[i]) 
  structure_factor_errs = calc_err_addition(Sk_errs[i], structure_factor_errs) 

  Sk_sorted.resize(Nx, Ny)
  Sk_sorted = np.transpose(Sk_sorted)
  Sk_sorted = np.flip(Sk_sorted, 0)

  # Plot the structure factor  
  plt.figure(figsize=(6.77166, 6.77166))
  plt.imshow(Sk_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
  plt.title(r'$S_{' + str(i) + str(i) + '} (\mathbf{k})$', fontsize = 30)
  plt.xlabel('$k_x$', fontsize = 32) 
  plt.ylabel('$k_y$', fontsize = 32)
  if('SOC' in system):
    plt.xlim(-4.*kappa,4.*kappa)
    plt.ylim(-4.*kappa,4.*kappa)
  plt.colorbar(fraction=0.046, pad=0.04)
  #plt.savefig('Sk.eps')
  plt.show()

  # Plot structure factor in log-scale  
  plt.figure(figsize=(6.77166, 6.77166))
  plt.imshow(Sk_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)], norm=LogNorm()) 
  plt.title(r'$S_{' + str(i) + str(i) + '} (\mathbf{k})$', fontsize = 30)
  plt.xlabel('$k_x$', fontsize = 32) 
  plt.ylabel('$k_y$', fontsize = 32)
  if('SOC' in system):
    plt.xlim(-4.*kappa,4.*kappa)
    plt.ylim(-4.*kappa,4.*kappa)
  plt.colorbar(fraction=0.046, pad=0.04)
  #plt.savefig('Sk_log.eps')
  plt.show()

  ##### Plot the angular average 
  # Calculate angular average 
  kr = np.sqrt(kx**2 + ky**2 + kz**2)
  theta = np.arctan(ky/kx) # rads 
  kr_plot, S_kr, S_kr_errs = compute_angular_average(kr, theta, Sk_unsorted, structure_factor_errs, 2) 
  
  # Plot angular average 
  plt.figure(figsize=(6.77166/2, 6.77166/2))
  plt.errorbar(kr_plot, S_kr.real, S_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='Langevin')
  plt.title('Angular averaged structure factor', fontsize = 22)
  plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
  plt.ylabel(r'$S(k_{r}) $', fontsize = 24, fontweight = 'bold')
  if('SOC' in system):
    plt.axvline(x = 2*kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2\tilde{\kappa} = ' + str(2.*kappa) + '$')
  #plt.savefig('S_k_angular_avg.eps')
  plt.legend()
  plt.show()

  # np.savetxt('S_k_00_figure.dat', S_k_sorted.real)
  #np.savetxt('S_k_00_angularAvg_data.dat', np.column_stack( [kr_uniq, S_kr.real, S_kr_errs.real] ))





# TODO: add off-diagonal and total structure factor 

