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
from csbosons_data_analysis.time_grid import TimeGrid

# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path_image = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_spins.txt') 
style_path_data = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_data.txt') 

#### Begin script #### 

# Script to load and plot correlation data 
params = import_parser('input.yml')

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
lattice = False
realspace = False

# Extract spatial grid details 
grid_pts, d = extract_grid_details(params, lattice) 

# Extract time grid details 
tgrid = extract_time_grid_details(params)

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
kgrid, S_kw, S_kw_errs = process_data(files, N_spatial, _CL, realspace, 2, len(tgrid))


print('Number of time points: ' + str(len(tgrid)))


N_species = len(files)

kx = kgrid[0]
ky = kgrid[1]
kz = kgrid[2]


w_grid = tgrid.get_reciprocol_grid()

w_0 = w_grid[0]
w_max = w_grid[-1]
assert(w_0 == 0.)

# Species loop to plot the structure factors 
for i, data in enumerate(S_kw[0:N_species]):
  plt.style.use(style_path_image)
  # Create a dictionary for each file, store the grid and necessary data 
  #_data = {'kx': kx, 'ky': ky, 'kz' : kz, 'S_k': Sk[i], 'S_k_errs': Sk_errs[i], 
  #        'rho_k' : rho_k[i], 'rho_k_errs' : rho_k_errs[i], 'rho_-k' : rho_negk[i], 'rho_-k_errs' : rho_negk_errs[i]}
  _data = {'kx': kx, 'ky': ky, 'kz' : kz, 'S_k': S_kw[i], 'S_k_errs': S_kw_errs[i]}

  # Create a data frame 
 #  d_frame = pd.DataFrame.from_dict(_data)
 #  d_frame.sort_values(by=['kx', 'ky', 'kz'], ascending = True, inplace=True) 

  # Redefine numpy array post sorting
  #Sk_sorted = np.array(d_frame['S_k']) 
  #Sk_sorted = np.array(d_frame['S_k']) - (np.array(d_frame['rho_k'])*np.array(d_frame['rho_-k'])) 

  # Make an unsorted copy of the flattened array for angular averaging 
  Sk_omega_unsorted = np.zeros_like(S_kw[i])
  Sk_omega_unsorted += (S_kw[i]) 
  # Compute errors  
  structure_factor_errs = np.zeros_like(Sk_omega_unsorted)
  #structure_factor_errs += calc_err_multiplication(rho_k[i], rho_negk[i], rho_k_errs[i],  rho_negk_errs[i]) 
  #structure_factor_errs = calc_err_addition(Sk_errs[i], structure_factor_errs) 
  structure_factor_errs += S_kw_errs[i] 

  # Perform angular averaging over the k index 
  ''' Plot the angular average''' 
  # Calculate angular average 
  kr = np.sqrt(kx**2 + ky**2 + kz**2)
  #theta = np.arctan(ky/kx) # rads 
  kr_plot, S_kr_omega, S_kr_omega_errs = compute_angular_average(kr, Sk_omega_unsorted, structure_factor_errs, False, len(tgrid)) 
  
  # Plot angular average 
  #plt.style.use(style_path_image)
  #plt.style.use(style_path_image)
  plt.figure(figsize=(6, 6))
  plt.imshow(S_kr_omega.real, origin = 'lower', aspect='auto', extent=[kr[0], kr[-1], w_0, w_max], cmap ='magma')
  plt.title(r'Dynamical Structure Factor: $S(k, \omega)$', fontsize = 22)
  plt.xlabel('$k$', fontsize = 32) 
  plt.ylabel(r'$\omega$', fontsize = 32, rotation = 0, labelpad = 16) 
  plt.colorbar()
  #plt.colorbar(fraction=0.046, pad=0.04)
  #plt.zlabel(r'$S(|k|, \omega) $', fontsize = 24, fontweight = 'bold')
 #  if('SOC' in system):
 #    plt.axvline(x = 2*kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2\tilde{\kappa} = ' + str(2.*kappa) + '$')
  #plt.savefig('S_k_angular_avg.eps')
  #plt.legend()
  plt.show()

 
 #
 #  Sk_sorted.resize(Nx, Ny)
 #  Sk_sorted = np.transpose(Sk_sorted)
 #  Sk_sorted = np.flip(Sk_sorted, 0)
 #
 #  # Plot the structure factor  
 #  plt.figure(figsize=(6.77166, 6.77166))
 #  plt.imshow(Sk_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
 #  plt.title(r'$S_{' + str(i) + str(i) + '} (\mathbf{k})$', fontsize = 30)
 #  plt.xlabel('$k_x$', fontsize = 32) 
 #  plt.ylabel('$k_y$', fontsize = 32)
 #  if('SOC' in system):
 #    plt.xlim(-4.*kappa,4.*kappa)
 #    plt.ylim(-4.*kappa,4.*kappa)
 #  plt.colorbar(fraction=0.046, pad=0.04)
 #  #plt.savefig('Sk.eps')
 #  plt.show()
 #
 #  # Plot structure factor in log-scale  
 #  plt.figure(figsize=(6.77166, 6.77166))
 #  plt.imshow(Sk_sorted.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)], norm=LogNorm()) 
 #  plt.title(r'$S_{' + str(i) + str(i) + '} (\mathbf{k})$', fontsize = 30)
 #  plt.xlabel('$k_x$', fontsize = 32) 
 #  plt.ylabel('$k_y$', fontsize = 32)
 #  if('SOC' in system):
 #    plt.xlim(-4.*kappa,4.*kappa)
 #    plt.ylim(-4.*kappa,4.*kappa)
 #  plt.colorbar(fraction=0.046, pad=0.04)
 #  #plt.savefig('Sk_log.eps')
 #  plt.show()

 #  ##### Plot the angular average 
 #  # Calculate angular average 
 #  kr = np.sqrt(kx**2 + ky**2 + kz**2)
 #  #theta = np.arctan(ky/kx) # rads 
 #  kr_plot, S_kr, S_kr_errs = compute_angular_average(kr, Sk_unsorted, structure_factor_errs) 
 #  
 #  # Plot angular average 
 #  plt.style.use(style_path_data)
 #  plt.figure(figsize=(6.77166, 6.77166))
 #  plt.errorbar(kr_plot, S_kr.real, S_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='Langevin')
 #  plt.title('Angular averaged structure factor', fontsize = 22)
 #  plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
 #  plt.ylabel(r'$S(k_{r}) $', fontsize = 24, fontweight = 'bold')
 #  if('SOC' in system):
 #    plt.axvline(x = 2*kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2\tilde{\kappa} = ' + str(2.*kappa) + '$')
 #  #plt.savefig('S_k_angular_avg.eps')
 #  plt.legend()
 #  plt.show()
 #
  # np.savetxt('S_k_00_figure.dat', S_k_sorted.real)
  #np.savetxt('S_k_00_angularAvg_data.dat', np.column_stack( [kr_uniq, S_kr.real, S_kr_errs.real] ))





# TODO: add off-diagonal and total structure factor 

