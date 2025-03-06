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
style_path_image = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_dynamic_structure_factor.txt') 

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
  # Create a dictionary for each file, store the grid and necessary data 
  _data = {'kx': kx, 'ky': ky, 'kz' : kz, 'S_k': S_kw[i], 'S_k_errs': S_kw_errs[i]}
  # Make an unsorted copy of the flattened array for angular averaging 
  Sk_omega_unsorted = np.zeros_like(S_kw[i])
  Sk_omega_unsorted += (S_kw[i]) 
  # Compute errors  
  structure_factor_errs = np.zeros_like(Sk_omega_unsorted)
  structure_factor_errs += S_kw_errs[i] 

  # Perform angular averaging over the k index 
  ''' Plot the angular average''' 
  kr = np.sqrt(kx**2 + ky**2 + kz**2)
  kr_plot, S_kr_omega, S_kr_omega_errs = compute_angular_average(kr, Sk_omega_unsorted, structure_factor_errs, False, len(tgrid)) 

  np.savetxt('dynamical_structure_factor_data.dat', S_kr_omega.real) 

  # S(k,w) vector has S[0,0] as w = 0 and k = 0. So the top left corner is true origin. We need to rotate counter clockwise by 90 degrees to get the correct behavior 
  S_kr_omega = np.rot90(S_kr_omega)

  saveFigs = False 
  
  # Plot angular average 
  #plt.style.use(style_path_image)
  plt.style.use('./plot_style_dynamic_structure_factor.txt')
  map_style = 'inferno'
  plt.figure(figsize=(6, 6))
  plt.imshow(S_kr_omega.real,  aspect='auto', extent=[kr_plot[0], kr_plot[-1], w_0, w_max], cmap = map_style)
  plt.title(r'Dynamical Structure Factor: $S(k, \omega)$', fontsize = 22)
  plt.xlabel('$k$', fontsize = 32) 
  plt.ylabel(r'$\omega$', fontsize = 32, rotation = 0, labelpad = 16) 
  plt.colorbar(fraction=0.046, pad=0.04)
  if(saveFigs):
    plt.savefig('S_k_omega.pdf', dpi=300)
  plt.show()

  plt.figure(figsize=(6, 6))
  plt.imshow(S_kr_omega.real,  aspect='auto', extent=[kr_plot[0], kr_plot[-1], w_0, w_max], cmap = map_style, norm=LogNorm())
  plt.title(r'Dynamical Structure Factor: $S(k, \omega)$', fontsize = 22)
  plt.xlabel('$k$', fontsize = 32) 
  plt.ylabel(r'$\omega$', fontsize = 32, rotation = 0, labelpad = 16) 
  plt.colorbar(fraction=0.046, pad=0.04)
  if(saveFigs):
    plt.savefig('S_k_omega_log.pdf', dpi=300)
  plt.show()



