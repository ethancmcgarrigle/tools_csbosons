import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
if 'Linux' in platform.platform():
  matplotlib.use('TkAgg')
else:
  matplotlib.rcParams['text.usetex'] = True
import pandas as pd 

# Import our custom package for Csbosons data analysis
from csbosons_data_analysis.field_analysis import *
from csbosons_data_analysis.import_parserinfo import *
from csbosons_data_analysis.error_propagation import *

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

files = ['n_k.dat']
kgrid, Nk, Nk_errs = process_data(files, N_spatial, _CL, realspace)

#_kappa = params['system']['kappa'] 
kx = kgrid[0]
ky = kgrid[1]
kz = kgrid[2]
kx_unique = np.unique(kx)
ky_unique = np.unique(ky)
kz_unique = np.unique(kz)


_data = {'kx': kx, 'ky': ky, 'kz' : kz, 'n_k': Nk[0], 'n_k_errs': Nk_errs[0]}
d_frame = pd.DataFrame.from_dict(_data)
d_frame.sort_values(by=['kx', 'ky', 'kz'], ascending = True, inplace=True) 

# TODO: Handle different dimensionalities 
# Redefine numpy array post sorting
n_k_sorted = np.array(d_frame['n_k']) 
n_k_sorted.resize(Nx, Ny)
n_k_sorted = np.transpose(n_k_sorted)
n_k_sorted = np.flip(n_k_sorted, 0)


plt.style.use('~/CSBosonsCpp/tools/python_plot_styles_examples/plot_style_spins.txt')

# Plot the N_k distribution in k-space  
plt.figure(1)
plt.imshow(n_k_sorted.real/np.sum(n_k_sorted.real), cmap = 'hot', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
plt.title('Momentum Distribution $n(k)$', fontsize = 22)
plt.xlabel('$k_x$', fontsize = 22) 
plt.ylabel('$k_y$', fontsize = 22)
if('SOC' in system):
  plt.xlim(-2.*kappa,2.*kappa)
  plt.ylim(-2.*kappa,2.*kappa)
plt.colorbar()
#plt.savefig('n_k_mixed_1K.eps')
plt.show()


# Compute the angular average and plot 
kr = np.sqrt(kx**2 + ky**2 + kz**2)
theta = np.arctan(ky/kx) # rads 

kr_plot, n_kr, n_kr_errs = compute_angular_average(kr, theta, Nk[0], Nk_errs[0], 2) 

# Plot angular average 
plt.figure(2)
plt.errorbar(kr_plot, n_kr.real/np.sum(n_k_sorted.real), n_kr_errs.real/np.sum(n_k_sorted.real), marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='Langevin')
plt.title('Angular averaged momentum distribution', fontsize = 22)
plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
plt.ylabel(r'$n(k_{r})/N $', fontsize = 24, fontweight = 'bold')
if('SOC' in system):
  plt.axvline(x = kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$\tilde{\kappa} = ' + str(kappa) + '$')
#plt.savefig('n_k_angular_averaged.eps')
plt.legend()
plt.show()

