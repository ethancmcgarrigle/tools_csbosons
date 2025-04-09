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
from scipy.fft import fft,ifft


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
realspace = True

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
files = glob.glob("Dynamic*")
files.sort()
print(files)
r_grid, rho_rt, rho_rt_errs = process_data(files, N_spatial, _CL, realspace, 2, len(tgrid))

print('Number of time points: ' + str(len(tgrid)))


N_species = len(files)

x = r_grid[0]
y = r_grid[1]
z = r_grid[2]

w_grid = tgrid.get_reciprocol_grid()

w_0 = w_grid[0]
w_max = w_grid[-1]
assert(w_0 == 0.)

# Species loop to plot the structure factors 
for i, data in enumerate(rho_rt[0:N_species]):
  # Create a dictionary for each file, store the grid and necessary data 
  _data = {'x': x, 'y': y, 'z' : z, 'rho_rt': rho_rt[i], 'rho_rt_errs': rho_rt_errs[i]}


   
  # Plot a few snapshots of the density profile 
  N_samples = 6 
  # Want to plot the profiles for each sample point in real time  
  sample_every = len(rho_rt[i][0, :])//(N_samples - 1)

  rho_rt_subarray = rho_rt[i][:, ::sample_every]
  times = tgrid[::sample_every]
  assert(len(rho_rt_subarray[0, :]) == (N_samples-1))
  assert(len(rho_rt_subarray[0, :]) == len(times))

  for n in range(N_samples-1):
    rho_data = rho_rt_subarray[:, n].flatten()
    avg_rho = 1. 
    if d > 1:
      if d == 2:
        rho_data.resize(Nx, Ny)
        rho_data = np.transpose(rho_data)
        rho_data = np.flip(rho_data, 0)

    plt.style.use(style_path_image)
    plt.figure(figsize=(3.38583, 3.38583))
    plt.imshow(rho_data.real/avg_rho, cmap = 'magma', interpolation='none', extent=[np.min(x) ,np.max(x) ,np.min(y),np.max(y)]) 
    if(N_species > 1):
      plt.title(r'$\rho_{' + str(i+1) + '}(r, t = ' + str(times[n]) + ')$' , fontsize = 16)
    else:
      plt.title(r'$\rho(r, t = ' + str(np.round(times[n], 4)) + ')$', fontsize = 16)
    #plt.savefig('Fig1_mu_86mm.eps', dpi = 300)
    plt.clim(0, np.max(rho_data.real/avg_rho))
    plt.colorbar()
    plt.show()
    


