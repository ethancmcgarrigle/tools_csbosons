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
from scipy.fftpack import dct 
from matplotlib.colors import LinearSegmentedColormap

# Import our custom package for Csbosons data analysis
from csbosons_data_analysis.field_analysis import *
from csbosons_data_analysis.import_parserinfo import *
from csbosons_data_analysis.error_propagation import *
from csbosons_data_analysis.time_grid import TimeGrid

# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path_image = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_dynamic_structure_factor.txt') 

def custom_dct(signal): 
    ''' Custom discrete cosine transform for a signal '''
    ''' - expects signal as a 1D array of length N,
    - uses 1 / N as the scaling factor 
    - cosine argument: t_{j} * omega_{n} = (2pi / N) * j * n
    - "j" is the time index, n is the frequency index '''
    output = np.zeros_like(signal)
    N = len(signal)
    for n in range(len(signal)):
        # Vectorize the sum over j:
        cosine_vector = np.zeros(N)
        cosine_vector = np.cos(np.pi * 2. * n * np.array(range(N))/ N)
        # Accumulate product into output 
        output[n] = np.sum(cosine_vector * signal) / N

    return output
 

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
kgrid, S_kt, S_kt_errs = process_data(files, N_spatial, _CL, realspace, 2, len(tgrid))

N_species = len(files)

files = glob.glob("rho_*")
files.sort()
print(files)
_, rho_kt_list, rho_kt_errs_list = process_data(files, N_spatial, _CL, realspace, 2, len(tgrid))

print('Number of time points: ' + str(len(tgrid)))

kx = kgrid[0]
ky = kgrid[1]
kz = kgrid[2]


w_grid = tgrid.get_reciprocol_grid()

w_0 = w_grid[0]
w_max = w_grid[-1]
assert(w_0 == 0.)

# Species loop to plot the structure factors 
for i, data in enumerate(S_kt[0:N_species]):
  # Create a dictionary for each file, store the grid and necessary data 
  _data = {'kx': kx, 'ky': ky, 'kz' : kz, 'S_k': S_kt[i], 'S_k_errs': S_kt_errs[i]}
  # Make an unsorted copy of the flattened array for angular averaging 
  Sk_t_unsorted = np.zeros_like(S_kt[i])
  Sk_t_unsorted += (S_kt[i]) 

  rho_kt = np.zeros_like(Sk_t_unsorted)
  rho_negkt = np.zeros_like(Sk_t_unsorted)

  rho_kt += rho_kt_list[2*i]
  rho_negkt += rho_kt_list[2*i+1]

  Sk_t_unsorted -= (rho_kt*rho_negkt)

  # Compute errors  
  structure_factor_errs = np.zeros_like(Sk_t_unsorted)
  structure_factor_errs += S_kt_errs[i] 


  # Perform angular averaging over the k index 
  ''' Plot the angular average''' 
  kr = np.sqrt(kx**2 + ky**2 + kz**2)
  kr_plot, S_kr_t, S_kr_t_errs = compute_angular_average(kr, Sk_t_unsorted, structure_factor_errs, False, len(tgrid)) 

  # Copy the time signal for later processing  
  S_kr_omega = np.copy(S_kr_t)
  #S_kr_omega = np.copy(S_kr_t[:, :len(tgrid)//2])

  #np.savetxt('dynamical_structure_factor_data.dat', S_kr_omega.real) 
  #np.savetxt('S_kr_t_data.dat', S_kr_omega.real) 

  # S(k,t) vector has S[0,0] as t = 0 and k = 0. So the top left corner is true origin. 
  # We need to rotate counter clockwise by 90 degrees to get the desired image  
  S_kr_t = np.rot90(S_kr_t)

  saveFigs = True 

  print('Plotting the structure factor in S(k,t) representation')
  
  # Plot angular average 
  plt.style.use(style_path_image)

  # Create a custom colormap to map negative and zero values to black  
  map_style = 'inferno'

  plt.figure(figsize=(6, 6))
  y_0 = 0. 
  y_max = tgrid.return_tmax()
  ylabel = '$t$'
  title = r'Dynamical Structure Factor: $S(k, t)$'
  plt.imshow(S_kr_t.real,  aspect='auto', extent=[kr_plot[0], kr_plot[-1], y_0, y_max], cmap = map_style)
  plt.title(title, fontsize = 22)
  plt.xlabel('$k$', fontsize = 32) 
  plt.ylabel(ylabel, fontsize = 32, rotation = 0, labelpad = 16) 
  plt.colorbar(fraction=0.046, pad=0.04)
  if(saveFigs):
    plt.savefig('dynamical_structure_factor_k_t.pdf', dpi=300)
  plt.show()

 #  plt.figure(figsize=(6, 6))
 #  plt.imshow(S_kr_t.real,  aspect='auto', extent=[kr_plot[0], kr_plot[-1], y_0, y_max], cmap = map_style, norm=LogNorm())
 #  plt.title(title, fontsize = 22)
 #  plt.xlabel('$k$', fontsize = 32) 
 #  plt.ylabel(ylabel, fontsize = 32, rotation = 0, labelpad = 16) 
 #  plt.colorbar(fraction=0.046, pad=0.04)
 #  if(saveFigs):
 #    plt.savefig('dynamical_structure_factor_log.pdf', dpi=300)
 #  plt.show()
  
  print('Plotting the structure factor in S(k,w) representation')
  map_style = 'inferno'
#  inferno = plt.cm.get_cmap('inferno')
#
#  # Create new colormap with black as the lowest value
#  colors = np.vstack(([0, 0, 0, 1], inferno(np.linspace(0, 1, 256))))
#  custom_inferno = LinearSegmentedColormap.from_list('custom_inferno', colors)
#  map_style = custom_inferno # 'inferno'

  S_kr_omega_thresholded = np.copy(S_kr_omega)
  _threshold = False 
  if(_threshold):
    threshold = 0. 
    S_kr_omega_thresholded[S_kr_omega < threshold] = 0.

  assert(len(S_kr_omega_thresholded[:, 0]) == len(kr_plot))
  for k in range(len(kr_plot)):
    S_kr_omega[k, :] = dct(S_kr_omega_thresholded[k, :], norm='forward') * (tgrid.return_tmax())
    #S_kr_omega[k, :] = custom_dct(S_kr_omega[k, :]) 

  S_kr_omega = np.rot90(S_kr_omega, k = 1) # Rotate by 90 degrees for desired image (origin at bottom left corner)  
  plt.figure(figsize=(6, 6))
  y_0 = w_0
  y_max = w_max
  ylabel = r'$\omega$'
  title = r'Dynamical Structure Factor: $S(k, \omega)$'
  plt.imshow(S_kr_omega.real,  aspect='auto', extent=[kr_plot[0], kr_plot[-1], y_0, y_max], cmap = map_style)
  plt.title(title, fontsize = 22)
  plt.xlabel('$k$', fontsize = 32) 
  plt.ylabel(ylabel, fontsize = 32, rotation = 0, labelpad = 16) 
  plt.ylim(0, 50.)
  plt.colorbar(fraction=0.046, pad=0.04)
  if(saveFigs):
    plt.savefig('dynamical_structure_factor_k_omega.pdf', dpi=300)
  plt.show()





