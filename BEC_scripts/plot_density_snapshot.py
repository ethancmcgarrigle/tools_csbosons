import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
if 'Linux' in platform.platform():
  matplotlib.use('TkAgg')
else:
  matplotlib.rcParams['text.usetex'] = True
import glob 

# Import our custom package for Csbosons data analysis
from csbosons_data_analysis.field_analysis import *
from csbosons_data_analysis.import_parserinfo import *
# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_spins.txt') 



### Begni script 
params = import_parser('input.yml')

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
lattice = False
grid_pts, d = extract_grid_details(params, lattice) 

Nx = grid_pts[0] 
Ny = grid_pts[1] 
Nz = grid_pts[2] 

system = params['system']['ModelType'] 
d = params['system']['Dim']
_CL = params['simulation']['CLnoise']

N_spatial = calculate_Nspatial(grid_pts, d)
T = 1./float(params['system']['beta']) 


# Get the number of density files (species) in the system 
files = glob.glob("density*")
files.sort()

spacegrid, rho_list, rho_err_list = process_data(files, N_spatial, _CL, True, 5)
for i in range(len(rho_list)):
  if d > 1:
    if d == 2:
      rho_list[i].resize(Nx, Ny)
      rho_list[i] = np.transpose(rho_list[i])
      rho_list[i] = np.flip(rho_list[i], 0)


plt.style.use(style_path)

x = spacegrid[0]
y = spacegrid[1]
z = spacegrid[2]

normalize = True
showTotal = True

if(showTotal):
  rho_total = np.zeros_like(rho_list[0].real)

for i, density_data in enumerate(rho_list):
  if(normalize):
    avg_rho = np.mean(density_data.real)
  else:
    avg_rho = 1.
  
  plt.figure(figsize=(3.38583, 3.38583))
  plt.imshow(density_data.real/avg_rho, cmap = 'magma', interpolation='none', extent=[np.min(x) ,np.max(x) ,np.min(y),np.max(y)]) 

  if(len(rho_list) > 1):
    plt.title(r'$\rho_{' + str(i+1) + '}(r)$' , fontsize = 16)
  else:
    plt.title(r'$\rho(r)$', fontsize = 16)
  #plt.savefig('Fig1_mu_86mm.eps', dpi = 300)
  plt.clim(0, np.max(rho_list[0].real/avg_rho))
  plt.colorbar()
  plt.show()

  rho_total += density_data.real


if(showTotal and len(rho_list) > 1): 
  plt.figure(figsize=(3.38583, 3.38583))
  if(normalize):
    avg_rho = np.mean(rho_total)
  else:
    avg_rho = 1.
  plt.imshow(rho_total/avg_rho, cmap = 'magma', interpolation='none', extent=[np.min(x) ,np.max(x) ,np.min(y),np.max(y)]) 
  
  if(len(rho_list) > 1):
    plt.title(r'Total density', fontsize = 16)
  plt.clim(0, np.max(rho_list[0].real/avg_rho))
  plt.colorbar()
  plt.show()
  

