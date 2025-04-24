import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
if 'Linux' in platform.platform():
  matplotlib.use('TkAgg')
else:
  matplotlib.rcParams['text.usetex'] = True
from scipy.stats import sem 
import glob 

# Import our custom package for CSbosons data analysis
from csbosons_data_analysis.field_analysis import *
from csbosons_data_analysis.import_parserinfo import *
from csbosons_data_analysis.error_propagation import *
from csbosons_data_analysis.time_grid import TimeGrid

# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path_image = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_dynamic_structure_factor.txt') 
style_path_data = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_data.txt') 


def visualize_rho_rt(displayPlots: bool, saveFigure: bool, N_samples_to_plot: int=10, z_fractional: float= 0.5):
  #### Begin script #### 
  # Script to load and plot correlation data 
  params = import_parser('input.yml')
  
  lattice = False  # Lattice vs. continuum model 
  realspace = True  # The operator we're trying to visualize is either in real space or k space 
  
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
  D = params['system']['Diffusivity']
  k = params['system']['Reaction_constant']
  
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
  
  # Species loop to plot the structure factors 
  for i, data in enumerate(rho_rt[0:N_species]):
    # Create a dictionary for each file, store the grid and necessary data 
    _data = {'x': x, 'y': y, 'z' : z, 'rho_rt': rho_rt[i], 'rho_rt_errs': rho_rt_errs[i]}
  
    # Plot a few snapshots of the density profile 
    # Want to plot the profiles for each sample point in real time  
    time_points = len(rho_rt[i][0,:])
    N_samples_to_plot += 1  
    if N_samples_to_plot > 1:
      indices = np.linspace(0, time_points-1, N_samples_to_plot).astype(int)
      rho_rt_subarray = rho_rt[i][:, indices]
      times = tgrid[indices]
    else:
      rho_rt_subarray = rho_rt[i][:, 0]
      times = tgrid[0]

    assert(len(rho_rt_subarray[0, :]) == N_samples_to_plot)
    assert(len(rho_rt_subarray[0, :]) == len(times))

    for n in range(N_samples_to_plot-1):
      rho_data = rho_rt_subarray[:, n].flatten()
      if(k == 0):
        title_postfix = ', $D = ' + str(D) + '$'
      else:
        title_postfix = ', $D = ' + str(D) + '$, $\mu = ' + str(k) + '$'

      if d > 1:
        if d == 2:
          rho_data.resize(Nx, Ny)
          rho_data = np.transpose(rho_data)
          rho_data = np.flip(rho_data, 0)
        else:
          assert(d==3)
          Lz = params['system']['CellLength-z'] 
          # Pick a slice along the z-axis and visualize a 2D cut; use z-fractional coordinate  
          # Default slice is in the middle of the cell along the z axis  
          rho_data = rho_data.reshape((Nx, Ny, Nz))
          z_slice_index = int(z_fractional*(Nz-1))
          rho_data = rho_data[:, :, z_slice_index]
          rho_data = np.transpose(rho_data) 
          title_postfix += ', $z = ' + str(z_fractional*Lz) + '$' 
 
      if(d > 1):
        plt.style.use(style_path_image)
        plt.figure(figsize=(4,4))
        plt.imshow(rho_data.real, cmap = 'magma', interpolation='none', extent=[np.min(x) ,np.max(x) ,np.min(y),np.max(y)]) 
        plt.xlabel('$x$', fontsize = 24)
        plt.ylabel('$y$', fontsize = 24, rotation=0, labelpad=10)
        plt.clim(0, np.max(rho_data.real))
        plt.colorbar()
      else:
        plt.style.use(style_path_data)
        plt.figure(figsize=(5,5))
        plt.plot(x, rho_data.real, linewidth = 2., color = 'k', label=r'$\rho(x)$')
        plt.xlabel('$x$', fontsize = 24)
        plt.ylabel(r'$\rho$', fontsize = 24, rotation = 0, labelpad = 12)
        plt.legend()

      if(tgrid[-1] > 1.): 
        round_to_dp = 1
      else:
        round_to_dp = 4

      if(N_species > 1):
        plt.title(r'$\rho_{' + str(i+1) + '}(r, t = ' + str(np.round(times[n], round_to_dp)) + ')$' + title_postfix, fontsize = 16)
      else:
        plt.title(r'$\rho(r, t = ' + str(np.round(times[n], round_to_dp)) + ')$' + title_postfix, fontsize = 16)

      if(saveFigure):
        plt.savefig('rho_r_teq' + str(np.round(times[n],round_to_dp)) + '.pdf', dpi=300)
      if(displayPlots):  
        plt.show()



if __name__ == "__main__":
  ''' Script to visualize the local density at various time points ''' 
  ''' Script to plot a few samples of the density profile in time. ''' 
  showPlots = True
  saveFigs = True
  N_samples = 8
  z_frac = 0.50  # between 0 and 1.
  visualize_rho_rt(showPlots, saveFigs, N_samples, z_frac)  


