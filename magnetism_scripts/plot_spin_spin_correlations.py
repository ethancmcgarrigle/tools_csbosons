from scipy.optimize import curve_fit 
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

# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path_data = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_data.txt') 



def power_law(x, a, b):
    ''' F(x) = a x^{b} '''
    return a * np.power(x, b)

def exponential_fit(x, a, b):
    ''' F(x) = a e^{bx} '''
    return a * np.exp(x * b)



def process_correlator_data(corr_file, N_spatial, d, CL):
  file_list = [corr_file]
  spacegrid, C_list, C_err_list = process_data(file_list, N_spatial, CL, 200)

  x = spacegrid[0]
  y = spacegrid[1]
  z = spacegrid[2]
  correlator_data = C_list[0]

  # Would like to calculate all unique "r" vectors 
  r_vector = np.sqrt(x**2 + y**2 + z**2) 
  r_plot, C_r, C_r_errs = compute_angular_average(r_vector, correlator_data, C_err_list[0], True)  # True for correlator

  return r_plot, C_r, C_r_errs



def plot_correlator(r, data, data_errs, Fit = 'None'):
  ''' Function for plotting the correlation function C(|r|), normalized so that C(0) = 1.
      Option to apply and show a best-fit procedure. Current choices: power-law, exponential. 
      Fit = 'Exp' denotes an exponential fit.
      Fit = 'Power' denotes a power-law fit.'''

  half_indx = len(r)//2

  # Determine a best fit, based on user specified.
  if(Fit != 'None'):
    if(Fit == 'Exp'):
      r_0 = 0.
      print('Determining fit using exponential decay form')
      pars,cov = curve_fit(f=exponential_fit, xdata=r[0:half_indx], ydata=data[0:half_indx].real/data[0].real, p0=[0,0], bounds=(-np.inf, np.inf))
      print(pars)
    elif(Fit == 'Power'):        
      r_0 = 0.25   # avoid singularity at r = 0
      print('Determining fit using power-law decay form')
      pars,cov = curve_fit(f=power_law, xdata=r[1:half_indx], ydata=data[1:half_indx].real/data[0].real, p0=[0,-0.1], bounds=(-np.inf, np.inf))
      print(pars)
    r_fit = np.linspace(r_0, r[half_indx], 1000)

  # Plot angular average 
  print('Plotting the correlation data.')
  plt.style.use(style_path_data)
  plt.figure(figsize=(5,5))
  plt.errorbar(r[0:half_indx], data[0:half_indx].real/data[0].real, data_errs[0:half_indx].real/data[0].real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='Langevin')

  # Plot the fit 
  if(Fit == 'Exp'):
    plt.plot(r_fit, pars[0] * exponential_fit(r_fit, pars[0], pars[1]), color = 'r', linestyle='solid', linewidth = 2.0, label = 'Exponential fit: ' + r'$\xi = ' + str(round(-1. * pars[1], 2)) + '$') 
  elif(Fit == 'Power'):
    plt.plot(r_fit, pars[0] * (r_fit)**(pars[1]), color='r', linestyle = 'solid',linewidth = 2.0, label = 'Power Law: $ r^{' + str(round(pars[1],2)) + '}$')

  plt.title('Isotropic Spin-Spin Correlation', fontsize = 16)
  plt.xlabel('$|r|$', fontsize = 24, fontweight = 'bold')
  plt.ylabel(r'$C(r)$', fontsize = 24, fontweight = 'bold')
  #plt.savefig('spin-spin_correlator.eps')
  plt.legend()
  plt.show()



if __name__ == "__main__":
  ''' Script to load and visualize spin-spin correlation data''' 
  # Script to load and plot correlation data 
  params = import_parser('input.yml')
  
  # Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
  lattice = True
  grid_pts, dim = extract_grid_details(params, lattice) 
  N_spatial = calculate_Nspatial(grid_pts, dim)
  
  Nx = grid_pts[0] 
  Ny = grid_pts[1] 
  Nz = grid_pts[2] 
  
  system = params['system']['ModelType'] 
  _CL = params['simulation']['CLnoise']
  
  # Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
  T = float(params['system']['beta'])
  T = 1./T
  lattice = params['system']['lattice'] 
  ntau = params['system']['ntau'] 
  _isPlotting = True
  
  dirs = {0 : 'x', 1 : 'y', 2 : 'z'}
  
  if(system == 'HONEYCOMB'):
    # Retrieve the spin textures for each sublattice, and then combine into a single plot 
    num_basis_sites = 2
    basis_site_labels = {0: 'A', 1: 'B'}
  else:
    num_basis_sites = 1
    basis_site_labels = {0: 'A'}

  # Main loop to process the correlation data for each sublattice and each spin
  for K in range(0, num_basis_sites):
    # loop over each spin direction 
    S_file = 'C_rprime' + str(K) + '.dat' 
    r, C_r, C_r_errs = process_correlator_data(S_file, N_spatial, dim, _CL)

    plot_correlator(r, C_r, C_r_errs, 'Power')
  

