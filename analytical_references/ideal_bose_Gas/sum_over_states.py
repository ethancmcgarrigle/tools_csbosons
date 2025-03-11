import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
import pdb
import pandas as pd 
from scipy.stats import sem 
import platform
if 'Linux' in platform.platform():
  matplotlib.use('TkAgg')
else:
  matplotlib.rcParams['text.usetex'] = True
import glob 

# Import our custom package for Csbosons data analysis
from csbosons_data_analysis.field_analysis import *
from csbosons_data_analysis.import_parserinfo import *
from csbosons_data_analysis.time_grid import TimeGrid

# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path_image = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_dynamic_structure_factor.txt') 

''' Script to compute an ideal Bose gas reference thermodynamics. Reads input parameters from local input.yml file
    Outputs the average particle number, energy for a given inverse temperature (beta), chemical potential, box size'''


def output_k2grid(L_list, N_list):
    dim = len(L_list)
    assert dim == len(N_list), 'Incorrect number of spatial grid parameters'
    ''' Assumes a cubic implementation: L = Lx = Ly = Lz, N = Nx = Ny = Nz for d == 3'''
    Nx = N_list[0]
    L = L_list[0]

    # Set up the spatial/k-grids
    if(Nx > 3):
      n_grid = np.append( np.arange(0, Nx/2 + 1, 1) , np.arange(-Nx/2 + 1, 0, 1) ) # artifical +1 required for 2nd argument 
    elif(Nx == 1):
      n_grid = np.array([0]) 
    dk = np.pi * 2 / L
    
    x_grid = np.arange(0., L, L/Nx) 
    kx_grid = n_grid * dk
    assert(len(x_grid) == len(kx_grid))
    if(dim > 1):
      if(dim > 2):
        z_grid = x_grid 
        kz_grid = kx_grid 
      y_grid = x_grid # assumes cubic mesh  
      ky_grid = kx_grid 

    # these are not synced up with k-grid 
    if(dim > 1):
      X,Y = np.meshgrid(x_grid, y_grid)
      if(dim > 2):
        X,Y,Z = np.meshgrid(x_grid, y_grid, z_grid)
    else:
      X = x_grid
    
    _k2_grid = np.zeros(Nx**dim)
    _k2_grid_v2 = np.zeros(Nx**dim)
    if(dim > 1):
      KX, KY = np.meshgrid(kx_grid, ky_grid) 
      _k2_grid_v2 += (KX*KX + KY*KY).flatten()
      if(dim > 2):
        KX, KY, KZ = np.meshgrid(kx_grid, ky_grid, kz_grid) 
        _k2_grid_v2 += (KX*KX + KY*KY + KZ*KZ).flatten()
    else:
      KX = kx_grid 
      _k2_grid_v2 += (KX*KX).flatten() 
    _k2_grid += _k2_grid_v2

    # return k^2 grid 
    return _k2_grid



def output_kgrid(L_list, N_list):
    dim = len(L_list)
    assert dim == len(N_list), 'Incorrect number of spatial grid parameters'
    ''' Assumes a cubic implementation: L = Lx = Ly = Lz, N = Nx = Ny = Nz for d == 3'''
    Nx = N_list[0]
    L = L_list[0]

    # Set up the spatial/k-grids
    if(Nx > 3):
      n_grid = np.append( np.arange(0, Nx/2 + 1, 1) , np.arange(-Nx/2 + 1, 0, 1) ) # artifical +1 required for 2nd argument 
    elif(Nx == 1):
      n_grid = np.array([0]) 
    dk = np.pi * 2 / L
    
    kx_grid = n_grid * dk
    if(dim > 1):
      ky_grid = kx_grid 
      if(dim > 2):
        kz_grid = kx_grid 
      else:
        kz_grid = np.array([0.])
    else:
      ky_grid = np.array([0.])
      kz_grid = np.array([0.])

    return np.meshgrid(kx_grid, ky_grid, kz_grid) 



def N_B_q(q, beta, mu, lamb):
  ''' Calculates the Bose occupation factor for a given 
      - beta (inverse temperature) 
      - momentum state "q"
      - chemical potential mu ''' 
  Eq = lamb * (q**2) 
  return 1./(np.exp(beta * (Eq - mu)) - 1.) 


def N_B_Eq(Eq, beta, mu):
  ''' Calculates the Bose occupation factor for a given 
      - beta (inverse temperature) 
      - momentum state "q"
      - chemical potential mu ''' 
  return 1./(np.exp(beta * (Eq - mu)) - 1.) 


def calculate_Sk_t(t_grid: TimeGrid, k_grid, beta, mu, lamb):
  ''' Calculates the ideal gas dynamical structure factor S(k,t)'''
  ''' Outputs a 2D array, where the columns are the time points and rows are wavevectors.'''
  ''' Assumes a cubic implementation ''' 
  kx = k_grid[0]
  ky = k_grid[1]
  kz = k_grid[2]

  print(kx.flatten())
  print(ky.flatten())
  print(kz.flatten())
  #print(len(kx.flatten()))
  #print(len(ky.flatten()))
  #print(len(kz.flatten()))

  k2_grid = (kx*kx + ky*ky + kz*kz).flatten()
  N_wavevectors = len(k2_grid)

  Sk_t = np.zeros((N_wavevectors, len(t_grid)), dtype=np.complex128) 

  print('Calculating S(k,t) for ' + str(N_wavevectors) + ' wavevectors')

  kpq = np.copy(k_grid) # container for wavevector grid
  tmpfield = np.zeros(N_wavevectors, dtype=np.complex128)

  # TODO: Currently brute forced. Need to vectorize 
  for j, time in enumerate(t_grid[:]):
    # loop through the wavevectors in each direction to fill in S(k) at a given time 
    for i in range(N_wavevectors):
      # Index i represents the ith wavevector in a list of unique, flattened wavevectors 
        # Need to map from the unique index i to a vector element of kx,ky,kz
      # Get the current wavevector
      kx_idx, ky_idx, kz_idx = map_flat_to_vector(i)
      k = np.array([kx.flatten()[kx_idx], ky.flatten()[ky_idx], kz.flatten()[kz_idx]])
 #      if(j == 0):
 #        print('k = ' + str(np.abs(k)))
#    for ix, kx_val in enumerate(kx.flatten()):
#      for iy, ky_val in enumerate(ky.flatten()):
#        for iz, kz_val in enumerate(kz.flatten()):
          # S(k,t) This is the particular wave vector value k 
      #k = np.array([kx_val, ky_val, kz_val])

          # For this k, sum on q (the whole k grid), considering states in the given k grid only 
      kpq = np.copy(k_grid) # q  
      for nu in range(3):
        kpq[nu] += k[nu]  # k + q 

      E_kpq = lamb*(kpq[0]*kpq[0] + kpq[1]*kpq[1] + kpq[2]*kpq[2]).flatten() 
      E_q = lamb*(k2_grid)
 #      if(j == 0):
 #        print((E_kpq - E_q))
      # Compute the S(k,t) contribution 
      #tmpfield.fill(0.)
      tmpfield = np.exp(-1j * (E_kpq - E_q)*time)
      tmpfield *= (1. + N_B_Eq(E_q, beta, mu)) 
      tmpfield *= N_B_Eq(E_kpq, beta, mu) 
#      if(j == 50):
#        print(tmpfield)
#        print(t_grid[j])
      Sk_t[i, j] += np.sum(tmpfield) # performs sum over q  

  return Sk_t


def map_flat_to_vector(flat_indx: int) -> int:
    ''' Maps an index from a flattened array to a vector
       - Returns the corresponding x, y, z indices
       - Assumes a cubic implementation '''  
    i = flat_indx // (Nz * Ny)
    j = (flat_indx // Nz) % Ny
    k = flat_indx % Nz
    return i, j, k 
    


if __name__ == "__main__":
  with open('input.yml') as infile:
    params = yaml.load(infile, Loader=yaml.FullLoader)
  
  # Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
  model = params['system']['ModelType']
  d = params['system']['Dim']
  mu = params['system']['mu']
  beta = params['system']['beta']

  try: 
    dimensionless = parser['system']['dimensionless']
    print('Running a dimensionless model? ' + str(dimensionless))
  except:
    print('Dimensionless keyword not found, setting to false.')
    dimensionless = False 

  if(dimensionless): 
    mu = 1. * np.sign(mu) 
  
  if(model == 'BOSE_HUBBARD_CUBIC'):
    Nx = params['system']['NSitesPer-x'] 
    Vol = Nx
    Ny = params['system']['NSitesPer-y'] 
    Nz = params['system']['NSitesPer-z'] 
    if d > 1 :
      Vol *= Ny
      if d > 2 :
        Vol *= Nz
  else:
    Lx = params['system']['CellLength-x']
    Vol = Lx
    Ly = params['system']['CellLength-y']
    Lz = params['system']['CellLength-z']
    Nx = params['simulation']['Nx'] 
    Ny = params['simulation']['Ny'] 
    Nz = params['simulation']['Nz'] 
    lengths = [Lx]
    gridpoints = [Nx]
    if d > 1 :
      lengths.append(Ly)
      gridpoints.append(Ny)
      Vol *= Ly
      if d > 2 :
        lengths.append(Lz)
        gridpoints.append(Nz)
        Vol *= Lz

  Nx_list = [Nx, Ny, Nz]

  lamb = 1.  

  if('KELDYSH' or 'keldysh' in model):
    # Extract time grid details
    try:
      tgrid = extract_time_grid_details(params)
    except:
      tgrid = np.zeros(1)
    tmax = tgrid[-1] 
    Nt = len(tgrid)

    kx, ky, kz = output_kgrid(lengths, gridpoints)
    k_grid = [kx, ky, kz]
    # Computes S(k,t) where k = 0, t = 0 corresponds to top left corner. k =0, t > 0 is on the top row, k \neq 0, t = 0 is the first column.
    # Therefore, must rotate by 90-degrees counterclockwise 
    Sk_t_out = calculate_Sk_t(tgrid, k_grid, beta, mu, lamb)
    saveFigs = False 

    ''' Plot the angular average''' 
    kr = np.sqrt(kx*kx + ky*ky + kz*kz).flatten()
    kr_plot, S_kr_t, S_kr_t_errs = compute_angular_average(kr, Sk_t_out, np.zeros_like(Sk_t_out), False, Nt) 

    S_kr_t = np.rot90(S_kr_t) 

    # Plot angular average 
    plt.style.use(style_path_image)
    map_style = 'inferno'
    plt.figure(figsize=(6, 6))
    #plt.imshow(Sk_t.real,  aspect='auto', extent=[kr_plot[0], kr_plot[-1], w_0, w_max], cmap = map_style)
    plt.imshow(S_kr_t.real, extent=[kr_plot[0], kr_plot[-1], 0., tmax], cmap = map_style)
    plt.title(r'Dynamical Structure Factor: $S(k, t)$', fontsize = 22)
    plt.xlabel('$k$', fontsize = 32) 
    plt.ylabel(r'$t$', fontsize = 32, rotation = 0, labelpad = 16) 
    plt.colorbar(fraction=0.046, pad=0.04)
    if(saveFigs):
      plt.savefig('S_kr_t.pdf', dpi=300)
    plt.show()

 #    plt.figure(figsize=(6, 6))
 #    #plt.imshow(Sk_t.real,  aspect='auto', extent=[kr_plot[0], kr_plot[-1], w_0, w_max], cmap = map_style)
 #    plt.imshow(S_kr_t.imag, extent=[kr_plot[0], kr_plot[-1], 0., tmax], cmap = map_style)
 #    plt.title(r'Dynamical Structure Factor: $S(k, t)$', fontsize = 22)
 #    plt.xlabel('$k$', fontsize = 32) 
 #    plt.ylabel(r'$t$', fontsize = 32, rotation = 0, labelpad = 16) 
 #    plt.colorbar(fraction=0.046, pad=0.04)
 #    if(saveFigs):
 #      plt.savefig('S_kr_t.pdf', dpi=300)
 #    plt.show()



  #k2map_data = np.loadtxt('k2map.dat')
  #k2_data = k2map_data[:,2]
  k2_data = output_k2grid(lengths, gridpoints)

  E = 0.
  S = 0.
  N_check = 0.

  E_k = k2_data * lamb
  N_k = N_B_Eq(E_k, beta, mu)
  N_total = np.sum(N_k)

  for k2 in k2_data:
    E_i = k2 * lamb   # hbar^2 /2m 
    n_i = N_B_Eq(E_i, beta, mu)
    E += (E_i * n_i) 
    S += (( (1. + n_i) * np.log(1 + n_i) ) - n_i * np.log(n_i) )
  
  print()
  print('Sum over states equilibrium results \n')  
  print('N sum over occuption: ' + str(np.round(N_total, 3)))
  
  #print('Energy CL: ' + str(round(intensive_internal_E, 2)))
  print('Energy reference: ' + str(round(E,2)))
  
  print('Entropy reference: ' + str(round(S,2)))
  
  #print('Pressure: ' + str(round(pressure, 6)))
  #print('Pressure reference: ' + str(round((2./3.) * E/Vol, 6)))


