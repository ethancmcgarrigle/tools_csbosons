import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
#matplotlib.rcParams['text.usetex'] = True
matplotlib.use('TkAgg')
import pdb
import pandas as pd 
from scipy.stats import sem 

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


if __name__ == "__main__":
  with open('input.yml') as infile:
    params = yaml.load(infile, Loader=yaml.FullLoader)
  
  # Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
  model = params['system']['ModelType']
  d = params['system']['Dim']
  mu = params['system']['mu']
  beta = params['system']['beta']
  
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

  #k2map_data = np.loadtxt('k2map.dat')
  #k2_data = k2map_data[:,6]
  k2_data = output_k2grid(lengths, gridpoints)
  
  E = 0.
  S = 0.
  N_check = 0.
  for k2 in k2_data:
    #E_i = -k2 * 6.05 # hbar^2 /2m for He-4
    E_i = k2 * 6.05 # hbar^2 /2m for He-4
    E += E_i / (np.exp(beta * (E_i - mu)) - 1.) 
  
    n_i = 1./(np.exp(beta * (E_i - mu)) - 1.)
    N_check += n_i
    S += (( (1. + n_i) * np.log(1 + n_i) ) - n_i * np.log(n_i) )
  
  
  print('N sum over occuption: ' + str(np.round(N_check, 3)))
  
  #print('Energy CL: ' + str(round(intensive_internal_E, 2)))
  print('Energy reference: ' + str(round(E,2)))
  
  print('Entropy reference: ' + str(round(S,2)))
  
  #print('Pressure: ' + str(round(pressure, 6)))
  #print('Pressure reference: ' + str(round((2./3.) * E/Vol, 6)))
  


