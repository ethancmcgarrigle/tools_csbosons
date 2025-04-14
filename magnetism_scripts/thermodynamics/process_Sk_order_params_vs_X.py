import csv
from mpmath import *
import subprocess
import os
import re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb
import yaml
import math
import pandas as pd
# Import our custom package for Csbosons data analysis
from csbosons_data_analysis.field_analysis import *


def extract_Sk_at_wavevector(Sk_data: np.ndarray, k_grid: list[np.ndarray], k: list[float]) -> float:
    ''' Function to return the unique structure factor value at a desired wavevector, i.e. returns S(k)'''
    ''' Sk_data is a 2D numpy array, first (d) columns are k data'''
    N_gridpoints = len(Sk_data[0])

    # Argmin() is not a great strategy overall and should be used minimally. Instead, find the values in each direction of k_grid closest to the desired ones  
    k_indices = [] # index where an approximate match is found 
    k_approx = []  # k value at that approximate match index in the k_grid  
    for nu in range(0, 3):
      k_indices.append(np.abs(k[nu] - k_grid[nu]).argmin())
      k_approx.append(k_grid[nu][k_indices[nu]])

    # Now, find all indices in each direction where k_grid == k_approx
    k_matching = []
    for nu in range(0, 3):
      k_matching.append(np.where(k_grid[nu] == k_approx[nu])[0]) 

    # Find index between these matching arrays where the values all match  
    match_values_xy = np.intersect1d(k_matching[0], k_matching[1], return_indices = True)[0][0]
    match_values_yz = np.where(k_matching[2] == match_values_xy)[0][0]
    assert match_values_xy == match_values_yz, 'Warning, could not locate the unique k wavevector.'
    # sanity check:
    print('Extracting Sk value at wavevector:')
    for nu in range(0, 3):
      print('k' + str(nu) + '  = ' + str(k_grid[nu][match_values_xy]))

    return Sk_data[0][match_values_xy] 




def gather_order_parameters(X, input_file, avg_spin_direction=True, X_is_beta=True): 

  with open(input_file) as infile:
    master_params = yaml.load(infile, Loader=yaml.FullLoader)

  # For each independent variable (e.g. temperature) extract the structure factor at appropriate k points; print a file with the value
  # Boolean to take average over all 3 spin directions' structure factors 
  print('Averaging over each spin direction? '  + str(avg_spin_direction) )
  
  # T S(pi, pi) S(0, pi) + errs?  
  if(X_is_beta):
    X_str = 'T'
  else:
    X_str = 'J2/J1'

  if(avg_spin_direction):
    head_str = X_str + ' S(pi, pi) S(0, pi)'
  else:
    head_str = X_str + ' Sx(pi, pi) Sy(pi, pi) Sz(pi, pi) Sx(0, pi) Sy(0, pi) Sz(0, pi)' 
  
  # Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
  Nx = master_params['system']['NSitesPer-x']
  Ny = master_params['system']['NSitesPer-y']
  Nz = master_params['system']['NSitesPer-z']
  dimension = master_params['system']['Dim']
  J1 = master_params['system']['Jnn']
  if(X_is_beta):
    J2 = master_params['system']['Jnnn']
    B = X
  else:
    B = master_params['system']['beta']
    J2 = X
  
  ratio = np.round(J2/J1, 3)

  Nspins = Nx 
  if dimension > 1  :
    Nspins *= Ny 
    if dimension > 2 : 
      Nspins *= Nz 
  
  if(X_is_beta):
    fname = str(Nspins) + '_J2J1_' + str(ratio) + '.dat'
  else:
    fname = str(Nspins) + '_beta_' + str(B) + '.dat'
  
  # Allocate data 
  if(avg_spin_direction):
    S_pi_pi = np.zeros_like(X)
    S_pi_0 = np.zeros_like(X)
  else:
    S_pi_pi = []
    S_pi_0 = []
    for nu in range(0, 3):
      S_pi_pi.append(np.zeros_like(X))
      S_pi_0.append(np.zeros_like(X))
  
  # loop over each spin direction for file name  
  Sk_files = [] 
  dirs = {0 : 'x', 1 : 'y', 2 : 'z'}
  
  K = 0 # only 1 basis site # TODO: extend to multiple basis sites (e.g. honeycomb lattice )
  for nu in range(0, 3):
    Sk_files.append('S' + str(dirs[nu]) + '_k_S' + str(dirs[nu]) + '_-k_' + str(K) + '.dat') 
  
  runtime_cutoff = 50.0
  
  for i, X_ in enumerate(X):
    # directory name 
    if(X_is_beta):
      inner_path = 'B_' + str(X_)
    else:
      inner_path = 'J2_' + str(X_)
    print(inner_path)   
  
    ops_data = np.loadtxt(inner_path + "/operators0.dat", unpack=True)
    if(ops_data.ndim > 1):
      runtime = ops_data[2,-1] # column 2, last entry 
    else: 
      runtime = 0.0 
  
    if(runtime < runtime_cutoff):
      print('Not enough runtime. Setting observables to nan')
      if(avg_spin_direction):
        S_pi_pi[i] = np.nan
        S_pi_0[i] = np.nan
      else:
        for nu in range(0, 3):
          S_pi_pi[i][nu] = np.nan
          S_pi_0[i][nu] = np.nan
    else:
      print('Extracting structure factor data')
      os.chdir(inner_path)
      for nu, spin_file in enumerate(Sk_files):
        #Sk_data = np.loadtxt(inner_path + "/" + Sk_files[nu], unpack=True)
        # Load data and extract k grid 
        k_grid, Sk_avg, Sk_errs = process_data([spin_file], Nspins, True, False)
        # Find pi,pi contribution
        S_pi_pi_tmp = extract_Sk_at_wavevector(Sk_avg, k_grid, [np.pi, np.pi, 0.]) 
        S_pi_0_tmp = extract_Sk_at_wavevector(Sk_avg, k_grid, [np.pi, 0., 0.]) 
        S_pi_0_tmp += extract_Sk_at_wavevector(Sk_avg, k_grid, [0., np.pi, 0.]) 
        
        if(avg_spin_direction):
          S_pi_pi[i] += S_pi_pi_tmp/3.
          S_pi_0[i] += S_pi_0_tmp/3.
        else:
          S_pi_pi[nu][i] = S_pi_pi_tmp
          S_pi_0[nu][i] = S_pi_0_tmp
  
      os.chdir('../')
  
  # Finally save all the data to text file, column layout 
  if(X_is_beta):
    if(avg_spin_direction):
      np.savetxt(fname, np.column_stack([1./B, S_pi_pi, S_pi_0]), header = head_str)
    else: 
      np.savetxt(fname, np.column_stack([1./B, *S_pi_pi, *S_pi_0]), header = head_str)
  else:
    if(avg_spin_direction):
      np.savetxt(fname, np.column_stack([ratio, S_pi_pi, S_pi_0]), header = head_str)
    else: 
      np.savetxt(fname, np.column_stack([ratio, *S_pi_pi, *S_pi_0]), header = head_str)




if __name__ == "__main__":
    # Script gather structure factor data for order parameter analysis 
    # Gather sweep parameter (independent variable) 
 #    T = np.linspace(0.5, 16.0, 20)
 #    T2 = np.linspace(0.1, 1.75, 12)
 #    T = np.append(T, T2)
 #    B = 1./T
 #    B = np.round(B, 7)
 #    B = np.sort(B)
    J2 = np.linspace(0.0, 0.8, 20)
    J2 = np.append(J2, np.linspace(0.9, 2.0, 20))
    J2 = np.round(J2, 4)
    J2 *= -1.

    input_file = 'input.yml'
    avg_spin_directions = True
    usingTemperature = False 

    if(usingTemperature):
      gather_order_parameters(B, input_file, avg_spin_directions, usingTemperature) 
    else:
      gather_order_parameters(J2, input_file, avg_spin_directions, usingTemperature) 


