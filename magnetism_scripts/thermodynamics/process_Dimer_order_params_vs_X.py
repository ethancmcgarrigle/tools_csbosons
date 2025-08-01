import csv
#from mpmath import *
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
    suppress_output = True

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
    if(not suppress_output):
      print('Extracting Sk value at wavevector:')
      for nu in range(0, 3):
        print('k' + str(nu) + '  = ' + str(k_grid[nu][match_values_xy]))

    return Sk_data[0][match_values_xy] 




def gather_order_parameters(X, input_file, X_is_beta=True): 

  with open(input_file) as infile:
    master_params = yaml.load(infile, Loader=yaml.FullLoader)

  subtract_terms = True 

  # For each independent variable (e.g. temperature) extract the structure factor at appropriate k points; print a file with the value
  # Boolean to take average over all 3 spin directions' structure factors 
  
  # T S(pi, pi) S(0, pi) + errs?  
  if(X_is_beta):
    X_str = 'T'
  else:
    X_str = 'J2/J1'

  head_str = X_str + ' Dx Dy Dx_err Dy_err'
  
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
    fname = 'Dimer_' + str(Nspins) + '_J2J1_' + str(ratio) + '.dat'
  else:
    fname = 'Dimer_' + str(Nspins) + '_beta_' + str(B) + '.dat'
  
  # Allocate data 
  Dx = np.zeros_like(X)
  Dy = np.zeros_like(X)
  Dx_err = np.zeros_like(X)
  Dy_err = np.zeros_like(X)
  
  # loop over each spin direction for file name  
  Dk_negk_files = [] 
  Dk_files = [] 
  Dnegk_files = [] 
  dirs = {0 : 'x', 1 : 'y'} 
  
  for nu in range(2):
    Dk_negk_files.append('Dimer_' + dirs[nu] + dirs[nu] + '_k_-k.dat') 
    Dk_files.append('Dimer_' + dirs[nu] + '_k.dat') 
    Dnegk_files.append('Dimer_' + dirs[nu] + '_-k.dat') 
  
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
        Dx[i] = np.nan
        Dy[i] = np.nan
        Dx_err[i] = np.nan
        Dy_err[i] = np.nan
    else:
      print('Extracting dimer structure factor data')
      os.chdir(inner_path)
      # Doing correlators first 
      for nu, dimer_file in enumerate(Dk_negk_files):
        #Sk_data = np.loadtxt(inner_path + "/" + Sk_files[nu], unpack=True)
        # Load data and extract k grid 
        k_grid, Dk_avg, Dk_errs = process_data([dimer_file], Nspins, True, False)

        # Estimate error from imaginary parts 
        err = 0. 

        # Find pi,pi contribution
        if(nu == 0):
          tmp = 0. + 1j*0.
          tmp += extract_Sk_at_wavevector(Dk_avg, k_grid, [np.pi, 0., 0.])  # Dx peak at k = (pi, 0)
          Dx[i] += tmp.real
          Dx_err[i] = np.abs(tmp.imag)
        else:
          tmp = 0. + 1j*0.
          tmp += extract_Sk_at_wavevector(Dk_avg, k_grid, [0, np.pi, 0.])  # Dy peak at k = (0, pi)
          Dy[i] += tmp.real
          Dy_err[i] = np.abs(tmp.imag)

      # Doing subtraction part  
      for nu in range(2):
        # import +k part 
        k_grid, Dk_nu_avg, Dk_nu_errs = process_data([Dk_files[nu]], Nspins, True, False ) 
        if(nu == 0):
          tmp_k = 0. + 1j*0.
          tmp_k += extract_Sk_at_wavevector(Dk_nu_avg, k_grid, [np.pi, 0., 0.])  # Dx peak at k = (pi, 0)
        else:
          tmp_k = 0. + 1j*0.
          tmp_k += extract_Sk_at_wavevector(Dk_nu_avg, k_grid, [0., np.pi, 0.])  # Dx peak at k = (pi, 0)

        k_grid, Dnegk_nu_avg, Dnegk_nu_errs = process_data([Dnegk_files[nu]], Nspins, True, False ) 
        if(nu == 0):
          tmp_negk = 0. + 1j*0.
          tmp_negk += extract_Sk_at_wavevector(Dnegk_nu_avg, k_grid, [np.pi, 0., 0.])  # Dx peak at k = (pi, 0)
        else:
          tmp_negk = 0. + 1j*0.
          tmp_negk += extract_Sk_at_wavevector(Dnegk_nu_avg, k_grid, [0., np.pi, 0.])  # Dx peak at k = (pi, 0)

        if(subtract_terms):
          if(nu == 0):
            Dx[i] -= tmp_k*tmp_negk
          else:
            Dy[i] -= tmp_k*tmp_negk

      os.chdir('../')
  
  # Finally save all the data to text file, column layout 
  if(X_is_beta):
    np.savetxt(fname, np.column_stack([1./B, Dx, Dy, Dx_err, Dy_err]), header = head_str)
  else:
    np.savetxt(fname, np.column_stack([ratio, Dx, Dy, Dx_err, Dy_err]), header = head_str)




if __name__ == "__main__":
    # Script gather structure factor data for order parameter analysis 
    # Gather sweep parameter (independent variable) 
    T = np.array([1.0, 2.0]) 
    B = 1./T
    B = np.round(B, 7)
    B = np.sort(B)

    input_file = 'input.yml'
    usingTemperature = True 

    if(usingTemperature):
      gather_order_parameters(B, input_file, usingTemperature) 
    else:
      gather_order_parameters(J2, input_file, usingTemperature) 


