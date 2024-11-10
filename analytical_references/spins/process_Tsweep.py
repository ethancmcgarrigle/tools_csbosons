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
import sys
helper_fxns_path = "/home/emcgarrigle/tools_csbosons/analytical_references/spins"
sys.path.append(helper_fxns_path)
from q_vs_classical_reference import * 

## This function runs statistics on the runs accessed (i.e. parameter sweep). Then it collects the relevant data and plots it at the end

def sech(x):
  return 1/(np.cosh(x))

def calc_err_division(x, y, x_err, y_err):
    # x/y 
    # assumes x and y are real 

    # Calculate error using standard error formula 
    result = np.sqrt( ((-x * y_err / (y**2))**2 ) + (x_err/y)**2)
    return result


def calc_err_multiplication(x, y, x_err, y_err, z):
    # z = x * y
    result = z * np.sqrt( ((x_err/x)**2)  + ((y_err/y)**2) ) 
    return result



def calc_err_addition(x_err, y_err):
    # Error propagation function for x + y 
    #result = 0.
    # assumes x and y are real 

    # Calculate error using standard error formula 
    result = np.sqrt( (x_err**2) + (y_err**2) )
    return result


with open('input.yml') as infile:
  master_params = yaml.load(infile, Loader=yaml.FullLoader)



T = np.arange(0.1, 14.0, 0.1)
#T = np.arange(0.1, 20.0, 0.1)
beta = 1./T 
beta = np.round(beta, 6) 

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
dt = master_params['simulation']['dt']  
ntau = master_params['system']['ntau']
hz = master_params['system']['hz']
S = master_params['system']['spin']
#Nx = master_params['system']['-x']
#N = master_params['system']['N']
#Vol = Lx * Ly
stepper = master_params['simulation']['DriverType']
ensemble = master_params['system']['ensemble']
if(ensemble == 'CANONICAL'):
  _isCanonical = True
else:
  _isCanonical = False

num_steps = master_params['simulation']['numtsteps']
dimension = master_params['system']['Dim']
_isCleaning = False


#apply_ADT = master_params['simulation']['apply_ADT']
apply_ADT = False 
total_CL_time = master_params['simulation']['Total_CL_Time']
if(apply_ADT):
  runlength_max = total_CL_time
else:
  runlength_max = num_steps * dt

# set up a data frame with observables as keys, and "beta" (1/T) as the rows 
# Set the index (rows) to be "L" since we want to conveniently return a list for each L, rather than grab 1 number at a time from B when plotting across L 

_isPrinting = False
# Read 1 operators file and grab the strings after operators0.dat
sample_ops_file_path = 'beta_' + str(beta[0]) + '/operators0.dat'
ops_file = open(sample_ops_file_path, 'r')
lines = ops_file.readlines()
topline = lines[0]
ops_file.close()

observables_list = topline.split(' ') 
observables_list = observables_list[5:] 
#observables_list[-1] = 'ImSF_density1'
observables_list[-1] = observables_list[-1].replace('\n', '')
print('Observables list: \n', observables_list)
print()

obs_list = observables_list

# Create data frame containing observables means (and errors) for each method 
method = ['CL S = 7/2']

means_dframe = pd.DataFrame(columns = observables_list, index = method) # contains the noise averages from Cmethod sampling  
errs_dframe = pd.DataFrame(columns = observables_list, index = method) # contains the noise averages from Cmethod sampling  

# for each key and row, fill it with a list of np.zeros() of length(L)  
for k, m_ in enumerate(method):
  means_dframe.loc[k] = list( np.zeros(( len(observables_list), len(beta) ) ))
  errs_dframe.loc[k] = list( np.zeros(( len(observables_list), len(beta) ) ))


print('Data frame example: \n\n', means_dframe)
# Other properties we want to calculate 
runtime_cutoff = 500.0

for k, m_ in enumerate(method):
  for i, beta_ in enumerate(beta):
    # directory name 
  
    inner_path = 'beta_' + str(beta_)
    print(inner_path)   
    ops_data = np.loadtxt(inner_path + "/operators0.dat", unpack=True)
    if(ops_data.ndim > 1):
      runtime = ops_data[2,-1] # column 2, last entry 
    else: 
      runtime = 0.0 
  
    output_filename = 'data0'
  
    if(apply_ADT):
      output_filename += '_ADT'
  
    output_filename += '.dat'
  
    if(_isCleaning):
      print('removing old processed output files')
      if(apply_ADT):
        subprocess.call('rm ' + inner_path + '/data0_ADT.dat', shell = True)
      else:
        subprocess.call('rm ' + inner_path + '/data0.dat', shell = True)
      #subprocess.call('rm ' + inner_path + '/N_SOC.dat', shell = True)
    
    # Run ADT conversion script if ADT  
    ADT_reweight = 'python3 ~/CSBosonsCpp/tools/stats_ADT.py -f ' + inner_path + '/operators0.dat -w 13 > ' + inner_path + '/' + output_filename 
  
    if(apply_ADT):
      cmd_string = ADT_reweight 
    else:
      cmd_string = "python3 ~/CSBosonsCpp/tools/stats.py -f " + inner_path + "/operators0.dat -o "
      for obs in observables_list:
        cmd_string += obs 
        cmd_string += ' ' 
      cmd_string += '-a -q > ' + inner_path + '/' + output_filename
  
    check_path = inner_path + '/' + output_filename
  
    if (not os.path.exists(check_path)): 
      if (runtime == 0 or runtime < runtime_cutoff):
        print("None or too little runtime, inserting NaN for observables")
        for obs in observables_list:
          means_dframe.loc[k][obs][i] = math.nan 
          errs_dframe.loc[k][obs][i] = math.nan 
  
      else:
        print("processing " + inner_path)
        if not os.path.exists(check_path):
          subprocess.call(cmd_string, shell = True)
    
    if(runtime != 0 and runtime > runtime_cutoff):
      in_file = open(inner_path + '/' + output_filename, "r")
  
      # TODO change to loop 
      tmp = in_file.read()
      in_file.close()
      tmp = re.split(r"\s+", tmp)
      tmp = tmp[0:-1]
      tmp = tuple(map(float, tmp))
    
      # Put the observables data into the dataframes
      ctr = 0
      for obs in observables_list:
        means_dframe.loc[k][obs][i] = tmp[ctr] 
        errs_dframe.loc[k][obs][i] = tmp[ctr+1]
        ctr += 2

ctr = 1

# Canonical ensemble -- plot the free energy 
plt.style.use('~/CSBosonsCpp/tools/python_plot_styles_examples/plot_style_data.txt')

#markers = ['o', 'x', '*']
markers = ['o', '*', 'p']
colors = ['r', 'b', 'k']

beta_ref = np.linspace(min(T), max(T), 500)
beta_ref = 1./beta_ref

Mz_ref = np.zeros(len(beta_ref))
U_ref = np.zeros(len(beta_ref))
Cv_ref = np.zeros(len(beta_ref))
chi_zz_ref = np.zeros(len(beta_ref))

Mz_ref_cl = np.zeros(len(beta_ref))
U_ref_cl = np.zeros(len(beta_ref))
Cv_ref_cl= np.zeros(len(beta_ref))
chi_zz_ref_cl = np.zeros(len(beta_ref))
for i, _B in enumerate(beta_ref):
  # Quantum 
  Mz_ref[i] = Mz(_B, hz, S) 
  U_ref[i] = U(Mz_ref[i], hz)
  Cv_ref[i] = U_squared(_B, hz, S)
  Cv_ref[i] -= (U_ref[i] ** 2)
  Cv_ref[i] *= (_B * _B)
  chi_zz_ref[i] = Mz_squared(_B, hz, S)
  chi_zz_ref[i] -= (Mz_ref[i] ** 2)
  chi_zz_ref[i] *= _B 
  # Classical
  Mz_ref_cl[i] = Mz_classical(_B, hz, S) 
  U_ref_cl[i] = -Mz_ref_cl[i] 
  Cv_ref_cl[i] = Cv_classical(_B, hz, S) 
  chi_zz_ref_cl[i] = chi_zz_classical(_B, hz, S)


plt.figure(figsize=(6,6))
for k, m_ in enumerate(method):
  _Mz = means_dframe.loc[k]['ReMz']
  Mz_errs = errs_dframe.loc[k]['ReMz']
  plt.errorbar(1./beta, _Mz, Mz_errs, marker=markers[k], color = colors[k], markersize = 6, elinewidth=0.5, linewidth = 0.0, label = method[k]) 
  plt.plot(1./beta_ref, Mz_ref, color = colors[0], linestyle = 'solid', linewidth = 1.0, label = 'Quantum Reference') 
  plt.plot(1./beta_ref, Mz_ref_cl, color = colors[1], linestyle = 'solid', linewidth = 1.0, label = 'Classical Reference') 
plt.xlabel(r'$T$',fontsize = 24) 
plt.ylabel(r'$M_{z}$', fontsize = 24) 
plt.legend()
plt.show()


plt.figure(figsize=(6,6))
for k, m_ in enumerate(method):
  _U = means_dframe.loc[k]['RebetaU']
  U_errs = errs_dframe.loc[k]['RebetaU']
  plt.errorbar(1./beta, _U / beta, U_errs / beta, marker=markers[k], color = colors[k], markersize = 6, elinewidth=0.5, linewidth = 0.0, label = method[k]) 
  plt.plot(1./beta_ref, U_ref, color = colors[0], linestyle = 'solid', linewidth = 1.0, label = 'Quantum Reference') 
  plt.plot(1./beta_ref, U_ref_cl, color = colors[1], linestyle = 'solid', linewidth = 1.0, label = 'Classical Reference') 
plt.xlabel(r'$T$',fontsize = 24)
plt.ylabel(r'$U$', fontsize = 24) 
plt.legend()
plt.show()


plt.figure(figsize=(6,6))
for k, m_ in enumerate(method):
  _Mz = means_dframe.loc[k]['ReMz']
  _Mz_sq = means_dframe.loc[k]['ReMz_squared']
  Mz_errs = errs_dframe.loc[k]['ReMz']
  Mz_sq_errs = errs_dframe.loc[k]['ReMz_squared']
  chi_zz = _Mz_sq - (_Mz**2)
  chi_zz *= beta
  chi_zz_errs = calc_err_addition(Mz_sq_errs, calc_err_multiplication(_Mz, _Mz, Mz_errs, Mz_errs, _Mz**2) ) * beta 
  plt.errorbar(1./beta, chi_zz, chi_zz_errs, marker=markers[k], color = colors[k], markersize = 6, elinewidth=0.5, linewidth = 0.0, label = method[k]) 
  plt.plot(1/beta_ref, chi_zz_ref, color = colors[0], linestyle = 'solid', linewidth = 1.0, label = 'Quantum Reference') 
  plt.plot(1./beta_ref, chi_zz_ref_cl, color = colors[1], linestyle = 'solid', linewidth = 1.0, label = 'Classical Reference') 
plt.xlabel(r'$T$',fontsize = 24)
plt.ylabel(r'$\chi_{zz}$', fontsize = 24)
plt.legend()
plt.show()


plt.figure(figsize=(6,6))
for k, m_ in enumerate(method):
  _U = means_dframe.loc[k]['RebetaU']
  U_errs = errs_dframe.loc[k]['RebetaU']
  U_sq = means_dframe.loc[k]['RebetaU_squared']
  U_sq_errs = errs_dframe.loc[k]['RebetaU_squared']
  Cv = U_sq - (_U**2)
  Cv_errs = calc_err_addition(U_sq_errs, calc_err_multiplication(_U, _U, U_errs, U_errs, _U**2) ) 
  plt.errorbar(1./beta, Cv, Cv_errs, marker=markers[k], color = colors[k], markersize = 6, elinewidth=0.5, linewidth = 0.0, label = method[k]) 
  plt.plot(1./beta_ref, Cv_ref, color = colors[0], linestyle = 'solid', linewidth = 1.0, label = 'Quantum Reference') 
  plt.plot(1./beta_ref, Cv_ref_cl, color = colors[1], linestyle = 'solid', linewidth = 1.0, label = 'Classical Reference') 
plt.xlabel(r'$T$',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$C_{V}$', fontsize = 20, fontweight = 'bold')
plt.legend()
plt.show()
