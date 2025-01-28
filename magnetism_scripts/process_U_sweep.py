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
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import sem
from scipy.optimize import curve_fit
## This function runs statistics on the runs accessed (i.e. parameter sweep). Then it collects the relevant data and plots it at the end

def sech(x):
  return 1/(np.cosh(x))

def calc_err_division(x, y, x_err, y_err):
    # x/y 
    # assumes x and y are real 

    # Calculate error using standard error formula 
    result = np.sqrt( ((-x * y_err / (y**2))**2 ) + (x_err/y)**2)
    return result


def calc_err_multiplication(x, y, x_err, y_err):
    z = x * y
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


method = ['1SI_TAMED_SPECTRALTAU']
method_label = ['1SI-Exp-Tamed'] 
U = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]) 
exp_damped = ['true']
method_label = ['exp_damped']

# Force norms 
force_norm = np.zeros(len(U))

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
ntau = master_params['system']['ntau']
beta = master_params['system']['beta']
stepper = master_params['simulation']['DriverType']
ensemble = master_params['system']['ensemble']
num_steps = master_params['simulation']['numtsteps']
dimension = master_params['system']['Dim']
S = master_params['system']['spin']
Jnn = master_params['system']['Jnn']
_isCleaning = False
dt = master_params['simulation']['dt']
Nx = 10*10

#apply_ADT = master_params['simulation']['apply_ADT']
apply_ADT = False 
# set up a data frame with observables as keys, and "beta" (1/T) as the rows 
# Set the index (rows) to be "L" since we want to conveniently return a list for each L, rather than grab 1 number at a time from B when plotting across L 

_isPrinting = False
# Read 1 operators file and grab the strings after operators0.dat
sample_ops_file_path = method_label[0] + '/U_' + str(U[0]) + '/operators0.dat'
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

means_dframe = pd.DataFrame(columns = observables_list, index = method) # contains the noise averages from Cmethod sampling  
errs_dframe = pd.DataFrame(columns = observables_list, index = method) # contains the noise averages from Cmethod sampling  

# for each key and row, fill it with a list of np.zeros() of length(L)  
for k, m_ in enumerate(method):
  means_dframe.loc[k] = list( np.zeros(( len(observables_list), len(U) ) ))
  errs_dframe.loc[k] = list( np.zeros(( len(observables_list), len(U) ) ))


print('Data frame example: \n\n', means_dframe)
# Other properties we want to calculate 
runtime_cutoff = 1.0

for k, m_ in enumerate(method):
  for i, U_ in enumerate(U):
    # directory name 
  
    inner_path = method_label[k] + '/U_' + str(U_)
    print(inner_path)   
    ops_data = np.loadtxt(inner_path + "/operators0.dat", unpack=True)
    if(ops_data.ndim > 1):
      runtime = ops_data[2,-1] # column 2, last entry 
    else: 
      runtime = 0.0 
  
    output_filename = 'data0'


    force_data = np.loadtxt(inner_path + "/forces.dat", unpack=True)
    force_norm[i] = np.mean(force_data[2])
  
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
markers = ['D', 's', 'o', 'p']
colors = ['k', 'r', 'b', 'g']


plt.figure(figsize=(6,6))
for k, m_ in enumerate(method):
  Mz = means_dframe.loc[k]['ReM_abs'] 
  Mz_errs = errs_dframe.loc[k]['ReM_abs']
 #  Mz2 = means_dframe.loc[k]['ReMz_squared'] 
 #  Mz2_errs = errs_dframe.loc[k]['ReMz_squared'] 
 #  chi_zz = Mz2 - (Mz*Mz*Nx)
 #  chi_zz *= B
 #  chi_zz_errs = calc_err_addition(Mz2_errs, calc_err_multiplication(Mz, Mz, Mz_errs, Mz_errs, Mz**2))
 #  chi_zz_errs *= B 
  plt.errorbar(U, Mz, Mz_errs, marker=markers[k], color = colors[k], markersize = 6, elinewidth=0.5, linewidth = 0.5, label = method_label[k]) 
plt.xlabel(r'$U$',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$|M|$', fontsize = 20, fontweight = 'bold')
#plt.xscale('log')
plt.ylim(-1.0, 5.0)
plt.legend()
plt.show()



plt.figure(figsize=(6,6))
for k, m_ in enumerate(method):
  plt.errorbar(U, force_norm, Mz_errs, marker=markers[k], color = colors[k], markersize = 6, elinewidth=0.5, linewidth = 0.5, label = method_label[k]) 
plt.xlabel(r'$U$',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$|F|$', fontsize = 20, fontweight = 'bold')
#plt.xscale('log')
plt.legend()
plt.show()


plt.figure(figsize=(6,6))
for k, m_ in enumerate(method):
  plt.errorbar(U, force_norm*dt, Mz_errs, marker=markers[k], color = colors[k], markersize = 6, elinewidth=0.5, linewidth = 0.5, label = method_label[k]) 
plt.xlabel(r'$U$',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$|F|\Delta t$', fontsize = 20, fontweight = 'bold')
plt.yscale('log')
plt.legend()
plt.show()



plt.figure(figsize=(6,6))
betaU_gs = -Jnn * (S**2) * dimension 
for k, m_ in enumerate(method):
  betaU = means_dframe.loc[k]['RebetaU'] 
  betaU_errs = errs_dframe.loc[k]['RebetaU']
  # Internal energy 
  plt.errorbar(U, betaU, betaU_errs, marker=markers[k], color = colors[k], markersize = 6, linewidth = 1.0, elinewidth=0.5, label = method_label[k]) 
plt.axhline(y = betaU_gs, color = 'black', linestyle = 'dashed', linewidth = 2.0)
plt.ylabel(r'$\beta U$', fontsize = 20, fontweight = 'bold')
plt.xlabel(r'$U$',fontsize = 20, fontweight = 'bold')
#plt.ylim(0., 0.50)
#plt.xscale('log')
plt.ylim(-50.0, 10.0)
plt.legend()
plt.show()

plt.figure(figsize=(6,6))
for k, m_ in enumerate(method):
  betaU = means_dframe.loc[k]['RebetaU'] 
  betaU_errs = errs_dframe.loc[k]['RebetaU']
  U2 = means_dframe.loc[k]['RebetaU_squared'] 
  U2_errs = errs_dframe.loc[k]['RebetaU_squared']
  Cv = (U2 - (betaU**2))*(Nx)
  Cv_errs = calc_err_addition(U2_errs, calc_err_multiplication(betaU, betaU, betaU_errs, betaU_errs)) 
  # Internal energy 
  plt.errorbar(U, Cv, Cv_errs, marker=markers[k], color = colors[k], markersize = 6, linewidth = 1.0, elinewidth=0.5, label = method_label[k]) 
plt.ylabel(r'$C_{v}$', fontsize = 20, fontweight = 'bold')
plt.xlabel(r'$U$',fontsize = 20, fontweight = 'bold')
#plt.ylim(0., 0.50)
plt.ylim(-1.0, 100.0)
#plt.xscale('log')
plt.legend()
plt.show()

