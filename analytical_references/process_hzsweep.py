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
## This function runs statistics on the runs accessed (i.e. parameter sweep). Then it collects the relevant data and plots it at the end

def sech(x):
  return 1/(np.cosh(x))


def Mz(beta, hz):
    m_i = np.array([7, 5, 3, 1])
    coshmi = np.cosh(beta * hz * m_i)
    Z =  np.sum(coshmi)# leave out the 2 
    Mz_i = np.sinh(beta * hz * m_i) * m_i
    Mz = np.sum(Mz_i)/Z
    return Mz



def U(Mz, hz):
    U = -hz * Mz
    return U


def Mz_squared(beta, hz):
    m_i = np.array([7, 5, 3, 1])
    coshmi = np.cosh(beta * hz * m_i)
    Z =  np.sum(coshmi) # leave out the 2 
    Mz_sq_i = np.cosh(beta * hz * m_i) * m_i * m_i
    Mz_sq = np.sum(Mz_sq_i)/Z
    return Mz_sq


def U_squared(beta, hz):
    m_i = np.array([7, 5, 3, 1])
    coshmi = np.cosh(beta * hz * m_i)
    Z =  np.sum(coshmi) # leave out the 2 
    U_sq_i = np.cosh(beta * hz * m_i) * m_i * m_i * hz * hz
    U_sq = np.sum(U_sq_i)/Z
    return U_sq



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


hz = np.arange(-2.0, 2.0, 0.05)
hz = np.round(hz, 4) 

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
dt = master_params['simulation']['dt']  
ntau = master_params['system']['ntau']
beta = master_params['system']['beta']
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
sample_ops_file_path = 'hz_' + str(hz[0]) + '/operators0.dat'
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
method = ['projected CL']

means_dframe = pd.DataFrame(columns = observables_list, index = method) # contains the noise averages from Cmethod sampling  
errs_dframe = pd.DataFrame(columns = observables_list, index = method) # contains the noise averages from Cmethod sampling  

# for each key and row, fill it with a list of np.zeros() of length(L)  
for k, m_ in enumerate(method):
  means_dframe.loc[k] = list( np.zeros(( len(observables_list), len(hz) ) ))
  errs_dframe.loc[k] = list( np.zeros(( len(observables_list), len(hz) ) ))


print('Data frame example: \n\n', means_dframe)
# Other properties we want to calculate 
runtime_cutoff = 500.0

for k, m_ in enumerate(method):
  for i, hz_ in enumerate(hz):
    # directory name 
  
    inner_path = 'hz_' + str(hz_)
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
plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_orderparams.txt')

#markers = ['o', 'x', '*']
markers = ['o', '*', 'p']
colors = ['r', 'b', 'k']

hz_ref = np.linspace(min(hz), max(hz), 250)
Mz_ref = np.zeros(len(hz_ref))
U_ref = np.zeros(len(hz_ref))
chi_zz_ref = np.zeros(len(hz_ref))
Cv_ref = np.zeros(len(hz_ref))
for i, _h in enumerate(hz_ref):
  Mz_ref[i] = Mz(beta, _h) 
  U_ref[i] = U(Mz_ref[i], _h)
  Cv_ref[i] = U_squared(beta, _h)
  Cv_ref[i] -= (U_ref[i] ** 2)*beta*beta
  chi_zz_ref[i] = Mz_squared(beta, _h)
  chi_zz_ref[i] -= (Mz_ref[i] ** 2)
  chi_zz_ref[i] *= beta



plt.figure(figsize=(6,6))
for k, m_ in enumerate(method):
  _Mz = means_dframe.loc[k]['ReMz']
  Mz_errs = errs_dframe.loc[k]['ReMz']
  plt.errorbar(hz, _Mz, Mz_errs, marker=markers[k], color = colors[k], markersize = 6, elinewidth=0.5, linewidth = 0.0, label = method[k]) 
  plt.plot(hz_ref, Mz_ref, color = colors[k], linestyle = 'solid', linewidth = 1.0, label = 'Reference') 
plt.xlabel(r'$h_z$',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$M_{z}$', fontsize = 20, fontweight = 'bold')
plt.legend()
plt.show()


plt.figure(figsize=(6,6))
for k, m_ in enumerate(method):
  _U = means_dframe.loc[k]['ReEnergy']
  U_errs = errs_dframe.loc[k]['ReEnergy']
  plt.errorbar(hz, _U, U_errs, marker=markers[k], color = colors[k], markersize = 6, elinewidth=0.5, linewidth = 0.0, label = method[k]) 
  plt.plot(hz_ref, U_ref, color = colors[k], linestyle = 'solid', linewidth = 0.5, label = 'Reference') 
plt.xlabel(r'$h_z$',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$U$', fontsize = 20, fontweight = 'bold')
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
  plt.errorbar(hz, chi_zz, chi_zz_errs, marker=markers[k], color = colors[k], markersize = 6, elinewidth=0.5, linewidth = 0.0, label = method[k]) 
  plt.plot(hz_ref, chi_zz_ref, color = colors[k], linestyle = 'solid', linewidth = 0.5, label = 'Reference') 
plt.xlabel(r'$h_z$',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$\chi_{zz}$', fontsize = 20, fontweight = 'bold')
plt.legend()
plt.show()


plt.figure(figsize=(6,6))
for k, m_ in enumerate(method):
  _U = means_dframe.loc[k]['ReEnergy']
  U_errs = errs_dframe.loc[k]['ReEnergy']
  U_sq = means_dframe.loc[k]['ReEnergySquared']
  U_sq_errs = errs_dframe.loc[k]['ReEnergySquared']
  Cv = U_sq - (_U**2)
  Cv *= beta**2
  Cv_errs = calc_err_addition(U_sq_errs, calc_err_multiplication(_U, _U, U_errs, U_errs, _U**2) ) * beta * beta
  plt.errorbar(hz, Cv, Cv_errs, marker=markers[k], color = colors[k], markersize = 6, elinewidth=0.5, linewidth = 0.0, label = method[k]) 
  plt.plot(hz_ref, Cv_ref, color = colors[k], linestyle = 'solid', linewidth = 0.5, label = 'Reference') 
plt.xlabel(r'$h_z$',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$C_{V}$', fontsize = 20, fontweight = 'bold')
plt.legend()
plt.show()
