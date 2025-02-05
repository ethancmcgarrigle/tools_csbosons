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


# Nx = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 30, 32, 36, 40, 50]

L = np.arange(25, 35, 2)
B = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5])

with open('input.yml') as infile:
  master_params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
dt = master_params['simulation']['dt']  
N_tau = master_params['system']['ntau']
#dt = master_params['simulation']['dt']
stepper = master_params['simulation']['DriverType']
#U_val = master_params['system']['U']
num_steps = master_params['simulation']['numtsteps']
#hz = master_params['system']['hz']
# cellLength = master_params['system']['CellLength']
dimension = master_params['system']['Dim']
# Nx = np.array(Nx)
# Vol = cellLength**(int(dimension))

 #T = master_params['system']['beta']
 #T = 1/T
apply_ADT = master_params['simulation']['apply_ADT']
total_CL_time = master_params['simulation']['Total_CL_Time']
if(apply_ADT):
  runlength_max = total_CL_time
else:
  runlength_max = num_steps * dt

# Read 1 operators file and grab the strings after operators0.dat
sample_ops_file_path = 'L_' + str(L[0]) + '/B_' + str(B[0]) + '/operators0.dat'
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
# set up a data frame with observables as keys, and "beta" (1/T) as the rows 
# Set the index (rows) to be "L" since we want to conveniently return a list for each L, rather than grab 1 number at a time from B when plotting across L 
means_dframe = pd.DataFrame(columns = observables_list, index = L) # contains the noise averages from CL sampling  
errs_dframe = pd.DataFrame(columns = observables_list, index = L) # contains the SEM (errors)  

# for each key and row, fill it with a list of np.zeros() of length(L)  
for L_ in L:
  means_dframe.loc[L_] = list( np.zeros(( len(obs_list), len(B) ) ))
  errs_dframe.loc[L_] = list( np.zeros(( len(obs_list), len(B) ) ))

print('Data frame example: \n\n', means_dframe)
# Other properties we want to calculate 

# Isothermal compressibility 
 #ReKappa_T = np.zeros(len(Nx))
 #errReKappa_T = np.zeros(len(Nx))
 #ImKappa_T = np.zeros(len(Nx))
 #errImKappa_T = np.zeros(len(Nx))

for L_ in L:
  pwd = "."
  path = pwd + "/" + "L_" + str(L_)

  for i, B_ in enumerate(B):
    # directory name 
    inner_path = path + '/B_' + str(B_)

    print(inner_path)   
    output_file = open(inner_path + "/output.out", 'r')
    # print("Getting runtime in " + outer_path)
    lines = output_file.readlines()
    last_lines = lines[-3:]
    s = " ".join(last_lines[1].split()) 
    pen_line = " ".join(last_lines[0].split())

    tmp = re.split(r"\s+", pen_line)

    # This is hardcoded, come up with a better solution some other time. This is stored in tmp[0] if there's no runtime
    if(tmp[0] == '----------------------------------------------------------------------------------'):
      runtime = 0
    else:
      runtime = float(tmp[0])

    output_file.close()

    # Remove data and reprocess if necessary 
     #  rmv_data = "rm " + outer_path + "data0.dat" 
   #    os.chdir(outer_path)
   #    subprocess.call("rm ./data0.dat", shell = True)
   #    os.chdir('../../')   
   #    check_path = outer_path + '/data0.dat'
  
    #cmd_string = "python3 ~/csbosonscpp/tools/stats.py -f " + path + "/operators0.dat -o Repartnum0 Impartnum0 Repartnum1 Impartnum1 ReE ImE ReE_squared ImE_squared -a -q > " + path + "/data0.dat" 
    cmd_string = "python3 ~/csbosonscpp/tools/stats.py -f " + inner_path + "/operators0.dat -o "
    for obs in observables_list:
      cmd_string += obs 
      cmd_string += ' ' 

    cmd_string += "-a -q > " + inner_path + "/data0.dat" 
    
    check_path = inner_path + '/data0.dat'

    if not os.path.exists(check_path):
      if (runtime == 0):
        print("No runtime, inserting NaN for observables")
        for obs in observables_list:
          means_dframe.loc[L_][obs][i] = math.nan 
          errs_dframe.loc[L_][obs][i] = math.nan 

      else:
        print("processing " + inner_path)
        subprocess.call(cmd_string, shell = True)
  
    if(runtime != 0):
      in_file = open(inner_path + "/data0.dat", "r")
      tmp = in_file.read()
      in_file.close()
      tmp = re.split(r"\s+", tmp)
      tmp = tmp[0:-1]
      tmp = tuple(map(float, tmp))
      # Put the observables data into the dataframes
      ctr = 0
      for obs in observables_list:
        means_dframe.loc[L_][obs][i] = tmp[ctr] 
        errs_dframe.loc[L_][obs][i] = tmp[ctr+1]
        ctr += 2


print('Data frame, post data processing: \n\n', means_dframe)

# plt.style.use('plot_style.txt')

# Plot 1: Nk0/N condensate fraction (Total) 
plt.figure(1)
for L_ in L:
  # Get the Nk0 and N totals for the length
  N_up = means_dframe['Repartnum0'][L_]
  N_dwn = means_dframe['Repartnum1'][L_]
  Nk0_up = means_dframe['ReNK0_0'][L_] 
  Nk0_dwn = means_dframe['ReNK0_1'][L_] 
  errN_up = errs_dframe['Repartnum0'][L_] 
  errN_dwn = errs_dframe['Repartnum1'][L_] 
  errNk0_up = errs_dframe['ReNK0_0'][L_] 
  errNk0_dwn = errs_dframe['ReNK0_1'][L_] 
  #plt.errorbar(1./B, (Nk0_up + Nk0_dwn)/(N_up + N_dwn),  (errNk0_up + errNk0_dwn)/(errN_up + errN_dwn), marker='x', markersize = 6, linewidth = 0.5, label = 'L = ' + str(L_))
  plt.errorbar(B, (Nk0_up + Nk0_dwn)/(N_up + N_dwn),  (errNk0_up + errNk0_dwn)/(errN_up + errN_dwn), marker='x', markersize = 6, linewidth = 0.5, label = 'L = ' + str(L_))
  #plt.plot(B, (Nk0_up + Nk0_dwn)/(N_up + N_dwn),  marker='x', markersize = 6, linewidth = 0.5, label = 'L = ' + str(L_))
# plt.plot(hz, np.tanh(hz), 'k', linewidth=2, label = 'Analytic Reference: ' + 'tanh$(h_z)$')
# params_title_str = '$dt$ = ' + str(dt) + ', $J_{nn}$ = ' + str(Nx) + ', $T$ =  ' + str(T) + ', ' + ', $N_{spins}$ = ' + str(N_sites) + ', ' + str(dimension) + 'D, ' + stepper 
title_str = 'Condensate Fraction vs. Inverse Temperature' 
plt.title(title_str, fontsize = 20)
plt.xlabel('$1/T$ [K]',fontsize = 20, fontweight = 'bold')
#plt.xlabel('$T$ [K]',fontsize = 20, fontweight = 'bold')
plt.ylabel('$N_{k=0}/N$', fontsize = 20, fontweight = 'bold')
plt.ylim(0, 1.0)
plt.legend()
#plt.show()


print('Data frame, post plot 1 : \n\n', means_dframe)
# Plot 2: superfluid density
plt.figure(2)
for L_ in L:
  # Get the Nk0 and N totals for the length
  N_up = means_dframe['Repartnum0'][L_]
  N_dwn = means_dframe['Repartnum1'][L_]
  rho_total = N_up + N_dwn 
  rho_total /= (L_ ** dimension) 
  SF_rho_up = means_dframe['ReSF_density0'][L_] 
  SF_rho_dwn = means_dframe['ReSF_density1'][L_] 
  err_SF_rho_up = errs_dframe['ReSF_density0'][L_] 
  err_SF_rho_dwn = errs_dframe['ReSF_density1'][L_]
  # plt.errorbar(1./B, (SF_rho_up + SF_rho_dwn)/rho_total,  (err_SF_rho_up + err_SF_rho_dwn)/rho_total, marker='x', markersize = 6, linewidth = 0.5, label = 'L = ' + str(L_))
  plt.errorbar(B, (SF_rho_up + SF_rho_dwn)/rho_total,  (err_SF_rho_up + err_SF_rho_dwn)/rho_total, marker='x', markersize = 6, linewidth = 0.5, label = 'L = ' + str(L_))
# plt.plot(hz, np.tanh(hz), 'k', linewidth=2, label = 'Analytic Reference: ' + 'tanh$(h_z)$')
# params_title_str = '$dt$ = ' + str(dt) + ', $J_{nn}$ = ' + str(Nx) + ', $T$ =  ' + str(T) + ', ' + ', $N_{spins}$ = ' + str(N_sites) + ', ' + str(dimension) + 'D, ' + stepper 
title_str = 'Superfluid Fraction vs. Inverse Temperature' 
plt.title(title_str, fontsize = 20)
#plt.xlabel('$T$ [K]',fontsize = 20, fontweight = 'bold')
plt.xlabel('$1/T$ [K]',fontsize = 20, fontweight = 'bold')
#plt.xlabel('$T$ [K]',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$\rho_{s} / \rho$', fontsize = 20, fontweight = 'bold')
# plt.ylim(0, 1.0)
plt.ylim(-0.1, 2.0)
plt.legend()
plt.show()

print('Data frame, post plot 2 : \n\n', means_dframe)
 # 
 #ReKappa_T = ReN2 - (ReN + 1j*ImN)**2
 #ReKappa_T /= ((ReN + 1j*ImN)**2)
 #ReKappa_T = ReKappa_T.real
 #errReKappa_T = errReN2 - (errReN2 + 1j * errImN2)**2
 #errReKappa_T = errReKappa_T.real 
 #
 #plt.figure(2)
 #plt.errorbar(Nx, ReKappa_T, errReKappa_T, marker='x', markersize = 6, linewidth = 1, label = 'Complex Langevin') 
 #compressibility_title = 'Isothermal compressibility, Nx Convergence '
 #plt.title(compressibility_title, fontsize = 20) 
 #plt.xlabel('$N_{x}$',fontsize = 20, fontweight = 'bold')
 #plt.ylabel('$\Re{(\kappa_{T})}$', fontsize = 20, fontweight = 'bold')
 #plt.legend()	
 #plt.show()
 #

 #plt.figure(3)
 #for k, item in enumerate(Ntau): 
 #  plt.errorbar(Ntau, ReEnergy,  errReEnergy[:, k], marker='x', markersize = 6, linewidth = 1, label = 'Complex Langevin: $U$ = ' + str(item))
 #  #plt.errorbar(hz, ReEnergy[:, k],  errReEnergy[:, k], marker='x', markersize = 4, linewidth = 0.5, label = 'Complex Langevin: $\gamma$ = ' + str(item))
 #plt.plot(hz, -1.0 * np.array(hz) * np.tanh(hz), 'k', linewidth=2, label = 'Analytic Reference: ' + '$E = h_z$ tanh$(h_z)$')
 #  #plt.errorbar(hz, np.ones(len(hz)) - ReMag[:, k]**2, N_spins*errReX[:, k], marker='x', markersize = 4, linewidth = 0.5, label = 'Complex Langevin: U = ' + str(item)) 
 #internal_energy_title = 'Internal Energy'
 ##plt.title(params_title_str, fontsize = 11) 
 #plt.title(internal_energy_title, fontsize = 20) 
 #plt.xlabel('$h_z$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('$\Re{(E)}$', fontsize = 20, fontweight = 'bold')
 #plt.legend()	
 #plt.ylim(0, -1.5)
 ## plt.savefig('X_Test.eps')
 ##plt.xscale('log')
 #plt.show()
 #
 #
 #plt.figure(4)
 #for k, item in enumerate(U):  
 #  plt.errorbar(hz, N_spins * ReCv[:, k], N_spins * errReEnergy2[:, k], marker='x', markersize = 6, linewidth = 1, label = 'Complex Langevin: $U$ = ' + str(item))
 #  #plt.errorbar(hz, N_spins * ReCv[:, k], N_spins * errReEnergy2[:, k], marker='x', markersize = 4, linewidth = 0.5, label = 'Complex Langevin: $\gamma$ = ' + str(item))
 #plt.plot(hz, (np.array(hz)**2) * (sech(hz)**2), 'k', linewidth=2, label = 'Analytic Reference: ' + '$C_v  = h_z$ sech$(h_z)^2$')
 #  #plt.errorbar(hz, np.ones(len(hz)) - ReMag[:, k]**2, N_spins*errReX[:, k], marker='x', markersize = 4, linewidth = 0.5, label = 'Complex Langevin: U = ' + str(item)) 
 #energy_fluc_title = 'Constant Volume Heat Capacity'
 ##plt.title(params_title_str, fontsize = 11) 
 #plt.title(energy_fluc_title, fontsize = 20) 
 #plt.xlabel('$h_z$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('$\Re{(C_v)}$', fontsize = 20, fontweight = 'bold')
 #plt.legend()	
 #plt.ylim(0, 1.5)
 ## plt.savefig('X_Test.eps')
 ##plt.xscale('log')
 #plt.show()
 #
 #
 #plt.figure(5)
 #for k, item in enumerate(U):  
 #  plt.errorbar(hz, N_spins*ReM2[:, k], N_spins*errReM2[:, k], marker='x', markersize = 4, linewidth = 0.5, label = 'CL: $U$ = ' + str(item)) 
 #  #plt.errorbar(hz, np.ones(len(hz)) - ReMag[:, k]**2, N_spins*errReX[:, k], marker='x', markersize = 4, linewidth = 0.5, label = 'Complex Langevin: U = ' + str(item)) 
 #title_ = 'Real part of $M^2$'
 ##plt.title(params_title_str, fontsize = 11) 
 #plt.title(title_, fontsize = 14) 
 #plt.plot(hz, (sech(hz))**2, 'k', linewidth=2, label = 'Analytic Reference: ' + 'sech$(h_z)^2$') 
 #plt.xlabel('$h_z$', fontsize = 16, fontweight = 'bold')
 #plt.ylabel('$\Re{(\chi_{zz})}$', fontsize = 14, fontweight = 'bold')
 #plt.legend()	
 #plt.ylim(0, 2.5)
 ## plt.savefig('X_Test.eps')
 ##plt.xscale('log')
 #plt.show()
 #
 #
 #plt.figure(6)
 #for k, item in enumerate(U):  
 #  plt.errorbar(hz, N_spins*ReEnergy2[:, k], N_spins*errReEnergy2[:, k], marker='x', markersize = 4, linewidth = 0.5, label = 'CL: $U$ = ' + str(item)) 
 #  #plt.errorbar(hz, np.ones(len(hz)) - ReMag[:, k]**2, N_spins*errReX[:, k], marker='x', markersize = 4, linewidth = 0.5, label = 'Complex Langevin: U = ' + str(item)) 
 #title_ = 'Real part of $E^2$'
 #plt.plot(hz, (np.array(hz)**2) * (sech(hz)**2), 'k', linewidth=1, label = 'Analytic Reference: ' + '$C_v = h_z$ sech$(h_z)^2$')
 ##plt.title(params_title_str, fontsize = 11) 
 #plt.title(title_, fontsize = 14) 
 #plt.xlabel('$h_z$', fontsize = 16, fontweight = 'bold')
 #plt.ylabel('$\Re{(C_{v})}$', fontsize = 14, fontweight = 'bold')
 #plt.legend()	
 #plt.ylim(0, 2.5)
 ## plt.savefig('X_Test.eps')
 ##plt.xscale('log')
 #plt.show()
 # #
 #color_str = ['blue', 'orange', 'green', 'red', 'cyan']

 #plt.figure(8)
 #plt.plot(Nx, runtime, marker='x', linewidth=0.5, label = 'CL') 
 #plt.title(params_title_str, fontsize = 11) 
 #plt.plot(Nx, runlength_max * np.ones(len(Nx)), 'k', linewidth=1, label = 'Max Runtime')
 #plt.xlabel('$N_{x}$', fontsize = 14, fontweight = 'bold')
 #plt.ylabel('CL Runtime', fontsize = 14, fontweight = 'bold')
 #plt.legend()
 ##plt.ylim(0, runlength_max + 200) 	
 ## plt.ylim(0, 3000) 	
 ## plt.savefig('X_Test.eps')
 ##plt.xscale('log')
 ##plt.yscale('log')
 #plt.show()
 #
 #
