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



def calc_err_division(x, y, x_err, y_err):
    # x/y 
    # assumes x and y are real 
    z = x/y
    # Calculate error using standard error formula 
    #result = np.sqrt( ((-x * y_err / (y**2))**2 ) + (x_err/y)**2)
    result = z * np.sqrt( ((x_err/x) **2) + ((y_err/y)**2) ) 
    return result


def calc_err_multiplication(x, y, x_err, y_err):
    # z = x * y
    z = x*y
    result = z * np.sqrt( ((x_err/x)**2)  + ((y_err/y)**2) ) 
    return result



def calc_err_addition(x_err, y_err):
    # Error propagation function for x + y 
    #result = 0.
    # assumes x and y are real 

    # Calculate error using standard error formula 
    result = np.sqrt( (x_err**2) + (y_err**2) )
    return result



def Process_Data(input_file, _isCleaning):

  print('is cleaning? ' + str(_isCleaning))
  
  # Lx array 
  K = np.arange(0.06, 0.25, 0.01) 
  K = np.sort(K)
  K = np.round(K, 2)
  K = K[0:-2] 
  
  
  B_sweepone = np.array([2.0, 1.0, 0.75, 0.33, 0.08, 0.07, 0.06, 0.055, 0.044, 0.033, 0.025, 0.02, 0.015, 0.01, 0.008,0.005]) # reference  
  #B_sweepone = np.array([2.0, 0.75, 0.33, 0.08, 0.07, 0.06, 0.055, 0.044, 0.033, 0.025, 0.02, 0.015, 0.01, 0.008,0.005]) # \beta = 1.0 removed 
  B_sweeptwo = np.array([0.5, 0.2, 0.15, 0.1])
  B_sweepthree = np.array([0.009, 0.006, 0.004, 0.002, 0.003, 0.001]) 
  B = np.hstack([B_sweeptwo, B_sweepone, B_sweepthree])
  B = np.sort(B)
  #L = np.array([32.986695])

  # need to calculate in each folder. one of the \kappa folders has a different mu_eff 
  kappa_tilde = np.zeros((len(K), len(B)))
  kappa_dim = np.zeros((len(K), len(B)))
  beta_tilde = np.zeros((len(K), len(B)))
  beta_dim = np.zeros((len(K), len(B)))
  
  # Get N-input required for constant conc:  # N is length L 
  #N_inputs = conc * (L**(float(2)))
  
  # Anticipate stripe or overlap area to be L*L*aspect

  input_file.seek(0)  
  input_filename = input_file.name  
  with open(input_filename) as infile:
    master_params = yaml.load(infile, Loader=yaml.FullLoader)
  
  # Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
  dt = master_params['simulation']['dt']  
  N_tau = master_params['system']['ntau']
  #dt = master_params['simulation']['dt']
  stepper = master_params['simulation']['DriverType']
  ensemble = master_params['system']['ensemble']
  if(ensemble == 'CANONICAL'):
    _isCanonical = True
  else:
    _isCanonical = False
  
  num_steps = master_params['simulation']['numtsteps']
  dimension = master_params['system']['Dim']
  Lx = master_params['system']['CellLength-x']
  Ly = master_params['system']['CellLength-y']
  Vol = Lx * Ly
  lam_ref = 6.0596534037 
  mass = master_params['system']['mass']
  lam = lam_ref * 4.0026 / mass
  print('hbar^2 / 2m  = ' + str(lam))
 
  apply_ADT = master_params['simulation']['apply_ADT']
  total_CL_time = master_params['simulation']['Total_CL_Time']
  if(apply_ADT):
    runlength_max = total_CL_time
  else:
    runlength_max = num_steps * dt
  
  # Read 1 operators file and grab the strings after operators0.dat
  sample_ops_file_path = 'K_' + str(K[0]) + '/B_' + str(B[0]) + '/operators0.dat'
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
  means_dframe = pd.DataFrame(columns = observables_list, index = K) # contains the noise averages from CK sampling  
  errs_dframe = pd.DataFrame(columns = observables_list, index = K) # contains the SEM (errors)  
  
  #SOC_obs_list = ['N_SOC', 'N_SOC_err', 'frac_SOC', 'frac_SOC_err']
  #SOC_dframe = pd.DataFrame(columns = SOC_obs_list, index = L) 
  
  
  
  # for each key and row, fill it with a list of np.zeros() of length(L)  
  for K_ in K:
    means_dframe.loc[K_] = list( np.zeros(( len(obs_list), len(B) ) ))
    errs_dframe.loc[K_] = list( np.zeros(( len(obs_list), len(B) ) ))
    #SOC_dframe.loc[L_] = list( np.zeros(( len(SOC_obs_list), len(B) ) ))
    #psi_dframe.loc[L_] = list( np.zeros(( len(psi_obs_list), len(B) ) ))
  
  
  print('Data frame example: \n\n', means_dframe)
  # Other properties we want to calculate 
  
  
  # TODO make these inputs to parse  
  #_isCleaning = True
  runtime_cutoff = 80.
  
  
  for j, K_ in enumerate(K):
    pwd = "."
    path = pwd + "/" + "K_" + str(K_)
  
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
  
  
      output_filename = 'data0'
  
      if(apply_ADT):
        output_filename += '_ADT'
  
      output_filename += '.dat'
 
      input_filename = inner_path + '/input.yml'
      # Fill in the dimensionless kappa and beta
      with open(input_filename) as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader) 

      # Calculate kappa tilde
      kappa_x_val = inputs['system']['kappa_x']
      assert(kappa_x_val == K_) 
      kappa_y_val = inputs['system']['kappa_y'] 
      mu_eff = inputs['system']['mu_0'] - (lam * ( (kappa_x_val**2) + kappa_y_val**2) ) 
      kappa_tilde[j,i] = kappa_x_val * np.sqrt(lam/mu_eff)  # should be the same list for each beta, i.e. column should be the same 
      kappa_dim[j,i] = kappa_x_val 
      beta_tilde[j,i] = B_ * mu_eff  # should be the same list for each kappa , i.e. each row is the same  
      beta_dim[j,i] = B_
 
  
      if(_isCleaning):
        # Remove data and reprocess if necessary 
        #os.chdir(inner_path)
        print('removing old processed output files')
        if(apply_ADT):
          subprocess.call('rm ' + inner_path + '/data0_ADT.dat', shell = True)
        else:
          subprocess.call('rm ' + inner_path + '/data0.dat', shell = True)
        subprocess.call('rm ' + inner_path + '/N_SOC.dat', shell = True)
        #os.chdir('../../')   
    
      # Run ADT conversion script if ADT  
      ADT_reweight = 'python3 ~/csbosonscpp/tools/stats_ADT.py -f ' + inner_path + '/operators0.dat -w 6 > ' + inner_path + '/' + output_filename 
  
      if(apply_ADT):
        cmd_string = ADT_reweight 
      else:
        cmd_string = "python3 ~/csbosonscpp/tools/stats.py -f " + inner_path + "/operators0.dat -o "
        for obs in observables_list:
          cmd_string += obs 
          cmd_string += ' ' 
        cmd_string += '-a -q > ' + inner_path + '/' + output_filename
  
  
      check_path = inner_path + '/' + output_filename
  
      if (not os.path.exists(check_path)): #or (not os.path.exists(inner_path + '/N_SOC.dat')) :
        if (runtime == 0 or runtime < runtime_cutoff):
          print("None or too little runtime, inserting NaN for observables")
          for obs in observables_list:
            means_dframe.loc[K_][obs][i] = math.nan 
            errs_dframe.loc[K_][obs][i] = math.nan 
  
          #for obs in SOC_obs_list:
            #SOC_dframe.loc[L_][obs][i] = math.nan 
            #psi_dframe.loc[L_][obs][i] = math.nan 
  
        else:
          print("processing " + inner_path)
          if not os.path.exists(check_path):
            subprocess.call(cmd_string, shell = True)
  
 #          if not os.path.exists(inner_path + '/N_SOC.dat'):
 #            #os.chdir(inner_path)
 #            #if(apply_ADT):
 #            subprocess.call('python3 ~/csbosonscpp/tools/stats_field.py -f ' + inner_path + '/n_k.dat -in ' + inner_path + '/input.yml ' + '-fop ' + inner_path + '/operators0.dat -d ' + inner_path + '/' + output_filename + ' -w 5 > ' + inner_path + '/N_SOC.dat', shell=True)
 #            #else:
 #            #  subprocess.call('python3 ~/csbosonscpp/tools/python_scripts/stats_field_ADT.py -f > ' + inner_path + '/N_SOC.dat', shell=True)
 #              #subprocess.call('python3 ~/csbosonscpp/tools/python_scripts/process_plot_Nkappa_OLD.py > ' + inner_path + '/N_SOC.dat', shell=True)
 #
 #            #subprocess.call('tail -1 '+ inner_path + '/N_SOC.dat > ' + inner_path + '/N_SOC2.dat', shell=True)
 #            #subprocess.call("python3 ~/csbosonscpp/tools/python_scripts/avg_psi_r.py ", shell=True) 
 #            #subprocess.call("tail -1 N_psi.dat > N_psi2.dat", shell=True)
 #            #os.chdir('../../')
    
      if(runtime != 0 and runtime > runtime_cutoff):
        in_file = open(inner_path + '/' + output_filename, "r")
        #in_file_SOC = open(inner_path + "/N_SOC.dat", "r")
        #in_file_psi = open(inner_path + "/N_psi2.dat", "r")
  
        # TODO change to loop 
        tmp = in_file.read()
        in_file.close()
        tmp = re.split(r"\s+", tmp)
        tmp = tmp[0:-1]
        tmp = tuple(map(float, tmp))
  
 #        tmp2 = in_file_SOC.read()
 #        in_file_SOC.close()
 #        tmp2 = re.split(r"\s+", tmp2)
 #        tmp2 = tmp2[0:-1]
 #        tmp2 = tuple(map(float, tmp2))
  
   #      tmp3 = in_file_psi.read()
   #      in_file_psi.close()
   #      tmp3 = re.split(r"\s+", tmp3)
   #      tmp3 = tmp3[0:-1]
   #      tmp3 = tuple(map(float, tmp3))
        # Put the observables data into the dataframes
        ctr = 0
        for obs in observables_list:
          means_dframe.loc[K_][obs][i] = tmp[ctr] 
          errs_dframe.loc[K_][obs][i] = tmp[ctr+1]
          ctr += 2
  
 #        for k, obs in enumerate(SOC_obs_list):
 #          SOC_dframe.loc[L_][obs][i] = tmp2[k]
  
   #      for j, obs in enumerate(psi_obs_list):
   #        psi_dframe.loc[L_][obs][i] = tmp3[j]
  
  
  #print('Data frame, post data processing: \n\n', means_dframe)
  
  # --------------- Plots ! -------------- 
  
  _isFiniteSizing = False
  #_nu = 1 # good 
  #_nu = -0.665
  #p = 1/2
  plt.style.use('~/csbosonscpp/tools/python_scripts/plot_style.txt')
  
  symbols_list = ['+', 'o', 'x', 'h', 'p']
  ctr = 1

  Cond_frac_array = np.zeros((len(K), len(B)))
  Mx_array = np.zeros((len(K), len(B))) # to be normalized 
  chi_xx_array = np.zeros((len(K), len(B))) 
  kappa_compress_array = np.zeros((len(K), len(B))) 

  # Plot 1: Nk0/N condensate fraction (Total) 
  plt.figure(ctr)
  for l, K_ in enumerate(K):
    # Get the Nk0 and N totals for the length
    N_tot = means_dframe['Repartnum'][K_]
    Nk0_tot = means_dframe['ReNK0_tot'][K_] 
    errN_tot = errs_dframe['Repartnum'][K_] 
    errNk0_tot = errs_dframe['ReNK0_tot'][K_]

    # Fill a 2D matrix with Nk0/N. 
    Cond_frac_array[l, :] = Nk0_tot/N_tot

  # Plot a heat map of the condensate fraction 
  plt.figure(ctr)
  plt.pcolor(kappa_tilde, 1./beta_tilde, Cond_frac_array, shading='auto')
  #plt.pcolor(kappa_dim, 1./beta_dim, Cond_frac_array, shading='auto')
  plt.title('$N_{k=0}/N$', fontsize = 24, fontweight = 'bold')
  plt.xlabel(r'$\tilde \kappa$', fontsize = 24, fontweight = 'bold')
  #plt.ylabel(r'$\tilde \beta$', fontsize = 24, fontweight = 'bold')
  plt.ylabel(r'$\tilde T$', fontsize = 24, fontweight = 'bold')
  #plt.zlabel('', fontsize = 20, fontweight = 'bold')
  #plt.xlim(np.min(kappa_dim),np.max(kappa_dim))
  #plt.ylim(np.min(1./beta_dim), np.max(1./beta_dim))
  plt.xlim(np.min(kappa_tilde),np.max(kappa_tilde))
  plt.ylim(np.min(1./beta_tilde), np.max(1./beta_tilde))
  plt.yscale('log')
  plt.colorbar()
  #plt.savefig('Nk0_stripe_diagram.eps')
  plt.show()

  ctr += 1
  
  
  
  
  for l, K_ in enumerate(K):
  
    Mx = means_dframe['ReMag_x'][K_]
    errMx = errs_dframe['ReMag_x'][K_]

    # Fill a 2D matrix with Nk0/N. 
    Mx_array[l, :] = Mx
    #Mx_array[l, :] = Mx/(np.max(Mx))  # normalized for each kappa 
  
    # plt the x-spin component - scale by K (1D)   
    #plt.errorbar(1./B, Mx, errMx, marker=symbols_list[l], markersize = 6, linewidth = 0.5, label = '$K$ = ' + str(K_))
  # Plot a heat map of the condensate fraction 
  plt.figure(ctr)
  plt.pcolor(kappa_tilde, 1./beta_tilde, Mx_array, shading='auto')
  #plt.title('Normalized $M_{x}$ ', fontsize = 24, fontweight = 'bold')
  plt.title('$M_{x}$ ', fontsize = 24, fontweight = 'bold')
  plt.xlabel(r'$\tilde \kappa$', fontsize = 24, fontweight = 'bold')
  #plt.ylabel(r'$\tilde \beta$', fontsize = 24, fontweight = 'bold')
  plt.ylabel(r'$\tilde T$', fontsize = 24, fontweight = 'bold')
  #plt.zlabel('', fontsize = 20, fontweight = 'bold')
  plt.xlim(np.min(kappa_tilde),np.max(kappa_tilde))
  plt.ylim(np.min(1./beta_tilde), np.max(1./beta_tilde))
  plt.yscale('log')
  plt.colorbar()
  # plt.legend()
  #plt.savefig('Mx_stripe_diagram.eps')
  plt.show()

  ctr += 1
  
  
  for l, K_ in enumerate(K):
  
    Mx = means_dframe['ReMag_x'][K_]
    errMx = errs_dframe['ReMag_x'][K_]
    # need imaginary parts for susceptibility 
    Mx_im = means_dframe['ImMag_x'][K_]
    #errMx_im = errs_dframe['ImMag_x'][K_]
  
    Mx_squared = means_dframe['ReMag_x_squared'][K_]
    Mx_squared_im = means_dframe['ImMag_x_squared'][K_]
    #errMx_squared = errs_dframe['ReMag_x_squared'][K_]
    chi_xx = Mx_squared + 1j*Mx_squared_im - ( (Mx + 1j * Mx_im) **2)
    # Fill a 2D matrix with Nk0/N. 
    chi_xx_array[l, :] = chi_xx
  
    # plt the x-spin component - scale by K (1D)   
    #plt.errorbar(1./B, Mx, errMx, marker=symbols_list[l], markersize = 6, linewidth = 0.5, label = '$K$ = ' + str(K_))
  # Plot a heat map of the condensate fraction 
  plt.figure(ctr)
  plt.pcolor(kappa_tilde, 1./beta_tilde, chi_xx_array, shading='auto')
  plt.title('$\chi_{xx}$ ', fontsize = 24, fontweight = 'bold')
  plt.xlabel(r'$\tilde \kappa$', fontsize = 24, fontweight = 'bold')
  #plt.ylabel(r'$\tilde \beta$', fontsize = 24, fontweight = 'bold')
  plt.ylabel(r'$\tilde T$', fontsize = 24, fontweight = 'bold')
  #plt.zlabel('', fontsize = 20, fontweight = 'bold')
  plt.xlim(np.min(kappa_tilde),np.max(kappa_tilde))
  #plt.ylim(np.min(beta_tilde), np.max(beta_tilde))
  plt.ylim(np.min(1./beta_tilde), np.max(1./beta_tilde))
  plt.yscale('log')
  plt.colorbar()
  plt.show()

  ctr += 1
  
  
  
  if(not _isCanonical):
    for l, K_ in enumerate(K):
  
      N_tot = means_dframe['Repartnum'][K_]
      N2_tot = means_dframe['Repartnum_squared'][K_] 
      errN_tot = errs_dframe['Repartnum'][K_] 
      errN2_tot = errs_dframe['Repartnum_squared'][K_] 
    
      N_tot_im = means_dframe['Impartnum'][K_]
      N2_tot_im = means_dframe['Impartnum_squared'][K_] 
      errN_tot_im = errs_dframe['Impartnum'][K_] 
      errN2_tot_im = errs_dframe['Impartnum_squared'][K_] 
      # Calculate the isothermal compressibility
      kappa_compress = np.zeros(len(N2_tot), dtype=np.complex_) 
      #kappa_compress += (N2_tot + 1j * N2_tot_im)
      #kappa_compress /= (N_tot + 1j * N_tot_im)
      #kappa_compress /= (N_tot + 1j * N_tot_im)
      kappa_compress += (N2_tot) 
      kappa_compress /= (N_tot) 
      kappa_compress /= (N_tot) 
      kappa_compress += -1. 
  
      # Propagate error: 
      #kappa_errs = calc_err_division(N2_tot.real, N_tot.real, calc_err_addition(errN2_tot, errN2_tot_im), calc_err_addition(errN_tot, errN_tot_im) )
      N_sq_err = calc_err_multiplication(N_tot.real, N_tot.real, errN_tot, errN_tot) 
  
      kappa_errs = calc_err_division(N2_tot.real, N_tot.real * N_tot.real, errN2_tot, N_sq_err )
      #kappa_compress /= N_tot  # divide by 1 factor of N_tot
      #kappa_compress /= (N_tot**2) # keep unscaled  

      kappa_compress_array[l, :] = kappa_compress.real

  plt.figure(ctr)
  plt.pcolor(kappa_tilde, 1./beta_tilde, kappa_compress_array, shading='auto')
  plt.title(r'$\kappa_{T}$' + ' ' + '$k_{B} T/V$ ', fontsize = 24, fontweight = 'bold')
  plt.xlabel(r'$\tilde \kappa$', fontsize = 24, fontweight = 'bold')
  #plt.ylabel(r'$\tilde \beta$', fontsize = 24, fontweight = 'bold')
  plt.ylabel(r'$\tilde T$', fontsize = 24, fontweight = 'bold')
  #plt.zlabel('', fontsize = 20, fontweight = 'bold')
  plt.xlim(np.min(kappa_tilde),np.max(kappa_tilde))
  plt.ylim(np.min(1./beta_tilde), np.max(1./beta_tilde))
  plt.yscale('log')
  plt.colorbar()
  #plt.savefig('kappa_compress_stripe_diagram.eps')
  plt.show()

  ctr += 1
    







  


if __name__ == '__main__':
    # For command-line runs, build the relevant parser
    import argparse as ap
    parser = ap.ArgumentParser(description='Statistical analysis of csbosonscpp data')
    parser.add_argument('-in','--input_file',default='./operators0.dat',type=ap.FileType('r'),help='Filename containing scalar statistical data.')
    parser.add_argument('-cl','--isCleaning',default=False,dest='isCleaning',action='store_false',help='Clean all previous processed data')
    args=parser.parse_args()
    Process_Data(args.input_file, args.isCleaning)



