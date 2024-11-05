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
## This function runs statistics on the runs accessed (i.e. parameter sweep). Then it collects the relevant data and plots it at the end

def sech(x):
  return 1/(np.cosh(x))

L = np.arange(2, 100, 2)

for L_ in L:
  # directory name 
  dir_name = 'L_' + str(L_)
  
  # mkdir command
  str_mkdir = 'mkdir ' + dir_name
  subprocess.call(str_mkdir, shell = True)
  # change directory
  os.chdir(dir_name)

  # copy the submit and inputs and graphing scripts/files 
  print(os.getcwd())
  cp_cmd = 'cp ../submit.sh ./submit.sh'
  cp_input = 'cp ../input.yml ./input.yml'
  cp_seed0 = 'cp ../phi_phistar_0_seed.bin ./phi_phistar_0_seed.bin'
  cp_seed1 = 'cp ../phi_phistar_1_seed.bin ./phi_phistar_1_seed.bin'
  cp_scripts = 'cp ../*.p ./'
  sed_command1 = 'sed -i "s/__L__/' + str(L_) + '/g" input.yml'
  sed_command2 = 'sed -i "s/__jobname__/' + dir_name + '/g" submit.sh'
  qsub_cmd = 'qsub submit.sh'

  str_list = [cp_cmd, cp_input, cp_seed0, cp_seed1, cp_scripts, sed_command1, sed_command2, qsub_cmd] 
  # execute all the commands in order 
  for s in str_list:
    subprocess.call(s, shell = True)

  os.chdir('../')



