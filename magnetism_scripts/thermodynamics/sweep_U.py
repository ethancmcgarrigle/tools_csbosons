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
## This function sweeps 
U = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]) 
exp_damped = ['true'] 
label = ['exp_damped']

for j, driver in enumerate(exp_damped):
  
  # mkdir command
  outer_dir_name = label[j] 

  str_mkdir = 'mkdir ' + outer_dir_name 
  subprocess.call(str_mkdir, shell = True)
  # change directory
  os.chdir(outer_dir_name)

  for i, U_ in enumerate(U):
    # directory name 
    inner_dir_name = 'U_' + str(U_) 
    
    # mkdir command
    str_mkdir = 'mkdir ' + inner_dir_name
    subprocess.call(str_mkdir, shell = True)
    # change directory
    os.chdir(inner_dir_name)

    jobname = label[j] + '_' + str(U_)
  
    # copy the submit and inputs and graphing scripts/files 
    print(os.getcwd())
    cp_cmd = 'cp ../../submit.sh ./submit.sh'
    cp_input = 'cp ../../input.yml ./input.yml'
  
    sed_command1 = 'sed -i "s/__U__/' + str(U_) + '/g" input.yml'
    sed_command2 = 'sed -i "s/__damped__/' + exp_damped[j] + '/g" input.yml'
    sed_command_jobname = 'sed -i "s/__jobname__/' + jobname + '/g" submit.sh'
    qsub_cmd = 'qsub submit.sh'
    
    str_list = [cp_cmd, cp_input, sed_command1, sed_command2, sed_command_jobname, qsub_cmd] 
    # execute all the commands in order 
    for s in str_list:
      subprocess.call(s, shell = True)
    os.chdir('../')

  os.chdir('../')


