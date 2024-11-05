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

#kappa = np.arange(0.1, 4, 20)


N_up = 100000.0
N_input = np.arange(0.1, 0.9, 0.05) * N_up 
N_input = np.round(N_input, 1)
# Script to process the resulting sweep 
# 1) in each directory, run the DAT2VTK script 
# 2) Rename the resulting output file , FMT: N_dx.vtk 
# 3) cp to the 3D_densities folder in \home\emcgarrigle 


with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['simulation']['Nx'] 
Ny = params['simulation']['Ny'] 
Nz = params['simulation']['Nz'] 

Lx = params['system']['CellLength-x'] 
Ly = params['system']['CellLength-y'] 
Lz = params['system']['CellLength-z'] 

dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

for N_ in N_input:
  # directory name 
  dir_name = 'N_' + str(N_)
  
  # mkdir command
  #str_mkdir = 'mkdir ' + dir_name
  #subprocess.call(str_mkdir, shell = True)
  # change directory
  os.chdir(dir_name)

  # copy the submit and inputs and graphing scripts/files 
  print(os.getcwd())
 
  output_fname = 'N_' + str(int(N_)) 
  # 1) run the script, same input density file for each directory  
  py_script = 'ipython3 ~/csbosonscpp/tools/python_scripts/rho_dat2vtk.py ' + output_fname + ' ' + str(dx) + ' ' + str(dy) + ' ' + str(dz)
  subprocess.call(py_script, shell=True) 

  # 2) Rename the output file; the outpout file is defaulted to 'filename_imag.vtk' .. remove the "imag" part, optionally replace with "real"  
  mv_cmd = 'mv ' + output_fname + '_imag.vtk' + ' ' + output_fname + '.vtk' 
  subprocess.call(mv_cmd, shell=True)

  # 3) cp to 3D_densities folder 
  cp_cmd = 'cp ' + output_fname + '.vtk ~/3D_densities'
  subprocess.call(cp_cmd, shell=True)
  #str_list = [cp_cmd, cp_input, sed_command1, sed_command2, qsub_cmd] 
  # execute all the commands in order 
  #for s in str_list:
  #  subprocess.call(s, shell = True)

  os.chdir('../')

  


