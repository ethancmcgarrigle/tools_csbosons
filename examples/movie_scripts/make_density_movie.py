import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
#matplotlib.use('TkAgg')
import pdb
import pandas as pd 
from scipy.stats import sem 
import sys

def calculate_field_average(field_data, Nx, Ny, dim, N_samples_to_avg): # assumes cubic/square mesh 
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data

    # Get number of samples 
    N_samples = len(field_data)/(Nx*Ny)
    assert(N_samples.is_integer())
    N_samples = int(N_samples)

    # Use split (np) to get arrays that represent each sample (1 array per sample) Throw out the first sample (not warmed up properly) 
    sample_arrays = np.split(field_data, N_samples) 
    sample_arrays = sample_arrays[len(sample_arrays) - N_samples_to_avg:len(sample_arrays)]

    # Final array, initialized to zeros. 
    averaged_data = np.zeros(len(sample_arrays[0]), dtype=np.complex_)
    averaged_data += np.mean(sample_arrays, axis=0) # axis=0 calculates element-by-element mean
    # Calculate the standard error 
    std_errs = np.zeros(len(sample_arrays[0]))
    std_errs += sem(sample_arrays, axis=0)
    return averaged_data, std_errs




# -------- Script to make a movie of the field files over fictitious CL time ---------  

# ------ HOW TO USE: ----  
# 1. Run this script with the print_density_png.p gnuplot script in the same folder  
# 2. Run the bash script to generate the movie from the folder of pictures   




# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['simulation']['Nx'] 
Ny = params['simulation']['Ny'] 
d = params['system']['Dim']

# import the data
up_file = 'density1.dat'
down_file = 'density2.dat'
cols_x = np.loadtxt(up_file, unpack=True)
cols_y = np.loadtxt(down_file, unpack=True)

#x = cols_x[0]
#y = cols_x[1]
# Extract 1 set of x and y column data 
x = cols_x[0][0:Nx*Ny]
y = cols_x[1][0:Nx*Ny]

up_real = cols_x[2]
up_imag = cols_x[3]

down_real = cols_y[2]
down_imag = cols_y[3]

list_x = np.unique(x)
list_y = np.unique(y)

total_density = up_real + down_real + (1j * (up_imag + down_imag))


N_samples = int(len(total_density)/(Nx*Ny))


# Factor the files into N samples
# Use split (np) to get arrays that represent each sample (1 array per sample) Throw out the first sample (not warmed up properly) 
sample_arrays = np.split(total_density, N_samples) 
# sample_arrays = sample_arrays[len(sample_arrays) - N_samples_to_avg:len(sample_arrays)] , option to filter or start at a certain point, e.g. not from t = 0 
# sample_arrays[0] represents one sample  



Num_samples_in_movie = 500 


if(Num_samples_in_movie > len(sample_arrays)):
  Num_samples_in_movie = len(sample_arrays)


gnuplot_script = 'print_density_png.p'
gnuplot_script_cp = 'print_density_png_cp.p'
gnuplot_command = 'gnuplot ' + gnuplot_script_cp

dir_ = 'density_images'
mkdir_command = 'mkdir ' + dir_

# Make the directory
subprocess.call(mkdir_command, shell=True)

print('Printing ' + str(Num_samples_in_movie) + ' image samples for a movie\n')


# Get number of digits required for fmpeg  

# if N_samples is 2 digits, need 

num_digits = len(str(Num_samples_in_movie))


# Loop through each sample, plot a 2D plot using the gnuplot script saving to png (or eps?), producing N_samples .png's  
for i, n in enumerate(sample_arrays[0:Num_samples_in_movie]):
  num_zeros = num_digits - len(str(i))

  # Need to add "N" zeros to the .png output so all digits are same 
  png_out_str = 'density_n_' + str(i).zfill(num_digits) 

  # reprint column data in a .dat file 
  dat_out_str = png_out_str + '.dat' 

  sed_command1 = 'sed -i .bak "s/foo/' + png_out_str + '/g" ' + gnuplot_script_cp 
  sed_command2 = 'sed -i .bak "s/bar/' + png_out_str + '/g" ' + gnuplot_script_cp 
  cp_command = 'cp ' + gnuplot_script + ' ' + gnuplot_script_cp 


  # Write to an output file 
  out_file = open(dat_out_str, 'w')
  out_file.write('# x y Re Im \n') # format
  for s in range(0, len(x)): 
    out_file.write('{} {} {} {}\n'.format(x[s], y[s], n[s].real, n[s].imag))
  out_file = np.genfromtxt(dat_out_str)

  # 1. Make a cp of the original gnuplot script with foo as the placeholder filename 
  subprocess.call(cp_command, shell = True)

  # 2. Run the 2 bash commands to change gnuplot output png file from foo --> filename and change the input.dat files  
  subprocess.call(sed_command1, shell = True)
  subprocess.call(sed_command2, shell = True)

  # 3. Plot in gnuplot to save to .png
  subprocess.call(gnuplot_command, shell = True)

  # 4. Move .png to a folder  
  #subprocess.call('mv *.png' +  ' ' + dir_, shell = True)
  subprocess.call('mv ' +  png_out_str + '.png ' + dir_, shell = True)

  # Clean up the intermediate data file and intermediate gnuplot script, and its backup 
  subprocess.call('rm ' + dat_out_str, shell = True)
  subprocess.call('rm ' + gnuplot_script_cp, shell = True)
  #subprocess.call('rm ' + gnuplot_script_cp + '.bak', shell = True)
  subprocess.call('rm *.bak', shell = True)

 #  if(i == 0):
 #    sys.exit() 

  

# After the loop, copy the bash script into the folder with the density images 
bash_script = 'density_movie.sh'
cp_cmd = 'cp ~/csbosonscpp/tools/movie_scripts/' + bash_script + ' ./' + dir_ 

subprocess.call(cp_cmd, shell=True)

# Execute the movie script command
os.chdir('./' + dir_)
subprocess.call('. ' + bash_script, shell=True)



