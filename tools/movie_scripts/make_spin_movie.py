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

def calculate_field_average(field_data, Nx, Ny,N_samples_to_avg): # assumes cubic/square mesh 
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
Sx_file = 'Sx.dat'
Sz_file = 'Sz.dat'
cols_x = np.loadtxt(Sx_file, unpack=True)
cols_y = np.loadtxt(Sz_file, unpack=True)

#x = cols_x[0]
#y = cols_x[1]
# Extract 1 set of x and y column data 
x = cols_x[0][0:Nx*Ny]
y = cols_x[1][0:Nx*Ny]

S_x_real = cols_x[2]
S_x_imag = cols_x[3]

S_z_real = cols_y[2]
S_z_imag = cols_y[3]

list_x = np.unique(x)
list_y = np.unique(y)

N_samples = len(S_x_real)/(Nx*Ny)
assert(N_samples.is_integer())

# Factor the files into N samples
# Use split (np) to get arrays that represent each sample (1 array per sample) Throw out the first sample (not warmed up properly) 
sample_arrays_Sx = np.split(S_x_real + 1j*S_x_imag, N_samples) 
sample_arrays_Sz = np.split(S_z_real + 1j*S_z_imag, N_samples) 


Num_samples_in_movie = 500 


if(Num_samples_in_movie > len(sample_arrays_Sx)):
  Num_samples_in_movie = len(sample_arrays_Sx)


dir_ = 'spin_images'
mkdir_command = 'mkdir ' + dir_

# Make the directory
subprocess.call(mkdir_command, shell=True)

print('Printing ' + str(Num_samples_in_movie) + ' image samples for a movie\n')


# Get number of digits required for fmpeg  

# if N_samples is 2 digits, need 

num_digits = len(str(Num_samples_in_movie))


# Loop through each sample, plot a 2D plot using the gnuplot script saving to png (or eps?), producing N_samples .png's  
for i in range(0,Num_samples_in_movie):
  num_zeros = num_digits - len(str(i))

  # Need to add "N" zeros to the .png output so all digits are same 
  png_out_str = 'spin_n_' + str(i).zfill(num_digits) 

  # reprint column data in a .dat file 
  dat_out_str = png_out_str + '.dat' 

  Sx_vector = sample_arrays_Sx[i] 
  Sz_vector = sample_arrays_Sz[i] 

  # Need to sort the 2D data 
  Sx_data = {'x': x, 'y': y, 'Sx': Sx_vector}
  Sz_data = {'x': x, 'y': y, 'Sz': Sz_vector}
  d_frame_Sx = pd.DataFrame.from_dict(Sx_data)
  d_frame_Sz = pd.DataFrame.from_dict(Sz_data)
  
  
  # Sort the data  
  d_frame_Sx.sort_values(by=['x', 'y'], ascending = True, inplace=True) 
  d_frame_Sz.sort_values(by=['x', 'y'], ascending = True, inplace=True) 
  
  assert(len(list_x) == Nx)
  
  # Redefine numpy array post sorting
  Sx_sorted = np.array(d_frame_Sx['Sx']) 
  Sx_sorted.resize(Nx, Ny)
  Sx_sorted = np.transpose(Sx_sorted)
  
  
  Sz_sorted = np.array(d_frame_Sz['Sz']) 
  Sz_sorted.resize(Nx, Ny)
  Sz_sorted = np.transpose(Sz_sorted)
  
  Sx_sorted /= np.sqrt(Sx_sorted**2 + Sz_sorted**2)
  Sz_sorted /= np.sqrt(Sx_sorted**2 + Sz_sorted**2)


  # 3. Plot in gnuplot to save to .png
  # 3. Plot using matplotlib
  plt.figure(i, figsize=(8,6), dpi=80) 
  #plt.quiver(list_x, list_y, Sx_sorted.real, Sz_sorted.real, color = 'b', units = 'xy', scale = 1.)
  plt.quiver(list_x, list_y, Sx_sorted.real, Sz_sorted.real, color = 'b', units = 'xy', scale = 4.10)
  plt.title('Spin vector field : $<S_{x} , S_{z}>$ ', fontsize = 16)
  plt.xlabel('$x$', fontsize = 20, fontweight = 'bold')
  plt.ylabel('$y$', fontsize = 20, fontweight = 'bold')
  # plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
  #plt.colorbar()
  #plt.xlim(0, 32)
  #plt.ylim(0, 32)
  #plt.legend()
  #plt.savefig(png_out_str + '.png') 
  plt.savefig(png_out_str + '.png', bbox_inches='tight') 
  plt.close()

  # 4. Move .png to a folder  
  #subprocess.call('mv *.png' +  ' ' + dir_, shell = True)
  subprocess.call('mv ' +  png_out_str + '.png ' + dir_, shell = True)


 #  if(i == 0):
 #    sys.exit() 


# After the loop, copy the bash script into the folder with the density images 
bash_script = 'spin_movie.sh'
cp_cmd = 'cp ~/csbosonscpp/tools/movie_scripts/' + bash_script + ' ./' + dir_ 

subprocess.call(cp_cmd, shell=True)

# Execute the movie script command
os.chdir('./' + dir_)
subprocess.call('. ' + bash_script, shell=True)



