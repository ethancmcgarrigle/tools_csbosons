import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
# matplotlib.use('TkAgg')
import pdb

# Script to load and plot correlation data 

# import the data
ops_file = 'operators0.dat'
cols = np.loadtxt(ops_file, unpack=True)

CL_time = cols[2] # 3rd column is simulation time  

# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
r_i = params['system']['operators']['i'] 
r_j = params['system']['operators']['j']

print()
print('site i: ' + str(r_i))
print('site j: ' + str(r_j))
print()

# number of sites to correlate with 
num_sites = r_j - r_i 
print('Number of sites to loop through: ' + str(num_sites))

# process the observables and get averages 

cmd_string = "python3 ~/csbosonscpp/tools/stats.py -f " + './' + ops_file + ' -o '

for k in range(0, num_sites):
  cmd_string += 'ReC_'
  cmd_string += str(r_i)
  cmd_string += '_' + str(k + r_i) + ' ' 
 #  cmd_string += 'ImC_'
 #  cmd_string += str(r_i)
 #  cmd_string += '_' + str(k + r_i) + ' ' 


C_ij_data_filename = 'C_ij_data.dat'
print('python command: ')
cmd_string += '-a -q > ' + './' + C_ij_data_filename 
print(cmd_string)

# Run the python command
if not os.path.exists('./' + C_ij_data_filename):
  subprocess.call(cmd_string, shell = True) 

# Open the resulting data file 
in_file = open("./" + C_ij_data_filename, "r")
tmp = in_file.read()
tmp = re.split(r"\s+", tmp)
tmp = tmp[0:-1]
tmp = tuple(map(float, tmp))

# Create a list/array to hold the C_ij averages, data and their errors 
C_ij_data = np.zeros(num_sites)
C_ij_errs = np.zeros(num_sites)

# tmp will have dimensions num_sites * 2 
ctr = 0
for k in range(0, num_sites):
  C_ij_data[k] = tmp[ctr]
  C_ij_errs[k] = tmp[ctr + 1]
  ctr += 2



# Plot the C_ij data 
plt.figure(1)
plt.errorbar(range(r_i, r_j), C_ij_data, C_ij_errs, linewidth=0.5, markersize=6, marker = '*', label = '$C_{i, j}$')
plt.title('Correlation Function: $C_{i, j}$ i = ' + str(r_i), fontsize = 11)
plt.xlabel('Site number - j', fontsize = 20, fontweight = 'bold')
plt.ylabel('$C_{i, j}$', fontsize = 20, fontweight = 'bold')
plt.legend()
plt.show()


