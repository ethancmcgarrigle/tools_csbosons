import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
#matplotlib.rcParams['text.usetex'] = True
matplotlib.use('TkAgg')
import pdb
import pandas as pd 
from scipy.stats import sem 
from scipy.optimize import curve_fit 



def power_law(x, a, b):
    return a * np.power(x, b)


def calculate_field_average(field_data, Nx, Ny, N_samples_to_avg): # assumes cubic/square mesh 
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



# Script to load and plot correlation data 

# RUN process.sh FIRST to get a condensate fraction estimate 
# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
model = params['system']['ModelType']
d = params['system']['Dim']

if(model == 'BOSE_HUBBARD_CUBIC'):
  Nx = params['system']['NSitesPer-x'] 
  Vol = Nx
  Ny = params['system']['NSitesPer-y'] 
  Nz = params['system']['NSitesPer-z'] 
  if d > 1 :
    Vol *= Ny
    if d > 2 :
      Vol *= Nz
else:
  Lx = params['system']['CellLength-x']
  Vol = Lx
  Ly = params['system']['CellLength-y']
  Lz = params['system']['CellLength-z']
  if d > 1 :
    Vol *= Ly
    if d > 2 :
      Vol *= Lz

  Nx = params['simulation']['Nx'] 
  Ny = params['simulation']['Ny'] 
  Nz = params['simulation']['Nz'] 


# Retrieve condensate fraction CL average from the run 
data_file = 'data0.dat'
try:
  print('Parsing ' + data_file)
except:
  print('please run process.sh first, script cannot find data0.dat')
else:
  noise_averaged_data = np.loadtxt(data_file, unpack=True) # parses into 2 columns, 1st column is means; 2nd is errorbars 


# Pull out the condensate fraction 
density = noise_averaged_data[0][0]/Vol 
N_kO = noise_averaged_data[0][4]
SF_frac = noise_averaged_data[0][10] / density
SF_frac_err = noise_averaged_data[1][10] / density
density_err = noise_averaged_data[1][0]/Vol  # 2nd column is error, 1st element is real(density)
#print(N_kO)

cond_frac = N_kO/(density * (Vol))


# Correlation data processing 
correlation_data = 'C_rprime.dat' # This is a function of "r'", the coordinate distance away from a reference site 'r' 

# Read column data from .dat file, loads all samples  
cols = np.loadtxt(correlation_data, unpack=True)

# Extract 1 set of x and y column data 
x = cols[0][0:Nx*Ny]
y = cols[1][0:Nx*Ny]

# Extract real and imaginary parts of the correlation data 
corr_real = cols[2]
corr_imag = cols[3]

list_x = np.unique(x)
list_y = np.unique(y)

C_rprime_data = np.zeros(len(corr_real), dtype=np.complex_)
C_rprime_data += corr_real + 1j * corr_imag 
errs = np.zeros(len(corr_real))

# Option to either take the block average of the data or the final data point  
_isAveraging = True

num_samples_to_avg = 50


# Average the data 
if(_isAveraging):
    C_rprime_data, errs = calculate_field_average(C_rprime_data, Nx, Ny, num_samples_to_avg)

 #else:
 #    C_rprime_data = np.split(C_rprime_data, len(C_rprime_data)/(Nx**d))
 #    C_rprime_data = C_rprime_data[-1] 
 #

# Put into a data frame 
C_rprime = {'x': x, 'y': y, 'corr': C_rprime_data, 'errs' : errs}
d_frame_corr = pd.DataFrame.from_dict(C_rprime)

# Sort the data 
d_frame_corr.sort_values(by=['x', 'y'], ascending = True, inplace=True) 


# Redefine numpy array post sorting
corr_sorted = np.array(d_frame_corr['corr'])
errs_sorted = np.array(d_frame_corr['errs'])
if d == 2: 
    corr_sorted.resize(Nx, Ny)
    errs_sorted.resize(Nx, Ny)
corr_sorted = np.transpose(corr_sorted)
errs_sorted = np.transpose(errs_sorted)

# TODO: do some averaging across one of the coordinates if possible 
 #corr_sorted_y_avg = np.zeros(len(list_x), dtype=np.complex_)
 #for j in range(0, len(list_y)):
 #  corr_sorted_y_avg += corr_sorted[:][j]
 #
 #corr_sorted_y_avg /= len(list_y)

# normalize by dividing off the density (r = 0 value)  
 #corr_sorted_y_avg /= corr_sorted_y_avg[0]
 #print(corr_sorted[:][0]/corr_sorted[0][0], 'not y-averaged array')

#stop_indx = int(7*(len(list_x)-1)/8)
stop_indx = int(len(list_x)/2)


plt.style.use('~/csbosonscpp/tools/python_scripts/plot_style.txt')
# Chooses an arbitrary direction in 2D
plt.figure(1)
#plt.plot(list_x, corr_sorted[:][0].real/corr_sorted[0][0].real, 'r*', label = '$C(r)$')
if d == 2:
    #plt.plot(list_x[0:stop_indx], corr_sorted[0][0:stop_indx].real, 'r*', label = '$C(r)$')
    #plt.errorbar(list_x[0:stop_indx], corr_sorted[0][0:stop_indx].real, errs_sorted[0][0:stop_indx], 'r*', label = '$C(r)$')
    plt.errorbar(list_x[0:stop_indx], corr_sorted[0][0:stop_indx].real, errs_sorted[0][0:stop_indx], elinewidth = 0.5, color = 'r', linewidth = 0.5, marker = '*', label = '$C(r)$')
elif d == 1:
    plt.errorbar(list_x[0:stop_indx], corr_sorted[0:stop_indx].real, errs_sorted[0:stop_indx], elinewidth = 0.5, linewidth = 0.5, color = 'r', marker = '*', label = '$C(r)$')
plt.title('Correlation Function: $C(r)$ ', fontsize = 16)
plt.xlabel('$r$', fontsize = 20, fontweight = 'bold') # actually "x"
plt.ylabel('$C$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
#plt.xlim(-1, 1)
plt.legend()
plt.show()


# Notion of a coherence function: https://www.nii.ac.jp/qis/first-quantum/e/forStudents/lecture/pdf/qis385/QIS385_chap2.pdf

# Calculate error using standard error formula 
normalized_errs = np.sqrt( ((-corr_sorted.real * density_err / density)**2 ) + (errs_sorted/density)**2) 


lam_T = 3.3796

plt.figure(2)
#plt.plot(list_x[0:midway_indx], corr_sorted_y_avg[0:midway_indx].real, linewidth=0.5, markersize=6, marker = '*', label = '$C(r)$')
if d == 2:
    #plt.plot(list_x[0:stop_indx], corr_sorted[0][0:stop_indx].real/corr_sorted[0][0].real, linewidth=0.5, markersize=6, marker = '*', label = '$C(r)$')
    plt.errorbar(list_x[0:stop_indx]/lam_T, corr_sorted[0][0:stop_indx].real/corr_sorted[0][0].real, normalized_errs[0][0:stop_indx], marker = 'o', markersize = 6, elinewidth=0.5, linewidth = 0.0,  color = 'r', label = 'CL')
elif d == 1:
    #plt.plot(list_x[0:stop_indx], corr_sorted[0:stop_indx].real/corr_sorted[0].real, linewidth=0.5, markersize=6, marker = '*', label = '$C(r)$')
    plt.errorbar(list_x[0:stop_indx]/lam_T, corr_sorted[0:stop_indx].real/corr_sorted[0].real, normalized_errs[0:stop_indx], elinewidth=0.5, linewidth=0.5, color = 'r', marker = '*', label = '$C(r)$')

# Fit the data to a powerlaw 
pars,cov = curve_fit(f=power_law, xdata=list_x[1:stop_indx]/lam_T, ydata=corr_sorted[0][1:stop_indx].real/corr_sorted[0][0].real, p0=[0,0], bounds=(-np.inf, np.inf))
#print(pars[0])
#print(pars[1])
#plt.plot(list_x[0:stop_indx]/lam_T, pars[0] * (list_x[0:stop_indx]/lam_T)**(-0.25), '-k', linewidth = 0.5, label = 'Power Law: $r^{-1/4}$')
plt.plot(list_x[0:stop_indx]/lam_T, pars[0] * (list_x[0:stop_indx]/lam_T)**(pars[1]), color='r', ls = '--', linewidth = 2.0, label = 'Power Law: $ r^{' + str(round(pars[1],2)) + '}$')
plt.axhline(y = cond_frac, ls='--',color = 'k', linewidth = 0.90, label = '$N_{k = 0}/N$')
#plt.plot(list_x[0:stop_indx]/8.73, cond_frac * np.ones(stop_indx), '-k', linewidth = 0.5, label = '$N_{k = 0}/N$') 
#plt.plot(list_x[0:stop_indx], SF_frac * np.ones(stop_indx), '-b', linewidth = 0.5, label = r'$\rho_{SF}/ \rho$') 
plt.title('Normalized Correlation Function ', fontsize = 16)
plt.xlabel('$r/\lambda_{T}$', fontsize = 20, fontweight = 'bold') # actually "x"
plt.ylabel('$C(r)/C(0)$', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
#plt.xlim(-1, 1)
plt.legend()
plt.savefig('Tc_2D_corr_fxn.eps')
plt.show()


