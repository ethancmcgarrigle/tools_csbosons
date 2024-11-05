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


def calculate_field_average(field_data, Nx, dim, N_samples_to_avg, apply_ADT): # assumes cubic/square mesh 
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data

    # Get number of samples 
    N_samples = len(field_data)/(Nx**dim)
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




def integrate_intensive(field_data):
    result = 0. + 1j*0.
    result = np.sum(field_data) # consider nan-sum 
    result /= len(field_data) # divide by num elements 
    return result



def calc_err_division(x, y, x_err, y_err):
    result = 0.
    # assumes x and y are real 

    # Calculate error using standard error formula 
    result = np.sqrt( ((-x * y_err / (y**2))**2 ) + (x_err/y)**2) 
    return result    





# Script to load and plot correlation data 

# RUN process.sh FIRST to get a condensate fraction estimate 
# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['simulation']['Nx'] 
apply_ADT = params['simulation']['apply_ADT'] 
d = params['system']['Dim']
kappa_x = params['system']['kappa_x']
L = params['system']['CellLength']
Vol = L**d 

# Retrieve condensate fraction CL average from the run 
data_file = 'data0.dat'
try:
  print('Parsing ' + data_file)
except:
  print('please run process.sh first, script cannot find data0.dat')
else:
  noise_averaged_data = np.loadtxt(data_file, unpack=True) # parses into 2 columns, 1st column is means; 2nd is errorbars 

N_mean = noise_averaged_data[0][8]
N_err = noise_averaged_data[1][8]

# Pull out the condensate fraction 
#density = noise_averaged_data[0][0] 
#N_kO = noise_averaged_data[0][4]
#density_err = noise_averaged_data[1][0] # 2nd column is error, 1st element is real(density)

#cond_frac = N_kO/(density * (Nx**d))
_isPlotting = False

# Correlation data processing 
n_k_data = 'n_k.dat' # This is a function of "r'", the coordinate distance away from a reference site 'r' 


# Pull out time data
ops_file = 'operators0.dat'
column_ops = np.loadtxt(ops_file, unpack=True)

CL_time_data = column_ops[2]
CL_iteration_data = column_ops[0]





N = column_ops[3] + column_ops[5] # total particle number 
NkO = column_ops[19] # real part 


# Read column data from .dat file, loads all samples  
cols = np.loadtxt(n_k_data, unpack=True)

# Assumes 2D
assert(d == 2)
# n(k) data is in k-space, must cols 1 and 2 are kx and ky;  cols 5 and 6 are n(k) data
# Extract 1 set of x and y column data 
kx = cols[0][0:Nx**d]
ky = cols[1][0:Nx**d]

# Extract real and imaginary parts of the n(k) data 
nk_real = cols[4]
nk_imag = cols[5]

nk_data = np.zeros(len(nk_real), dtype=np.complex_)
nk_data = nk_real + 1j*nk_imag


list_kx = np.unique(kx)
list_ky = np.unique(ky)


## Search for kx = \kappa (nearest) and ky = 0 index positions (should be 2, for \pm \kappa)
# first instance of kappa_x in list will be ky=0  
# the kx we are looking for is near one of the kx's. k
#kappa_indx = np.argmin(np.abs(list_kx - kappa_x)) 
kappa_indx = np.argmin(np.abs(kx - kappa_x)) 

if(kappa_indx == 0):
    print('k = 0 is closest to kappa, moving to the next highest k_x')
    tmp = np.where(list_kx == 0)
    zero_indx = tmp[0][0] # 0 returns the list
    # Get next kx 
    k_kappa = list_kx[zero_indx + 1] # get the next element, adjacent to the right  
    print('Adjacent, rightward element:')
    print(k_kappa)
    # Find index in kx where this k_kappa starts 
    tmp = np.where(kx == k_kappa)
    kappa_indx = tmp[0][0]
    # repeat for negative  
    tmp = np.where(kx == -k_kappa)
    kappa_neg_indx = tmp[0][0] 
else:
    # want the first instanace -- corresponds to ky = 0 
    kappa_indx = np.argmin(np.abs(kx - kappa_x)) # gets first instance of kx = kappa_x 
    #kappa_indx_loc = np.where(

    kappa_neg_indx = np.argmin(np.abs(kx + kappa_x))  # gets first instance 
    #kappa_neg_indx_loc = np.where(kx == kx_neg_star)


# kx_star is the kx value that is closest to kappa_x 
kx_star = kx[kappa_indx] 
kx_neg_star = kx[kappa_neg_indx]


N_samples = int(len(nk_real)/(Nx**d))
Nk_samples = np.split(nk_data, N_samples)
NKappa_trace = np.zeros(N_samples, dtype=np.complex_)

# Len from ops should be the same as the sampling from the n_k data
#assert(len(CL_time_data) == len(Nk_samples))
#assert(len(CL_time_data) == N_samples)

# in each sample, extract the kx = \pm \kappa particle number 
for i in range(0, N_samples):
    NKappa_trace[i] += Nk_samples[i][kappa_neg_indx]  # -\kappa_x contribution
    NKappa_trace[i] += Nk_samples[i][kappa_indx] # +\kappa_x contribution


if(_isPlotting):
  plt.figure(1)
  plt.plot(CL_time_data, NKappa_trace.real/N, '-r', linewidth = 0.5, label = '$N_{k \pm \kappa_{x}}/N$') 
  plt.plot(CL_time_data, NkO/N, '-b', linewidth = 0.5, label = '$N_{k = 0}/N$') 
  plt.plot(CL_time_data, (NkO + NKappa_trace.real)/N, '-k', linewidth = 0.5, label = 'Sum') 
  plt.title('SOC, $T = 1K$, CL Simulation ', fontsize = 16)
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold') # actually "x"
  plt.ylabel('Particle Fractions', fontsize = 20, fontweight = 'bold')
  plt.legend()
  plt.show()
  
  
  plt.figure(2)
  plt.plot(CL_time_data, NKappa_trace.real/N, '-r', linewidth = 0.5, label = '$N_{k \pm \kappa_{x}}/N$') 
  plt.title('$n(k_{x} = \kappa_x , k_{y} = 0)/N$, CL Simulation ', fontsize = 16)
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold') # actually "x"
  plt.ylabel('Particle Fraction', fontsize = 20, fontweight = 'bold')
  plt.ylim(0.4, 0.6)
  plt.legend()
  plt.show()
  
  
  plt.figure(3)
  plt.plot(CL_time_data, NKappa_trace.imag/N, '-r', linewidth = 0.5, label = 'Im($N_{k \pm \kappa_{x}}/N$)') 
  plt.title('Im($n(k_{x} = \kappa_x , k_{y} = 0)$), CL Simulation ', fontsize = 16)
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold') # actually "x"
  plt.ylabel('Im(SOC Fraction)', fontsize = 20, fontweight = 'bold')
  #plt.ylim(0.4, 0.6)
  plt.legend()
  plt.show()
  
  plt.figure(4)
  plt.plot(CL_time_data, NKappa_trace.real, '-r', linewidth = 0.5, label = '$N_{k \pm \kappa_{x}}/N$') 
  plt.title('$n(k_{x} = \kappa_x , k_{y} = 0)$, CL Simulation ', fontsize = 16)
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold') # actually "x"
  plt.ylabel('N', fontsize = 20, fontweight = 'bold')
  plt.legend()
  plt.show()
  
  
# Calculate the average N_{\kappa_x} and its error 
if(apply_ADT):
  avgd_data, errs = calculate_field_average(nk_data, Nx, d, N_warmup_to_throw, apply_ADT)
else:
  avgd_data, errs = calculate_field_average(nk_data, Nx, d, int(0.80*N_samples))


# Grab the k_x = \kappa_x ones
noise_avgd_Nkappa = 0. + 1j*0. 
noise_avgd_Nkappa += avgd_data[kappa_neg_indx]
noise_avgd_Nkappa += avgd_data[kappa_indx]

err_NKappa = 0. + 1j* 0.
err_NKappa += errs[kappa_neg_indx]
err_NKappa += errs[kappa_indx]

SOC_frac = noise_avgd_Nkappa.real/(integrate_intensive(N.real).real)
print('CL averaged SOC particle number: ' + str(round(noise_avgd_Nkappa.real, 2)))
print('CL averaged SOC fraction: ' + str(round(noise_avgd_Nkappa.real/(integrate_intensive(N.real).real), 2)))

# propagate error 
SOC_frac_err = calc_err_division(noise_avgd_Nkappa.real, N_mean, err_NKappa.real, N_err)

# Print the N_SOC and its error to a data file 
outfile = 'N_SOC.dat' 
with open(outfile, 'w') as filehandle:
  filehandle.write("# N_SOC N_SOC_err SOC_frac SOC_frac_err\n") 
  filehandle.write("{} {} {} {}\n".format(noise_avgd_Nkappa.real, err_NKappa.real, SOC_frac, SOC_frac_err))



