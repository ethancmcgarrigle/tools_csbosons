import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.use('TkAgg')
import pdb
import pandas as pd 
from scipy.stats import sem 
import sys


def calc_sem(X, X_avgs, weights, summed_weights): 
    # X is a 2D array of observations, each column is an operator, rows are "time"
    # X_avgs is a vector of weighted average of len(N_operators)
    # Weights are len(observations)
    # Summed weights is a scalar 

    # Checks
    assert(len(X_avgs) == len(X[0, :])) # Num_operators
    assert(len(weights) == len(X[:, 0])) # Num_observations 

    # Calculate and return the standard error (weighted) of the mean 
    # Term1 is a scalar, same scalar applied to each operator's Sem 
    term1 = np.sum(weights**2)
    term2 = summed_weights**2 
    term2 -= term1
    term1 /= term2

    # Term3 is a vector of len(N_operators)
    # Make weights a N_obs x N_ops matrix 
    # TODO do we need a transpose on the weights? probably 
    weights_matrix = np.transpose([weights] * len(X_avgs))
    term3 = np.sum(weights_matrix * X**2, axis=0)   # reduces 2d array input to 1d array  
    term3 /= summed_weights
    term3 -= X_avgs**2
    term3 *= term1

    return np.sqrt(term3)





def calculate_field_average(field_data, Nx, dim, N_samples_to_avg, apply_ADT=False, scale_factor=np.ones(100), weights = np.ones(100), summed_weights = 1000): # assumes cubic/square mesh 
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data

    # Get number of samples 
    N_samples = len(field_data)/(Nx**dim)
    assert(N_samples.is_integer())
    N_samples = int(N_samples)


    # Use split (np) to get arrays that represent each sample (1 array per sample) Throw out the first sample (not warmed up properly) 
    sample_arrays = np.split(field_data, N_samples) 
    sample_arrays = sample_arrays[N_samples - N_samples_to_avg:N_samples]

    # --- Can't just multiply by N_samples and use stats.py beceause stats.py uses autowarmup correction and throws out some samples 
    
    # Create 2D array to apply to all columns, only 2 columns of data (Re and Im part) 
    assert(len(scale_factor) == len(sample_arrays))
    
    # Scale all of the observables after column 3
    # TODO remove forloop implementation
    N_field_elements = Nx**dim    # Assumes CUBIC/SQUARE mesh 
    O_obs = np.zeros((len(sample_arrays), N_field_elements), dtype=np.complex_) 
    O_obs += sample_arrays
    for j in range(0, len(sample_arrays)):
        sample_arrays[j] *= scale_factor[j]

    # Final array, initialized to zeros. 
    averaged_data = np.zeros(len(sample_arrays[0]), dtype=np.complex_)
    if(apply_ADT):
        averaged_data += np.sum(sample_arrays, axis=0)
    else:    
        averaged_data += np.mean(sample_arrays, axis=0) # axis=0 calculates element-by-element mean

    # Calculate the standard error 
    std_errs = np.zeros(len(sample_arrays[0]), dtype=np.complex_)
    if(apply_ADT):
        # Calculate standard errors via formula from ( https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf )
        std_errs += calc_sem(O_obs, averaged_data, weights, summed_weights) 
    else:    
        std_errs += sem(sample_arrays, axis=0)

    # Need to compute a weighted error 
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




def doStats(filename, N_input_to_throw, ops_file, input_file, scalar_avgs_file):
    # Pull out time data

    # Script to load and plot correlation data 
    
    # RUN process.sh FIRST to get a condensate fraction estimate 
    # import the input parameters, specifically the i and j indices 

    scalar_avgs_file.seek(0)
    data_file = scalar_avgs_file.name

    input_file.seek(0)
    input_filename = input_file.name

    with open(input_filename) as infile:
      params = yaml.load(infile, Loader=yaml.FullLoader)
    
    # Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
    Nx = params['simulation']['Nx'] 
    #Ny = params['simulation']['Ny'] 
    apply_ADT = params['simulation']['apply_ADT'] 
    d = params['system']['Dim']
    kappa_x = params['system']['kappa_x']
    
    # Retrieve condensate fraction CL average from the run
 #    if(apply_ADT):
 #      data_file = 'data0_ADT.dat' 
 #    else:
 #      data_file = 'data0.dat'


 #    try:
 #      #print('Parsing ' + data_file)
 #    except:
 #      #print('please run process.sh first, script cannot find data0.dat')
 #    else:
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
    filename.seek(0)
    n_k_data = filename.name 

    ops_file.seek(0)
    ops_filename = ops_file.name
    
    # Pull out time data
    #ops_file = 'operators0.dat'
    column_ops = np.loadtxt(ops_filename, unpack=True)
    
    CL_time_data = column_ops[2]
    CL_timesteps = column_ops[1]
    CL_iteration_data = column_ops[0]
    
    
    N = column_ops[3] + column_ops[5] # total particle number 
    NkO = column_ops[19] # real part 
    
    
    # Read column data from .dat file, loads all samples  
    cols = np.loadtxt(n_k_data, unpack=True)
    
    # Assumes 2D
    #assert(d == 2)
    # n(k) data is in k-space, must cols 1 and 2 are kx and ky;  cols 5 and 6 are n(k) data
    # Extract 1 set of x and y column data 
    kx = cols[0][0:Nx**d]
    ky = cols[1][0:Nx**d]
    
    # Extract real and imaginary parts of the n(k) data (4 and 5 for d = 2) 
    nk_real = cols[2*d]
    nk_imag = cols[2*d + 1]
    
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
      
      


    # For ADT averaging 
    # create a vector of elapsed time s.t. t_elap[0] = 0, t_elap[1] = CL-time[1] - CL-time[0], etc. 
    t_elapsed = np.zeros(len(CL_timesteps))
    
    for j in range(1, len(CL_time_data)):
      t_elapsed[j] = CL_time_data[j] - CL_time_data[j-1]
    
    t_thrown = CL_time_data[N_input_to_throw]
    total_t_elapsed = CL_time_data[-1] - t_thrown 
    
    scale_factor = t_elapsed[N_input_to_throw+1:len(t_elapsed)]/ total_t_elapsed # N * dt_j / T 

    #print('Checking weighting sum (should be 1.00): ' + str(sum(scale_factor)))

    # Calculate the average N_{\kappa_x} and its error 
    if(apply_ADT):
        avgd_data, errs = calculate_field_average(nk_data, Nx, d, N_samples - N_input_to_throw - 1, apply_ADT, scale_factor, t_elapsed[N_input_to_throw+1:len(t_elapsed)], total_t_elapsed)
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
 #    print('CL averaged SOC particle number: ' + str(round(noise_avgd_Nkappa.real, 2)))
 #    print('CL averaged SOC fraction: ' + str(round(noise_avgd_Nkappa.real/(integrate_intensive(N.real).real), 2)))
    
    # propagate error 
    SOC_frac_err = calc_err_division(noise_avgd_Nkappa.real, N_mean, err_NKappa.real, N_err)
    
    # Print the N_SOC and its error to a data file 
 #    outfile = 'N_SOC.dat' 
 #    with open(outfile, 'w') as filehandle:
 #        filehandle.write("# N_SOC N_SOC_err SOC_frac SOC_frac_err\n") 
 #        filehandle.write("{} {} {} {}\n".format(noise_avgd_Nkappa.real, err_NKappa.real, SOC_frac, SOC_frac_err))
    sys.stdout.write("{0} {1} {2} {3}\n".format(noise_avgd_Nkappa.real, err_NKappa.real, SOC_frac, SOC_frac_err))
    #sys.stdout.write("\n")
    
    




if __name__ == "__main__":
    # For command-line runs, build the relevant parser
    import argparse as ap
    parser = ap.ArgumentParser(description='Statistical analysis of csbosonscpp data')
    parser.add_argument('-f','--file',default='./n_k.dat',type=ap.FileType('r'),help='Filename containing field statistical data.')
    parser.add_argument('-fop','--file_ops',default='./operators0.dat',type=ap.FileType('r'),help='Filename containing scalar statistical data.')
    parser.add_argument('-in','--file_input',default='./input.yml',type=ap.FileType('r'),help='Filename containing input parameters .')
    parser.add_argument('-d','--data_file',default='./data0.dat',type=ap.FileType('r'),help='Filename containing scalar_avgs.')
    parser.add_argument('-w', '--warmup',default=6,type=int,help='Number of samples to eliminate from the beginning of the data.')
    args=parser.parse_args()
    doStats(args.file, args.warmup, args.file_ops, args.file_input, args.data_file)


