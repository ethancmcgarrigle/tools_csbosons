
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
    weights_matrix = np.transpose([weights] * len(X_avgs))
    term3 = np.sum(weights_matrix * X**2, axis=0)   # reduces 2d array input to 1d array  
    term3 /= summed_weights
    term3 -= X_avgs**2
    term3 *= term1

    return np.sqrt(term3)



def doStats(filename, N_input_to_throw):
    # Pull out time data
    #ops_file = 'operators0.dat'
    filename.seek(0)
    ops_file = filename.name
    column_ops = np.loadtxt(ops_file, unpack=True)
    column_ops = np.transpose(column_ops) 
    
    CL_time_data = column_ops[:,2]
    CL_timesteps = column_ops[:,1] # column 2 is the timestep output, these are block-averaged 
    
    #    <O> = np.mean( O(t) * dt(t) * N_samples / sum(dt(t)) )
    #  -- Weighted average doesn't require a factor of 1/T -- 
    #    <O> = np.mean( O(t) * dt(t) * iointerval * N_samples )
    # likely better to weight by the elapsed time (i.e. dt \approx  iointerval * N_samples vs. dt = CL_time[j] - CL_time[j-1]  
    
    # create a vector of elapsed time s.t. t_elap[0] = 0, t_elap[1] = CL-time[1] - CL-time[0], etc. 
    t_elapsed = np.zeros(len(CL_timesteps))
    
    for j in range(1, len(CL_time_data)):
      t_elapsed[j] = CL_time_data[j] - CL_time_data[j-1]
    
    
    # Need to know the number of samples to throw out from warmup 
    # Default, throwout 0 to 5 samples
    # first sample is already thrown out automatically, in principle 
    
    # e.g. N_input_to_Throw = 1 corresponds to the first 2 samples being thrown out! First sample is always thrown out
    
    #N_input_to_throw = 1  # \in [2, N_samples] 
    #print('Throwing out ' + str(N_input_to_throw) + ' samples')
    N_samples = len(CL_timesteps) - N_input_to_throw - 1
    
    t_thrown = CL_time_data[N_input_to_throw]
    total_t_elapsed = CL_time_data[-1] - t_thrown 
    
    #scale_factor = t_elapsed * (N_samples)/ total_t_elapsed # N * dt_j / T 
    
    scale_factor = t_elapsed[N_input_to_throw+1:len(t_elapsed)]/ total_t_elapsed # N * dt_j / T 
    
    #scale_factor[0] = 1./ total_t_elapsed # weight the first sample perfectly  
    #scale_factor[0] = N_samples * 1./ total_t_elapsed # weight the first sample perfectly  
    
    # --- Can't just multiple by N_samples beceause stats.py uses autowarmup correction and throws out some samples 
    
    # Create 2D array to apply to all columns 
    scale_factors = np.transpose([scale_factor] * (len(column_ops[1]) - 3) )
    
    
    num_observables = len(column_ops[0]) - 3
    # Scale all of the observables after column 3
    O_obs = np.zeros((len(CL_timesteps) - N_input_to_throw - 1 , num_observables)) 
    O_obs += column_ops[N_input_to_throw+1:len(CL_timesteps), 3:len(column_ops[0])]
    column_ops[N_input_to_throw+1:len(CL_timesteps), 3:len(column_ops[0])] *= scale_factors
    
    # need header string for the operators0.dat file 
    #subprocess.call('head -1 operators0.dat > ops_header.dat', shell=True)
    with open(ops_file) as ops:
      lines = ops.read()
      first_line = lines.split('\n', 1)[0]
    
    first_line = first_line[2:-1]
    
    # Output into operators0.dat file 
    #np.savetxt('operators0_ADT.dat', column_ops, delimiter=' ', header = first_line)
    
 #    print(sum(column_ops[N_input_to_throw+1:len(CL_timesteps),3]))
 #    print(sum(column_ops[N_input_to_throw+1:len(CL_timesteps),11]))
 #    print(sum(column_ops[N_input_to_throw+1:len(CL_timesteps),12]))
 #    #print(sum(t_elapsed[N_input_to_throw+1:len(t_elapsed)]/total_t_elapsed))
 #    print(sum(scale_factor))
 #    
    
    # Make a data0_ADT.dat file with new means and errs 
    output_data = np.zeros((num_observables,2)) # 2 columns, 1 for mean and 1 for sem 
    
    # Want to sum the weighted COLUMNS to get the means 
    output_data[:, 0] = sum(column_ops[N_input_to_throw+1:len(CL_timesteps), 3:num_observables + 3], 0) # axis = 0 gives sum of columns

    # Calculate standard errors via formula from ( https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf )
    sems = np.zeros(num_observables) 
    sems = calc_sem(O_obs, output_data[:,0], t_elapsed[N_input_to_throw+1:len(t_elapsed)], total_t_elapsed) 

    # Need to compute a weighted error 
    #output_data[:, 1] = sem(column_ops[N_input_to_throw+1:len(CL_timesteps), 3:num_observables + 3], axis=0, ddof=0)
    output_data[:, 1] = sems 
    
    for i in range(0, num_observables):
      sys.stdout.write("{0} {1}\n".format(output_data[i, 0],output_data[i, 1]))
    sys.stdout.write("\n")
    # np.savetxt('data0_ADT.dat', output_data, delimiter=' ')
    




if __name__ == "__main__":
    # For command-line runs, build the relevant parser
    import argparse as ap
    parser = ap.ArgumentParser(description='Statistical analysis of csbosonscpp data')
    parser.add_argument('-f','--file',default='./operators0.dat',type=ap.FileType('r'),help='Filename containing scalar statistical data.')
    parser.add_argument('-w', '--warmup',default=6,type=int,help='Number of samples to eliminate from the beginning of the data.')
    args=parser.parse_args()
    doStats(args.file, args.warmup)

 
