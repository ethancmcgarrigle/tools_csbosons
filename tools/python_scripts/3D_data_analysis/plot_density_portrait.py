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

def calculate_field_average(field_data, N_spatial, N_samples_to_avg, start_index): # assumes cubic/square mesh
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data

    # Get number of samples
    N_samples = len(field_data)/(N_spatial)
    assert(N_samples.is_integer())
    N_samples = int(N_samples)

    # Use split (np) to get arrays that represent each sample (1 array per sample) Throw out the first sample (not warmed up properly)
    sample_arrays = np.split(field_data, N_samples)
    # Save only the last N_samples_to_avg
    # sample_arrays = sample_arrays[len(sample_arrays) - N_samples_to_avg:len(sample_arrays)]
    # Save only index start - start + N_samples_to_avg -- allows us to take averaged snapshots of the density
    sample_arrays = sample_arrays[start_index:start_index + N_samples_to_avg]

    # Final array, initialized to zeros.
    averaged_data = np.zeros(len(sample_arrays[0]), dtype=np.complex_)
    print('Averaging ' + str(int(len(sample_arrays))) + ' samples')
    print('Start index ' + str(start_index))
    averaged_data += np.mean(sample_arrays, axis=0) # axis=0 calculates element-by-element mean
    # Calculate the standard error
    std_errs = np.zeros(len(sample_arrays[0]))
    std_errs += sem(sample_arrays, axis=0)
    return averaged_data, std_errs

def main(Nsamples_to_avg, start_index, fig_storage_prefix):
  # Script to load and plot correlation data
  # import the input parameters, specifically the i and j indices
    with open('input.yml') as infile:
      params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
    Nx = params['simulation']['Nx']
    Ny = params['simulation']['Ny']
    Nz = params['simulation']['Nz']
    Lx = params['system']['CellLength-x']
    Ly = params['system']['CellLength-y']
    Lz = params['system']['CellLength-z']
    dx = Lx/float(Nx)
    dy = Ly/float(Ny)
    dz = Lz/float(Nz)
    d = params['system']['Dim']
    T_ = 1./np.float(params['system']['beta'])
    kappa = params['system']['kappa']

# import the data
    Sx_file = 'density1.dat'
    Sz_file = 'density2.dat'
    cols_x = np.loadtxt(Sx_file, unpack=True)
    cols_y = np.loadtxt(Sz_file, unpack=True)

    N_spatial = Nx*Ny*Nz
# Extract 1 set of x and y column data
    x = cols_x[0][0:N_spatial]
    y = cols_x[1][0:N_spatial]
    z = cols_x[2][0:N_spatial]

    S_x_real = cols_x[3]
    S_x_imag = cols_x[4]

    S_z_real = cols_y[3]
    S_z_imag = cols_y[4]

    list_x = np.unique(x)
    list_y = np.unique(y)
    list_z = np.unique(z)

    N_samples = int(len(S_x_real)/(N_spatial))

    print('Total number of samples: ' + str(int(N_samples)))

    Sx_vector = np.zeros(len(x), dtype=np.complex_)
    Sz_vector = np.zeros(len(x), dtype=np.complex_)

# Average the data
    Sx_vector, Sx_errs = calculate_field_average(S_x_real + 1j*S_x_imag, N_spatial, Nsamples_to_avg, start_index)
    Sz_vector, Sz_errs = calculate_field_average(S_z_real + 1j*S_z_imag, N_spatial, Nsamples_to_avg, start_index)

# Store a 2D array of <Sx, Sy> data
    Sx_data = {'x': x, 'y': y, 'z' : z, 'Sx': Sx_vector, 'Sz': Sz_vector}
    d_frame_Sx = pd.DataFrame.from_dict(Sx_data)

# Don't Sort the data for vtk production 
    #d_frame_Sx.sort_values(by=['x', 'y', 'z'], ascending = True, inplace=True)

    assert(len(list_x) == Nx)

# Redefine numpy array post sorting
    Sx_sorted = np.array(d_frame_Sx['Sx']) # should be a flattened function of space  
    #Sx_sorted.resize(Nx, Nx)
    #Sx_sorted = np.transpose(Sx_sorted)

    Sz_sorted = np.array(d_frame_Sx['Sz'])
    #Sz_sorted.resize(Nx, Nx)
    #Sz_sorted = np.transpose(Sz_sorted)

    density_up = Sx_sorted
    density_dwn = Sz_sorted

    density_profiles = [density_up, density_dwn]
    suffix = 'T_' + str(np.round(T_,2)) + '_start_' + str(start_index) + '_avglen' + str(Nsamples_to_avg) + '_'
    fname_str = ['rho_0_', 'rho_1_']
    # Now save to VTK 
    for i, profile in enumerate(density_profiles):
      # Do real part  
      with open (fig_storage_prefix + fname_str[i] + suffix + "real.vtk", "w") as fout:
        fout.write("# vtk DataFile Version 2.0\n")
        fout.write("CT Density\n")
        fout.write("ASCII\n")
        fout.write("\n")
        fout.write("DATASET STRUCTURED_POINTS\n")
        fout.write("DIMENSIONS " + str(Nz) + " " + str(Ny) + " " + str(Nx) + "\n")
        fout.write("ORIGIN 0.000000 0.000000 0.000000\n")
        fout.write("SPACING " + str(dz) + " " + str(dy) + " " + str(dx) +"\n")
        fout.write("\n")
        fout.write("POINT_DATA " + str(int(Nx)*int(Ny)*int(Nz)) + "\n")
        fout.write("SCALARS scalars float\n")
        fout.write("LOOKUP_TABLE default\n\n")
      
        for x in profile:
          fout.write(str(x.real) + "\n")
      
      # Do imag part  
      with open (fig_storage_prefix + fname_str[i] + suffix + "imag.vtk", "w") as fout:
        fout.write("# vtk DataFile Version 2.0\n")
        fout.write("CT Density\n")
        fout.write("ASCII\n")
        fout.write("\n")
        fout.write("DATASET STRUCTURED_POINTS\n")
        fout.write("DIMENSIONS " + str(Nz) + " " + str(Ny) + " " + str(Nx) + "\n")
        fout.write("ORIGIN 0.000000 0.000000 0.000000\n")
        fout.write("SPACING " + str(dz) + " " + str(dy) + " " + str(dx) +"\n")
        fout.write("\n")
        fout.write("POINT_DATA " + str(int(Nx)*int(Ny)*int(Nz)) + "\n")
        fout.write("SCALARS scalars float\n")
        fout.write("LOOKUP_TABLE default\n\n")
      
        for x in profile:
          fout.write(str(x.imag) + "\n")


if __name__ == '__main__':
  main(5, 10, '')
