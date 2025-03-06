import numpy as np
import pandas as pd 
from scipy.stats import sem 
from .error_propagation import * 
from .time_grid import TimeGrid

def calculate_field_average(field_data: np.ndarray, N_spatial: int, N_samples_to_avg: int) -> tuple: 
    ''' Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    field_data is data of N_samples * Nx*Ny*Nz, for d-dimensions. Typically complex data'''

    # Get number of samples 
    N_samples = len(field_data)/(N_spatial)
    assert N_samples.is_integer(), 'Error: Number of samples is not integer.'
    N_samples = int(N_samples)

    if(N_samples == 1):
      print('1 sample detected. Processing the snapshot instead of averaging')
      return field_data, np.zeros_like(field_data)
    else:
      print('Computing sample average. Averaging: ' + str(N_samples_to_avg) + ' samples. ')

    # Use split (np) to get arrays that represent each sample (1 array per sample) Throw out the first sample (not warmed up properly) 
    sample_arrays = np.split(field_data, N_samples) 
    sample_arrays = sample_arrays[len(sample_arrays) - N_samples_to_avg:len(sample_arrays)]

    # Final array, initialized to zeros. 
    averaged_data = np.zeros(len(sample_arrays[0]), dtype=np.complex128)
    averaged_data += np.mean(sample_arrays, axis=0) # axis=0 calculates element-by-element mean
    # Calculate the standard error 
    std_errs = np.zeros(len(sample_arrays[0]))
    std_errs += sem(sample_arrays, axis=0)
    return averaged_data, std_errs



def extract_grid(field_data: np.ndarray, N_spatial: int, inRealSpace: bool=True, dimension: int = 1) -> tuple:
    ''' - Takes in a field file, formatted with its first d columns containing spatial coordinates. 
        - Outputs a tuple of np.ndarrays that contain the spatial grid coordinates. 
        - For dimensions less than 3, the unused coordinates are stored as zeros.
        For a k-space field, this returns the reciprocol space grid x <==> kx, y <==> ky, z <==> kz.'''
    # Extract the spatial grid  
    if(not inRealSpace):
      multiplier = 2 # kspace grid includes integer plane-wave indices 
    else:
      multiplier = 1

    x = field_data[0*multiplier][0:N_spatial]
    if(dimension > 1):
      y = field_data[1*multiplier][0:N_spatial]
      if( dimension  > 2 ):
        z = field_data[2*multiplier][0:N_spatial]
      else:
        z = np.zeros_like(y)
    else:
      y = np.zeros_like(x)
      z = np.zeros_like(x)

    return [x, y, z]


 #
 #def process_CSfield_data(file_list: list, N_spatial: int, tgrid: TimeGrid, CL: bool, inRealSpace: bool=False, FrequencyRep: bool=True, N_samples_to_avg: int = 5) -> tuple:
 #    ''' Imports the CSfield data, where the file-names are specified in the file_list and performs any averaging. 
 #        - N_spatial specifies the total number of spatial grid points in real or k-space. 
 #        - "CL" is a boolean, indicating whether the field files contain many samples. 
 #        - "inRealSpace" is a boolean, indicating whether the field files are in real space or k-space. 
 #     Assumes all files in the file list have the same space-time grid.''' 
 #
 #    ''' CSfield data is d+1 dimensional (d spatial dimensions + a time-dimension) 
 #
 #
 #    Return: 
 #      1. The corresponding real-space grid (extracted from the file).  
 #      2. a CSfield object in an array  
 #      3. a CSfield object in an array corresponding to errors, if needed  
 #    '''
 #
 #    if not file_list:
 #      raise ValueError("File list cannot be empty") 
 #
 #    file_data = [] # Create a list to hold the data for each file 
 #    for file in file_list: 
 #      file_data.append( np.loadtxt(file, unpack=True) )
 #      print('Processing data in file ' + file) 
 #
 #    assert len(file_data) == len(file_list)
 #
 #    # Extract the dimension and the number of samples from the first file 
 #    # We require the time grid information to extract the dimensionality. 
 #    # In real space: the file contains 2*_Nt columns + d columns  
 #    # In k space: the file contains 2*_Nt columns + 2*d columns  
 #    # (factor of 2 comes from complex data types at each t-point) 
 #    # Num columns = len(file_data[0])
 #    nt_points = len(tgrid)
 #    if nt_points <= 0:
 #      raise ValueError("Number of time points cannot be zero or negative.")
 #
 #    if(inRealSpace):
 #      dim = len(file_data[0]) - 2*nt_points
 #    else:
 #      dim = (len(file_data[0]) - 2*nt_points)//2
 #
 #    # Extract the spatial (r or k) grid from the first file, using first 3 columns 
 #    # Expects a 2D-array input formatted where the first index represents the column, second represents rows  
 #    #   - for CSfield object, the num rows will be the same as a field object. 
 #    #   - a CSfield object will have more columns. 
 #    #   - the extract_grid() function only uses first 3 columns, so this can handle CSfield array types as well 
 #    grid = extract_grid(file_data[0], N_spatial, inRealSpace) 
 # 


def process_data(file_list: list, N_spatial: int, CL: bool, inRealSpace: bool=True, N_samples_to_avg: int = 5, nt_points: int = 1) -> tuple:
    ''' Imports the field data, where the file-names are specified in the file_list and performs any averaging. 
        - N_spatial specifies the total number of spatial grid points in real or k-space. 
        - "CL" is a boolean, indicating whether the field files contain many samples. 
        - "inRealSpace" is a boolean, indicating whether the field files are in real space or k-space. 
        - nt_points is the number of time points to account for if the field is a CSfield object. 
     Assumes all files in the file list have the same grid. 

     For real space files, a small average over snapshots is desirable. 
     For k-space files, a long average over many snapshots is desirable (if snapshots are there).

     Returns the spatial (or k-space) grid, as well as lists containing the data (for each file) and its standard errors. ''' 

    if not file_list:
      raise ValueError("File list cannot be empty") 

    file_data = [] # Create a list to hold the data for each file 
    for file in file_list: 
      file_data.append( np.loadtxt(file, unpack=True) )
      print('Processing data in file ' + file) 

    assert len(file_data) == len(file_list)

    # Extract the dimension and the number of samples from the first file 
    if nt_points <= 0:
      raise ValueError("Number of time points cannot be zero or negative.")

    if(inRealSpace):
      dim = len(file_data[0]) - 2*nt_points
    else:
      dim = (len(file_data[0]) - 2*nt_points)//2

    print('Dimension of the grid in file: ' + str(dim))

    # Extract the spatial (r or k) grid from the first file. 
    grid = extract_grid(file_data[0], N_spatial, inRealSpace, dim) 

    assert(len(grid[0]) == N_spatial)
    # Take any column of the first file and divide out the number of spatial degrees of freedom to get N_samples    
    N_samples = int(len(file_data[0][0])/N_spatial) 

    # Extract real and imaginary part of each file and compute the average if necessary 
    data_vectors = [] 
    data_errs = []

    # Average the data 
    if(not CL):
      N_sample_to_avg = 1 

    if(CL and N_samples_to_avg > N_samples and inRealSpace):
      # Default to 5 samples 
      N_samples_to_avg = 5

    if(CL and (not inRealSpace)):
      N_samples_to_avg = int(0.75 * N_samples)
    
    for i, data in enumerate(file_data): 
      # Allocate space for extracted vectors
      if(nt_points == 1):
        data_vectors.append(np.zeros(N_spatial, dtype=np.complex128))
        data_errs.append(np.zeros(N_spatial, dtype=np.complex128))
        # Calculate average  
        if(inRealSpace):
          data_vectors[i], data_errs[i] = calculate_field_average(data[dim] + 1j*data[dim+1], N_spatial, N_samples_to_avg)   
        else: 
          data_vectors[i], data_errs[i] = calculate_field_average(data[2*dim] + 1j*data[2*dim+1], N_spatial, N_samples_to_avg)   
      else:
        data_vectors.append(np.zeros((N_spatial, nt_points), dtype=np.complex128))
        data_errs.append(np.zeros((N_spatial, nt_points), dtype=np.complex128))
        # Calculate average 
        for j in range(nt_points): 
          if(inRealSpace):
            data_vectors[i][:, j], data_errs[i][:, j] = calculate_field_average(data[dim + 2*j] + 1j*data[dim+1 + 2*j], N_spatial, N_samples_to_avg)   
          else: 
            data_vectors[i][:, j], data_errs[i][:, j] = calculate_field_average(data[2*dim + 2*j] + 1j*data[2*dim+1 + 2*j], N_spatial, N_samples_to_avg)   

    return grid, data_vectors, data_errs




def compute_angular_average(kr: np.ndarray, data_k: np.ndarray, data_k_errs: np.ndarray, correlations: bool=False, nt_points: int = 1):
    ''' An optional flag "Correlations" will indicate whether a real-space correlation function is being passed through.
        Since periodic boundary conditions are assumed, we must NOT average over the entire domain if correlations=True. 
        Instead, perform the angular average only in the lower left quadrant of the system (r \in [(0,0), (Nx/2 , Ny/2)] )'''
    ''' For field arrays, data_k represents a flattened array of data as a function of k. 
        For CSfield arrays, data_k represents a 2D array of data where the rows represent k-dependence and cols represent t-dependence.'''

    kr_uniq = np.unique(kr)

    # If passing through C(|r|) data, find the halfway indx 
    if(correlations):
      halfway_indx = np.where(kr_uniq >= np.max(kr_uniq)/2)[0][0]
      kr_uniq = kr_uniq[0:halfway_indx] # truncate halfway 

    # Allocate 1D arrays for angular average     
    if(nt_points > 1):
      data_kr = np.zeros((len(kr_uniq), nt_points), dtype=np.complex128)
      data_kr_errs = np.zeros((len(kr_uniq), nt_points), dtype=np.complex128)
    else:
      data_kr = np.zeros_like(kr_uniq)
      data_kr_errs = np.zeros_like(kr_uniq)

    if(nt_points == 1):
      _polar_data = {'kr': kr, 'data_k': data_k, 'data_k_errs': data_k_errs}
      polar_d_frame = pd.DataFrame.from_dict(_polar_data)
      polar_d_frame.sort_values(by=['kr'], ascending = True, inplace=True) 

      # Manually handle kr = 0 element. 
      data_kr[0] += polar_d_frame['data_k'].iloc[0].real
      data_kr_errs[0] += polar_d_frame['data_k_errs'].iloc[0].real
      i = 0
      for kr_ in kr_uniq[1:len(kr_uniq)]:
        i += 1
        tmp_frame = (polar_d_frame['kr'] == kr_)
        indices = np.where(tmp_frame == True)[0] 
        #indices = indices[0] # 0th element is the list of true indices 
        assert(polar_d_frame['kr'].iloc[indices[0]] == kr_)
        # 2. Extract 
        data_kr[i] += polar_d_frame['data_k'].iloc[indices].mean().real
        # propagate error across the average 
        data_kr_errs[i] += calc_err_average(polar_d_frame['data_k_errs'].iloc[indices].values).real 
    else:
      for j in range(nt_points):
        _polar_data = {'kr': kr, 'data_k': data_k[:, j], 'data_k_errs': data_k_errs[:, j]}
        polar_d_frame = pd.DataFrame.from_dict(_polar_data)
        polar_d_frame.sort_values(by=['kr'], ascending = True, inplace=True) 
  
        # Manually handle kr = 0 element. 
        data_kr[0, j] += polar_d_frame['data_k'].iloc[0].real
        data_kr_errs[0, j] += polar_d_frame['data_k_errs'].iloc[0].real
        i = 0
        for kr_ in kr_uniq[1:len(kr_uniq)]:
          i += 1
          tmp_frame = (polar_d_frame['kr'] == kr_)
          indices = np.where(tmp_frame == True)[0] 
          #indices = indices[0] # 0th element is the list of true indices 
          assert(polar_d_frame['kr'].iloc[indices[0]] == kr_)
          # 2. Extract 
          data_kr[i, j] += polar_d_frame['data_k'].iloc[indices].mean().real 
          # propagate error across the average 
          data_kr_errs[i, j] += calc_err_average(polar_d_frame['data_k_errs'].iloc[indices].values).real 

    return kr_uniq, data_kr, data_kr_errs
