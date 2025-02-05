import numpy as np
import pandas as pd 
from scipy.stats import sem 
from error_propagation import * 

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



def extract_grid(field_data: np.ndarray, N_spatial: int, inRealSpace: bool=True) -> tuple:
    ''' - Takes in a field file, formatted with its first d columns containing spatial coordinates. 
        - Outputs a tuple of np.ndarrays that contain the spatial grid coordinates. 
        - For dimensions less than 3, the unused coordinates are stored as zeros.
        For a k-space field, this returns the reciprocol space grid x <==> kx, y <==> ky, z <==> kz.'''
    if(inRealSpace):
      dimension = len(field_data) - 2   # Last two columns are the data (real and complex values) 
    else:
      dimension = (len(field_data) - 2)//2  # k-grid includes kx,ky,kz integer indices  

    # Extract the spatial grid  
    x = field_data[0][0:N_spatial]
    if(dimension > 1):
      y = field_data[1][0:N_spatial]
      if( dimension  > 2 ):
        z = field_data[2][0:N_spatial]
      else:
        z = np.zeros_like(y)
    else:
      y = np.zeros_like(x)
      z = np.zeros_like(x)

    return [x, y, z]


def process_data(file_list: list, N_spatial: int, CL: bool, inRealSpace: bool=True, N_samples_to_avg: int = 5) -> tuple:
    ''' Imports the field data, where the file-names are specified in the file_list and performs any averaging. 
        - N_spatial specifies the total number of spatial grid points in real or k-space. 
        - "CL" is a boolean, indicating whether the field files contain many samples. 
        - "inRealSpace" is a boolean, indicating whether the field files are in real space or k-space. 
     Assumes all files in the file list have the same grid. 


     For real space files, a small average over snapshots is desirable. 
     For k-space files, a long average over many snapshots is desirable (if snapshots are there).

        Returns the spatial (or k-space) grid, as well as lists containing the data (for each file) and 
         its standard errors. ''' 

    if not file_list:
      raise ValueError("File list cannot be empty") 

    file_data = [] # Create a list to hold the data for each file 
    for file in file_list: 
      file_data.append( np.loadtxt(file, unpack=True) )
      print('Processing data in file ' + file) 

    assert len(file_data) == len(file_list)

    # Extract the dimension and the number of samples from the first file 
    if(inRealSpace):
      dim = len(file_data[0]) - 2 
    else:
      dim = (len(file_data[0]) - 2)//2

    # Extract the spatial (r or k) grid from the first file. 
    grid = extract_grid(file_data[0], N_spatial, inRealSpace) 

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
      data_vectors.append(np.zeros(len(grid[0]), dtype=np.complex128))
      data_errs.append(np.zeros(len(grid[0]), dtype=np.complex128))
      # Calculate average  
      if(inRealSpace):
        data_vectors[i], data_errs[i] = calculate_field_average(data[dim] + 1j*data[dim+1], N_spatial, N_samples_to_avg)   
      else: 
        data_vectors[i], data_errs[i] = calculate_field_average(data[2*dim] + 1j*data[2*dim+1], N_spatial, N_samples_to_avg)   

    return grid, data_vectors, data_errs




def compute_angular_average(kr: np.ndarray, theta: np.ndarray, data_k: np.ndarray, data_k_errs: np.ndarray, dim: int=2):
    # TODO: Extend to 3D 
    if(dim == 1):
      raise ValueError("Dimension needs to be at least 2 for angular averaging.") 

    if(dim == 3):
      raise ValueError("Function currently expects dimension = 2. Update for 3D in-progress.") 

    kr_uniq = np.unique(kr)

    # Allocate 1D arrays for angular average     
    data_kr = np.zeros_like(kr_uniq)
    data_kr_errs = np.zeros_like(kr_uniq)

    if(dim == 2):    
      _polar_data = {'kr': kr, 'theta': theta, 'data_k': data_k, 'data_k_errs': data_k_errs}
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

    return kr_uniq, data_kr, data_kr_errs
