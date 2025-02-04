import numpy as np
from scipy.stats import sem 

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



def extract_grid(field_data: np.ndarray, N_spatial: int) -> tuple:
    ''' - Takes in a field file, formatted with its first d columns containing spatial coordinates. 
        - Outputs a tuple of np.ndarrays that contain the spatial grid coordinates. 
        - For dimensions less than 3, the unused coordinates are stored as zeros.
        For a k-space field, this returns the reciprocol space grid x <==> kx, y <==> ky, z <==> kz.'''
    dimension = len(field_data) - 2   # Last two columns are the data (real and complex values) 

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


def process_data(file_list: list, N_spatial: int, CL: bool, N_samples_to_avg: int = 5) -> tuple:
    ''' Imports the field data, where the file-names are specified in the file_list and performs any averaging. 
        - N_spatial specifies the total number of spatial grid points in real or k-space. 
        - "CL" is a boolean, indicating whether the field files contain many samples. 
        - "plots" specifies whether plots will be created and displayed. ''' 
    ''' Assumes all files have the same grid. 

        Returns the spatial (or k-space) grid, as well as lists containing the data (for each file) and 
         its standard errors. ''' 

    if not file_list:
      raise ValueError("File list cannot be empty") 

    file_data = [] # Create a list to hold the data for each file 
    for file in file_list: 
      file_data.append( np.loadtxt(file, unpack=True) )

    assert len(file_data) == len(file_list)

    # Extract the spatial (r or k) grid from the first file. 
    grid = extract_grid(file_data[0], N_spatial) 

    # Extract the dimension and the number of samples from the first file 
    dim = len(file_data[0]) - 2 
    
    N_samples = int(len(file_data[0][dim])/N_spatial) 

    # Extract real and imaginary part of each file and compute the average if necessary 
    data_vectors = [] 
    data_errs = []

    # Average the data 
    if(not CL):
      N_sample_to_avg = 1 

    if(CL and N_samples_to_avg > N_samples):
      # Default to 5 samples 
      N_samples_to_avg = 5

    for i, data in enumerate(file_data): 
      # Allocate space for extracted vectors
      data_vectors.append(np.zeros(len(grid[0]), dtype=np.complex128))
      data_errs.append(np.zeros(len(grid[0]), dtype=np.complex128))
      # Calculate average  
      data_vectors[i], data_errs[i] = calculate_field_average(data[dim] + 1j*data[dim+1], N_spatial, N_samples_to_avg)   

    return grid, data_vectors, data_errs
