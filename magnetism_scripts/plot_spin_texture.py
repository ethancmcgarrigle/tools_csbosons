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
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, to_rgba
import seaborn as sns

def calculate_field_average(field_data, N_spatial, N_samples_to_avg): # assumes cubic/square mesh 
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data

    # Get number of samples 
    N_samples = len(field_data)/(N_spatial)
    assert(N_samples.is_integer())
    N_samples = int(N_samples)

    if(N_samples == 1):
      print('1 sample detected. Processing the snapshot instead of averaging')
      return field_data, np.zeros_like(field_data)

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




def analyze_data(Sx_file, Sy_file, Sz_file, N_spatial, d, CL, plots):
  # Return a list of [Sx, Sy, Sz] in the final spin texture, normalized  
  cols_x = np.loadtxt(Sx_file, unpack=True)
  cols_y = np.loadtxt(Sy_file, unpack=True)
  cols_z = np.loadtxt(Sz_file, unpack=True)
  
  # Extract the spatial grid  
  x = cols_x[0][0:N_spatial]
  if(d > 1):
    y = cols_x[1][0:N_spatial]
    if( d > 2 ):
      z = cols_x[2][0:N_spatial]
    else:
      z = np.zeros_like(y)
  else:
    y = np.zeros_like(x)
    z = np.zeros_like(x)
  
  S_x_real = cols_x[d]
  S_x_imag = cols_x[d+1]
  
  S_y_real = cols_y[d]
  S_y_imag = cols_y[d+1]
  
  S_z_real = cols_z[d]
  S_z_imag = cols_z[d+1]
  
  N_samples = int(len(S_x_real)/(N_spatial))
  
  Sx_vector = np.zeros(len(x), dtype=np.complex128)
  Sy_vector = np.zeros(len(x), dtype=np.complex128)
  Sz_vector = np.zeros(len(x), dtype=np.complex128)
  
  # Average the data 
  if(_CL):
    pcnt = 5./N_samples 
  else:
    pcnt = 1./N_samples 
  
  Sx_vector, Sx_errs = calculate_field_average(S_x_real + 1j*S_x_imag, N_spatial, int(pcnt * N_samples))   
  Sy_vector, Sy_errs = calculate_field_average(S_y_real + 1j*S_y_imag, N_spatial, int(pcnt * N_samples))   
  Sz_vector, Sz_errs = calculate_field_average(S_z_real + 1j*S_z_imag, N_spatial, int(pcnt * N_samples))   
  
  if(_CL):
    print('Averaging: ' + str(int(pcnt * N_samples)) + ' samples')
  
  S_data = {'x': x, 'y': y, 'z': z, 'Sx': Sx_vector, 'Sy': Sy_vector, 'Sz': Sz_vector}
  
  d_frame_S = pd.DataFrame.from_dict(S_data)
  
  # Redefine numpy array post sorting
  Sx_sorted = np.array(d_frame_S['Sx']) 
  Sy_sorted = np.array(d_frame_S['Sy']) 
  Sz_sorted = np.array(d_frame_S['Sz']) 
  if(d > 1):
    Sx_sorted.resize(Nx, Ny)
    
    Sy_sorted.resize(Nx, Ny)
    
    Sz_sorted.resize(Nx, Ny)
    #Sz_sorted = np.transpose(Sz_sorted) # not needed for quiver 
    #Sz_sorted = np.flip(Sz_sorted, 0) ##  np.flip is only needed for imshow(); for quiver, it is NOT necessary
  
  if(d == 3):
    print('WARNING: visualization script not updated for d = 3 dimensions')
  
  planar_norm = np.sqrt(Sx_sorted.real**2 + Sy_sorted.real**2)
  # Total norm is a field N(r) where we have calculated the normalizing factor at each site (r)
  total_norm = np.sqrt(Sx_sorted.real**2 + Sy_sorted.real**2 + Sz_sorted.real**2)
  
  #np.savetxt('X_data.dat', list_x)
  #np.savetxt('Y_data.dat', list_y)
   #np.savetxt('spinX_Data.dat', Sx_sorted.real / total_norm)
   #np.savetxt('spinY_Data.dat', Sy_sorted.real / total_norm)
   #np.savetxt('spinZ_Data.dat', Sz_sorted.real / total_norm)
  
  ## Plot the spin map where the arrow represents (Mx, My); the color represents Mz 
  
  if(plots):
    sns_cmap = ListedColormap(sns.color_palette("RdBu", 256)) 
    plt.style.use('~/CSBosonsCpp/tools/python_plot_styles_examples/plot_style_spins.txt')
    #plt.figure(1)
    plt.figure(figsize=(3.38583*2, 3.38583*2))
    
    for i in range(N_spatial):
      if( i + Ny < N_spatial):
        plt.plot([x[i], x[i+Ny]], [y[i], y[i+Ny]], color='k', lw = 1.0, zorder = 1)
    
      if( (i % Ny) < (Ny - 1)):
        plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color='k', lw = 1.0, zorder = 1)
    
    plt.scatter(x, y, color = 'k', s = 2)
    plt.quiver(x, y, Sx_sorted.real/total_norm, Sy_sorted.real/total_norm, Sz_sorted.real/total_norm, units = 'xy', cmap=sns_cmap, pivot = 'middle', zorder = 2) 
    cbar = plt.colorbar()
    cbar.set_label(label = r'$M_{z}$', size = 30, rotation = 2)
    cbar.set_ticks([-1, -0.5, 0.0, 0.5, 1])
    cbar.set_ticklabels([-1, -0.5, 0.0, 0.5, 1])
    cbar.ax.tick_params(labelsize=20)
    if(d == 1): # manually set the y window so that we can see the spins
      plt.ylim(-4, 4)
    plt.clim(-1.0,1.0) 
    plt.gca().set_aspect('equal')
    plt.show()
    
    ## Plot the spin map where the arrow is binary and represents Mz: up <==> positive  
    _showBinaryColors = True
    
    Mz = Sz_sorted.real/np.abs(Sz_sorted.real)
    ## Plot the spin map where the arrow is binary and represents Mz: up <==> positive; additionally, we add color=> blue is up, red is down
    Mz_flattened = Mz.flatten() 
    spin_colors = np.empty((N_spatial, 4))
    spin_colors[Mz_flattened > 0] = to_rgba('blue')
    spin_colors[Mz_flattened < 0] = to_rgba('red')
    spin_colors[Mz_flattened == 0] = to_rgba('gray')
    
    plt.figure(figsize=(3.38583*2, 3.38583*2))
    plt.title('Ising spin ordering', fontsize = 20)
    for i in range(N_spatial):
      if( i + Ny < N_spatial):
        plt.plot([x[i], x[i+Ny]], [y[i], y[i+Ny]], color='k', lw = 1.0, zorder = 1)
    
      if( (i % Ny) < (Ny - 1)):
        plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color='k', lw = 1.0, zorder = 1)
    
    #plt.scatter(x, y, color = 'k', s = 2, zorder = 2)
    if(_showBinaryColors):
      plt.quiver(x, y, np.zeros_like(Mz), Mz, color = spin_colors, units = 'xy', pivot = 'middle', zorder = 2) 
    else:
      plt.quiver(x, y, np.zeros_like(Mz), Mz, units = 'xy', pivot = 'middle', zorder = 2) 
    plt.gca().set_aspect('equal')
    if(d == 1): # manually set the y window so that we can see the spins
      plt.ylim(-5, 5)
    plt.show()
  
  
    plt.quiver(x, y, Sx_sorted.real/total_norm, Sy_sorted.real/total_norm, Sz_sorted.real/total_norm, units = 'xy', cmap=sns_cmap, pivot = 'middle', zorder = 2) 
  return [x, y, Sx_sorted.real, Sy_sorted.real, Sz_sorted.real]





# Script to load and plot correlation data 
# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['system']['NSitesPer-x'] 
Ny = params['system']['NSitesPer-y'] 
Nz = params['system']['NSitesPer-z'] 
system = params['system']['ModelType'] 
d = params['system']['Dim']
_CL = params['simulation']['CLnoise']

N_spatial = Nx

if(d > 1):
  N_spatial *= Ny
  if( d > 2):
    N_spatial *= Nz
  else:
    Nz = 1
else:
  Ny = 1
  Nz = 1

num_basis_sites = 1
if(system == 'HONEYCOMB'):
  # Retrieve the spin textures for each sublattice, and then combine into a single plot 
  num_basis_sites = 2
  lattice_textures = [] # list of length sublattice basis sites (2 for honeycomb) 
  for i in range(0, num_basis_sites):
    # import the data for the i'th sublattice 
    Sx_file = 'Mx_' + str(i) + '.dat'
    Sy_file = 'My_' + str(i) + '.dat'
    Sz_file = 'Mz_' + str(i) + '.dat'

    # element of list is: [x, y, Sx_sorted.real, Sy_sorted, Sz_sorted]
    lattice_textures.append(analyze_data(Sx_file, Sy_file, Sz_file, N_spatial, d, _CL, False))

  # Calculate a total norm for the spin texture 
  # Total norm is a field N(r) where we have calculated the normalizing factor at each site (r)
  # therefore apply the norm to each sublattice 
  total_norm = [np.zeros_like(lattice_textures[0][2]), np.zeros_like(lattice_textures[0][2])] 
  for k in range(0, num_basis_sites):
    for nu in range(0, 3):
      total_norm[k] += (lattice_textures[k][2 + nu].real**2) 
    total_norm[k] = np.sqrt(total_norm[k]) 
  
  # Plot the spin texture  
  sns_cmap = ListedColormap(sns.color_palette("RdBu", 256)) 

  plt.style.use('~/CSBosonsCpp/tools/python_plot_styles_examples/plot_style_spins.txt')

  plt.figure(figsize=(3.38583*2, 3.38583*2))
  # plot sublattices 
  # Shift the b sub lattice vertically by 1/sqrt(3) for a equilateral triangular honeycomb 
  shift = np.array([0., 1./np.sqrt(3)]) 

  for k in range(0, num_basis_sites):
    x = lattice_textures[k][0]
    y = lattice_textures[k][1] + shift[k]
    Sx_sorted = lattice_textures[k][2]
    Sy_sorted = lattice_textures[k][3]
    Sz_sorted = lattice_textures[k][4]

      # Calculate nearest neighbof 

    # plot the underlying sublattice topology
    plt.quiver(x, y, Sx_sorted.real/total_norm[k], Sy_sorted.real/total_norm[k], Sz_sorted.real/total_norm[k], units = 'xy', cmap=sns_cmap, pivot = 'middle', zorder = 1) 
  # plot the nearest neighbor connectivity of the honeycomb lattice 
  x_A = lattice_textures[0][0]
  y_A = lattice_textures[0][1]
  x_B = lattice_textures[1][0]
  y_B = lattice_textures[1][1]

  # NN from adjacent sublattices (assuming 2D): (1/2 , \pm 1/2(sqrt(3)) )
  assert(d > 1), 'This code is meant for d == 2'
  for i in range(len(x)):
    neighbors_x = [x_A[i], x_A[i] + 0.5, x_A[i] - 0.5]
    neighbors_y = [y_A[i] + shift[1], y_A[i] - 0.5 * shift[1], y_A[i] - 0.5*shift[1]]

    # plot a bond for each neighbor  
    for j in range(3): # 3 NN in honeycomb lattice 
      plt.plot([x_A[i], neighbors_x[j]], [y_A[i], neighbors_y[j]], color = 'k', linewidth = 0.75)

  cbar = plt.colorbar()
  cbar.set_label(label = r'$M_{z}$', size = 30, rotation = 2)
  cbar.set_ticks([-1, -0.5, 0.0, 0.5, 1])
  cbar.set_ticklabels([-1, -0.5, 0.0, 0.5, 1])
  cbar.ax.tick_params(labelsize=20)
  if(d == 1): # manually set the y window so that we can see the spins
    plt.ylim(-4, 4)
  plt.clim(-1.0,1.0) 
  plt.gca().set_aspect('equal', adjustable='box')
  plt.show()
    
  ## Plot the spin map where the arrow is binary and represents Mz: up <==> positive  
  _showBinaryColors = True
  
  plt.figure(figsize=(3.38583*2, 3.38583*2))
  plt.title('Ising spin ordering', fontsize = 20)
  # Calculate a total norm for the spin texture 
  total_norm = [np.zeros_like(lattice_textures[0][2]), np.zeros_like(lattice_textures[0][2])] 
  for k in range(0, num_basis_sites):
    # Want only Sz or Mz (this is index 4)
    total_norm[k] += (lattice_textures[k][4].real**2) 
    total_norm[k] = np.sqrt(total_norm[k])


  for k in range(0, num_basis_sites):
    x = lattice_textures[k][0]
    y = lattice_textures[k][1] + shift[k]
    Mz = lattice_textures[k][4].real/total_norm[k]

    ## Plot the spin map where the arrow is binary and represents Mz: up <==> positive; additionally, we add color=> blue is up, red is down
    Mz_flattened = Mz.flatten() 
    spin_colors = np.empty((N_spatial, 4))
    spin_colors[Mz_flattened > 0] = to_rgba('blue')
    spin_colors[Mz_flattened < 0] = to_rgba('red')
    spin_colors[Mz_flattened == 0] = to_rgba('gray')

    if(_showBinaryColors):
      plt.quiver(x, y, np.zeros_like(Mz), Mz, color = spin_colors, units = 'xy', pivot = 'middle', zorder = 2) 
    else:
      plt.quiver(x, y, np.zeros_like(Mz), Mz, units = 'xy', pivot = 'middle', zorder = 2) 

  for i in range(len(x)):
    neighbors_x = [x_A[i], x_A[i] + 0.5, x_A[i] - 0.5]
    neighbors_y = [y_A[i] + shift[1], y_A[i] - 0.5 * shift[1], y_A[i] - 0.5*shift[1]]

    # plot a bond for each neighbor  
    for j in range(3): # 3 NN in honeycomb lattice 
      plt.plot([x_A[i], neighbors_x[j]], [y_A[i], neighbors_y[j]], color = 'k', linewidth = 0.75)


  plt.gca().set_aspect('equal')
  if(d == 1): # manually set the y window so that we can see the spins
    plt.ylim(-5, 5)
  plt.show()



else:
  Sx_file = 'Mx.dat'
  Sy_file = 'My.dat'
  Sz_file = 'Mz.dat'
  _isPlotting = True
  analyze_data(Sx_file, Sy_file, Sz_file, N_spatial, d, _CL, _isPlotting)





