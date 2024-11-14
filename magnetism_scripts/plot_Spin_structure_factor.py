import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import plotly.graph_objects as go
import plotly.figure_factory as ff
#matplotlib.rcParams['text.usetex'] = True
matplotlib.use('TkAgg')
import pdb
import pandas as pd 
from scipy.stats import sem 
from matplotlib.colors import LogNorm


def calculate_field_average(field_data, N_spatial, averaging_pcnt): 
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data

    # Get number of samples
    N_samples = len(field_data)/(N_spatial)

    assert(N_samples.is_integer())
    N_samples = int(N_samples)

    # Use split (np) to get arrays that represent each sample (1 array per sample) Throw out the first sample (not warmed up properly) 
    sample_arrays = np.split(field_data, N_samples)

    N_samples_to_avg = int(averaging_pcnt * N_samples)
    sample_arrays = sample_arrays[len(sample_arrays) - N_samples_to_avg:len(sample_arrays)]
    print('Averaging over ' + str(N_samples_to_avg) + ' samples') 

    # Final array, initialized to zeros. 
    averaged_data = np.zeros(len(sample_arrays[0]), dtype=np.complex_)
    averaged_data += np.mean(sample_arrays, axis=0) # axis=0 calculates element-by-element mean
    # Calculate the standard error 
    std_errs = np.zeros(len(sample_arrays[0]))
    std_errs += sem(sample_arrays, axis=0)
    return averaged_data, std_errs



def calc_err_multiplication(x, y, x_err, y_err):
    # z = x * y
    z = x*y
    result = z * np.sqrt( ((x_err/x)**2)  + ((y_err/y)**2) ) 
    return result



def calc_err_addition(x_err, y_err):
    # Error propagation function for x + y 
    #result = 0.
    # assumes x and y are real 

    # Calculate error using standard error formula 
    result = np.sqrt( (x_err**2) + (y_err**2) )
    return result


def calc_err_average(vector):
   # error propagation for summing over a whole vector of numbers. The input vector is the 1D list of errors to be propagated  
   # returns the resulting error
   err = 0. + 1j*0. 
   err += (1./len(vector)) * np.sqrt( np.sum( vector**2  ) )
   return err 



def process_data(spin_file, N_gridpoints, dim, _Langevin):
    # Load the data 
    print('Processing data in file ' + spin_file) 
    Sk_raw_data = np.loadtxt(spin_file, unpack=True)
    Sk_data = Sk_raw_data[2*(dim)] + 1j*Sk_raw_data[2*(dim) + 1]

    if(_Langevin):
      # Average the data 
      pcnt_averaging = 0.60
      Sk_avg, Sk_errs = calculate_field_average(Sk_data, N_gridpoints, pcnt_averaging)
    else:
      Sk_avg = Sk_data
      Sk_errs = np.zeros_like(Sk_avg)

    #rho_k_avg_0, rho_k_err_0 = calculate_field_average(rho_k_data_0, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
    #rho_negk_avg_0, rho_negk_err_0 = calculate_field_average(rho_negk_data_0, Nx, Ny, Nz, dim, int(len(Sk_data) * pcnt_averaging))
    #Structure_factor -= (rho_k_avg_0 * rho_negk_avg_0)

    # 1. calc error multiplication for rho(k) and rho(-k)
    # 2. calc error addition for 1) and then <rho(k) rho(-k)> 
    #S_k_errs += calc_err_multiplication(rho_k_avg_0, rho_negk_avg_0, rho_k_err_0,  rho_negk_err_0) 
    #S_k_errs = calc_err_addition(S_k_errs, corr_err) 
    #S_k_errs = corr_err 

    # Extract the reciprocal (k) grid 
    kx = Sk_raw_data[0][0:N_gridpoints]
    if(dim > 1):
      ky = Sk_raw_data[1][0:N_gridpoints]
      if(dim > 2):
        kz = Sk_raw_data[2][0:N_gridpoints]
      else:
        kz = np.zeros_like(ky)
    else:
      ky = np.zeros_like(kx)
      kz = np.zeros_like(kx)

    processed_data = {'kx': kx, 'ky': ky, 'kz': kz, 'S(k)': Sk_avg, 'S(k)_errs': Sk_errs} 

    d_frame_Sk = pd.DataFrame.from_dict(processed_data)

    if(lattice == 'square'): # process for later usage of imshow()
      d_frame_Sk.sort_values(by=['kx', 'ky'], ascending = True, inplace=True)
      # Redefine numpy array post sorting
      Sk_processed = np.array(d_frame_Sk['S(k)']) 
      Sk_processed.resize(Nx, Ny)
      Sk_processed = np.transpose(Sk_processed)
      Sk_processed = np.flip(Sk_processed, 0)
    else:
      Sk_processed = np.array(d_frame_Sk['S(k)']) 


    return [np.array(d_frame_Sk['kx']), np.array(d_frame_Sk['ky']), np.array(d_frame_Sk['kz']), Sk_processed]
  
def Rotation_matrix(in_vec, theta):
    ''' Return the vector rotated by theta [degrees]'''    
    # Convert to radians 
    theta *= (np.pi / 180.)
    R = np.array([[np.cos(theta) , -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.matmul(R, in_vec) 



def return_BZ1_points():
    ''' function to reconstruct the first brillouin zone as a regular hexagon from the kx, ky data '''
    ''' Outputs the Gamma, K, K', and right M point ''' 
    ''' Assumes 30-degree triangular lattice Bravais lattice '''
    # Need to replace each kx, ky by equivalent (kx, ky) in 1st BZ 
    # Reciprocal lattice vectors are defined globally 
    assert(Rotation_matrix([0,1], 90)[0] == -1) # Check validity 
    assert(Rotation_matrix([0, 0.5241] , 360)[1] == 0.5241)

    # List of high symmetry points in the 1st-BZ: 
    high_symmetry_pts = {'Gamma' : [0., 0.] } # will add the others 

    # Make a list of all the equivalent K prime points  
    K_primes = []
    K_prime_distance = np.linalg.norm(b1)*0.5 / np.cos(np.pi / 6.)
    K_prime_vector = np.array([K_prime_distance, 0.]) 
    K_primes.append(K_prime_vector)
    K_primes.append(Rotation_matrix(K_primes[0], 120))
    K_primes.append(Rotation_matrix(K_primes[0], 240))

    K_vector = np.array([K_prime_distance * np.cos(np.pi / 3.) , K_prime_distance * np.sin(np.pi / 3.)]) 
    #K_vector = Rotation_matrix(K_vector, 30) # global rotation
    K_points = []
    K_points.append(K_vector)
    K_points.append(Rotation_matrix(K_vector, 120))
    K_points.append(Rotation_matrix(K_vector, 240))
    
    M_points = []
    M_points.append(b3/2.)
    M_points.append(b3/2. - b3)

    high_symmetry_pts['K_points'] = K_points 
    high_symmetry_pts['K_prime_points'] = K_primes 
    high_symmetry_pts['M_points'] = M_points 

    return high_symmetry_pts

 

def global_rotation(kx, ky, theta):
    ''' Function that performs a global rotaton of the k grid by theta degrees ''' 
    # Loop through all points and rotate 
    for q in range(0, len(kx)):
      kx[q], ky[q] = Rotation_matrix(np.array([kx[q], ky[q]]), theta) 



def global_translation(kx, ky, Q):
    ''' Function that performs a global translation of the k grid by Q, a reciprocol lattice vector''' 
    # Loop through all points and translate 
    for q in range(0, len(kx)):
      kx[q], ky[q] = np.array([kx[q] + Q[0], ky[q] + Q[1]])



def extend_plot(kx, ky, Sk, linear_scale = True):
    ''' In-place function that is entended to use in a figure environment. 
        - Extends the plot by performing various reciprocol lattice vector translations  
        - Extends the plot by performing various 120-degree rotations, for each reciprocal lattice vector
        - Takes in a bool (default to True) to show intensity values on a linear scale (False for log scale)
        - ASSUMES TRIANGULAR LATTICE (120-degree symmetry) ''' 
       
 #    for i in range(0, 2):
 #      #global_rotation(kx, ky, 120)
 #      triangles = tri.Triangulation(kx, ky)
 #      if(linear_scale):
 #        plt.tricontourf(triangles, Sk.real, cmap = 'inferno') 
 #      else:
 #        plt.tricontourf(triangles, Sk.real, cmap = 'inferno', norm=LogNorm()) 
    
    #global_rotation(kx, ky, 120) # to return back to original grid 
  
    # 2. Plot translations (and rotations for each translation) 
    for Q in [b1, b2, b3]:
      global_translation(kx, ky, Q)
      triangles = tri.Triangulation(kx, ky)
      if(linear_scale):
        plt.tricontourf(triangles, Sk.real, cmap = 'inferno', levels = 100) 
      else:
        plt.tricontourf(triangles, Sk.real, cmap = 'inferno', norm=LogNorm(), levels = 100) 

 #      for i in range(0, 2):
 #        #global_rotation(kx, ky, 120)
 #        triangles = tri.Triangulation(kx, ky)
 #        if(linear_scale):
 #          plt.tricontourf(triangles, Sk.real, cmap = 'inferno') 
 #        else:
 #          plt.tricontourf(triangles, Sk.real, cmap = 'inferno', norm=LogNorm()) 
    
      #global_rotation(kx, ky, 120) # to return back to original grid 
  
      global_translation(kx, ky, -2*Q) # return grid back to original 
  
      triangles = tri.Triangulation(kx, ky)
      if(linear_scale):
        plt.tricontourf(triangles, Sk.real, cmap = 'inferno', levels = 100) 
      else:
        plt.tricontourf(triangles, Sk.real, cmap = 'inferno', norm=LogNorm(), levels = 100) 
 #      for i in range(0, 2):
 #        #global_rotation(kx, ky, 120)
 #        triangles = tri.Triangulation(kx, ky)
 #        if(linear_scale):
 #          plt.tricontourf(triangles, Sk.real, cmap = 'inferno') 
 #        else:
 #          plt.tricontourf(triangles, Sk.real, cmap = 'inferno', norm=LogNorm()) 
      #global_rotation(kx, ky, 120) # to return back to original grid 
  
      global_translation(kx, ky, Q) # return grid back to original 



def plot_BZ1(BZ1_dict):
  ''' Function to plot an outline of the first brillouin zone '''
  for i, pts in enumerate(BZ1_dict['K_points']):
     plt.plot( np.array([BZ1_dict['K_prime_points'][i][0], BZ1_dict['K_points'][i][0]]), np.array([BZ1_dict['K_prime_points'][i][1], BZ1_dict['K_points'][i][1] ]), linestyle = 'solid', color = 'red', linewidth = 0.5)
     ip1 = (i + 1 ) % 3 
     plt.plot( np.array([BZ1_dict['K_points'][i][0], BZ1_dict['K_prime_points'][ip1][0]]), np.array([BZ1_dict['K_points'][i][1], BZ1_dict['K_prime_points'][ip1][1] ]), linestyle = 'solid', color = 'red', linewidth = 0.5)





def plot_structure_factor(Sk_alpha_tmp, save_data, save_plot, basis_site_indx=1, basis_sites = 1):
    ''' Takes in a list Sk which contains Sk_xx, Sk_yy, Sk_zz (structure factor for each spin direction). 
        - Sk_nu,nu is a list of form [kx, ky, kz, Sk_data] 
        - e.g. Sk_list[0] is the list: [kx, ky, kz, Sk_xx] 
        - e.g. Sk_list[1] is the list: [kx, ky, kz, Sk_yy] '''
    plt.style.use('~/tools_csbosons/python_plot_styles/plot_style_spins.txt')

    if(ntau > 1):
      _LogPlots = False
    else:
      _LogPlots = True

    for nu in range(0, 3):
      Sk = Sk_alpha_tmp[nu][3]
      kx = Sk_alpha_tmp[nu][0]
      ky = Sk_alpha_tmp[nu][1]
      kz = Sk_alpha_tmp[nu][2]

      file_str = 'Sk_' + dirs[nu] + dirs[nu] + '_' + str(basis_site_indx) 

      plt.figure(figsize=(6.77166, 6.77166))
      if(lattice == 'square'):
        plt.imshow(Sk.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
      else:
        # Need to reconstruct the 1st-BZ for visualization 
        BZ1_dict = return_BZ1_points()
        # Distance from origin to any of the hexagon's vertices  
        K_prime_distance = np.linalg.norm(b1)*0.5 / np.cos(np.pi / 6.)
        BZ1_end = K_prime_distance
        extension_factor = 1.25
        # Plot the hexagon depicting the first Brillouin Zone 
        plot_BZ1(BZ1_dict)
        # Need to populate the entire first BZ. The easiest way to do this is to perform various grid shifts (rotataions, translations) and overlay them    
        triangles = tri.Triangulation(kx, ky)
        plt.tricontourf(triangles, Sk.real, cmap = 'inferno', levels = 100) 
        extend_plot(kx, ky, Sk, True)
        # Plot the domain (kx, ky) \in [-1.25BZ, 1.25BZ]
        #plt.xlim(-BZ1_end * extension_factor, BZ1_end * extension_factor)
        #plt.ylim(-BZ1_end * extension_factor, BZ1_end * extension_factor)
        #plt.scatter(kx, ky, Sk.real)

      if(basis_sites > 1):
        plt.title(r'$S^{' + basis_site_labels[basis_site_indx] + '}_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)
      else:
        plt.title(r'$S_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)
      plt.xlabel('$k_{x}$', fontsize = 32)
      plt.ylabel('$k_{y}$', fontsize = 32)
 #      # plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
      plt.colorbar(fraction=0.046, pad=0.04)
      if(save_plot):
        plt.savefig(file_str + '.eps', dpi=300)
        plt.savefig(file_str + '.pdf', dpi=300)
      plt.show()

      if(save_data): 
        np.savetxt(file_str + '_figure.dat', Sk.real)
      
      # -------- Plot on a log scale------- 
      if(_LogPlots):
        plt.figure(figsize=(6.77166, 6.77166))
        if(basis_sites > 1):
          plt.title(r'$S^{' + basis_site_labels[basis_site_indx] + '}_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)
        else:
          plt.title(r'$S_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)
  
        if(lattice == 'square'):
          plt.imshow(Sk.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)], norm=LogNorm()) 
        else:
          # Need to reconstruct the 1st-BZ for visualization 
          BZ1_dict = return_BZ1_points()
          # Plot the hexagonal depicting the first Brillouin Zone 
          plot_BZ1(BZ1_dict)
          triangles = tri.Triangulation(kx, ky)
          plt.tricontourf(triangles, Sk.real, cmap = 'inferno', norm=LogNorm(), levels = 100) 
          extend_plot(kx, ky, Sk, False)
        plt.xlabel('$k_{x}$', fontsize = 32)
        plt.ylabel('$k_{y}$', fontsize = 32)
        # plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
        plt.colorbar(fraction=0.046, pad=0.04)
        if(save_plot):
          plt.savefig(file_str + '_log.eps', dpi=300)
          plt.savefig(file_str + '_log.pdf', dpi=300)
        plt.show()





if __name__ == "__main__":
  ''' Script to load and visualize spin-spin correlation data''' 
  
  # import the input parameters, specifically the i and j indices 
  with open('input.yml') as infile:
    params = yaml.load(infile, Loader=yaml.FullLoader)
  
  # Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
  Nx = params['system']['NSitesPer-x']
  Ny = params['system']['NSitesPer-y']
  Nz = params['system']['NSitesPer-z']
  T = float(params['system']['beta'])
  T = 1./T
  dim = params['system']['Dim']
  system = params['system']['ModelType'] 
  lattice = params['system']['lattice'] 
  ntau = params['system']['ntau'] 
  _CL = params['simulation']['CLnoise']
  
  N_spatial = Nx
  _isPlotting = True
  
  if(dim > 1):
    N_spatial *= Ny
    if( dim > 2):
      N_spatial *= Nz
    else:
      Nz = 1
  else:
    Ny = 1
    Nz = 1
  
  dirs = {0 : 'x', 1 : 'y', 2 : 'z'}
  
  
  if(system == 'HONEYCOMB'):
    # Retrieve the spin textures for each sublattice, and then combine into a single plot 
    num_basis_sites = 2
    basis_site_labels = {0: 'A', 1: 'B'}
  else:
    num_basis_sites = 1
    basis_site_labels = {0: 'A'}
  
  Sk_list = [] # list of length sublattice basis sites (2 for honeycomb) 
  # Main loop to process the correlation data for each sublattice and each spin
  for K in range(0, num_basis_sites):
    Sk_alpha = []
    # loop over each spin direction 
    for nu in range(0, 3):
      S_file = 'S' + str(dirs[nu]) + '_k_S' + str(dirs[nu]) + '_-k_' + str(K) + '.dat' 
      Sk_alpha.append(process_data(S_file, N_spatial, dim, _CL))
    Sk_list.append(Sk_alpha)

  
  # Define global lattice vectors:
  if(lattice == 'triangular'):
    b1 = np.array([1, -1./np.sqrt(3.)]) 
    b1 *= (2. * np.pi) 
    b2 = np.array([0, 2. / np.sqrt(3.)])
    b2 *= (2. * np.pi)
    b3 = b1 + b2
  else:
    b1 = np.array([1., 0])
    b1 *= (2. * np.pi) 
    b2 = np.array([0, 1.])
    b2 *= (2. * np.pi)
    b3 = b1 + b2

  
  for K in range(0, num_basis_sites):
    plot_structure_factor(Sk_list[K], False, False, K, num_basis_sites) 
  

# Calculate angular average 
#kr = np.sqrt(kx**2 + ky**2)
#theta = np.arctan(ky/kx) # rads 
#
#kr_uniq = np.unique(kr)
#
#S_kr = np.zeros(len(kr_uniq), dtype=np.complex_)
#S_kr_errs = np.zeros(len(kr_uniq), dtype=np.complex_)
#
#_polar_data = {'kr': kr, 'theta': theta, 'S_k': Structure_factor, 'S_k_errs': S_k_errs}
#polar_d_frame = pd.DataFrame.from_dict(_polar_data)
#polar_d_frame.sort_values(by=['kr'], ascending = True, inplace=True) 
#
#
#S_kr[0] += polar_d_frame['S_k'].iloc[0]
#S_kr_errs[0] += polar_d_frame['S_k_errs'].iloc[0]
#i = 0
#print(kr[0])
#for kr_ in kr_uniq[1:len(kr_uniq)]:
#  i += 1
#  tmp_frame = (polar_d_frame['kr'] == kr_)
#  indices = np.where(tmp_frame == True)[0] 
#  #indices = indices[0] # 0th element is the list of true indices 
#  assert(polar_d_frame['kr'].iloc[indices[0]] == kr_)
#  # 2. Extract 
#  S_kr[i] += polar_d_frame['S_k'].iloc[indices].mean()
#  # propagate error across the average 
#  S_kr_errs[i] += calc_err_average(polar_d_frame['S_k_errs'].iloc[indices].values) 
#
#
#
## Plot angular average 
#plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_orderparams.txt')
#plt.figure(3)
#plt.errorbar(kr_uniq, S_kr.real, S_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='CL')
#plt.title('Angular Averaged S(k), ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 22)
#plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
#plt.ylabel(r'$S(k_{r}) $', fontsize = 24, fontweight = 'bold')
#plt.axvline(x = 2*_kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2 \tilde{\kappa} = ' + str(2*_kappa) + '$')
#plt.legend()
#plt.savefig('S_k_angular_avg.eps')
#plt.show()
# 
##
## Export list_x[1:stop_indx], corr_sorted[0][1:stop_indx]
#np.savetxt('S_k_00_angularAvg_data.dat', np.column_stack( [kr_uniq, S_kr.real, S_kr_errs.real] ))
# #
# #plt.figure(4)
# #plt.errorbar(kr_uniq, np.log10(S_kr.real), S_kr_errs.real, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label='CL')
# #plt.title('Angular Averaged S(k), ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 22)
# #plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
# #plt.ylabel(r'log($S(k_{r}) $)', fontsize = 24, fontweight = 'bold')
# #plt.axvline(x = 2*_kappa, color = 'r', linewidth = 2.0, linestyle='dashed', label = r'$2 \tilde{\kappa} = ' + str(_kappa) + '$')
# #plt.legend()
# #plt.savefig('S_k_angular_avg_log.eps')
# #plt.show()
