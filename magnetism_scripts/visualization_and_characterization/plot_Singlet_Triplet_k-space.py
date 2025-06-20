import numpy as np
import matplotlib
import yaml
import os 
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import plotly.graph_objects as go
import plotly.figure_factory as ff
import platform
if 'Linux' in platform.platform():
  matplotlib.use('TkAgg')
else:
  matplotlib.rcParams['text.usetex'] = True
import pdb
import pandas as pd 
from scipy.stats import sem 
from matplotlib.colors import LogNorm

# Import our custom package for Csbosons data analysis
from csbosons_data_analysis.field_analysis import *
from csbosons_data_analysis.import_parserinfo import *
from csbosons_data_analysis.error_propagation import *

# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_spins.txt') 


# TODO Generalize to 1D and 3D 
def process_kspace_data(file, N_gridpoints, dim, _Langevin):
    # Load the data 
    k_grid, fk_avg, fk_errs = process_data([file], N_gridpoints, _Langevin, False)

    kx = k_grid[0]    
    ky = k_grid[1]
    kz = k_grid[2]

    kx, ky, kz, fk_avg, fk_errs = extend_orthorhombic_grid(kx, ky, kz, fk_avg, fk_errs)

    processed_data = {'kx': kx, 'ky': ky, 'kz': kz, 'f(k)': fk_avg, 'S(k)_errs': fk_errs} 

    d_frame_fk = pd.DataFrame.from_dict(processed_data)

    d_frame_fk.sort_values(by=['kx', 'ky', 'kz'], ascending = True, inplace=True)
    if(lattice == 'square'): # process for later usage of imshow()
      # Redefine numpy array post sorting
      fk_processed = np.array(d_frame_fk['f(k)']) 
      augmentation_factor = int(np.sqrt(int(len(kx) / (Nx * Ny * Nz))))
      fk_processed.resize(Nx * augmentation_factor, Ny * augmentation_factor)
      fk_processed = np.transpose(fk_processed)
      fk_processed = np.flip(fk_processed, 0)
    else:
      fk_processed = np.array(d_frame_fk['f(k)']) 

    return [np.array(d_frame_fk['kx']), np.array(d_frame_fk['ky']), np.array(d_frame_fk['kz']), fk_processed]
 

 
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



def global_translation(k_list, Q):
    ''' Function that performs a global translation of the k grid by Q, a reciprocol lattice vector''' 
    ''' In-place modification of k_list = [kx, ky, kz] input grid '''
    # Loop through all points and translate 
    for q in range(0, len(k_list[0])):
      k_list[0][q], k_list[1][q], k_list[2][q] = np.array([k_list[0][q] + Q[0], k_list[1][q] + Q[1], k_list[2][q] + Q[2]])



def extend_orthorhombic_grid(kx, ky, kz, Sk, Sk_errs):
    ''' Function to extend the reciprocal space grid to \pm b1, b2, b3, b4. Returns the arrays having been augmented 
        Plan: 
           1. Append the (k) data with translated grid 
           2. Append the corresponding Sk data to the original Sk data set    
           3. Sort in a data frame later
    '''  
    kx_original = np.copy(kx)
    ky_original = np.copy(ky) 
    kz_original = np.copy(kz) 
    Sk_original = np.copy(Sk)
    Sk_errs_original = np.copy(Sk)

    for Q in [b1, b2, b3, b4, -b1, -b2, -b3, -b4]: 
      k_translate = [np.copy(kx_original), np.copy(ky_original), np.copy(kz_original)]
      global_translation(k_translate, Q)
      kx, ky, kz = np.append(kx, k_translate[0]) , np.append(ky, k_translate[1]), np.append(kz, k_translate[2]) 
      Sk = np.append(Sk, Sk_original)
      Sk_errs = np.append(Sk_errs, Sk_errs_original)

    return kx, ky, kz, Sk, Sk_errs



def plot_BZ1(BZ1_dict = {}):
  ''' Function to plot an outline of the first brillouin zone '''
  if(lattice == 'square'):
    # plot a 2D square for the BZ 
    pts_x = np.array([-np.pi, -np.pi, np.pi, np.pi])
    pts_y = np.array([-np.pi, np.pi, np.pi, -np.pi])
    for i, pts in enumerate(pts_x):
      ip1 = (i + 1 ) % 4 
      plt.plot( np.array([pts_x[i], pts_x[ip1]]), np.array([pts_y[i], pts_y[ip1]]), linestyle = 'solid', color = 'white', linewidth = 0.5)
  else:
    for i, pts in enumerate(BZ1_dict['K_points']):
       plt.plot( np.array([BZ1_dict['K_prime_points'][i][0], BZ1_dict['K_points'][i][0]]), np.array([BZ1_dict['K_prime_points'][i][1], BZ1_dict['K_points'][i][1] ]), linestyle = 'solid', color = 'white', linewidth = 0.5)
       ip1 = (i + 1 ) % 3 
       plt.plot( np.array([BZ1_dict['K_points'][i][0], BZ1_dict['K_prime_points'][ip1][0]]), np.array([BZ1_dict['K_points'][i][1], BZ1_dict['K_prime_points'][ip1][1] ]), linestyle = 'solid', color = 'white', linewidth = 0.5)
  


def plot_BZ_square():
  # plot a 2D square for the BZ 
  pts_x = np.array([-np.pi, -np.pi, np.pi, np.pi])
  pts_y = np.array([-np.pi, np.pi, np.pi, -np.pi])
  for i, pts in enumerate(pts_x):
    ip1 = (i + 1 ) % 4 
    plt.plot( np.array([pts_x[i], pts_x[ip1]]), np.array([pts_y[i], pts_y[ip1]]), linestyle = 'solid', color = 'white', linewidth = 0.5)



def plot_kspace_data(data_tmp, save_data, save_plot, basis_site_indx=1, basis_sites = 1):
    ''' Takes in a list Sk which contains Sk_xx, Sk_yy, Sk_zz (structure factor for each spin direction). 
        - Sk_nu,nu is a list of form [kx, ky, kz, Sk_data] 
        - e.g. Sk_list[0] is the list: [kx, ky, kz, Sk_xx] 
        - e.g. Sk_list[1] is the list: [kx, ky, kz, Sk_yy] '''
    plt.style.use(style_path)

    if(ntau > 1 or lattice == 'triangular'):
      _LogPlots = False
    else:
      _LogPlots = True

    fk = data_tmp[3]
    kx = data_tmp[0]
    ky = data_tmp[1]
    kz = data_tmp[2]

    #file_str = 'fk_' + dirs[nu] + dirs[nu] + '_' + str(basis_site_indx) 

    extension_factor = 1.5   # for 1st BZ


    plt.figure(figsize=(6.77166, 6.77166))
    if(lattice == 'square'):
      #plot_BZ_square()
      plot_BZ1()
      plt.imshow(fk.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
      BZ1_end = np.pi 
      plt.xlim(-BZ1_end * extension_factor, BZ1_end * extension_factor)
      plt.ylim(-BZ1_end * extension_factor, BZ1_end * extension_factor)
    else:
      # Need to reconstruct the 1st-BZ for visualization 
      BZ1_dict = return_BZ1_points()
      # Distance from origin to any of the hexagon's vertices  
      K_prime_distance = np.linalg.norm(b1)*0.5 / np.cos(np.pi / 6.)
      BZ1_end = K_prime_distance
      # Plot the hexagon depicting the first Brillouin Zone 
      plot_BZ1(BZ1_dict)
      # Plot the structure factor 
      triangles = tri.Triangulation(kx, ky)
      plt.tricontourf(triangles, fk.real, cmap = 'inferno', levels = 200, zorder=1) 

      # Plot the domain (kx, ky) \in [-1.25BZ, 1.25BZ]
      plt.xlim(-BZ1_end * extension_factor, BZ1_end * extension_factor)
      plt.ylim(-BZ1_end * extension_factor, BZ1_end * extension_factor)
      #plt.scatter(kx, ky, Sk.real)

 #    if(basis_sites > 1):
 #      plt.title(r'$S^{' + basis_site_labels[basis_site_indx] + '}_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)
 #    else:
 #      plt.title(r'$S_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)
    plt.xlabel('$k_{x}$', fontsize = 32)
    plt.ylabel('$k_{y}$', fontsize = 32)
 #    # plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
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
 #      if(basis_sites > 1):
 #        plt.title(r'$S^{' + basis_site_labels[basis_site_indx] + '}_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)
 #      else:
 #        plt.title(r'$S_{' + dirs[nu] + dirs[nu] + '} (\mathbf{k})$', fontsize = 30)
  
      if(lattice == 'square'):
        plt.imshow(fk.real, cmap = 'inferno', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)], norm=LogNorm()) 
        BZ1_end = np.pi 
        plt.xlim(-BZ1_end * extension_factor, BZ1_end * extension_factor)
        plt.ylim(-BZ1_end * extension_factor, BZ1_end * extension_factor)
      else:
        # Need to reconstruct the 1st-BZ for visualization 
        BZ1_dict = return_BZ1_points()
        # Plot the hexagonal depicting the first Brillouin Zone 
        plot_BZ1(BZ1_dict)
        triangles = tri.Triangulation(kx, ky)
        plt.tricontourf(triangles, fk.real, cmap = 'inferno', norm=LogNorm(), levels = 100) 
        #extend_plot(kx, ky, kz, Sk, False)
      plt.xlim(-BZ1_end * extension_factor, BZ1_end * extension_factor)
      plt.ylim(-BZ1_end * extension_factor, BZ1_end * extension_factor)
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
  # Script to load and plot correlation data 
  params = import_parser('input.yml')
  
  # Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
  lattice = True
  grid_pts, dim = extract_grid_details(params, lattice) 
  N_spatial = calculate_Nspatial(grid_pts, dim)
  
  Nx = grid_pts[0] 
  Ny = grid_pts[1] 
  Nz = grid_pts[2] 
  
  system = params['system']['ModelType'] 
  _CL = params['simulation']['CLnoise']
  
  # Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
  T = float(params['system']['beta'])
  T = 1./T
  lattice = params['system']['lattice'] 
  ntau = params['system']['ntau'] 
  _isPlotting = True
  
  dirs = {0 : 'x', 1 : 'y', 2 : 'z'}
  
  
  if(system == 'HONEYCOMB'):
    # Retrieve the spin textures for each sublattice, and then combine into a single plot 
    num_basis_sites = 2
    basis_site_labels = {0: 'A', 1: 'B'}
  else:
    num_basis_sites = 1
    basis_site_labels = {0: 'A'}

  # Define global lattice vectors:
  if(lattice == 'triangular'):
    b1 = np.array([1, -1./np.sqrt(3.), 0.]) 
    b1 *= (2. * np.pi) 
    b2 = np.array([0, 2. / np.sqrt(3.), 0.])
    b2 *= (2. * np.pi)
    b3 = b1 + b2
    b4 = b1 - b2
  else:
    b1 = np.array([1., 0., 0.])
    b1 *= (2. * np.pi) 
    b2 = np.array([0., 1., 0.])
    b2 *= (2. * np.pi)
    b3 = b1 + b2
    b4 = b1 - b2

  fileNames = ['Singlet_OP.dat', 'Triplet_OP.dat']
  
  OP_list = [] # list of length sublattice basis sites (2 for honeycomb) 

  # Main loop to process the correlation data for each sublattice and each spin
  for f in fileNames:
    OP_list.append(process_kspace_data(f, N_spatial, dim, _CL))

  for O in OP_list: 
    plot_kspace_data(O, False, False, 0, num_basis_sites) 
  

