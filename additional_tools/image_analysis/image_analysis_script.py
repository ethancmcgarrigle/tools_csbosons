import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
import platform
if 'Linux' in platform.platform():
  matplotlib.use('TkAgg')
else:
  matplotlib.rcParams['text.usetex'] = True
import glob 

# Import our custom package for Csbosons data analysis
from csbosons_data_analysis.field_analysis import *
from csbosons_data_analysis.import_parserinfo import *
# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
data_style_path = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_data.txt') 
generic_style_path = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style.txt') 

import pdb
import pandas as pd 
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import cmath
from scipy.stats import sem 
from scipy.ndimage import label, center_of_mass
import scipy.fft
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import sem
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.spatial import KDTree

########## -- helper functions -- #####################
def calc_orientational_order_parameter(points, triangulation, _isPlotting, symmetry_int, lattice_constant): 
    # Function to calculate the orientational order parameter for hexagonally packed system
    # returns a fxn of spatial points; each point in the Voronoi tesselation 
    #symmetry_int = 6
    psi_M_vals = np.zeros(len(points), dtype=np.complex_) 

    if(symmetry_int == 4):
      # Create a KDTree from the transformed points
      tree = KDTree(points)

    # Iterate through the pts to find neighbors 
    for idx, point in enumerate(points):
      if(symmetry_int == 6): # from Delaunay triangles 
        # Find indices of triangles 
        indptr = np.where(triangulation.simplices == idx)[0]
        # Find all vertices connected to this point through those triangles 
        neighbors = np.unique(triangulation.simplices[indptr])
        #neighbors = vor.ridge_points[vor.points_region[idx] == vor.ridge_points].flatten()
        neighbors = neighbors[neighbors != idx]
      else: # M == 4 
        assert symmetry_int == 4, 'Abort. This assumes M = 4 square symmetry' 
 
        # Query the tree for the 5 nearest neighbors of each point (the first one is the point itself)
        distances, indices = tree.query(point, k=5)
 
        # Filter out the first column which represents the point itself
        neighbors = indices[1:5]  # Only take the four actual nearest neighbors

        #print(nearest_neighbors_indx)
        # Optionally, convert indices to actual coordinates
        #neighbors = [points[idx] for idx in nearest_neighbors_indx]
    
      # Calculate angles between the points 
      weights = []
      for neighbor_idx in neighbors:
        dx, dy = points[neighbor_idx] - point
        if(symmetry_int == 4):
          distance = np.sqrt(dx**2 + dy**2)
          if(distance < np.sqrt(2.) * lattice_constant * 0.95):
            angle = cmath.phase(complex(dx,dy))
            weights.append(np.exp(1j * symmetry_int * angle))
        else:
          angle = cmath.phase(complex(dx, dy))
          weights.append(np.exp(1j * symmetry_int * angle))
      # take mean over the neighbors sum to get psi_M at that center 
      psi_M = np.mean(weights) if len(weights) > 0 else 0
      psi_M_vals[idx] = psi_M

    # remove the ends??
    # truncate the first and last section, defined by this percentage 
    pcnt = 0.15
    start_indx = int(pcnt*len(points))
    end_indx = int((1.- pcnt)*len(points))
    psi_M_vals = psi_M_vals[start_indx:end_indx] # psi_{M}(r)
 
    if(_isPlotting):
      plot_psi6(psi_M_vals, symmetry_int)     

    return psi_M_vals, (np.abs(np.mean(psi_M_vals))**2.)  # psi(r), |1/N \sum_{r} psi(r)|^2



def calc_translational_order_parameter(points, _isPlotting, symmetry_int, qx, qy): 
    # Function to calculate the translational order parameter for hexagonally packed system
    # returns a fxn of spatial points; each point in the Voronoi tesselation 

    ''' - psi_T_l (r) = e^{-q_l dot r} for a given reciprocol lattice vector q_l 
        - points is an array with num_rows = num_points; and cols = Y and X coordinates for that row 
        - Outputs a 2D numpy array, rows correspond to the reciprocol lattice vectors 
            cols correspond to the spatial coordinate argument '''
    num_lattice_vectors = len(qx)
    assert(num_lattice_vectors == symmetry_int)
    # rows represent lattice vector; cols reperesent spatial points
    psi_T_vals = np.zeros((num_lattice_vectors, len(points)), dtype=np.complex_) 

    # For each reciprocol lattice vector set, loop over all the points  
    for i, _qx in enumerate(qx):
      for idx, point in enumerate(points):
        psi_T_vals[i, idx] = np.exp(-1j * (_qx * point[0] + qy[i] * point[1]) ) # 0 for x, 1 for y in the points array 

    # There is no need to ever plot this since it is always modulus = 1  
 #    if(_isPlotting):
 #      plot_psi_q(psi_T_vals) # plot the first one      

    return psi_T_vals


def process_sample(x_grid, y_grid, Nx, Ny, sample):
    # package into data frame 
    rho_data = {'x': x_grid, 'y': y_grid, 'rho_up': sample}
    d_frame_rho = pd.DataFrame.from_dict(rho_data)
    # Sort the data  
    d_frame_rho.sort_values(by=['x', 'y'], ascending = True, inplace=True) 
    
    # Redefine numpy array post sorting
    rho_up_sorted = np.array(d_frame_rho['rho_up']) 
    rho_up_sorted.resize(Nx, Ny)
    rho_up_sorted = np.transpose(rho_up_sorted)
    rho_up_sorted = np.flip(rho_up_sorted, 0)
    density_up = rho_up_sorted

    avg_rho = np.mean(density_up.real)
    # Find regions of high density to denote "centers"
    # 2.4 is a good number 
    num_stds = 1.85
    #num_stds = 1.25
    #num_stds = 1.35
    threshold_val = avg_rho + num_stds * np.std(density_up.real)
    binary_array = density_up.real > threshold_val
    
    # Use Scipy.ndimage label functions to label the clumps of higher density  
    labeled_array, num_features = label(binary_array)

    # Calculate the centroids of the regions 
    centroids = center_of_mass(density_up.real, labeled_array, range(1, num_features+1))
    
    # Convert the centroids to an array for the Voronoi tessellation functions 
    points = np.array(centroids)

    # Get dx and dy 
    dx = np.max(x_grid) / Nx
    dy = np.max(y_grid) / Ny

    # Points are in units of pixels or indices; to convert to real space, multiply x by dx/pixel and y by dy/pixel 
    # This image processing flips x and y for some reason?????!!!, i.e. points[:,0] contain the y coords and points[:,1], the x coords  
    points[:,0] *= dy 
    points[:,0] += dy
    points[:,1] *= dx
    # Transformed points now contains the proper ordering: transformed_points[:, 0] <==> x coordinates; transformed_points[:,1] <==> y coordinates
    transformed_points = np.flip(points, axis=1)

    # output the processed sample, the centers for that sample, and the re-arranged centers vector 
    return density_up, points, transformed_points



def plot_image(density_map, transformed_points, x, y, vor):
    plt.style.use(generic_style_path)
    avg = np.mean(density_map.real)
    plt.figure(figsize=(3.38583, 3.38583))
    axis = plt.gca()
    # Plot density heat map  
    #plt.imshow(density_up.real/avg_rho, cmap = 'magma', interpolation='none', extent=[np.min(x) ,np.max(x) ,np.min(y),np.max(y)]) 
    im = axis.imshow(density_map.real/avg, cmap = 'magma', interpolation='none', vmin=0, vmax=np.max(density_map.real/avg), extent=[np.min(x), np.max(x), np.min(y), np.max(y)]) 
    plt.plot(transformed_points[:,0], transformed_points[:,1], marker = 'o', color = 'r', linewidth = 0)
    voronoi_plot_2d(vor, ax = axis, show_vertices=False, line_colors='red')
    #plt.clim(0, np.max(density_up.real/avg_rho))
    #plt.colorbar()
    #plt.savefig('Fig1_mu_86mm.eps', dpi = 300)
    plt.show()



def calc_correlation_fxn_GT(psi_T, _isPlotting):
    # Calculate the translataional order parameter as a function of the spatial psi_6 field 

    num_coords = len(psi_T[0,:])    
    num_lattice_vectors = len(psi_T[:,0])    
    G_T = np.zeros(num_coords, dtype=np.complex_)
   
    # loop over lattice vectors
    for l in range(0, num_lattice_vectors): 
      # Calculate the correlation function
      psi_T_ql = psi_T[l, :] 
      psi_T_star = np.conj(psi_T_ql)
     
      # FFT both, one to k, one to -k 
      psi_T_k = scipy.fft.fft(psi_T_ql)
      # To get psi_T fft from r to neg k, pretend we're in k space and use the ifft function, but balance the scaling factor 
      #  Numerical FFT r to -k is very similar to Numerical iFFT operation  
      psi_T_star_negk = scipy.fft.ifft(psi_T_star) * len(psi_T_star) # scale since ifft uses scaling factor
      
      # Multiply and iffT to get G_T(r), and avg over the lattice vectors 
      G_T += scipy.fft.ifft(psi_T_k * psi_T_star_negk) / num_lattice_vectors 

    # Because of the periodic boundary conditions, only plot the first half of the function     
    if(_isPlotting):
      plot_GT(G_T)
    return G_T    


def calc_correlation_fxn_G6(psi_6, _isPlotting, M):
    # Calculate the orientational order parameter as a function of the spatial psi_6 field 
    #psi_6 = calc_orientational_order_parameter(transformed_points, tri) 
    # Calculate the correlation functions 
    psi_6_star = np.conj(psi_6)

    # FFT both, one to k, one to -k 
    psi_6_k = scipy.fft.fft(psi_6)
    # To get psi_6 fft from r to neg k, pretend we're in k space and use the ifft function, but balance the scaling factor 
    #  Numerical FFT r to -k is very similar to Numerical iFFT operation   
    psi_6_star_negk = scipy.fft.ifft(psi_6_star) * len(psi_6_star) # scale since ifft uses scaling factor
    
    # Multiply
    G_6 = psi_6_k * psi_6_star_negk
    
    # iffT to get G_6(r)
    G_6 = scipy.fft.ifft(G_6) 

    # Because of the periodic boundary conditions, only plot the first half of the function     
    if(_isPlotting):
      plot_G6(G_6, M)

    return G_6    


def plot_psi6(psi_6, M):
    plt.style.use(data_style_path)
    plt.figure(figsize=(8,8))
    #plt.plot(range(0, len(psi_6)), psi_6.real, marker='o', color = 'b')
    plt.plot(range(0, len(psi_6)), np.abs(psi_6), marker='o', color = 'b')
    plt.xlabel('$|r|$', fontsize = 20)
    plt.ylabel('$|\psi_{' + str(M) + '}|$', fontsize = 20)
    plt.show()


def power_law(x, A, eta):
    return A /(x**eta) 

def exp_decay(x, A, L):
    return A * (np.exp(-x/L)) 



def extract_decay_parameter(G):
    ''' Function to extract the decay parameter. Returns whether exponential or power law worked best'''
    q1_indx = 0 
    q1_indx = int(len(G)/6) 
    q2_indx = int(len(G)/2) - 1 
    r_coords = np.array(range(0, int(len(G))))
    r_coords = r_coords[q1_indx:q2_indx] + 1E-2
    #r_coords = r_coords[q1_indx:q2_indx] 

    fit_opts = ['power law', 'exponential']
    MSE_fits = []
    decay_params = []
    decay_params_err = []
    for fit in fit_opts:
      decay_param, decay_param_err, MSE = fit_function_decay(G[q1_indx:q2_indx].real/np.max(G.real), r_coords, fit) 
      MSE_fits.append(MSE)
      decay_params.append(decay_param)
      decay_params_err.append(decay_param_err)
       
    return decay_params[0], decay_params_err[0], fit_opts[0] 
    if(MSE_fits[0] < MSE_fits[1]):
      print('Choosing power law fit') 
      return decay_params[0], decay_params_err[0], fit_opts[0] 
    else:
      print('Choosing exponential fit') 
      return decay_params[1], decay_params_err[1], fit_opts[1] 



def fit_function_decay(F, x, fit):
    ''' Function to compute the power law fit. Returns the MSE and decay parameter'''
    X_fit = np.linspace(x[0] + 1E-3, x[-1], 1000)
    if(fit == 'power law'):
      #print('Fitting with ' + fit) # dbg 
      p0 = [0.5, -0.1] # this is an initial guess
      try: 
        props, pcov = curve_fit(power_law, x, F.real, p0, method='lm')
        decay_param = props[1]
        decay_param_err = np.sqrt(pcov[1,1])
      except:
        print('Power law fit failed')
        props = np.array([1., 1.])
        pcov = np.array([1., 1.])
        decay_param = np.nan 
        decay_param_err = np.nan 
      try:
        MSE = mean_squared_error(power_law(x, *props), F.real)
      except: 
        MSE = np.nan 
    else:
      #print('Fitting with ' + fit) #Dbg
      p0 = [0.5, 1.] # this is an initial guess
      props, pcov = curve_fit(exp_decay, x, F.real, p0, method='lm') 
      decay_param = 1./props[1] 
      decay_param_err = np.sqrt(pcov[1,1])/(decay_param**2) 
      MSE = mean_squared_error(exp_decay(x, *props), F.real)

    return decay_param, decay_param_err, MSE
    



def plot_G6(G_6, M):
    kappa = 16
    area = 204.84143616157473  
    lengthscale = 4. * np.pi / np.sqrt(3.) / kappa
    plt.style.use(data_style_path)
    plt.figure(figsize=(6,6))
    #plt.plot(range(0, int(len(G_6)/2)), G_6.real/np.max(G_6.real), marker='o', color='b')
    plt.plot(np.array(range(0, int(len(G_6)/2)))*lengthscale, G_6[0:int(len(G_6)/2)].real/np.max(G_6.real), marker='o', color='b')
    plt.xlabel('$|r|$', fontsize = 24)
    plt.ylabel('$G_{' + str(M) + '}$', fontsize = 24)
    plt.ylim(-0.05, 1.1)
    plt.show()

    plt.figure(figsize=(6,6))
    #plt.plot(range(0, int(len(G_6)/2)), G_6.real/np.max(G_6.real), marker='o', color='b')
    plt.plot(np.array(range(0, int(len(G_6)/2)))*lengthscale, G_6[0:int(len(G_6)/2)].real/np.max(G_6.real), marker='o', color='b')
    plt.xlabel('$|r|$', fontsize = 24)
    plt.ylabel('$G_{' + str(M) + '}$', fontsize = 24)
    plt.xlim(1., int(len(G_6)/2))
    plt.ylim(0.001, 10.)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    # get the large r decay exponent via a linear fit 
    # use 2nd quarter of the G6(r) data (1st quarter is too short ranged, 3rd and 4th quarters are periodic images of 1st and 2nd)
    G_6 /= np.max(G_6.real)
    q1_indx = int(len(G_6)/6) 
    q2_indx = int(len(G_6)/2) - 1 
    r_coords = np.array(range(0, int(len(G_6))))*lengthscale
    r_coords = r_coords[q1_indx:q2_indx]

    _powerLaw = True 
    X = np.linspace(r_coords[0], r_coords[-1], 1000)
    fit_opts = ['power law', 'exponential']
    plt.figure(figsize=(6,6))
    plt.plot(range(0, int(len(G_6)/2))*lengthscale, G_6[0:int(len(G_6)/2)].real, marker='o', color='b', label='data')
    plt.plot(r_coords, r_coords**(-1/4), linewidth = 1,  color='r', linestyle = 'dashed', label = '$\eta_{' + str(M) + '} = 1/4$') # avoid divison by zero 
    if(_powerLaw):
      props, pcov = curve_fit(power_law, r_coords, G_6[q1_indx:q2_indx], [1.0, 1.0], method='lm')
      Y = power_law(X, *props)
      plt.plot(X, Y, linewidth = 1,  color='k', linestyle = 'dashed', label = '$\eta_{' + str(M) + '} = ' + str(np.round(props[1], 4)) + '$ fit') 
    else:
      props, pcov = curve_fit(exp_decay, r_coords, G_6[q1_indx:q2_indx], [1.0, 1.0], method='lm')
      Y = exp_decay(X, *props)
      plt.plot(X, Y, linewidth = 1,  color='k', linestyle = 'dashed', label = 'exp decay: $L = ' + str(np.round(props[1], 3)) + '$') 
    #plt.plot(r_coords[1:], np.exp(fit[1])*r_coords[1:]**(eta_6), linewidth = 2,  color='k', label = '$\eta_{6} = ' + str(np.round(eta_6, 3)) + '$ fit') # avoid divison by zero 
    #plt.plot(r_coords, G_6[q1_indx:q2_indx].real, linewidth = 2,  color='k', label = '$\eta_{6} = ' + str(np.round(eta_6, 2)) + '$ fit') # avoid divison by zero 
    plt.xlabel('$|r|$', fontsize = 24)
    plt.ylabel('$G_{' + str(M) + '}$', fontsize = 24)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.legend()
    plt.show()



def plot_GT(G_T):
    plt.style.use(data_style_path)
    plt.figure(figsize=(6,6))
    #plt.plot(range(0, int(len(G_6)/2)), G_6.real/np.max(G_6.real), marker='o', color='b')
    plt.plot(range(0, int(len(G_T)/2)), G_T[0:int(len(G_T)/2)].real/np.max(G_T.real), marker='o', color='b')
    plt.xlabel('$|r|$', fontsize = 24)
    plt.ylabel('$G_{T}$', fontsize = 24)
    plt.show()
    
    # get the large r decay exponent via a linear fit 
    # use 2nd quarter of the G6(r) data (1st quarter is too short ranged, 3rd and 4th quarters are periodic images of 1st and 2nd)
    G_T /= np.max(G_T.real)
    q1_indx = 0 
    q2_indx = int(len(G_T)/2) - 1 
    r_coords = np.array(range(0, int(len(G_T))))
    r_coords = r_coords[q1_indx:q2_indx]

    _powerLaw = True
    X = np.linspace(r_coords[0], r_coords[-1], 1000)
    if(_powerLaw):
      p0 = [0.5, 0.1] # this is an initial guess
      #p0 = [0.5, 0.1] # this is an initial guess
      props, pcov = curve_fit(power_law, r_coords, G_T[q1_indx:q2_indx], p0, method='lm')
      Y = power_law(X, *props)
      print('MSE: ' + str(mean_squared_error(power_law(r_coords, *props), G_T[q1_indx:q2_indx].real)))
    else:
      p0 = [0.5, 10.] # this is an initial guess
      props, pcov = curve_fit(exp_decay, r_coords, G_T[q1_indx:q2_indx], p0, method='lm') 
      Y = exp_decay(X, *props)
      print('MSE: ' + str(mean_squared_error(exp_decay(r_coords, *props), G_T[q1_indx:q2_indx].real)))
    #fit, cov = np.polyfit(np.log(r_coords), np.log(G_6[q1_indx:q2_indx].real), 1, cov=True)
    #print(fit[0])
    #print(props)

    plt.figure(figsize=(6,6))
    plt.plot(range(0, int(len(G_T)/2)), G_T[0:int(len(G_T)/2)].real, marker='o', color='b', label='data')
    plt.plot(r_coords, r_coords**(-1/3), linewidth = 1,  color='r', linestyle = 'dashed', label = '$\eta_{T} = 1/3$') # avoid divison by zero 
    if(_powerLaw):
      plt.plot(X, Y, linewidth = 1,  color='k', linestyle = 'dashed', label = '$\eta_{T} = ' + str(np.round(-props[1], 5)) + '$ fit') 
    else:
      plt.plot(X, Y, linewidth = 1,  color='k', linestyle = 'dashed', label = 'exp decay: $L = ' + str(np.round(props[1], 3)) + '$') 
    #plt.plot(r_coords[1:], np.exp(fit[1])*r_coords[1:]**(eta_T), linewidth = 2,  color='k', label = '$\eta_{T} = ' + str(np.round(eta_T, 3)) + '$ fit') # avoid divison by zero 
    #plt.plot(r_coords, G_T[q1_indx:q2_indx].real, linewidth = 2,  color='k', label = '$\eta_{T} = ' + str(np.round(eta_T, 2)) + '$ fit') # avoid divison by zero 
    plt.xlabel('$|r|$', fontsize = 24)
    plt.ylabel('$G_{T}$', fontsize = 24)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.legend()
    plt.show()



def do_analysis(density_file, pcnt_avg, input_file, _plot_time_series, outdir, _isTotal):
    ########## -- Main script -- #####################
    
    # Script to load and plot correlation data 
    # import the input parameters, specifically the i and j indices 
    #assert type(symmetry_number) is int, 'Please specify an integer for the M-fold symmetry number' 
    
    # Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
    params = import_parser('input.yml')

    lattice = False # continuum model 
    realspace = True # will be handling real-space data 
    grid_pts, d = extract_grid_details(params, lattice) 
    N_spatial = calculate_Nspatial(grid_pts, d)
    system = params['system']['ModelType'] 
    d = params['system']['Dim']
    _CLNoise = params['simulation']['CLnoise']
    T = 1./float(params['system']['beta']) 
    
    Nx = grid_pts[0] 
    Ny = grid_pts[1] 
    Nz = grid_pts[2] 
    
    assert N_spatial == Nx*Ny

    _Nx = params['simulation']['Nx'] 
    _Ny = params['simulation']['Ny'] 
    assert _Nx == Nx
    assert _Ny == Ny

    dt = params['simulation']['dt'] 
    iofreq = params['simulation']['iointerval'] 
    Lx = params['system']['CellLength-x'] 
    Ly = params['system']['CellLength-y']
    assert d == 2, 'Script assumes two-dimensional images.' 
    T = 1./float(params['system']['beta']) 
    if('SOC' in system):
      kappa = params['system']['kappa']
    
    # Suppress output? 
    _suppress_output = True 

    # Lattice constant for sq system  
    a = np.pi * np.sqrt(2.) / kappa

    # Perform analysis for several symmetry numbers: 4, 6, and 8 
    #M_list = np.array([4, 6]) # TODO expand for M == 8
    M_list = np.array([6]) # TODO expand for M == 8
 
    dx = Lx / Nx
    dy = Ly / Ny
    # import the data
    #density_file = 'density1.dat'
    density_file.seek(0)
    density_filename = density_file.name
    cols_x = np.loadtxt(density_filename, unpack=True)
    if(_isTotal):
      print('Importing the density profile for the down-spin atoms for total denisty profile analysis')
      cols_dwn = np.loadtxt(outdir + '/density2.dat', unpack=True)
    
    # Extract 1 set of x and y column data 
    x = cols_x[0][0:N_spatial]
    y = cols_x[1][0:N_spatial]
    
    rho_real = cols_x[2]
    rho_imag = cols_x[3]
    
    list_x = np.unique(x)
    list_y = np.unique(y)
    
    N_samples = int(np.round((len(rho_real)/(N_spatial)), 3))
    
    print('Total number of samples: ' + str(int(N_samples)))
    
    rho_vector = np.zeros(len(cols_x[2]), dtype=np.complex_)
    rho_vector += rho_real + 1j*rho_imag
    if(_isTotal):
      rho_vector += cols_dwn[2] + 1j*cols_dwn[3]

    density_samples = np.zeros(len(cols_x[2]), dtype=np.complex_) 
    density_samples += rho_vector

    for symmetry_number in M_list:
      # Calculate reciprocol lattice vectors 
      #theta = np.array(range(0,symmetry_number)) * 2. * np.pi / symmetry_number
      # optional phase shift
      if(symmetry_number == 4):
        phase_shift = np.pi/4.
      else:
        phase_shift = 0. 
      theta = np.array(range(0,symmetry_number)) * 2. * np.pi / symmetry_number
      theta += phase_shift
  
      # 1st-order bragg peaks will be given by a circle with radius kappa with M equally spaced points  
      if(_isTotal):
        qx = kappa * np.cos(theta)
        qy = kappa * np.sin(theta)
      else:  # component density varies like wavevector k = 2kappa
        if(symmetry_number == 4):
          factor = np.sqrt(2.) 
        else:
          factor = 1.
        qx = kappa * factor *np.cos(theta)
        qy = kappa * factor *np.sin(theta)
      if(not _suppress_output):
        print('Reciprocol lattice vectors using for translational order analysis:')
        print(qx)
        print(qy)
      
      #if(N_samples == 1 and not _CLNoise):
      if(N_samples == 1): 
        # Format the sample 
        processed_sample, points, transformed_points = process_sample(x, y, Nx, Ny, rho_vector) # outputs density field now in 2D array 
      
        # Perform Voronoi tesselation and delaunay triangulation 
        #vor = Voronoi(points)
        assert len(transformed_points) > 3, 'System size is too small!'
        vor = Voronoi(transformed_points)
        
        # Perform Ddaulany triangulation, needed to get nearest-neighbors 
        tri = Delaunay(transformed_points)
      
        # plot the image 
        plot_image(processed_sample, transformed_points, x, y, vor)
      
        # Calculate the orientational order parameter 
        psi_6, psi_6_spatial_avg = calc_orientational_order_parameter(transformed_points, tri, True, symmetry_number, a) 
  
        # Calculate the translational order parameter 
        psi_T = calc_translational_order_parameter(transformed_points, True, symmetry_number, qx, qy) 
        #psi_T = calc_translational_order_parameter(points, True, symmetry_number, qx, qy) 
        # Print an average translational order parameter
        psi_T_overall = 0. 
        for i, q in enumerate(qx):
          psi_T_overall += np.abs(np.mean(psi_T[i, :]) )/len(qx) # divide by length for a mean across the diff'nt major reciprocol vectors   
  
        #if(not _suppress_output):
        #print(psi_6)
        print('Overall orientational order parameter: (M = ' + str(symmetry_number) + ') :' + str(psi_6_spatial_avg)) 
        print('Overall translational order parameter: (M = ' + str(symmetry_number) + '): ' + str(psi_T_overall)) 
   
        # Calculate the correlation fxn 
        G_M = calc_correlation_fxn_G6(psi_6, True, symmetry_number)
        np.savetxt('G' + str(symmetry_number) + '.dat', G_M.real)
        if(np.max(G_M) == 0.):
          print('No correlations, skipping G_M correlation analysis')
        else:
          eta_M, eta_M_err, fit_str = extract_decay_parameter(G_M)
          print('Orientational order decay exponent: (M = ' + str(symmetry_number) + '): ' + str(np.round(eta_M, 3)) + ' +/- ' + str(np.round(eta_M_err, 3) ) )
  
        # Calculate the correlation fxn 
        G_T = calc_correlation_fxn_GT(psi_T, False)
        if(np.max(G_T) == 0.):
          print('No correlations, skipping G_M correlation analysis')
        else:
          eta_T, eta_T_err, fit_str = extract_decay_parameter(G_T)
          print('Positional order decay exponent (M = ' + str(symmetry_number) + '): ' + str(np.round(eta_T, 3)) + ' +/- ' + str(np.round(eta_T_err, 3) ) )
      
      else: # Langevin simulation #
        # Split the samples to calculate the orientational OP and correlation fxn for each sample  
        # Get the first sample we would like to use. We are cutting off all samples prior 
        sample_num = int((1. - pcnt_avg)*N_samples) # exclude this amount  
        #print('Considering samples ' + int(sample_num) + ' through ' + str(N_samples)) 
        print('Starting at sample # ' + str(sample_num) + '')
        print('Using ' + str(N_samples - sample_num) + ' samples')
        rho_vector = np.split(density_samples, N_samples)
        psi_6_samples = []
        G_6_samples = []
        eta_M_samples = []
        eta_M_err_samples = []
        psi_T_samples = []
        G_T_samples = []
        eta_T_samples = []
        eta_T_err_samples = []
        # Loop through all of the CL samples 
        for image in rho_vector[sample_num:]:
          # Format the sample 
          processed_sample, points, transformed_points = process_sample(x, y, Nx, Ny, image) # outputs density field now in 2D array 
        
          # Perform Voronoi tesselation and delaunay triangulation 
          #vor = Voronoi(points)
          assert len(transformed_points) > 3, 'System size is too small!'
          vor = Voronoi(transformed_points)
          
          # Perform Ddaulany triangulation, needed to get nearest-neighbors 
          tri = Delaunay(transformed_points)
        
          # plot the image 
          _plot = True 
          if(_plot):
            plot_image(processed_sample, transformed_points, x, y, vor)
       
          # Calculate the order parameter, suppress plotting if we are working with many images  
          psi_6, psi_6_spatial_avg = calc_orientational_order_parameter(transformed_points, tri, _plot, symmetry_number, a)
          #psi_6, psi_6_spatial_avg = calc_orientational_order_parameter(points, tri, _plot, symmetry_number, a)
      
          # Calculate the translational order parameter 
          psi_T = calc_translational_order_parameter(transformed_points, True, symmetry_number, qx, qy) 
          psi_T_overall = 0. 
          for i, q in enumerate(qx):
            psi_T_overall += np.abs(np.mean(psi_T[i, :]) )/len(qx) # divide by length for a mean across the diff'nt major reciprocol vectors   
  
          if(not _suppress_output):
            print('Overall orientational order parameter: ' + str(psi_6_spatial_avg)) 
            print('Overall translational order parameter: ' + str(psi_T_overall)) 
          #psi_T = calc_translational_order_parameter(points, True, symmetry_number, qx, qy) 
          psi_6_samples.append(psi_6_spatial_avg)
          psi_T_samples.append(psi_T_overall)
          # Calculate the correlation fxn, suppress plotting if we are working with many images  
          #if(symmetry_number == specified_symmetry_number):
          G_6 = calc_correlation_fxn_G6(psi_6, _plot, symmetry_number)
          np.savetxt('G' + str(symmetry_number) + '.dat', G_6.real)
          if(np.max(G_6) == 0.):
            print('No correlations, skipping G_M correlation analysis')
            eta_M = np.nan
            eta_M_err = np.nan
          else:
            eta_M, eta_M_err, fit_str = extract_decay_parameter(G_6)
          G_6_samples.append(G_6)

          G_T = calc_correlation_fxn_GT(psi_T, _plot)
          if(np.max(G_T) == 0.):
            print('No correlations, skipping G_M correlation analysis')
            eta_T = np.nan
            eta_T_err = np.nan
          else:
            eta_T, eta_T_err, fit_str = extract_decay_parameter(G_T)
          G_T_samples.append(G_T)
          # Compute the decay coefficient 
          eta_M_samples.append(eta_M)
          eta_T_samples.append(eta_T)
          eta_M_err_samples.append(eta_M_err)
          eta_T_err_samples.append(eta_T_err)

        #psi_6_r_avg = np.mean(psi_6_samples[sample_num:])  # check axis 
 #        spatial_means = []
 #        for psi in psi_6_samples[sample_num:] :
 #          spatial_means.append(np.mean(psi))
        #spatial_means = np.array(spatial_means)
        spatial_means = np.array(psi_6_samples)
        psi_6_avg = np.mean(spatial_means)
        psi_6_avg_err = sem(spatial_means)

        eta_M_samples = np.array(eta_M_samples)
        eta_M_err_samples = np.array(eta_M_err_samples)
        eta_M_mean = np.nanmean(eta_M_samples) 
        eta_M_err =  calc_err_average(eta_M_err_samples) 
        print('CL-time-averaged, Spatially averaged orientational order parameter' + str(np.round(psi_6_avg, 3)) + ' +/- ' + str(np.round(psi_6_avg_err, 3) ))
        print('CL-time-averaged, orientational order decay exponent ' + str(np.round(eta_M_mean, 3)) + ' +/- ' + str(np.round(eta_M_err, 3) ) )
        fname = 'psi_' + str(symmetry_number) + '_avg'
        fname_err = 'psi_' + str(symmetry_number) + '_err'
        _header = fname + '_real ' + fname_err + '_real ' + fname + '_imag ' + fname_err + '_imag ' + 'eta_M eta_M_err'  
        # Save orientational OP data to a file 
        np.savetxt(outdir + fname + '.dat', np.column_stack([psi_6_avg.real, psi_6_avg_err.real, psi_6_avg.imag, psi_6_avg_err.imag, eta_M_mean.real, eta_M_err.real]), 
                                   header=_header)
        # Save positional OP data to a file 
        psi_T_samples = np.array(psi_T_samples) 
        psi_T_mean = np.mean(psi_T_samples)
        psi_T_err = sem(psi_T_samples)
        eta_T_samples = np.array(eta_T_samples)
        eta_T_err_samples = np.array(eta_T_err_samples)
        eta_T_mean = np.nanmean(eta_T_samples) 
        eta_T_err =  calc_err_average(eta_T_err_samples) 
        print('CL-time-averaged, Spatially averaged positional order parameter' + str(np.round(psi_T_mean, 3)) + ' +/- ' + str(np.round(psi_T_err, 3) ))
        print('CL-time-averaged, positional order decay exponent ' + str(np.round(eta_T_mean, 3)) + ' +/- ' + str(np.round(eta_T_err, 3) ) )
        fname = 'psi_T_' + str(symmetry_number) + '_avg'
        fname_err = 'psi_T_' + str(symmetry_number) + '_err'
        _header = fname + '_real ' + fname_err + '_real eta_T eta_T_err' 

        np.savetxt(outdir + fname + '.dat', np.column_stack([psi_T_mean, psi_T_err, eta_T_mean.real, eta_T_err.real]), 
                                   header=_header)

        #G6_r_avg = np.mean(G_6_samples[sample_num:])  # check axis 
      
        # plot the averaged results 
        #plot_psi6(psi_6_r_avg)     
        #plot_G6(G6_r_avg)
        
        # plot the averaged results
        ylabel = r'$\bar{\psi_{' + str(symmetry_number) + '}}$' 
        plt.style.use('~/CSBosonsCpp/tools/python_plot_styles_examples/plot_style_data.txt')
        if(_plot_time_series):
          plt.figure(figsize=(8,8))
          plt.plot(np.linspace(0, len(spatial_means) * dt * iofreq, len(spatial_means)), spatial_means.real, marker = 'o', color = 'k')
          plt.plot(np.linspace(0, len(spatial_means) * dt * iofreq, len(spatial_means)), spatial_means.imag, marker = 'o', color = 'm')
          plt.xlabel('CL time', fontsize = 20)
          plt.ylabel(ylabel, fontsize = 20)
          #plt.ylabel(r'$\bar{\psi_{6}}$', fontsize = 20)
          plt.show()



if __name__ == "__main__":
    # For command-line runs, build the relevant parser
    import argparse as ap
    parser = ap.ArgumentParser(description='Statistical analysis of density csbosonscpp data for orientational order')
    parser.add_argument('-f','--file',default='./density1.dat',type=ap.FileType('r'),help='Filename containing field statistical data.')
    parser.add_argument('-in','--file_input',default='./input.yml',type=ap.FileType('r'),help='Filename containing input parameters .')
    #parser.add_argument('-S','--specified_symmetry_number',default=6,type=int,help='M-fold symmetry')
    parser.add_argument('-w', '--pcnt',default=0.5,type=float,help='Number of samples to eliminate from the beginning of the data.')
    parser.add_argument('-plot', '--isPlotting',action='store_true', default=False, help='plot the time series data (for noise runs only).')
    parser.add_argument('-outdir', '--outdir_str',default='./',help='directory where the file is stored')
    parser.add_argument('-total', '--isTotal',action='store_true', default=False, help='do the analysis on the total density profile')
    args=parser.parse_args()
    do_analysis(args.file, args.pcnt, args.file_input, args.isPlotting, args.outdir_str, args.isTotal)
    #do_analysis(args.file, args.pcnt, args.file_input, args.isPlotting, args.outdir_str, args.isTotal)

    # sample command: 
    # python3 calc_orientational_OP.py -f density1.dat -in input.yml -S 6 -w 0.8 -plot False

