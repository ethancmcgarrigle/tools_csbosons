import numpy as np
import matplotlib
import yaml
import os 
import subprocess 
import re
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
# matplotlib.use('TkAgg')
import pdb
import pandas as pd 


# Script to load and plot correlation data 

# import the data
ops_file = 'S_k_tot.dat'
cols = np.loadtxt(ops_file, unpack=True)

k_x = cols[0]
k_y = cols[1]
n_x = cols[2]
n_y = cols[3]

S_k_real = cols[4]
S_k_imag = cols[5]


S_k = np.zeros(len(S_k_real), dtype=np.complex_)
S_k += S_k_real + 1j * S_k_imag

_data = {'kx': k_x, 'ky': k_y, 'S_k': S_k}
d_frame = pd.DataFrame.from_dict(_data)


print('Sorting the data frame into ascending order')

d_frame.sort_values(by=['kx', 'ky'], ascending = True, inplace=True) 


# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
Nx = params['simulation']['Nx'] 



# Redefine numpy array post sorting
S_k_sorted = np.array(d_frame['S_k']) 
S_k_sorted.resize(Nx, Nx)
S_k_sorted = np.transpose(S_k_sorted)

kx_list = np.unique(k_x)
ky_list = np.unique(k_y)
assert(len(kx_list) == Nx)
#print(S_k_sorted, "S(k) sorted")

S_k_yavgd = np.zeros(Nx, dtype=np.complex_) # assumes cubic wave-vector mesh 
S_k_xavgd = np.zeros(Nx, dtype=np.complex_) # assumes cubic wave-vector mesh 
# I think rows ar kx, columns are ky 
# try to compute an average over y for each x
#print(kx_list, "kx")
for j in range(0, Nx):
  S_k_yavgd[j] = np.nanmean(S_k_sorted[j][:])
  S_k_xavgd[j] = np.nanmean(S_k_sorted[:][j])

#print(S_k_yavgd, "S(k_x)")

# import the input parameters, specifically the i and j indices 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
 #r_i = params['system']['operators']['i'] 
 #r_j = params['system']['operators']['j']


# Import rho(k) and rho(-k) data
rhok_file = 'rho_k.dat'
rhonegk_file = 'rho_-k.dat'
cols = np.loadtxt(rhok_file, unpack=True)

rhok_real = cols[4]
rhok_imag = cols[5]

cols = np.loadtxt(rhok_file, unpack=True)
rhonegk_real = cols[4]
rhonegk_imag = cols[5]

# sort the data first 



rho_k = np.zeros(len(rhok_real), dtype=np.complex_)
rho_k += rhok_real + 1j * rhok_imag

rho_negk = np.zeros(len(rhok_real), dtype=np.complex_)
rho_negk += rhonegk_real + 1j * rhonegk_imag

rhok_data = {'kx': k_x, 'ky': k_y, 'rho_k': rho_k, 'rho_-k': rho_negk}
d_frame_rhok = pd.DataFrame.from_dict(rhok_data)

 #rhonegk_data = {'kx': k_x, 'ky': k_y, 
 #d_frame_rhonegk = pd.DataFrame.from_dict(rhonegk_data)


print('Sorting the data frame into ascending order')

d_frame_rhok.sort_values(by=['kx', 'ky'], ascending = True, inplace=True) 
#d_frame_rhonegk.sort_values(by=['kx', 'ky'], ascending = True, inplace=True) 


# Redefine numpy array post sorting
rhok_sorted = np.array(d_frame_rhok['rho_k']) 
rho_negk_sorted = np.array(d_frame_rhok['rho_-k']) 

rhok_sorted.resize(Nx, Nx)
rhok_sorted = np.transpose(rhok_sorted)

#rho_negk_sorted = np.array(d_frame_rhonegk['rho_-k']) 
rho_negk_sorted.resize(Nx, Nx)
rho_negk_sorted = np.transpose(rho_negk_sorted)


rho_product = np.zeros((Nx,Nx), dtype=np.complex_) 

# Consider doing this operation in 1D, i.e. before resizing to an Nx x Nx array 
rho_product += rhok_sorted
rho_product *= rho_negk_sorted



# Calculate structure factor 
S_k_final = np.zeros((Nx,Nx), dtype=np.complex_)
S_k_final += S_k_sorted 
S_k_final -= rho_product 

# Plot the structure factor 
# Average over the y-coordinate to plot only S(k_x) 
#K Plot the S(kx) distribution in k-space  
plt.figure(1)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
#plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.plot(kx_list, S_k_final[:][0].real, 'r', label = '$S(k_{x})$')
plt.title('Structure Factor: $S(k_{x})$, $k_{y} = 0$', fontsize = 16)
plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
plt.ylabel('Re($S(k_{x})$)', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
plt.xlim(-1, 1)
plt.legend()
plt.show()

plt.figure(2)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
#plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.plot(kx_list, S_k_final[:][0].imag, 'r', label = '$S(k_{x})$')
plt.title('Structure Factor: $S(k_{x})$, $k_{y} = 0$', fontsize = 16)
plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
plt.ylabel('Im($S(k_{x})$)', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
plt.xlim(-1, 1)
plt.legend()
plt.show()
 
plt.figure(3)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
#plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.plot(ky_list, S_k_final[0][:].real, 'r', label = '$S(k_{y})$')
plt.title('Structure Factor: $S(k_{y})$, $k_{x} = 0$', fontsize = 16)
plt.xlabel('$k_y$', fontsize = 20, fontweight = 'bold')
plt.ylabel('Re($S(k_{y})$)', fontsize = 20, fontweight = 'bold')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
#plt.colorbar()
plt.xlim(-1, 1)
plt.legend()
plt.show()

plt.figure(4)
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
plt.plot(ky_list, S_k_final[0][:].imag, 'r', label = '$S(k_{y})$')
plt.title('Structure Factor: $S(k_{y})$ , $k_{x} = 0$', fontsize = 16)
plt.xlabel('$k_y$', fontsize = 20, fontweight = 'bold')
plt.ylabel('Im($S(k_{y})$)', fontsize = 20, fontweight = 'bold')
plt.xlim(-1, 1)
plt.legend()
plt.show()



#K Plot the S(kx) distribution in k-space  
 #plt.figure(1)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #plt.plot(kx_list, S_k_yavgd.real, 'r*', label = '$S(k_{x})$')
 #plt.title('Structure Factor: $S(k_{x})$ (y-averaged)', fontsize = 16)
 #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Re($S(k_{x})$)', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 #
 #plt.figure(2)
 ## plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 ##plt.imshow(S_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
 #plt.plot(kx_list, S_k_yavgd.imag, 'r*', label = '$S(k_{x})$')
 #plt.title('Structure Factor: $S(k_{x})$ (y-averaged)', fontsize = 16)
 #plt.xlabel('$k_x$', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Im($S(k_{x})$)', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 #
 #
 #
 ##Plot a vertical slice: of the S(k_y) structure factor 
 #plt.figure(3)
 #plt.plot(ky_list, S_k_sorted[0][:].real, 'r*', label = '$S(k_{x} = 0, k_{y})$')
 #plt.title('Structure Factor cut: $S(k_{y})$' , fontsize = 16)
 #plt.xlabel('$k_y$', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Re($S(k_{y})$)', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 #
 ##Plot S(k_y), averaged over x  
 #plt.figure(4)
 #plt.plot(ky_list, S_k_xavgd.real, 'r*', label = '$S(k_{y})$')
 #plt.title('Structure Factor: $S(k_{y})$' , fontsize = 16)
 #plt.xlabel('$k_y$', fontsize = 20, fontweight = 'bold')
 ## plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
 #plt.ylabel('Re($S(k_{y})$)', fontsize = 20, fontweight = 'bold')
 ##plt.colorbar()
 #plt.xlim(-1, 1)
 #plt.legend()
 #plt.show()
 #
