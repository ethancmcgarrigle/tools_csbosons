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


def calculate_field_average(field_data, Nx, Ny, dim, N_samples_to_avg): # assumes cubic/square mesh 
    # Calculates the average of a field given sample data, assumes .dat file imported with np.loadtxt, typically field formatting  
    # field_data is data of N_samples * len(Nx**d), for d-dimensions. Can be complex data

    # Get number of samples 
    N_samples = len(field_data)/(Nx*Ny)
    assert(N_samples.is_integer())
    N_samples = int(N_samples)

    # Use split (np) to get arrays that represent each sample (1 array per sample) Throw out the first sample (not warmed up properly) 
    sample_arrays = np.split(field_data, N_samples) 
    sample_arrays = sample_arrays[len(sample_arrays) - N_samples_to_avg:len(sample_arrays)]

    # Final array, initialized to zeros. 
    print('Averaging ' + str(int(len(sample_arrays))) + ' samples')
    averaged_data = np.zeros(len(sample_arrays[0]), dtype=np.complex_)
    averaged_data += np.mean(sample_arrays, axis=0) # axis=0 calculates element-by-element mean
    # Calculate the standard error 
    std_errs = np.zeros(len(sample_arrays[0]))
    std_errs += sem(sample_arrays, axis=0)
    return averaged_data, std_errs



def calc_err_division(x, y, x_err, y_err):
    # x/y 
    # assumes x and y are real 
    z = x/y
    # Calculate error using standard error formula 
    #result = np.sqrt( ((-x * y_err / (y**2))**2 ) + (x_err/y)**2)
    #result = z * np.sqrt( ((x_err/x)**2) + ((y_err/y)**2) ) 
    result =  z * np.sqrt( ((x_err/x)**2) + ((y_err/y)**2) ) 
    return result


def calc_err_average(vector):
   # error propagation for summing over a whole vector of numbers. The input vector is the 1D list of errors to be propagated  
   # returns the resulting error
   err = 0. + 1j*0. 
   err += (1./len(vector)) * np.sqrt( np.sum( vector**2  ) )
   return err 

# Script to load and plot correlation data 

plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style.txt')

n_k = np.loadtxt('n_k_figure_data.dat')

plt.figure(figsize=(3.38583, 3.38583))
# plt.imshow(n_k_sorted.real, cmap = plt.get_cmap('ocean'), interpolation='none', extent=[np.min(k_x) ,np.max(k_x) ,np.min(k_y),np.max(k_y)]) 
#plt.imshow(n_k_sorted.real/np.sum(n_k_sorted.real), cmap = 'hot', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
#plt.imshow(n_k_sorted.real, cmap = 'hot', interpolation='none', extent=[np.min(kx) ,np.max(kx) ,np.min(ky),np.max(ky)]) 
plt.imshow(n_k, cmap = 'hot', interpolation='none', extent = [-10., 10., -10., 10.]) 
plt.xlabel('$k_x$', fontsize = 28, fontweight = 'bold')
plt.ylabel('$k_y$', fontsize = 28, fontweight = 'bold')
plt.annotate(r'$\tilde{T} = ' + str(17.9) + '$', xy = (-2.0, 1.6), fontsize = 20, color = 'white')
# plt.zlabel('real part', fontsize = 20, fontweight = 'bold')
plt.xlim(-2*1.2,2.*1.2)
plt.ylim(-2*1.2,2.*1.2)
#plt.colorbar()
plt.savefig('Fig2a_86mm_n_k.eps', dpi = 320)
#plt.savefig('Fig5c_86mm_n_k.eps', dpi = 300)
# plt.legend()
#plt.savefig('n_k_mixed_1K.eps')
plt.show()



#plt.style.use('~/csbosonscpp/tools/Figs_scripts/plot_style_phase_diagram.txt')


data = np.loadtxt('fig2b_angularAvg_data_kappa12.dat')
kr_unique = data[:, 0]
n_kr = data[:, 1]
n_kr_errs = data[:, 2]
# Plot angular average 
#plt.figure(4)
plt.figure(figsize=(3.38583, 3.38583))
plt.errorbar(kr_unique, n_kr, n_kr_errs, marker='o', markersize = 6, elinewidth=2.00, linewidth = 0.00, color = 'black', label=r'$\langle N(k_{r}) \rangle$')
#plt.title('Polar Averaged Momentum Distribution, ' + r'$\tilde T = ' + str(np.round(T_,2)) + '$', fontsize = 22)
plt.xlabel('$k_{r}$', fontsize = 24, fontweight = 'bold')
#plt.ylabel(r'$\langle N(k_{r}) \rangle $', fontsize = 28, fontweight = 'bold')
plt.ylabel(r'Atom Number', fontsize = 24, fontweight = 'bold')
#plt.axvline(x = _kappa, color = 'r', linewidth = 1.5, linestyle='dashed', label = r'$\tilde{\kappa} = ' + str(_kappa) + '$')
plt.axvline(x = 1.2 , color = 'r', linewidth = 1.5, linestyle='dashed', label = r'$k_{r} = \tilde{\kappa} $')
#plt.savefig('n_k_plane_wave_1D.eps')
plt.xlim(0., 4.*1.2)
plt.legend()
#plt.savefig('Fig2a_86mm_inset.eps', dpi = 300)
plt.show()


# Export list_x[1:stop_indx], corr_sorted[0][1:stop_indx]
#np.savetxt('fig2b_angularAvg_data_kappa12.dat', np.column_stack( [kr_uniq, n_kr.real, n_kr_errs.real] ))

