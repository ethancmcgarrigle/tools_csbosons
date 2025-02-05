
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


data = np.loadtxt('stripe_T2_figure_data.dat')


plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style.txt')

plt.figure(figsize=(3.38583, 3.38583)) # 86mm ?
#plt.imshow(density_up.real/avg_rho, cmap = 'magma', interpolation='none', extent=[np.min(x) ,np.max(x) ,np.min(y),np.max(y)]) 
plt.imshow(data, cmap = 'magma', interpolation='none') 
#plt.colorbar()
#plt.clim(0, np.max(density_up.real/avg_rho))
plt.annotate(r'$\tilde{T} = ' + str(2.0) + '$', (10, 50), fontsize = 22, color = 'white', fontweight='bold')
plt.annotate(r'$\tilde{\kappa} = ' + str(0.4) + '$', (10, 100), fontsize = 22, color = 'white', fontweight='bold')
plt.clim(0, 4.2)
plt.savefig('Fig1a_Str_T2_86mm_normalized.eps', dpi = 300)
plt.show()
