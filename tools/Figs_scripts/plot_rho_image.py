
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


data = np.loadtxt('emu_T300nK_trapped_data.dat')

zoom = 36
#upper_left_corner = (24,24)
#lower_right = (300, 300)
upper_left_corner = (zoom,zoom)
lower_right = (324 - zoom, 324 - zoom)
cropped_data = data[upper_left_corner[0]:lower_right[0] + 1, upper_left_corner[1]:lower_right[1] + 1]

plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style.txt')

plt.figure(figsize=(3.38583*2, 3.38583*2)) # 86mm ?
#plt.imshow(density_up.real/avg_rho, cmap = 'magma', interpolation='none', extent=[np.min(x) ,np.max(x) ,np.min(y),np.max(y)]) 
plt.imshow(cropped_data, cmap = 'magma', interpolation='none') 
#plt.colorbar()
cbar = plt.colorbar()
#cbar.set_label(label = r'$M_{z}$', size = 38)
ticks = [0, 4, 8, 12]
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticks)
cbar.ax.tick_params(labelsize=32)
#cbar.ax.tick_params(labelsize=20)
#plt.clim(-1.0,1.0) # decent 
plt.clim(0, np.max(cropped_data))
plt.annotate(r'$T = ' + str(300) + '$ nK', (10, 25), fontsize = 32, color = 'white')
plt.savefig('Fig3a_rho.eps', dpi = 300)
plt.show()
