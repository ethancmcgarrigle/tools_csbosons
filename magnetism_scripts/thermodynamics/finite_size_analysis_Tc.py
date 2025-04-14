import csv
from mpmath import *
import subprocess
import os
import re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb
import yaml
import math
import pandas as pd
import glob 
# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_data.txt') 

# Create data frame containing observables means (and errors) for each method 
with open('input.yml') as infile:
  master_params = yaml.load(infile, Loader=yaml.FullLoader)

J1 = master_params['system']['Jnn']
J2 = master_params['system']['Jnnn']

ratio = np.round(J2/J1,3)
sizes = np.array([8, 12, 20])
ndim = 2
Nspins = sizes**ndim


if(J2/J1 > 0.5):
  stripePhase = True
else:
  stripePhase = False


# Import data from the file, plot and do finite size analysis to extract Tc  
plt.style.use(style_path)

markers = ['o', '*', 'p']
colors = ['r', 'b', 'k']

plt.figure(figsize=(6,6))
for s, size in enumerate(sizes): 
  dir_name = str(size) + 'x' + str(size)

  prefix = dir_name + '/' + str(Nspins[s]) + '_J2J1_*'
  matches = glob.glob(prefix)
  #fname = prefix + str(ratio) + '.dat'
  fname = matches[0] 
  d = np.loadtxt(fname, unpack=True)
  T = d[0]
  S_pi_pi = d[1]
  S_pi_0 = d[2]
  if(stripePhase):
    Y = S_pi_0
  else:
    Y = S_pi_pi
  plt.errorbar(T, Y, np.zeros_like(Y), marker=markers[s], color = colors[s], markersize = 6, elinewidth=0.5, linewidth = 1.0, label = Nspins[s]) 
plt.xlabel(r'$T$',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$S(\pi, \pi)$', fontsize = 20, fontweight = 'bold')
plt.legend()
plt.show()


p = 0.62
#p = 0.31
nu = 0.707
power = p/nu
power = p/nu
plt.figure(figsize=(6,6))
for s, size in enumerate(sizes): 
  dir_name = str(size) + 'x' + str(size)

  prefix = dir_name + '/' + str(Nspins[s]) + '_J2J1_*'
  matches = glob.glob(prefix)
  #fname = prefix + str(ratio) + '.dat'
  fname = matches[0] 
  d = np.loadtxt(fname, unpack=True)
  T = d[0]
  S_pi_pi = d[1]
  S_pi_0 = d[2]
  scale = size**(power)
  if(stripePhase):
    Y = S_pi_0
  else:
    Y = S_pi_pi
  plt.errorbar(T, Y*scale, np.zeros_like(Y), marker=markers[s], color = colors[s], markersize = 6, elinewidth=0.5, linewidth = 1.0, label = '$L = ' + str(size) +'$') 
plt.xlabel(r'$T$',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$L^{p} S(\pi, \pi)$', fontsize = 20, fontweight = 'bold')
plt.legend()
plt.show()


Tc = 2.675
plt.figure(figsize=(6,6))
for s, size in enumerate(sizes): 
  dir_name = str(size) + 'x' + str(size)

  prefix = dir_name + '/' + str(Nspins[s]) + '_J2J1_*'
  matches = glob.glob(prefix)
  #fname = prefix + str(ratio) + '.dat'
  fname = matches[0] 
  d = np.loadtxt(fname, unpack=True)
  T = d[0]
  S_pi_pi = d[1]
  S_pi_0 = d[2]
  if(stripePhase):
    Y = S_pi_0
  else:
    Y = S_pi_pi
  scale = size**(power)
  plt.errorbar((T-Tc)*(size**(1./nu)), Y*scale, np.zeros_like(Y), marker=markers[s], color = colors[s], markersize = 6, elinewidth=0.5, linewidth = 1.0, label = '$L = ' + str(size) +'$') 
plt.xlabel(r'$T$',fontsize = 20, fontweight = 'bold')
plt.ylabel(r'$L^{p} S(\pi, 0)$', fontsize = 20, fontweight = 'bold')
plt.legend()
plt.show()
