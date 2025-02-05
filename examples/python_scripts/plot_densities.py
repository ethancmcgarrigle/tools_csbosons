 #!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

def main():
  xs, ys, rho_real, rho_imag = np.loadtxt('./density_up_FINAL.dat', comments='#', delimiter='')
  print(xs)
  # args = np.zeros(1, len(xs))
  # for x, y, args in zip(xs,ys,args):
  #  arg_tmp = np.arctan(y/x)
  #  if (arg_tmp < 0):
  #    arg_tmp = arg_tmp + 2*3.14159165
  #plt.imshow(rx, ry, arg)


if __name__ == '__main__':
  main()

