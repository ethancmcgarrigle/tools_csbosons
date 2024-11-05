import numpy as np
import yaml


def dispersion(k, S, hz, Jnn, ndim, a, Vol):
    # returns the dispersion E(k) 
    Jk = 0.    
    for d in range(1, ndim+1):
      Jk += (1. - np.cos(k*a)) 
    Jk *= (Jnn * 2 * S)

    Ek = (2.*hz) - Jk 
    return Ek


def U_shift(S, hz, Jnn, ndim): # intensive energy shift 
    tmp = -2. * hz * S 
    tmp -= 2. * ndim * Jnn * (S**2.)/2.
    return tmp




with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
_beta = params['system']['beta']  
_Jnn = params['system']['Jnn']
_S = params['system']['spin']
_hz = params['system']['hz']
ndim = params['system']['Dim']
assert ndim == 1 , 'We are not in 1D. The reference expects 1D'
Nx = params['system']['NSitesPer-x']

_a = 1.

if(ndim == 1):
  N_spins = Nx 

Vol = N_spins

U_avg = 0.
Mz_avg = 2.*_S 


kx_grid = np.linspace(-N_spins/2 + 1, N_spins/2, N_spins) * np.pi * 2. / N_spins


# Sum over states 
for k_ in kx_grid:
  # compute Ek 
  E_k = dispersion(k_, _S, _hz, _Jnn, ndim, _a, Vol)

  # compute n_B(k) Bose distribution
  N_k = 1./(np.exp(_beta * E_k) - 1.)

  # compute internal energy 
  U_avg += E_k * N_k

  # compute z-magnetization 
  Mz_avg -= (2.*N_k/(Vol))


U_avg /= Vol
U_avg += U_shift(_S, _hz, _Jnn, ndim) 


print('Average internal energy: ' + str(np.round(U_avg, 3)) ) 
print('Average z-magnetization: ' + str(np.round(Mz_avg, 3)) ) 



