import numpy as np
import yaml


def coth(x):
  return np.cosh(x)/np.sinh(x)


def csch(x):
  return 1./np.sinh(x)





## non-interacting classical reference 
def Mz_classical(beta, hz, S):
    if(hz == 0.):
      return 0. 
    else:
      return S * (2. * coth(2. * beta * hz * S) - (1./(hz*beta*S))) 


def chi_zz_classical(beta, hz, S):
    tmp = -2. * beta * (S**2) * (csch(2. * beta * hz * S)**2)  
    return 2.*(tmp + 1./(2. * beta * (hz**2)))

def Cv_classical(beta, hz, S):
    if(hz == 0):
      return 0.
    else:
      return 1. - (((2. * beta * hz * S)**2) / (np.sinh(2. * beta * hz * S)**2))





## Non-interacting quantum reference 
def Mz(beta, hz, S):
    if(int(S) == S): # integer spin
      m_i = np.arange(2, int(np.round(2*S, 3)) + 2, 2)
    else:
      m_i = np.arange(1, int(np.round(2*S, 3)) + 2, 2) # skip every  
    coshmi = np.cosh(beta * hz * m_i)
    Z = np.sum(coshmi)# leave out the 2 
    if(int(S) == S): # integer spin
      Z += 1.

    Mz_i = np.sinh(beta * hz * m_i) * m_i
    Mz = np.sum(Mz_i)/Z
    return Mz



def U(Mz, hz):
    U = -hz * Mz 
    return U 


def Mz_squared(beta, hz, S):
    #m_i = np.arange(1, int(np.round(2*S, 3)) + 2, 2) # skip every  
    if(int(S) == S): # integer spin
      m_i = np.arange(2, int(np.round(2*S, 3)) + 2, 2)
    else:
      m_i = np.arange(1, int(np.round(2*S, 3)) + 2, 2) # skip every  
    coshmi = np.cosh(beta * hz * m_i)
    Z =  np.sum(coshmi) # leave out the 2 
    if(int(S) == S): # integer spin
      Z += 1.
    Mz_sq_i = np.cosh(beta * hz * m_i) * m_i * m_i
    Mz_sq = np.sum(Mz_sq_i)/Z
    return Mz_sq


def U_squared(beta, hz, S):
    #m_i = np.arange(1, int(np.round(2*S, 3)) + 2, 2) # skip every  
    if(int(S) == S): # integer spin
      m_i = np.arange(2, int(np.round(2*S, 3)) + 2, 2)
    else:
      m_i = np.arange(1, int(np.round(2*S, 3)) + 2, 2) # skip every  
    coshmi = np.cosh(beta * hz * m_i)
    Z =  np.sum(coshmi) # leave out the 2 
    if(int(S) == S): # integer spin
      Z += 1.
    U_sq_i = np.cosh(beta * hz * m_i) * m_i * m_i * hz * hz
    U_sq = np.sum(U_sq_i)/Z
    return U_sq
## end Non-interacting reference 




## Interacting Ising chain reference  Mz = -df / d(hz), where f equiv F/N the intensive free energy or free energy per site  
def Mz_Ising(beta, S, J, hz):
    # function for Mz (the z-magnetization)
    result = 0.
    result += beta*hz*S*((1 + S)**2)/(3.*(1. + S))
    result -= (2./9.)*beta*beta*hz*J*(S**2)*(1. + 2*S + S**2)
    return result 


def U_Ising(beta, S, J, hz):
    # function for internal energy  
    result = 0.
    result += (1./3.)*beta*S
    result *= ((-1./3.)*S*((J*(1. + S))**2) + (hz**2)*(-1. - S + beta*S*J  + 2.*beta*J*(S**2) + beta*J*(S**3) ) )
    return result 


def chi_zz_Ising(beta, S, J, hz):
    result = 0.
    result += beta*S*((1 + S)**2)/(3.*(1. + S)) 
    result -= (2./9.)*beta*beta*J*(S**2)*(1. + 2*S + S**2)
    return result 


def Cv_Ising(beta, S, J, hz):
    # function for internal energy  
    result = 0.
    result += S*((-1./9.)*(J**2)*S*((1+S)**2) + (hz**2)*(-1./3.)*(1. + S*(1. - 2.*beta*J) - (4.*beta*J)*(S**2) - 2.*beta*J*(S**3) ) )
    return result 


## Classical (large S) results for isotropic Heisenberg models 
def Cv_Heisenberg(beta, S, J, hz):
    # function for the heat capacity 
    result = 1.
    beta_eff = S * (S+1.) * beta
    J_eff = J*(S)*(S)
    #K = beta_eff * J
    K = beta * J_eff
    result -= (K)**2 / (np.sinh(K)**2) 
    return result # equates to beta^2 * [<U^2> - <U>^2] or Cv  


def Chi_zz_Heisenberg(beta, S, J, hz):
    # function for susceptibility  
    #beta_eff = S*(S+1.)*beta
    beta_eff = S*(S)*beta
    result = beta_eff*(1./3.)     # factor of 1/3 was wrong???  # * g**2 * 1/4 ?? g== 2 so this cancels out? 
    #U_fxn = u_K(J, beta_eff)
    J_eff = J*(S)*(S)
    U_fxn = u_K(J_eff, beta)
    result *= (1. - U_fxn)/(1 + U_fxn) 
    return result   # chi_zz / (beta*beta)

def u_K(J, beta_eff):
    u = 0.
    #K = J*beta_eff
    K = beta_eff * J
    u += 1./K
    u -= coth(K) ## uses mpmath function library 
    return u

def U_Heisenberg(beta, S, J, hz):
    # function for the internal energy, Fisher
    if(not hz == 0.):
      print('Warning, results are only accurate for hz = 0 ')
    result = 0.
    beta_eff = S*(S + 1.)*beta
    #J_eff = J*(S)*(S+1.)*0.75
    J_eff = J*(S)*(S)
    #J_eff = J*(S)*(S+1)*0.5
    #beta_eff = beta * (S**2)
    #result += J_eff*u_K(J_eff, beta)
    #result += u_helper(J, beta)
    result = 1. - (beta*J_eff)*coth(beta*J_eff)
    return result # intensive internal energy   


# import system parameters 
with open('input.yml') as infile:
  params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
_beta = params['system']['beta']  
_J = params['system']['Jnn']
_S = params['system']['spin']
_hz = params['system']['hz']
ndim = params['system']['Dim']
assert ndim == 1 , 'We are not in 1D. The reference expects 1D'
Nx = params['system']['NSitesPer-x']
ntau = params['system']['ntau']

beta_eff = _S * (_S + 1.) * _beta

print('effective beta for spin-S : \n')
print(beta_eff)
print()


## Calc references 
if(_J == 0):
  print('J is zero so spins are non-interacting, using the non-interacting reference: \n')
  if(ntau == 1):
    print('Using classical reference')
    Mag_z = Mz_classical(_beta, _hz, _S)
    U_ = -Mag_z * _hz 
    chi_zz = chi_zz_classical(_beta, _hz, _S)
    Cv = Cv_classical(_beta, _hz, _S)
  else:
    print('Using quantum reference')
    Mag_z = Mz(_beta, _hz, _S)
    U_ = U(Mag_z, _hz)
    chi_zz = _beta * (Mz_squared(_beta, _hz, _S) - Mag_z**2)
    Cv = _beta * _beta * (U_squared(_beta, _hz, _S) - U_**2)
else:
  print('J is non-zero so spins are interacting, using the interacting Ising reference: \n')
  Mag_z = Mz_Ising(_beta, _S, _J, _hz)
  #chi_zz = chi_zz_Ising(_beta, _S, _J, _hz)
  chi_zz = Chi_zz_Heisenberg(_beta, _S, _J, _hz)
  U_ = U_Heisenberg(_beta, _S, _J, _hz)
  Cv = Cv_Heisenberg(_beta, _S, _J, _hz) 
  #U_ = U_Ising(_beta, _S, _J, _hz)
  #Cv = Cv_Ising(_beta, _S, _J, _hz)



print('Spin S = ' + str(_S) + '\n\n')
print('z-Magnetization for spin-S = ' + str(np.round(Mag_z, 6)) + ' at field strength hz = ' + str(_hz))
print()
print('Internal energy for spin-S = ' + str(np.round(U_, 6)) + ' at field strength hz = ' + str(_hz))
print()
print('z-Susceptibility for spin-S = ' + str(np.round(chi_zz, 6)) + ' at field strength hz = ' + str(_hz))
print()
print('Heat capacity for spin-S = ' + str(np.round(Cv, 6)) + ' at field strength hz = ' + str(_hz) + '\n')

