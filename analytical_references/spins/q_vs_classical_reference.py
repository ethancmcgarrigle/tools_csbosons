import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# helper trig functions
def coth(x):
  return np.cosh(x)/np.sinh(x)


def csch(x):
  return 1./np.sinh(x)


'''Script to perform a comparison of non-interacting quantum vs. classical references'''



## non-interacting classical reference 
def Mz_classical(beta, hz, S):
    return S * (2. * coth(2. * beta * hz * S) - (1./(hz*beta*S))) 


def chi_zz_classical(beta, hz, S):
    tmp = -2. * beta * (S**2) * (csch(2. * beta * hz * S)**2)  
    return 2.*(tmp + 1./(2. * beta * (hz**2)))

def Cv_classical(beta, hz, S):
    # includes beta, i.e. Cv / kB = beta^2 * (<U^2> - <U>) 
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



if __name__ == "__main__":
  # Let's sweep magnetic field at a few temperatures, for several values of S  
  S = np.arange(0.5, 10.5) 
  #S = np.arange(50.5, 100.5) 
  #beta = np.array([0.1, 0.075, 0.05, 0.01]) 
  Temperature = np.arange(0.025, 30., 0.025)
  beta = 1./Temperature 
  
  # avoid hz == 0 since classical references do not have limiting procedure accounted for  
  hz = np.array([2.5]) 
  #hz = np.append(hz, np.arange(0.005, 20., 0.005))
  #hz = np.sort(hz)
  
  colors = ['b', 'r']
  markers = ['o', 'p']
  
  # loop for property vs. temperature 
  for field in hz:
    for spin in S:
      # Functions take in the whole hz vector
      Mz_class_data = Mz_classical(beta, field, spin)
      U_class_data = -Mz_class_data * field 
      chi_zz_class_data = chi_zz_classical(beta, field, spin)
      Cv_class_data = Cv_classical(beta, field, spin)
  
      # quantum functions are not vectorized 
      Mz_quantum_data = np.zeros(len(beta))
      U_quantum_data = np.zeros(len(beta))
      chi_zz_quantum_data = np.zeros(len(beta))
      Cv_quantum_data = np.zeros(len(beta))
  
      for i, inv_T in enumerate(beta):
        Mz_quantum_data[i] = Mz(inv_T, field, spin)
        U_quantum_data[i] = -Mz_quantum_data[i] * field 
        chi_zz_quantum_data[i] = inv_T * (Mz_squared(inv_T, field, spin) - Mz_quantum_data[i]**2)
        Cv_quantum_data[i] = inv_T * inv_T * (U_squared(inv_T, field, spin) - U_quantum_data[i]**2)
  
      # Plot the comparisons, one plot for each observable  
      plt.style.use('~/tools_csbosons/tools/Figs_scripts/plot_style_orderparams.txt')
      plt.figure(figsize = (5,5)) 
      plt.title('$S = ' + str(spin) + '$, $h_{z} = '  + str(np.round(field, 3)) + '$', fontsize = 20)
      plt.plot(Temperature, Mz_class_data, marker=markers[0], color = colors[0], markersize = 6, linewidth = 0.5, label = 'Classical')
      plt.plot(Temperature, Mz_quantum_data, marker=markers[1], color = colors[1], markersize = 6, linewidth = 0.5, label = 'Quantum')
      plt.xlabel('$T$', fontsize = 24)
      plt.ylabel('$M_{z}$', fontsize = 24)
      plt.legend()
      plt.show()
  
      plt.figure(figsize = (5,5)) 
      plt.title('$S = ' + str(spin) + '$, $h_{z} = '  + str(np.round(field, 3)) + '$', fontsize = 20)
      plt.plot(Temperature, U_class_data, marker=markers[0], color = colors[0], markersize = 6, linewidth = 0.5, label = 'Classical')
      plt.plot(Temperature, U_quantum_data, marker=markers[1], color = colors[1], markersize = 6, linewidth = 0.5, label = 'Quantum')
      plt.xlabel('$T$', fontsize = 24)
      plt.ylabel('$U$', fontsize = 24)
      plt.legend()
      plt.show()
  
      plt.figure(figsize = (5,5)) 
      plt.title('$S = ' + str(spin) + '$, $h_{z} = '  + str(np.round(field, 3)) + '$', fontsize = 20)
      plt.plot(Temperature, chi_zz_class_data, marker=markers[0], color = colors[0], markersize = 6, linewidth = 0.5, label = 'Classical')
      plt.plot(Temperature, chi_zz_quantum_data, marker=markers[1], color = colors[1], markersize = 6, linewidth = 0.5, label = 'Quantum')
      plt.xlabel('$T$', fontsize = 24)
      plt.ylabel('$\chi_{zz}$', fontsize = 24)
      plt.legend()
      plt.show()
  
      plt.figure(figsize = (5,5)) 
      plt.title('$S = ' + str(spin) + '$, $h_{z} = '  + str(np.round(field, 3)) + '$', fontsize = 20)
      plt.plot(Temperature, Cv_class_data, marker=markers[0], color = colors[0], markersize = 6, linewidth = 0.5, label = 'Classical')
      plt.plot(Temperature, Cv_quantum_data, marker=markers[1], color = colors[1], markersize = 6, linewidth = 0.5, label = 'Quantum')
      plt.xlabel('$T$', fontsize = 24)
      plt.ylabel('$C_{v} / k_{B}$', fontsize = 24)
      plt.legend()
      plt.show()
  
  
  beta = np.array([10.0, 5.0, 2.5, 2., 1., 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01]) 
  
  # Loop for property vs. magnetic fields 
  for spin in S:
    for inv_T in beta:
      # Functions take in the whole hz vector
      Mz_class_data = Mz_classical(inv_T, hz, spin)
      U_class_data = -Mz_class_data * hz 
      chi_zz_class_data = chi_zz_classical(inv_T, hz, spin)
      Cv_class_data = Cv_classical(inv_T, hz, spin)
  
      # quantum functions are not vectorized 
      Mz_quantum_data = np.zeros(len(hz))
      U_quantum_data = np.zeros(len(hz))
      chi_zz_quantum_data = np.zeros(len(hz))
      Cv_quantum_data = np.zeros(len(hz))
  
      for i, field in enumerate(hz):
        Mz_quantum_data[i] = Mz(inv_T, field, spin)
        U_quantum_data[i] = -Mz_quantum_data[i] * field 
        chi_zz_quantum_data[i] = inv_T * (Mz_squared(inv_T, field, spin) - Mz_quantum_data[i]**2)
        Cv_quantum_data[i] = inv_T * inv_T * (U_squared(inv_T, field, spin) - U_quantum_data[i]**2)
  
      # Plot the comparisons, one plot for each observable  
      T = np.round(1./inv_T, 4)
      plt.style.use('~/tools_csbosons/tools/Figs_scripts/plot_style_orderparams.txt')
      plt.figure(figsize = (5,5)) 
      plt.title('$S = ' + str(spin) + '$, $T = '  + str(T) + '$', fontsize = 20)
      plt.plot(hz, Mz_class_data, marker=markers[0], color = colors[0], markersize = 6, linewidth = 0.5, label = 'Classical')
      plt.plot(hz, Mz_quantum_data, marker=markers[1], color = colors[1], markersize = 6, linewidth = 0.5, label = 'Quantum')
      plt.xlabel('$h_{z}$', fontsize = 24)
      plt.ylabel('$M_{z}$', fontsize = 24)
      plt.legend()
      plt.show()
  
      plt.figure(figsize = (5,5)) 
      plt.title('$S = ' + str(spin) + '$, $T = '  + str(T) + '$', fontsize = 20)
      plt.plot(hz, U_class_data, marker=markers[0], color = colors[0], markersize = 6, linewidth = 0.5, label = 'Classical')
      plt.plot(hz, U_quantum_data, marker=markers[1], color = colors[1], markersize = 6, linewidth = 0.5, label = 'Quantum')
      plt.xlabel('$h_{z}$', fontsize = 24)
      plt.ylabel('$U$', fontsize = 24)
      plt.legend()
      plt.show()
  
      plt.figure(figsize = (5,5)) 
      plt.title('$S = ' + str(spin) + '$, $T = '  + str(T) + '$', fontsize = 20)
      plt.plot(hz, chi_zz_class_data, marker=markers[0], color = colors[0], markersize = 6, linewidth = 0.5, label = 'Classical')
      plt.plot(hz, chi_zz_quantum_data, marker=markers[1], color = colors[1], markersize = 6, linewidth = 0.5, label = 'Quantum')
      plt.xlabel('$h_{z}$', fontsize = 24)
      plt.ylabel('$\chi_{zz}$', fontsize = 24)
      plt.legend()
      plt.show()
  
      plt.figure(figsize = (5,5)) 
      plt.title('$S = ' + str(spin) + '$, $T = '  + str(T) + '$', fontsize = 20)
      plt.plot(hz, Cv_class_data, marker=markers[0], color = colors[0], markersize = 6, linewidth = 0.5, label = 'Classical')
      plt.plot(hz, Cv_quantum_data, marker=markers[1], color = colors[1], markersize = 6, linewidth = 0.5, label = 'Quantum')
      plt.xlabel('$h_{z}$', fontsize = 24)
      plt.ylabel('$C_{v} / k_{B}$', fontsize = 24)
      plt.legend()
      plt.show()
  
