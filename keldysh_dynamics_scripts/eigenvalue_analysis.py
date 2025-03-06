import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, csr_matrix
import os 
import platform
if 'Linux' in platform.platform():
  matplotlib.use('TkAgg')
else:
  matplotlib.rcParams['text.usetex'] = True
# Import our custom package for plot styles  
# Get plot styles from custom package 
from csbosons_data_analysis import __file__ as package_file
style_path_data = os.path.join(os.path.dirname(package_file), 'plot_styles', 'plot_style_data.txt') 

def construct_matrix(N, N_t, x, delta1, delta2):
    """
    Construct an N×N matrix with the specified structure.
    
    Parameters:
    -----------
    N : int
        Size of the matrix (must be even)
    N_t : int
        Number of entries in sections 1 and 2 (must satisfy 2*N_t <= N)
    x : float
        Scaling parameter for subdiagonal entries
    delta1 : float
        Parameter for sections 1 and 2
    delta2 : float
        Parameter for section 3
    
    Returns:
    --------
    A : numpy.ndarray
        The constructed N×N matrix
    """
    # Verify input conditions
    if N % 2 != 0:
        raise ValueError("N must be even")
    if 2*N_t > N:
        raise ValueError("2*N_t must be less than or equal to N")
    
    # Calculate M (entries in section 3)
    M = N - 2*N_t
    
    # Initialize matrix with zeros
    A = np.zeros((N, N), dtype=complex)
    
    # Set ones on the main diagonal
    np.fill_diagonal(A, 1)
    
    # Section 1: -1 + i*delta1*x
    for i in range(N_t):
        A[i+1, i] = -1 + 1j * delta1 * x
    
    # Section 2: -1 - i*delta1*x
    for i in range(N_t, 2*N_t):
        A[i+1, i] = -1 - 1j * delta1 * x
    
    # Section 3: -1 + delta2*x
    for i in range(2*N_t, N-1):
        A[i+1, i] = -1 + delta2 * x
    
    # Wrap-around element (periodic boundary condition)
    A[0, N-1] = -1 + delta2 * x
    
    return A

def plot_eigenvalues(N, N_t, x, delta1, delta2, title=None):
    """
    Compute and plot the eigenvalues of the matrix.
    
    Parameters:
    -----------
    N : int
        Size of the matrix
    N_t : int
        Number of entries in sections 1 and 2
    x : float
        Scaling parameter for subdiagonal entries
    delta1 : float
        Parameter for sections 1 and 2
    delta2 : float
        Parameter for section 3
    title : str, optional
        Title for the plot
    
    Returns:
    --------
    eigenvalues : numpy.ndarray
        Array of eigenvalues
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Construct the matrix
    A = construct_matrix(N, N_t, x, delta1, delta2)
    
    # Compute eigenvalues
    eigenvalues = linalg.eigvals(A)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(eigenvalues.real, eigenvalues.imag, c='red', marker='o', s=100)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_aspect('equal')
    
    if title is None:
        title = f'Eigenvalues for N={N}, N_t={N_t}, x={x}, delta1={delta1:.5f}, delta2={delta2:.5f}'
    ax.set_title(title)
    
    plt.tight_layout()
    
    return eigenvalues, fig

def analyze_eigenvalues_for_parameters(N, N_t, x, delta1_values, delta2_values):
    """
    Analyze eigenvalues for multiple parameter values with sequential plotting.
    
    Parameters:
    -----------
    N : int
        Size of the matrix
    N_t : int
        Number of entries in sections 1 and 2
    x : float
        Scaling parameter for subdiagonal entries
    delta1_values : list of float
        Values of delta1 to analyze
    delta2_values : list of float
        Values of delta2 to analyze
    
    Returns:
    --------
    None
    """
    # Create separate figures for each parameter combination
    for delta1 in delta1_values:
        for delta2 in delta2_values:
            # Create a new figure for each parameter combination
            #fig, ax = plt.subplots(figsize=(8, 8))
            
            # Construct the matrix
            A = construct_matrix(N, N_t, x, delta1, delta2)
            
            # Compute eigenvalues
            eigenvalues = linalg.eigvals(A)
            print('Delta1 = ' + str(delta1) + ', min(eig.real): ' + str(min(eigenvalues.real)) + ' ' )
            #eigval = eigs(A, k=1, which='SR', return_eigenvectors=False)

def find_crossover_delta1(N, N_t, x, dtau):
    delta1 = np.linspace(0.0001, 0.05, 100) 
    min_eigs = np.zeros(len(delta1))
    # Create separate figures for each parameter combination
    for i, d in enumerate(delta1):
      # Create a new figure for each parameter combination
      #fig, ax = plt.subplots(figsize=(8, 8))
      #print('Delta1: ' + str(d))
      
      # Construct the matrix
      A = construct_matrix(N, N_t, x, d, dtau)
            
      # Compute eigenvalues
      eigenvalues = linalg.eigvals(A)
      min_eigs[i] = min(eigenvalues.real)
      #print('Delta1 = ' + str(delta1) + ', min(eig.real): ' + str(min(eigenvalues.real)) + ' ' )
    # find and print first instance where the minimum eigenvalues go negative 
    if any(x < 0 for x in min_eigs) :
      indx_crossover = np.where(min_eigs < 0.)[0][0]
    else:
      indx_crossover = 0
    tmax_possible = N_t*delta1[indx_crossover]
    print('dt value where crossover occurs: ' + str(delta1[indx_crossover]) )
    print('Corresponding t_max = : ' + str(tmax_possible))
    print()
    return tmax_possible





def print_matrix(A):
    """
    Print a matrix with formatting for complex numbers.
    
    Parameters:
    -----------
    A : numpy.ndarray
        The matrix to print
    
    Returns:
    --------
    None
    """
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                if A[i, j].imag == 0:
                    print(f"{A[i, j].real:6.2f}", end="  ")
                else:
                    print(f"{A[i, j]:6.2f}", end="  ")
            else:
                print(f"{0:6.2f}", end="  ")
        print()

# Example usage
if __name__ == "__main__":
    # Parameters
    M = 20
    N_t = 100   # Number of entries in sections 1 and 2 (must satisfy 2*N_t <= N)
    N = 2*N_t + M

    print('Imaginary time slices: ' + str(M))
    print('Real time points N: ' + str(N_t))

    # Energetics of the model and the space grid  
    beta = 0.1
    includeMu = True
    trap_energy_max = 50.0
    E_K = 3.1582734083e+00
    mu = -1.0
    if(includeMu):
      E_0 = E_K - mu + trap_energy_max 
    else:
      E_0 = E_K + trap_energy_max 

    dtau = beta/M
    delta2_values = [dtau]
    best_tmax = find_crossover_delta1(N, N_t, E_0, delta2_values[0])
   

    sweep_gridsizes = True 
    saveFigure = True 

    if(sweep_gridsizes):
      # Generate a plot for this grid, computing the necessary tmax as a function of N increasing 
      plt.style.use(style_path_data)
      #N_list = np.array([6, 8, 12, 16, 20, 24, 32, 40, 56, 64, 80, 100, 120, 156, 180, 220, 264, 350, 450, 600, 800, 1200]) 
      N_list = np.array([8, 12, 16, 20, 24, 32, 40, 56, 64]) 
      tmax_N_list = np.zeros(len(N_list))
      for i, _N in enumerate(N_list):
        tmax_N_list[i] = find_crossover_delta1(2*_N + M, _N, E_0, dtau)
  
      plt.figure(figsize = (4,4))
      plt.title('Linear stability', fontsize = 20)
      plt.plot(N_list, tmax_N_list, color = 'k', linestyle = 'solid', linewidth = 1.5, marker = 'o', label = 'best possible')
      plt.xlabel('$N$',fontsize=24) 
      plt.ylabel('$t_{max}$', fontsize = 24, rotation = 0, labelpad=15) 
      plt.legend()
      if(saveFigure):
        plt.savefig('tmax_vs_N.pdf', dpi=300)
      plt.show()

      plt.figure(figsize = (4,4))
      plt.title('Linear stability', fontsize = 20)
      plt.plot(tmax_N_list, N_list, color = 'k', linestyle = 'solid', linewidth = 1.5, marker = 'o', label = 'Required gridpoints')
      plt.ylabel('$N$',fontsize=24, rotation = 0, labelpad=15)
      plt.xlabel('$t_{max}$', fontsize = 24) 
      plt.legend()
      if(saveFigure):
        plt.savefig('N_vs_tmax.pdf', dpi=300)
      plt.show()

      np.savetxt('Nreq_vs_tmax.dat', np.column_stack([tmax_N_list, N_list]), header='tmax N_required')
