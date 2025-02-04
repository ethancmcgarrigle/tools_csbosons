import numpy as np 
from scipy.stats import sem 

''' Helper functions for error propagation for handling stochastic data '''
def calc_err_multiplication(x: float, y: float, x_err: float, y_err: float) -> float:
    ''' Error propagation for z = x*y, given x and y and their errors.'''
    # z = x * y
    z = x*y
    result = z * np.sqrt( ((x_err/x)**2)  + ((y_err/y)**2) ) 
    return result


def calc_err_addition(x_err: float, y_err: float) -> float:
    ''' Error propagation for z = x + y, given x and y and their errors.'''
    # assumes x and y are real 
    # Calculate error using standard error formula 
    result = np.sqrt( (x_err**2) + (y_err**2) )
    return result


def calc_err_average(vector: np.ndarray) -> float:
   ''' Error propagation for summing over a whole vector of numbers. The input vector is the 1D list of errors to be propagated ''' 
   # returns the resulting error
   err = 0. + 1j*0. 
   err += (1./len(vector)) * np.sqrt( np.sum( vector**2  ) )
   return err


def calc_err_division(x: float, y: float, x_err: float, y_err: float) -> float:
    ''' Error propagation for z = x / y, given x and y and their errors.'''
    # assumes x and y are real 
    # Calculate error using standard error formula 
    result = np.sqrt( ((-x * y_err / (y**2))**2 ) + (x_err/y)**2)
    return result
