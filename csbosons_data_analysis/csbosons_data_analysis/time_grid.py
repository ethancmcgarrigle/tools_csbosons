import numpy as np 


class TimeGrid :
  ''' 
    A class that represents a time grid, used with CSfield output. 
    
    CSfield outputs do not print the time grid information, they simply have extra columns for each time point.
  ''' 

  def __init__(self, tmax, Nt, dt=None):
    self._tmax = tmax
    self._Nt = Nt 

    if(dt == None):    
      self._dt = tmax / Nt
    else:
      self._dt = dt


    self.grid = np.linspace(0., tmax, Nt)

    # Construct a reciprocol w_grid 
    dw = np.pi * 2. / tmax
    w_max = dw * (Nt - 1)
    self.w_grid = np.linspace(0., w_max, Nt)


  # Standard methods to define 
  def __len__(self):
    return self._Nt

  def return_tmax(self):
    return self._tmax

  def return_dt(self):
    return self._dt

  def __getitem__(self, idx: int):
    return self.grid[idx]

  def get_nearest_index(self, t: float):
    ''' Returns index closest to desired time "t" ''' 
    return np.abs(self.grid - t).argmin()

  def get_time_range(self, t_start: int, t_end: int):
    ''' Returns time grid, inclusive [t_start, t_end] '''
    idx1 = self.get_nearest_index(t_start)
    idx2 = self.get_nearest_index(t_end)
    return self.grid[idx:idx2+1]


  def get_reciprocol_grid(self): 
    return self.w_grid 

  def return_dw(self):
    return (np.pi * 2. / self._tmax) 


