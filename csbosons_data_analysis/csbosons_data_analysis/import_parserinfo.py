import numpy as np
import os
import yaml
from .time_grid import TimeGrid 

def import_parser(input_filename: str):
      if os.path.exists(input_filename):
        with open(input_filename) as infile:
          params = yaml.load(infile, Loader=yaml.FullLoader)
        return params 
      else:
        error("File not found") 


def extract_grid_details(parser, lattice: bool = True) -> tuple:
    if(lattice):
      Nx = parser['system']['NSitesPer-x']
      try:
        Ny = parser['system']['NSitesPer-y']
      except:
        Ny = 1 
      try:
        Nz = parser['system']['NSitesPer-z'] 
      except:
        Nz = 1 
    else:
      Nx = parser['simulation']['Nx'] 
      try:
        Ny = parser['simulation']['Ny'] 
      except:
        Ny = 1 
      try:
        Nz = parser['simulation']['Nz'] 
      except:
        Nz = 1 
    dimension = parser['system']['Dim'] 

    # Convention to ignore N_{\nu} of {\nu} > dim, where \nu = x, y, z. 
      # i.e. if d = 2, ignore value for Nz. 
    if(dimension < 3):
      Nz = 1
      if(dimension < 2):
        Ny = 1 

    grid_pts = [Nx, Ny, Nz]
    return grid_pts, dimension 



def extract_cell_details(parser, lattice: bool = True) -> list:
    try:
      dim = parser['system']['Dim']
    except:
      ValueError('Could not find dimension from parser.') 

    L_list = []
    if(lattice):
      L_list, dim = extract_grid_details(parser, lattice)
    else:
      Lx = parser['system']['CellLength-x'] 
      if(dim > 1):
        if(dim > 2):
          try:
            Lz = parser['system']['CellLength-z'] 
          except:
            Lz = 1.
        else:
          Lz = 1. 

        try:
          Ly = parser['system']['CellLength-y'] 
        except:
          Ly = 1.
      else:
        Ly = 1. 
        Lz = 1. 
      L_list.append(Lx)
      L_list.append(Ly)
      L_list.append(Lz)
    
    # Convention to ignore N_{\nu} of {\nu} > dim, where \nu = x, y, z. 
      # i.e. if d = 2, ignore value for Lz. 
    if(dim < 3):
      L_list[2] = 1
      if(dim < 2):
        L_list[1] = 1 

    return L_list 


def extract_time_grid_details(parser) -> TimeGrid:
  ''' - Assumes evenly spaced time grid'''
  try: 
    dimensionless = parser['system']['dimensionless']
    print('Running a dimensionless model? ' + str(dimensionless))
  except:
    print('Dimensionless keyword not found, setting to false.')
    dimensionless = False 
  
  if(dimensionless):
    tmax_key = 'tmax'
  else:
    tmax_key = 'tmax_ns'

  try:
    tmax = parser['system'][tmax_key] 
  except:
    if(tmax_key == 'tmax'):
      tmax = parser['system']['tmax_ns']
    else:
      tmax = parser['system']['tmax']
 
  Nt = parser['simulation']['nt'] 
  dt = tmax/Nt

  return TimeGrid(tmax, Nt, dt)

    


def calculate_Nspatial(grid_pts: list[int], dim: int) -> int:
    ''' Calculate the number of spatial grid points.
        Convention for lower dimensionality d < 3 is to store unused dimensionality as 1. 
         i.e. Nz = 1 for d < 3. '''
    N_spatial = 1
    N_spatial *= grid_pts[0]
    if(dim > 1):
      N_spatial *= grid_pts[1] 
      if( dim > 2):
        N_spatial *= grid_pts[2]
    return N_spatial

