## tools_csbosons: A repository for analyzing the output of coherent states boson C++ simulations. 
The repository is organized into 1) BEC scripts, 2) magnetism scripts, 3) analytical references, 4) examples, 5) python_plot_styles, and 6) csbosons_data_analysis.

1) BEC and 2) magnetism scripts are python scripts that are routinely used for analyzing output data from Bose fluid simulations and spin lattice simulations, respectively.
   Those scripts depend on helper functions in the package csbosons_data_analysis. 

## csbosons_data_analysis — A python package  

A collection of helper functions for analyzing field output data from a coherent states boson field theory c++ code, collected into a package that can be installed via pip:

## Installation of csbosons_data_analysis
```bash
pip install git+https://github.com/yourusername/tools_csbosons.git#subdirectory=csbosons_data_analysis
```

To upgrade the package, you can use the command 
```bash
pip install --upgrade git+https://github.com/yourusername/tools_csbosons.git#subdirectory=csbosons_data_analysis
```
