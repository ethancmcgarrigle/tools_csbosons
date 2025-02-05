## tools_csbosons: A repository for analyzing the output of coherent states boson C++ simulations. 
The repository is organized into 1) BEC scripts, 2) magnetism scripts, 3) analytical references, 4) example, 5) python_plot_styles, 6) additional tools, and 7) csbosons_data_analysis.

`BEC_scripts` and `magnetism_scripts` contain python scripts that are routinely used for analyzing output data from Bose fluid simulations and spin lattice simulations, respectively.
   Those scripts depend on helper functions in the package csbosons_data_analysis. 

`Analytical references` contains various python scripts for analytical or exact references for physical problems of interest, for helpful benchmarking. For example, there are single spin references and ideal Bose gas reference scripts. 

`python_plot_styles` contains various plot style scripts for usage with matplotlib. 

`example` contains an example usage of a python script, which depends on helper functions defined in the package below (csbosons_data_analysis). The example plots the structure factor for a triangular lattice spin model. 

`additional_tools` contains many other scripts that are helpful for data visualization, movie generation, submitting jobs on a slurm or PBS cluster management system, etc. 

Send inquiries or bug reports to ethancmcg@gmail.com 


## csbosons_data_analysis — A python package  

A collection of helper functions for analyzing field output data from a coherent states boson field theory c++ code, collected into a package that can be installed via pip:

## Installation of csbosons_data_analysis
```bash
pip3 install git+https://github.com/ethancmcgarrigle/tools_csbosons.git#subdirectory=csbosons_data_analysis
```

To upgrade the package, you can use the command 
```bash
pip3 install --upgrade git+https://github.com/ethancmcgarrigle/tools_csbosons.git#subdirectory=csbosons_data_analysis
```
