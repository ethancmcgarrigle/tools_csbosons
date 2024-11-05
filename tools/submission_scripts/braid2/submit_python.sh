#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=1000:00:00
#SBATCH --job-name=__pythonplot__
######################################
######################################
######################################

export PATH="/home/emcgarrigle/anaconda3/bin:$PATH" 

cd $SLURM_SUBMIT_DIR
python3 plot_one_n_k.py 
