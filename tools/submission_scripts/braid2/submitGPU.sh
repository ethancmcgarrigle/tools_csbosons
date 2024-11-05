#!/bin/bash
#SBATCH -N 1 --partition=gpu --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --job-name=__jobname__
######################################
inputfile=input.yml
outputfile=output.out
csbosonsdir=~/CSBosonsCpp/bin/Release
######################################

cd $SLURM_SUBMIT_DIR
outdir=${SLURM_SUBMIT_DIR}
rundir=${outdir}
username=`emcgarrigle`

############# TO USE LOCAL SCRATCH FOR INTERMEDIATE IO, UNCOMMENT THE FOLLOWING
#if [ ! -d /scratch_local/${username} ]; then
#  rundir=/scratch_local/${username}/${PBS_JOBID}
#  mkdir -p $rundir
#  cp ${PSB_O_WORKDIR}/* $rundir
#  cd $rundir
#fi
#####################################################

cat $SLURM_JOB_NODELIST > nodes

# Run the job
srun --gres=gpu:1 ${csbosonsdir}/CSBosonsGPU.x ${inputfile} > ${outdir}/${outputfile}
# Copy back results
if [ "$rundir" != "$outdir" ]; then
  mv * ${outdir}
fi

# Force good exit code here - e.g., for job dependency
exit 0
