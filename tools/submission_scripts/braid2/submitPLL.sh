#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=12   ### number of threads per task (OMP threads)
#SBATCH --time=1000:00:00
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

#cat $SLURM_JOB_NODELIST > nodes

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # redundant

# Run the job
srun ${csbosonsdir}/CSBosonsPLL.x ${inputfile} > ${outdir}/${outputfile}
# Copy back results
if [ "$rundir" != "$outdir" ]; then
  mv * ${outdir}
fi

# Force good exit code here - e.g., for job dependency
exit 0
