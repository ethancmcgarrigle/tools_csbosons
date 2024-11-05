#!/bin/bash
#PBS -q gpuq
#PBS -l nodes=1:ppn=1
#PBS -l walltime=1000:00:00
#PBS -V
#PBS -j oe
#PBS -N __jobname__
######################################
inputfile=input.yml
outputfile=output.out
csbosonsdir=~/CSBosonsCpp
######################################

cd $PBS_O_WORKDIR
outdir=${PBS_O_WORKDIR}
rundir=${outdir}
username=`whoami`

############# TO USE LOCAL SCRATCH FOR INTERMEDIATE IO, UNCOMMENT THE FOLLOWING
#if [ ! -d /scratch_local/${username} ]; then
#  rundir=/scratch_local/${username}/${PBS_JOBID}
#  mkdir -p $rundir
#  cp ${PSB_O_WORKDIR}/* $rundir
#  cd $rundir
#fi
#####################################################

#cat $PBS_NODEFILE > nodes
GPUDEV=`cat $PBS_GPUFILE | awk '{print $1}'`
if [ -z $GPUDEV ]; then
  echo "ERROR finding $PBS_GPUFILE; using default GPU deviceid=0"
  GPUDEV=0
fi

# Run the job
${csbosonsdir}/bin/Release/CSBosonsGPU.x ${inputfile} > ${outdir}/${outputfile}
# Copy back results
if [ "$rundir" != "$outdir" ]; then
  mv * ${outdir}
fi

exit 0
