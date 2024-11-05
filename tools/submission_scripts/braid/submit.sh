#!/bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=1
#PBS -l walltime=900:00:00
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

cat $PBS_NODEFILE > nodes

# Run the job
${csbosonsdir}/bin/Release/CSBosons.x ${inputfile} > ${outdir}/${outputfile}
# Copy back results
if [ "$rundir" != "$outdir" ]; then
  mv * ${outdir}
fi

exit 0
