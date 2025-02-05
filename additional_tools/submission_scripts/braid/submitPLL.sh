#!/bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=12
#PBS -l walltime=50:00:00
#PBS -V
#PBS -j oe
#PBS -N test_
######################################
inputfile=input.yml
outputfile=output.out
csbosonsdir=~/CSBosonsCpp
######################################

cd $PBS_O_WORKDIR
outdir=${PBS_O_WORKDIR}
rundir=${outdir}
username=`whoami`

# Generate the nodes file and compute the number of
# nodes and processors per node that we have requested
cat $PBS_NODEFILE > nodes
# How many cores total do we have?
NCORES=`cat $PBS_NODEFILE | egrep -v '^#'\|'^$' | wc -l | awk '{print $1}'`
NNODES=`cat $PBS_NODEFILE | sort | uniq -c | wc -l | awk '{print $1}'`
#NTDS=`cat $PBS_NODEFILE | sort | uniq -c | head -n 1 | awk '{print $1}'`
NTDS=${PBS_NUM_PPN}
############# TO USE LOCAL SCRATCH FOR INTERMEDIATE IO, UNCOMMENT THE FOLLOWING
#if [ ! -d /scratch_local/${username} ]; then
#  rundir=/scratch_local/${username}/${PBS_JOBID}
#  mkdir -p $rundir
#  cp ${PSB_O_WORKDIR}/* $rundir
#  cd $rundir
#fi
#####################################################

# Run the job
##${csbosonsdir}/bin/Release/CSBosonsPLL.x ${inputfile} > ${outdir}/${outputfile}
# Run the job
if [ ${NNODES} -gt 1 ]; then
  mpirun -np $NNODES -machinefile nodes ${csbosonsdir}/bin/Release/CSBosonsPLL.x ${inputfile} > ${outdir}/${outputfile}
else
  ${csbosonsdir}/bin/Release/CSBosonsPLL.x ${inputfile} > ${outdir}/${outputfile}
fi

# Copy back results
if [ "$rundir" != "$outdir" ]; then
  mv * ${outdir}
fi

exit 0
