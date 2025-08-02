#!/bin/bash

# need this wrapper to get the destinations for the output of slurm scripts right

mkdir -p $(ws_find automl)/cluster_logs/${1}/

# submit job
JOB_ID=$(sbatch --job-name=${1} job.sh)

echo $JOB_ID
