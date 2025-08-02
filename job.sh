#!/bin/bash
##SBATCH --partition=gpu_4
#SBATCH --partition=single
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (or processes) per node
##SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:1                # Number of GPUS
#SBATCH --time=08:00:00             # Wall time limit (hh:mm:ss)
<<<<<<< HEAD
#SBATCH --mem-per-cpu=4GB           # Memory per CPU core
=======
##SBATCH --mem-per-cpu=4GB           # Memory per CPU core
#SBATCH --mem-per-cpu=5GB
>>>>>>> origin/zainab
#SBATCH --error=/pfs/work7/workspace/scratch/fr_zs53-automl/cluster_logs/%x/%j.err
#SBATCH --output=/pfs/work7/workspace/scratch/fr_zs53-automl/cluster_logs/%x/%j.out
# ^^ adapt the error and output of slurm such that it points to your own workspace

source ${HOME}/.bashrc
conda activate automl

# extract the dataset from the ws to the tmpdir at the node
# compare https://wiki.bwhpc.de/e/BwUniCluster2.0/Hardware_and_Architecture#.24TMPDIR
tar -C ${TMPDIR}/ -xzf $(ws_find automl)/data.tgz

# copy the metadata from the workspace to the tmpdir
cp -r $(ws_find automl)/metadata ${TMPDIR}/metadata

#set up all the args
kwargs="--experiment-name=$SLURM_JOB_NAME --n-workers=$SLURM_CPUS_PER_TASK --device="cpu" --working-dir=$(ws_find automl) \
--datasetpath=${TMPDIR}/data --metadata-file=${TMPDIR}/metadata/deepweedsx_balanced-epochs-trimmed.csv" 
#--warmstart-smbo"

#kwargs="--experiment-name=$SLURM_JOB_NAME --n-workers=$SLURM_CPUS_PER_TASK --device="cuda" --working-dir=$(ws_find automl) \
#--datasetpath=${TMPDIR}/data --metadata-file=${TMPDIR}/metadata/deepweedsx_balanced-epochs-trimmed.csv" 

python3 -m warmstart_template $kwargs # --experiment-name=$SLURM_JOB_NAME --n-workers=$SLURM_CPUS_PER_TASK --device="cuda" # --warmstart-smbo
