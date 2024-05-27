#!/bin/bash
#
#SBATCH --job-name=myJobarrayTest
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=1GB
#SBATCH --mail-type=END
#SBATCH --mail-user=jk7362@nyu.edu

module purge
module load python/intel/3.8.6
python slurm_main_job_v6.py $SLURM_ARRAY_TASK_ID