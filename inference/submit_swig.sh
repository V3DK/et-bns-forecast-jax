#!/bin/bash
#! ---------- sbatch directives: must come before any executable line ----------
#SBATCH -J swig-12p-gpu                  # job name (%x below)
#SBATCH -A ASTRONOMY-SL3-GPU       
#SBATCH -p ampere                    # or icelake (see below)
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # one Python process
#SBATCH --gres=gpu:1                 # mandatory; Slurm adds 32 CPUs automatically
#SBATCH --time=0:45:00              # hard kill at this point
#SBATCH --output=/home/%u/rds/hpc-work/et-bns-forecast-jax/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue                 # see "Requeue" below

#! ---------- environment ----------
. /etc/profile.d/modules.sh
module purge

case "$SLURM_JOB_PARTITION" in
ampere*)  module load rhel8/default-amp ;;
cclake*)  module load rhel8/default-ccl ;;
icelake*) module load rhel8/default-icl ;;
*) echo "Unexpected partition: $SLURM_JOB_PARTITION"; exit 1 ;;
esac

# conda: batch shells are non-interactive, so ~/.bashrc (where `conda init` put its hook) isn't read
source ~/.bashrc                     # CSD3's documented method
conda activate neutron-jax-gpu       # activate AFTER modules so conda's python/libs come first on PATH

export OMP_NUM_THREADS=1             # stops numpy/BLAS spawning 56 threads that compete with XLA
export PYTHONUNBUFFERED=1            # print() appears in the log immediately, not at exit
export MPLBACKEND=Agg                # no display on compute nodes

export RUNS_DIR=$HOME/rds/hpc-work/et-bns-forecast-jax/runs
export REQUIRE_GPU=1
export N_LIVE=8192                   

#! ---------- run ----------
CODE_DIR=$HOME/et-bns-forecast-jax/inference          # where swig.py and run_io.py live
cd "$CODE_DIR"
echo "Job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi
python -u swig.py