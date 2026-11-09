#!/bin/bash
#! ---------- sbatch directives: must come before any executable line ----------
#SBATCH -J swig-12p-gpu                  # job name (%x below)
#SBATCH -A ASTRONOMY-SL3-GPU       
#SBATCH -p ampere                    # or icelake (see below)
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # one Python process
#SBATCH --gres=gpu:1                 # mandatory; Slurm adds 32 CPUs automatically
#SBATCH --array=0-17
#SBATCH --time=0:30:00              # hard kill at this point
#SBATCH --output=/home/%u/rds/hpc-work/et-bns-forecast-jax/logs/%x_%A_%a.out # previously %x_%j.out
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
# export N_LIVE=1024                   

# #! ---------- run ----------
# CODE_DIR=$HOME/et-bns-forecast-jax/inference          # where swig.py and run_io.py live
# cd "$CODE_DIR"
# echo "Job $SLURM_JOB_ID on $(hostname), $(date)"
# nvidia-smi
# python -u swig.py

CODE_DIR=$HOME/et-bns-forecast-jax/inference          # where swig.py and run_io.py live

STEPS_LIST=(3 5 10); FRAC_LIST=(0.5 0.75); SEED_LIST=(2 3 4)
i=$SLURM_ARRAY_TASK_ID
export STEPS_PER_DIM=${STEPS_LIST[$(( i % 3 ))]}
export DELETE_FRAC=${FRAC_LIST[$(( (i / 3) % 2 ))]}
export SAMPLER_SEED=${SEED_LIST[$(( i / 6 ))]}
export N_LIVE=1024
export JAX_CACHE_DIR=$HOME/rds/hpc-work/jax_cache; mkdir -p "$JAX_CACHE_DIR"
[ "$i" -eq 0 ] && export PROFILE=1

LOGDIR=$HOME/rds/hpc-work/et-bns-forecast-jax/logs
nvidia-smi dmon -s u -d 1 > "$LOGDIR/gpu_util_${SLURM_ARRAY_JOB_ID}_${i}.txt" &
DMON=$!
cd "$CODE_DIR"
python -u swig.py
kill $DMON