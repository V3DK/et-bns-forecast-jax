#!/bin/bash
#! ---------- sbatch directives: must come before any executable line ----------
#SBATCH -J swig-12p-gpu                  # job name (%x below)
#SBATCH -A ASTRONOMY-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # one Python process
#SBATCH --gres=gpu:1                 # mandatory; Slurm adds 32 CPUs automatically
#SBATCH --array=0-12                 # 0-2: heterodyne check; 3-12: n_live test (see TASKS below)
#SBATCH --time=1:00:00               # n_live 8192 took ~310 s sampling + a few min setup at 62 bins; 500-bin cost untested
#SBATCH --output=/home/%u/rds/hpc-work/et-bns-forecast-jax/logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

#! ---------- environment ----------
# pixi env is self-contained (python, jaxlib cuda12, cudnn, nccl from conda-forge), so no CSD3 modules or
# conda. Loading rhel8/default-amp would put the system CUDA on LD_LIBRARY_PATH and can shadow pixi's libs.
. /etc/profile.d/modules.sh
module purge

export PATH=$HOME/.pixi/bin:$PATH    # batch shells don't read ~/.bashrc; don't source it either (it may activate conda base)
REPO_DIR=$HOME/et-bns-forecast-jax   # holds pixi.toml / pixi.lock / .pixi
PIXI_ENV=neutron-gpu
# Install the env ONCE on a login node before submitting (compute nodes may have no internet, and 13 tasks
# installing into the same .pixi at once would race). Login nodes have no GPU, so fake the CUDA driver:
#   cd ~/et-bns-forecast-jax && CONDA_OVERRIDE_CUDA=12.9 pixi install --frozen -e neutron-gpu
# On the compute node pixi detects the real driver (__cuda), so no override here.
# --frozen: use pixi.lock exactly, never re-solve.
PIXI_RUN=(pixi run --frozen --manifest-path "$REPO_DIR/pixi.toml" -e "$PIXI_ENV")
[ -d "$REPO_DIR/.pixi/envs/$PIXI_ENV" ] || { echo "pixi env $PIXI_ENV not installed; see comment above"; exit 1; }

export OMP_NUM_THREADS=1             # stops numpy/BLAS spawning 56 threads that compete with XLA
export PYTHONUNBUFFERED=1            # print() appears in the log immediately, not at exit
export MPLBACKEND=Agg                # no display on compute nodes

export RUNS_DIR=$HOME/rds/hpc-work/et-bns-forecast-jax/runs
export REQUIRE_GPU=1
export JAX_CACHE_DIR=$HOME/rds/hpc-work/jax_cache; mkdir -p "$JAX_CACHE_DIR"

CODE_DIR=$REPO_DIR/inference          # where swig.py, run_io.py and heterodyne.py live
[ -f "$CODE_DIR/heterodyne.py" ] || { echo "missing $CODE_DIR/heterodyne.py"; exit 1; }

#! ---------- tasks (results doc, swig-gpu-experiment-results.md) ----------
# All at STEPS_PER_DIM=5, DELETE_FRAC=0.5, N_BINS=500 (heterodyne fix).
#   0-2   heterodyne check (sec. 2, to-do 1): n_live 1024, seeds 2-4. Compare with grid array 40316928
#         (same config, 62 bins): check passes at CHECK_TOL=0.1? single_ms / per_trip_ms cost of 500 bins?
#   3-12  n_live test (sec. 1b) rerun on the corrected likelihood (sec. 2, to-do 4): n_live 4096 and 8192,
#         seeds 2-6. Compare with array 40400355 (62 bins).
TASKS=(
  "1024 2" "1024 3" "1024 4"
  "4096 2" "4096 3" "4096 4" "4096 5" "4096 6"
  "8192 2" "8192 3" "8192 4" "8192 5" "8192 6"
)
i=$SLURM_ARRAY_TASK_ID
read -r N_LIVE SAMPLER_SEED <<< "${TASKS[$i]}"
export N_LIVE SAMPLER_SEED
export STEPS_PER_DIM=5
export DELETE_FRAC=0.5
export N_BINS=500

# previous grid (array 40316928), for reference:
# STEPS_LIST=(3 5 10); FRAC_LIST=(0.5 0.75); SEED_LIST=(2 3 4)   # --array=0-17
# export STEPS_PER_DIM=${STEPS_LIST[$(( i % 3 ))]}
# export DELETE_FRAC=${FRAC_LIST[$(( (i / 3) % 2 ))]}
# export SAMPLER_SEED=${SEED_LIST[$(( i / 6 ))]}
# [ "$i" -eq 0 ] && export PROFILE=1

#! ---------- run ----------
echo "Job ${SLURM_ARRAY_JOB_ID}_${i} on $(hostname), $(date)"
echo "N_LIVE=$N_LIVE SAMPLER_SEED=$SAMPLER_SEED STEPS_PER_DIM=$STEPS_PER_DIM DELETE_FRAC=$DELETE_FRAC N_BINS=$N_BINS"
nvidia-smi

LOGDIR=$HOME/rds/hpc-work/et-bns-forecast-jax/logs
nvidia-smi dmon -s u -d 1 > "$LOGDIR/gpu_util_${SLURM_ARRAY_JOB_ID}_${i}.txt" &
DMON=$!
trap 'kill $DMON 2>/dev/null' EXIT

cd "$CODE_DIR"
"${PIXI_RUN[@]}" python -c "import jax; print(jax.__version__, jax.devices())" || exit 1
"${PIXI_RUN[@]}" python -u swig.py
