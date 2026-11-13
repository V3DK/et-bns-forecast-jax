# SwiG GPU speed + bias experiment

## Context
- `swig.py`: jimgw / BlackJAX nested slice-within-Gibbs (SwiG) PE on one injected ET BNS (MPA1, 1.5+1.25 M☉, d_L=500 Mpc, network SNR ≈ 56), 12 params, heterodyned likelihood with phase marginalisation.
- Advisor expects ~10 s on an A100. Observed:

| run | n_live | wall | evals | ln Z ± σ |
|---|---|---|---|---|
| A100 | 8192 | 320 s | 72.5 M | 1471.83 ± 0.08 |
| A100 | 1024 | 246 s | 8.86 M | 1471.19 ± 0.23 |
| M4 Mac | 1024 | 321 s | 8.72 M | 1470.43 ± 0.25 |

## Diagnosis so far
- Runtime model: T ≈ T_compile + N_iter · S · γ · k̄ · (τ(B) + δ)
  - N_iter ≈ 58: set by information H ≈ 34 nats; independent of n_live at fixed n_delete_frac.
  - S = steps/dim × D = 60 sequential slice steps per iteration.
  - k̄ ≈ 5 calls per step; shrink:step-out ≈ 22:1, so the slice brackets are too wide.
  - γ: straggler factor; vmapped while_loop waits for the slowest lane.
  - τ(B): batched likelihood latency; δ: per-loop-trip overhead.
- GPU is latency-bound, not FLOP-bound:
  - 8× fewer evals cut wall time only 1.3×.
  - Ideal (γ=1, δ=0) at n_live=1024 is 13.7 s; actual is 18× that.
  - Two-point fit suggests δ ≈ 1.2 ms per trip, comparable to τ. Unverified.
- Compile time is inside `sampling_time`, unmeasured. jimgw deletes `ckpt/jax_cache` mid-run, so the persistent cache is useless.
- **ln Z values disagree at ~5σ (1.4 nats) on identical data.** Quoted errors miss mixing bias, so 5 steps/dim may already be too few.

## This experiment (SLURM array, 18 tasks, n_live = 1024)
- Grid: STEPS_PER_DIM ∈ {3, 5, 10} × DELETE_FRAC ∈ {0.5, 0.75} × SAMPLER_SEED ∈ {2, 3, 4}. Knobs are read from env.
- New instrumentation in `swig.py`:
  - Median single-call benchmark, `single_ms[B]`.
  - In-`while_loop` per-trip cost, `per_trip_ms[B]`; δ ≈ per_trip − single.
  - Compile time, `compile_s`, parsed from `jax_log_compiles`.
  - Derived: `t_iter`, `kbar`, `n_trip`, `gamma_eff = n_trip / (S·kbar)`.
  - Perfetto trace on task 0 (`PROFILE=1`); `nvidia-smi dmon` GPU-utilisation log per task.
- Stable `JAX_CACHE_DIR`. Check `jax_cache_dir_after_sample` to see whether jimgw overrides it.
- `run_io.runtime_environment` records the new env vars.

## Goals
1. Attribute runtime to compile time, per-trip overhead δ, stragglers γ and chain length S.
2. Measure speedup from fewer steps (3) and more aggressive deletion (0.75 halves N_iter).
3. Measure bias: compare ln Z mean and empirical scatter across seeds against the steps=10 reference. Accept a cheaper configuration only if it agrees.

Expected best case is ~60–80 s, still not 10 s.

## Likely next steps (sampler-level)
- Multi-proposal shrink: evaluate m ≈ 4–6 bracket draws per trip. B = 512 is well below saturation (B* ≈ 3000), so this is cheap, and it is exact if draws are processed in order.
- Slice widths from live-point scale in whitened coordinates (target k̄ ≈ 3).
- Optional: XLA command buffers for while loops (`--xla_gpu_enable_command_buffer=...,WHILE,CONDITIONAL`); flag support unverified.