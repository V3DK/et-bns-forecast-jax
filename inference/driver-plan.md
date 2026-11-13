# Compile-once driver for many events: plan

Goal: run ~1000 injected BNS events (possibly 10,000 later) on CSD3 GPUs while compiling the sampler kernel once, not once per event. Context: `swig-gpu-experiment-results.md`, section 3. There, compile (~200–225 s, mostly `kernel`) was 73–92% of `sampling_time`.

Development and testing happen locally (RHEL CPU box, M4 Mac) with `INTRINSIC_ONLY = True` and small n_live. The code must run unchanged on CSD3.

Based on jimgw 0.5.2 (pinned in `pixi.toml`). Paths below are relative to `site-packages/jimgw`.

## Decisions (settled)
- **gps:** random per event. `gmst`, `trigger_time` and `detector.start_time` are therefore event data, not constants.
- **M_c prior:** injected M_c ± 1% (as in `swig.py`). The bounds are event data.
- **Exact-likelihood check:** run on a random 5% of events, not on every event.
- **Population:** uniform in comoving volume out to z = 0.1, with reasonable ranges for masses and angles; EOS fixed (MPA1).

## 1. Why every event recompiles today
`step_fn = jax.jit(nested_sampler.step)` (`samplers/blackjax/ns/nss.py`) closes over the likelihood and the prior. These per-event Python values become constants in the compiled program:

| per-event value | where |
|---|---|
| `freq_grid_low`, `freq_grid_high`, `bin_widths` | `core/single_event/likelihood.py`, `_mask_and_set_frequency_arrays` |
| `waveform_low_ref[det]`, `waveform_high_ref[det]` | `HeterodynedTransientLikelihoodFD.__init__` |
| `summary_data[det]` (A0, A1, B0, B1) | `__init__` / `_compute_coefficients` |
| `trigger_time`, `gmst` | `SingleEventLikelihood._prepare_parameters` |
| `detector.start_time` (depends on gps and duration) | `core/single_event/detector.py:407` (`fd_response`) |
| `fixed_parameters` (the injected extrinsics when `INTRINSIC_ONLY`) | `apply_fixed_parameters` |
| M_c prior bounds | `UniformPrior` Python floats |
| **Shape:** `n_bins` can shrink when bins beyond the reference waveform's support are dropped | `_mask_and_set_frequency_arrays` |

Other causes:
- `step_fn` wraps fresh closures each call, so repeated `jim.sample()` calls recompile even within one process.
- jimgw redirects the persistent cache to `ckpt/jax_cache` and deletes it after the run (`samplers/config.py`, `samplers/blackjax/utils.py:188`).

These are the same for every event: the waveform, detector geometry, the other priors, n_live, n_delete, blocks, steps per dimension and the number of Gibbs sweeps.

**The kernel never sees the full frequency grid**, only arrays the size of `n_bins`. Events can have different data durations without changing the compiled program. The only shape to pin is `n_bins`; no other padding is needed. Each event's data is about 3 detectors × 6 arrays × 500 complex values, around 150 kB.

## 2. Design

### 2.1 `EventData` pytree
A NamedTuple of jnp arrays, built on the host from a fully set-up `BinnedHeterodynedLikelihoodFD` (`inference/heterodyne.py`):
- `f_lo`, `f_hi`, `bin_w`: `[n_bins]`
- `ref_lo`, `ref_hi`: `[n_det, n_bins]` complex
- `coeffs`: `[n_det, 4, n_bins]` complex (A0, A1, B0, B1)
- `t_offset`: `[n_det]` = `trigger_time − detector.start_time` (keep offsets small; don't pass the absolute GPS time into the kernel, for float precision)
- `gmst`: scalar
- `mc_lo`, `mc_hi`: scalars
- `fixed`: `[n_fixed]`, the injected extrinsics in a fixed order (only used when `INTRINSIC_ONLY`)

**Enforce exactly `N_BINS`.** Build bins only where the reference waveform is non-zero, and assert `likelihood.n_bins == N_BINS`; never let it shrink silently. `f_max_likelihood` already stops before the merger taper (`merger_cutoff_frequency` in `swig.py`), so this should be easy.

### 2.2 Pure functions of `(x, ev)`
- `log_likelihood(x, ev)`, `build_cache(x, ev)`, `ll_from_cache(x, cache, ev)`: copy jimgw's `HeterodynedTransientLikelihoodFD._likelihood`, `_generate_waveform` and `_evaluate_from_waveform` (~40 lines), reading every per-event array from `ev`.
  - Reuse the detectors' `antenna_pattern` and `delay_from_geocenter`; they depend only on geometry.
  - Keep phase marginalisation (`log_i0`) and the cache's unit-distance scaling as in jimgw.
- `log_prior(x, ev)`: small pure-JAX prior: uniform with traced bounds for M_c, plus sine (ι), cosine (dec), power law (d_L) and fixed-bound uniforms. Must match `CombinePrior.log_prob` from `swig.py`.
- Apply the `q → eta` likelihood transform (`MassRatioToSymmetricMassRatioTransform`) inside `log_likelihood`, as `Jim._sampling_array_to_likelihood_parameters` does. `swig.py` has no sample transforms.
- **Rule:** no event-derived Python float or array may be closed over. Everything event-specific comes in through `ev`.

### 2.3 Kernel
Use a module-level jitted step that builds the sampler inside the traced function, so the closures capture tracers rather than constants:

```python
def ns_step(key, state, ev):
    cstep = _build_swig_constrained_step(
        log_prior_fn=partial(log_prior, ev=ev),
        build_cache=partial(build_cache, ev=ev),
        log_likelihood_from_cache_fn=partial(ll_from_cache, ev=ev),
        rebuild_required_by_block=REBUILD,   # computed once, via jimgw's _build_rebuild_required_by_block
        num_gibbs_sweeps=..., num_inner_steps_per_dim=..., periodic=None, n_dims=N_DIMS,
    )
    kernel = build_from_mcmc_kernel(cstep, num_inner_steps=1,
                                    update_inner_kernel_params_fn=update_block_covariances,
                                    num_delete=N_DELETE)
    return kernel(key, state)

step = jax.jit(ns_step, donate_argnums=1)
```

- Build an `init(positions, ev)` the same way, mirroring `initialize_nested_sampler_state` in `nss.py` (`_ns_adaptive_init` with `lax.map`, `batch_size=n_delete`).
- `_build_swig_constrained_step` (`samplers/blackjax/ns/swig.py`) and `_build_rebuild_required_by_block` (`core/single_event/blocked_likelihood.py`) are private jimgw functions. That is acceptable because jimgw is pinned.
- Copy `update_block_covariances` from `BlackJAXSwiGSampler.__init__`.
- **One device per process** (`mesh=None` path). On CSD3, run one process per GPU; on the Mac, use `N_DEVICES=1`. The sharded path (`build_sharded_from_mcmc_kernel`) is a different program and is out of scope for now.

### 2.4 Catalogue: `make_catalogue.py`
Write one row per event: `event_id`, z, d_L, m1/m2 (source frame), ι, ra, dec, ψ, gps (random, e.g. uniform over one year), noise seed, sampler seed.
- The driver rebuilds each injection deterministically from its row, so no strain data is stored.
- To extend to 10k events, append rows 1000–9999 with the same generator and new seeds.

### 2.5 Driver: `driver.py`
- Takes a catalogue path, task index and task count; processes events `i::N`.
- Skips events that already have a results file.
- Per event, on the host:
  - Build the injection (follow `swig.py`: duration from the chirp time, `inject_signal`, `merger_cutoff_frequency`).
  - Build the likelihood with `N_BINS` and convert it to `EventData`.
  - Draw initial positions from the prior on the host.
- Sample with `init`, then loop on `step` until termination in Python, as in `nss.py` (`should_terminate`, about 49 host syncs per event). Then run blackjax `finalise` and anesthetic `NestedSamples` (copy from `nss.py`).
- Write each event's results atomically (write to a temporary file, then rename): ln Z, ln Z error, posterior samples, diagnostics, timings, and a `compiled_fresh` flag.
- Before starting each event, check the remaining Slurm time (job end time from `squeue -h -j $SLURM_JOB_ID -o %e`, or a `--deadline` argument); stop cleanly if an event won't fit. No checkpoint within an event is needed: events take about a minute and restart from scratch.
- **Exact check on 5% of events** (chosen by a hash of `event_id`): `TransientLikelihoodFD` closes over the full-grid data and its shapes depend on the duration, so it recompiles for each event. Accept that cost on the sampled events only. Use `CHECK_TOL = 0.1` as in `swig.py`.
- Drop the timing benchmarks from `swig.py` (`chain`, `med_time`) and turn `jax_log_compiles` off in production.
- **Measure host setup time per event.** With compile gone, setup may rival the ~46 s of sampling at 4096 live. If it does, prepare event i+1 in a background thread while event i samples (later optimisation; not in the first version).

## 3. Reusing the compile across Slurm jobs
Both layers below work only after 2.1–2.3, once the compiled program no longer depends on the event.

### 3a. JAX persistent compilation cache (do first)
- Set `jax_compilation_cache_dir` to a directory on `/rds` and `jax_persistent_cache_min_compile_time_secs = 0`.
- The driver never calls jimgw's `_sample`, so jimgw's cache redirect and deletion never run.
- This skips the XLA compile (~100 s). Tracing and lowering (~100 s) still run in every process.
- The cache key includes the program hash, jax/jaxlib version, GPU type, XLA flags and compile options.

### 3b. `jax.export` (second)
- Serialise `jax.export.export(step)(key_spec, state_spec, ev_spec)` and the same for `init` to files.
  - Loading skips tracing and lowering.
  - `exported.call(...)` still compiles, and that compile hits the cache from 3a, so a process starts in seconds.
- At the export boundary, pass the key as raw `jax.random.key_data`, and flatten the state into its arrays (unflatten outside, with a treedef from `jax.eval_shape`). This avoids `register_namedtuple_serialization` for blackjax's state types.
- Export on the target platform, i.e. on the CSD3 GPU node in the warm job.
- **Manifest** stored with the artifact: jax/jaxlib/jimgw/blackjax versions, `pixi.lock` hash, git hash, n_live, `N_BINS`, steps per dimension, delete fraction, blocks, `INTRINSIC_ONLY`, waveform, GPU type (`jax.devices()[0].device_kind`). Load the artifact only if the manifest matches; otherwise trace normally and log a warning.

### Priority
With one long-lived process per GPU, compile is paid once per task. For example, 250 events × ~1 min ≈ 4 h per task, so a ~200 s compile is under 2%. The driver plus 3a gets most of the gain. 3b matters when tasks are short (time limits, preemption, many small array tasks).

### Slurm workflow
1. **Warm job** (1 GPU, ~15 min): trace, export, compile into the shared cache, run one smoke-test event and write the manifest.
2. **Array job** with `--dependency=afterok:<warm>`: N tasks × 1 GPU, each looping over its share of events. Only one process ever writes to the cache, which avoids concurrent writes on Lustre.
3. **Extending to 10k:** if the manifest still matches, submit the array job again without a warm job. Changing any sampler setting, `N_BINS`, the waveform, the pixi lock or the GPU partition invalidates both layers; pin the partition.

## 4. Local tests (RHEL CPU or Mac, `INTRINSIC_ONLY = True`, n_live 256–512)
The caching mechanism doesn't depend on `INTRINSIC_ONLY`, so the 4-parameter setup is a valid fast test bed. T7 then confirms it on the real 12-parameter program.

| # | test | pass if |
|---|---|---|
| T1 | Pure likelihood and prior vs jimgw (`Jim.evaluate_posterior`) at 1000 prior draws, 3 catalogue events | \|Δ\| < 1e-9 |
| T2 | 3 events in one process | `ns_step` traced and compiled exactly once (count with a `jax_log_compiles` handler; `jax_explain_cache_misses` for diagnosis) |
| T3 | Driver vs `jim.sample()`, same event and seed | ln Z agrees within quoted errors (identical results are not expected: passing data as arguments instead of constants changes floating-point details) |
| T4 | Two separate processes, shared cache directory | second logs a persistent-cache hit for `ns_step`; its compile time collapses |
| T5 | Third process loads the export | no trace of `ns_step`; results identical to T2 for the same seed and machine |
| T6 | Kill mid-run and restart | finished events are skipped; no partial result files |
| T7 | T2–T5 with `INTRINSIC_ONLY = False`, small n_live, `jax_log_compiles` off | works for the 12-parameter program; record trace, lowering and compile times (compile-plan step 1 in the results doc) |

Then on CSD3: the warm job plus a 2-task array of 3 events each. Check the logs for cache hits, the per-event wall time and the host setup time.

## 5. Out of scope here
- The ι mode-mixing fix (results doc, section 1b): block layout changes the compiled program. Settle the blocks before the production warm job.
- Multi-device sharding, multi-proposal shrink, running several events at once with `vmap` (results doc, section 3, GPU plan).
