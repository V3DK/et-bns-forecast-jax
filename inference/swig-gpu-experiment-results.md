# SwiG GPU experiment: results

Plan: `swig-gpu-experiment.md`. Outputs in `rds/hpc-work/et-bns-forecast-jax/{runs,logs}`.
- Grid (steps/dim × delete frac × seed, n_live = 1024): SLURM array 40316928, 18/18 tasks finished, 2026-11-08.
- n_live test (4096 and 8192, 5 seeds each, 5 steps/dim, frac 0.5): SLURM array 40400355, 10/10 tasks finished, 2026-11-09.

**Not comparable to the table in the plan:** these runs use d_L = 1000 Mpc (`swig.py:141`) and a different sky position, so network SNR ≈ 31 and ln Z ≈ 450, not 1471.

## Summary
- **Config:** keep `STEPS_PER_DIM=5`, `DELETE_FRAC=0.5`. Do not settle `N_LIVE` yet: fix mixing first (section 1b), then choose between 4096 and 8192.
- **Accuracy:** no tested configuration is converged. The face-on/face-off split in ι (and sky weights) changes from seed to seed at every n_live (1024–8192) and steps/dim (3–10). Hypothesis: the Gibbs blocks cannot move between ι modes.
- **Heterodyne:** the likelihood check fails in 27/28 runs. Root cause: too few bins (jimgw default ε = 0.5 gives 62), not the f_max cutoff. Fixed in code (`heterodyne.py`, `N_BINS=500`); not yet rerun on GPU (section 2).
- **Speed:** compile (~200–225 s, mostly one function, `kernel`) is 73–92% of `sampling_time`. Net sampling: 32 s at 1024, 46 s at 4096, 84 s at 8192, with the GPU almost idle.

## 1. Sampling config

### Recommendation (interim)

| setting | value | why |
|---|---|---|
| `N_LIVE` | open: 4096 or 8192 | The n_live test (1b) showed n_live alone does not fix mode weights. Choose after the mixing fix. Measured net: 46 s at 4096, 84 s at 8192. |
| `STEPS_PER_DIM` | 5 | Data cannot separate 3, 5, 10, and 10 steps does not fix the ι modes. Judgement call: 10 adds ~26 s; 3 saves ~11 s and removes margin. |
| `DELETE_FRAC` | 0.5 | 0.75 saves ~8 s at 5 steps, reseeds from only 25% of live points, and has larger quoted σ (0.24 vs 0.21). |

1000 runs ≈ 13 GPU-hours (4096) or ≈ 23 GPU-hours (8192) of net sampling, once compile is shared.

### 1a. Steps and delete fraction (means over seeds 2, 3, 4; n_live = 1024)

| steps/dim | delete frac | wall (s) | compile (s) | net (s) | N_iter | evals (M) | ln Z | seed scatter | quoted σ |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.5 | 247.5 | 226.3 | 21.2 | 48 | 4.1 | 450.00 | 1.00 | 0.20 |
| 5 | 0.5 | 257.4 | 225.0 | 32.5 | 49 | 7.4 | 449.36 | 0.58 | 0.21 |
| 10 | 0.5 | 269.6 | 211.1 | 58.5 | 49 | 14.6 | 449.79 | 0.81 | 0.21 |
| 3 | 0.75 | 235.8 | 216.0 | 19.8 | 25 | 3.4 | 449.39 | 0.45 | 0.24 |
| 5 | 0.75 | 241.8 | 216.9 | 24.8 | 24 | 5.3 | 449.49 | 0.77 | 0.24 |
| 10 | 0.75 | 252.7 | 213.5 | 39.2 | 24 | 11.3 | 449.72 | 1.83 | 0.24 |

Net = wall − compile.

ln Z shifts (linear model, all 18 runs); the test only resolves ≳1 nat:

| effect on ln Z | shift |
|---|---|
| 3 vs 10 steps | −0.06 ± 0.55 |
| 5 vs 10 steps | −0.33 ± 0.55 |
| frac 0.75 vs 0.5 | −0.18 ± 0.45 |

- **ln Z scatter:** 0.88 overall vs mean quoted σ 0.22 (χ²/dof ≈ 16). More steps do not reduce it (sd 0.77, 0.61, 1.27 for 3, 5, 10 steps).
- **Sky posterior is not reproducible:** per-seed ra means span 1.2–5.6 across the grid. At 10 steps/0.5, ra = 2.47 ± 1.11, 4.69 ± 1.45, 3.20 ± 1.24 and dec mean = 0.04, 0.69, −0.57 for seeds 2, 3, 4. (Mean/sd are crude for periodic angles, but the spreads differ too.)
- **Config differences ≈ seed differences:** per-seed posterior means scatter by ~0.5 posterior sd on average (up to 1.5, in t_c).
- **Few posterior samples:** 91–1065 per run.
- **Original interpretation (superseded by 1b):** uneven allocation of live points across a few sky modes.

### 1b. n_live test (5 steps/dim, frac 0.5)

| n_live | seeds | wall (s) | compile (s) | net (s) | ln Z mean | ln Z scatter | quoted σ | scatter ÷ σ | posterior samples | P(ι < π/2) by seed |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 2–4 | 257.4 | 225.0 | 32.5 | 449.36 | 0.58 | 0.21 | 2.8× | 306–505 | 0.70, 0.30, 0.93 |
| 4096 | 2–6 | 249.1 | 203.4 | 45.7 | 449.04 | 0.73 | 0.10 | 7.5× | 1091–1822 | 0.74, 0.30, 0.16, 0.32, 0.77 |
| 8192 | 2–6 | 308.6 | 224.1 | 84.4 | 449.78 | 0.22 | 0.07 | 3.0× | 2303–3782 | 0.58, 0.83, 0.40, 0.56, 0.65 |

- **Timing:** N_iter ≈ 49 at every n_live; each doubling costs ~1.6–1.85× in net time. Compile does not change with n_live. k̄ ≈ 5.0, γ_eff ≈ 4.5.
- **ln Z:** scatter falls only at 8192 and is still ~3× the quoted σ. At 4096 it is worse than at 1024 (seed 6 at 447.98 pulls it down). Means at 4096 and 8192 differ by 0.74 ± 0.34.
- **Sky is nearly all-sky, not a few discrete modes:** ra circular resultant R ≈ 0.04–0.39 at 4096 and 8192. Plausibly because ET's three detectors share one site; not checked.
- **ι is bimodal:** face-on (ι ≈ 0.5, the injected value) vs face-off (ι ≈ 2.6). The weight between them changes from seed to seed at every n_live (table). Sky-cell fractions (ra quarter × sign of dec) disagree the same way.
- **Disagreement is far beyond sampling noise:** homogeneity χ²/dof across seeds is in the hundreds for both ι and sky at every n_live. NS posterior samples are correlated, so χ² overstates it, but a 0.40 vs 0.83 split is significant even for an effective sample size of ~100.
- **More steps do not help:** P(ι < π/2) at n_live = 1024 by seed:

| steps/dim | frac 0.5 | frac 0.75 |
|---|---|---|
| 3 | 0.44, 1.00, 0.00 | 0.54, 0.48, 0.56 |
| 5 | 0.70, 0.30, 0.93 | 0.99, 0.01, 0.70 |
| 10 | 0.52, 0.96, 0.55 | 0.15, 0.09, 0.95 |

### Hypothesis: Gibbs blocks prevent mixing between ι modes
- Moving between face-on and face-off needs a joint change in ι and ψ, and probably sky. ψ's circular mean shifts with the mode (0.28 face-on vs 0.50 face-off, 8192 seed 3).
- `swig.py` puts `iota` in block `["iota","d_L"]` and `psi` in `["ra","dec","psi"]`. A slice step within one block cannot make the joint move.
- Mode weights are then set by the seed rather than the likelihood. This would explain why neither n_live nor steps/dim fixes them, and part of the ln Z scatter.
- Untested.

### Tests still needed
All at the current injection, `STEPS_PER_DIM=5`, `DELETE_FRAC=0.5`. These compare sampling of the same heterodyned likelihood, so they can run before the heterodyne issue is fixed.

| # | test | runs | status / decides |
|---|---|---|---|
| 1 | n_live = 4096 | 5 seeds | Done: fails (1b) |
| 2 | n_live = 8192 | 5 seeds | Done: fails (1b) |
| 3 | Merge `iota`, `d_L`, `ra`, `dec`, `psi` into one block, n_live = 1024 | 5 seeds | Does mixing fix the ι mode weights? Cheapest setting that shows the problem. |
| 4 | Alternative: add a reflection move (ι → π − ι, ψ → ψ + π/2), n_live = 1024 | 5 seeds | Same question, if merging blocks is slow or insufficient |
| 5 | Best fix at n_live = 4096, then 8192 if needed | 5 seeds each | Settles N_LIVE |
| 6 | n_live = 2048 | 5 seeds, only if 4096 passes easily | Can it be cheaper? |
| 7 | steps = 10 at the chosen n_live | 3 seeds | Is 5 steps still enough after the fix? |

Proposed acceptance criteria (adjust to the science need):
- ln Z scatter across seeds ≤ 2× quoted σ.
- P(ι < π/2) and sky-cell fractions agree across seeds within sampling noise.
- Mean ln Z agrees with the 8192 reference within errors.
- Posterior sample count is high enough for the downstream use.

## 2. Heterodyne failure (root-caused; fix not yet run on GPU)

### Symptom
- max |logL_het − logL_exact| over 51 samples: 0.44–1.10 (median 0.78), above the 0.5 tolerance in 17/18 grid runs. `swig.py` prints "posterior unreliable".
- n_live test: 0.67–1.02 at 4096 and 0.64–1.05 at 8192; fails in 10/10. More live points do not change it.
- Earlier runs: 0.29–0.36 at 500 Mpc (pass); 0.57 in run 40314245 (fail).

### Cause: too few heterodyne bins
`HeterodynedTransientLikelihoodFD` defaults to `epsilon=0.5`. That gives **62 bins over 5–1469 Hz**; one bin spans 141–197 Hz. The heterodyne assumes h/h_ref is linear within each bin, and over bins this wide that fails for samples away from the reference. bilby's default (ε = 0.025) would give 1250.

How this was checked (CPU, 2026-11-12):
- Rebuilt the injection from the same seeds; per-detector SNRs match the CSD3 runs to 1e-9.
- Reimplemented the heterodyne with jimgw's formulas. It reproduces the stored `log_likelihood` to 1e-6.
- Compared it with the exact likelihood, bin by bin, at 80 posterior samples spread over logL (`12p_n8192_s5_f0.5_seed4`).

| variant (80 samples) | bins | max \|ΔlogL\| | mean ΔlogL |
|---|---|---|---|
| current: ε = 0.5, f_max 1469, ref = injection | 62 | 0.48 | +0.11 |
| ref = max-likelihood sample | 62 | 0.49 | +0.11 |
| f_max 1000 Hz | 62 | 0.22 | +0.06 |
| ε = 0.1 | 312 | 0.025 | +0.005 |
| ε = 0.025 | 1250 | 0.0013 | +0.0003 |

**Not the f_max cutoff.** The reference amplitude does taper steeply: 0.47× its inspiral trend at 1400 Hz and 0.2× at the 1469 Hz cutoff. Samples' own taper frequencies span 1175–1703 Hz, so |h/h_ref| at the top edge ranges from 0 to 3.3. But bins above 1000 Hz hold < 0.1% of ⟨h,h⟩ and contribute ≤ 0.02 to the error. Lowering f_max helps only because it squeezes the same 62 bins into a narrower band.

Where the error comes from:

| bins starting in | share of ⟨h,h⟩ | mean error | max error |
|---|---|---|---|
| 5–20 Hz | 0.53 | +0.008 | 0.05 |
| 20–100 Hz | 0.37 | +0.044 | 0.15 |
| 100–500 Hz | 0.09 | +0.030 | 0.23 |
| 500–1000 Hz | 0.005 | +0.028 | 0.24 |
| ≥ 1000 Hz | < 0.001 | +0.001 | 0.02 |

**Not the reference point.** Using the max-likelihood sample as reference changes nothing.

**The error is a bias, not noise.** It is always positive (the heterodyne overestimates logL) and correlates with Λ₁ and Λ₂ (r = 0.58, 0.48). So it pushes the posterior towards high Λ.

**This explains the earlier observations:**
- **Error grows as SNR falls:** broader posteriors reach further from the reference, where the straight-line approximation within each bin fails.
- **Peak logL below SNR_mf²/2 is expected:** phase marginalisation uses log I₀(|Z|) ≈ |Z| − ½ ln(2π|Z|). That is 4.4 nats below |Z| at |Z| ≈ 977, which matches the 4.3-nat gap (484.5 vs 488.8). It is not a heterodyne diagnostic.

### n_bins sweep
Max |ΔlogL| over 40 posterior samples per run (mean bias in brackets):

| n_bins | SNR 30 (n8192 seed4) | SNR 43 (40314245, other sky) | SNR 56 (40142298, 500 Mpc, other masses, f_max 1674) |
|---|---|---|---|
| 62 | 0.42 (+0.11) | 0.63 (+0.17) | 0.37 (+0.09) |
| 128 | 0.12 (+0.030) | 0.14 (+0.032) | 0.076 (+0.016) |
| 256 | 0.029 (+0.008) | 0.036 (+0.009) | 0.019 (+0.004) |
| 384 | 0.010 (+0.003) | 0.017 (+0.005) | 0.010 (+0.003) |
| **500** | **0.008 (+0.002)** | **0.010 (+0.003)** | **0.005 (+0.001)** |
| 768 | 0.003 | 0.004 | 0.002 |
| 1000 | 0.002 | 0.002 | 0.001 |

- The error falls as roughly n_bins⁻².
- At a given n_bins it barely depends on SNR. Posterior samples sit a similar number of nats below the peak whatever the SNR, and the error tracks that distance.
- **Stress test (failed as a posterior proxy):** I made pseudo-posteriors for SNR 15 and 10 by stretching the SNR-30 samples 2× and 3× from the injection. They land 20–200 nats below the peak, so no real posterior would contain them. There the error is large even at 1000 bins (max 1.1–2.2). This shows the error grows steeply away from the reference. It also means **low-SNR posteriors that rail against the Λ, q or spin priors are untested**.

### Fix
- **`inference/heterodyne.py` → `BinnedHeterodynedLikelihoodFD`.** jimgw already accepts `n_bins`, but its `_compute_coefficients` builds dense (n_bins × n_freq) arrays. That is ~0.2 GB per bin for 8192 s from 5 Hz, so ~100 GB at 500 bins. The subclass computes the same coefficients with a per-bin segment sum. It matches jimgw to 1e-6 at 62 bins and my reimplementation at 500.
- **`swig.py`:** uses the subclass. Bin count comes from `N_BINS` (default `DEFAULT_N_BINS = 500`) and is recorded as `likelihood.n_bins` in `config.json`.

**Recommendation: `n_bins = 500`.**
- Max error ≤ 0.01 on all three injections: 50× below the 0.5 tolerance, and below the quoted ln Z σ (0.07–0.2).
- Gives room for posteriors further from the reference than those tested (lower SNR, other sources).
- A fixed bin count, rather than ε, keeps array shapes the same across events, which section 3 needs to reuse one compiled step.
- 256 bins is the cheaper fallback (max 0.04, bias +0.01). Use 1000 if low-SNR checks fail.

### To do
1. **GPU rerun:** rerun one grid config with `N_BINS=500` and confirm the post-run check passes. Compare `single_ms` and `per_trip_ms` with the 62-bin runs; each call now evaluates the waveform at 2 × 501 points instead of 2 × 63. Cost is untested, but the GPU is latency-bound (section 3), so it should be small.
2. ~~**Tighten the check:** set `CHECK_TOL` to ~0.1.~~ Done: `CHECK_TOL = 0.1` in `swig.py`. The 0.5 threshold was above the whole sweep, and the error is a Λ-correlated bias, not noise.
3. **Low-SNR check:** one real low-SNR run (SNR ~ 10–15, ideally with Λ railing) at 500 bins.
4. **Rerun the mixing tests** (section 1) on the corrected likelihood, since the old one biased towards high Λ. The ι mode-weight problem is a sampler issue and should persist.

## 3. Compile and GPU optimisation

### Where the time goes
| component | cost |
|---|---|
| `kernel` trace | ~88 s (independent of steps/dim) |
| `kernel` lowering to MLIR | ~11 s |
| `kernel` XLA compile | ~100 s |
| Other compiles within `sampling_time` | ~15–30 s |
| Sampling at 1024 live | 20–60 s |
| Sampling at 4096 / 8192 live (5 steps, frac 0.5) | 46 s / 84 s |
| Benchmarks and setup (outside `sampling_time`) | a few minutes per job; timing benchmarks can be dropped in production |

`kernel` is the jitted NS iteration: `step_fn = jax.jit(nested_sampler.step)` in jimgw `samplers/blackjax/ns/nss.py`. It contains the full likelihood, waveform and all five Gibbs blocks.

### Why it recompiles
- `step_fn` wraps fresh closures inside `_sample`, so repeated `jim.sample()` calls recompile even in one process.
- The persistent cache hit `kernel` in 0/18 runs: jimgw redirects it to `ckpt/jax_cache` (`samplers/config.py:115`) and deletes it after the run. `jax_cache_dir_after_sample` was never recorded.
- Event data reaches `kernel` through closures, so it is probably embedded as constants. Different events would then be different programs and would miss the cache anyway (inferred, not tested).
- The persistent cache only skips XLA; trace and lowering (~100 s) still run.

### Compile plan, in order
1. **Baseline:** rerun with `jax_log_compiles` off and time from timestamps. The logging (~250k lines per run) may inflate the 88 s trace.
2. **Reuse:** custom driver with `step(key, state, event)`, where `event` holds the heterodyne coefficients, data and any per-event prior bounds. Run one long-lived process per GPU looping over events, so 1000 runs compile once per GPU. Needs a resume record for Slurm time limits.
3. **Check shapes:** heterodyne bins and detector masks must match across events, or be padded or grouped. Reuse also requires the same sampler settings, waveform, JAX/XLA version and GPU type.
4. **Reduce the trace:** profile why it produces ~250k tracing events; look for nested small jits.
5. **Optional:** `jax.export` plus persistent cache, if fresh processes are needed per event. Untested.

### GPU measurements
- Utilisation 0.4–0.9% mean (9 tasks checked), so the GPU is latency-bound.
- δ ≈ 0: in-loop 0.51 ms vs single call 0.58 ms at B = 512. This refutes the 1.2 ms estimate (bare likelihood loop only).
- γ ≈ 3.7–3.9 (upper bound): 18–19 trips per step at k̄ ≈ 4.8. Includes any non-likelihood kernel work.
- k̄ ≈ 4.8; shrink:step-out ≈ 21:1.
- t_iter ≈ 0.11 s + 9.1 ms·S (frac 0.5); 0.46 s + 9.6 ms·S (frac 0.75).
- Without compile, 5 steps/0.5 takes ~32 s at 1024 live, under the 60–80 s expected.

### GPU plan, in order
1. Analyse the unused Perfetto trace in `runs/12p_s3_f0.5_seed2_40317062/trace` to split γ into stragglers and non-likelihood kernel work.
2. Multi-proposal shrink (m ≈ 4–6 draws per trip), to use spare batch capacity and cut k̄ and γ.
3. Slice widths from live-point scale in whitened coordinates (target k̄ ≈ 3).
4. XLA command buffers for while loops (flag support unverified).
5. `vmap` several events through one GPU. Harder, since events need different numbers of iterations.

The GPU measurements above are from 1024 live. At 4096 / 8192: t_iter 0.92 / 1.73 s, k̄ ≈ 5.0, γ_eff ≈ 4.5, per-trip 0.68 / 1.28 ms at B = 2048 / 4096. GPU utilisation was not logged for these runs. Compile is still 82% (4096) and 73% (8192) of `sampling_time`.
