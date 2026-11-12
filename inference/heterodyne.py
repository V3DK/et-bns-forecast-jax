""" Heterodyned (relative-binning) likelihood with a user-chosen bin count and memory-light setup.

jimgw's HeterodynedTransientLikelihoodFD already accepts `n_bins`, but its default (`epsilon=0.5`)
gives only ~60 bins over 5 Hz to merger, which is too coarse (|logL_het - logL_exact| up to ~1 at
SNR 30; see inference/swig-gpu-experiment-results.md, section 2). More bins are unusable as shipped:
`_compute_coefficients` builds dense (n_bins x n_freq) arrays, ~n_bins * 0.2 GB each for an 8192 s
segment from 5 Hz. This subclass computes the same summary data with a per-bin segment sum instead.

A fixed `n_bins` (rather than `epsilon`) also keeps array shapes identical across events, which is
needed to reuse one compiled sampler step for many runs.
"""
import jax
import jax.numpy as jnp

from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD

DEFAULT_N_BINS = 500


class BinnedHeterodynedLikelihoodFD(HeterodynedTransientLikelihoodFD):
    """HeterodynedTransientLikelihoodFD with `n_bins` as the bin control and O(n_freq) setup memory.

    Bins are equal-phase bins from jimgw's `_make_binning_scheme`; bins entirely beyond the reference
    waveform's support are dropped by jimgw, so the final `self.n_bins` can be slightly smaller.
    """

    def __init__(self, *args, n_bins: int = DEFAULT_N_BINS, **kwargs):
        if kwargs.get("epsilon") is not None:
            raise ValueError("pass n_bins, not epsilon")
        super().__init__(*args, n_bins=n_bins, **kwargs)

    @staticmethod
    def _compute_coefficients(detector, h_ref, f_bins):
        """Same A0, A1, B0, B1 as jimgw (half-open bins, last bin closed), via segment sums."""
        data = detector.sliced_fd_data
        psd = detector.sliced_psd
        freqs = detector.sliced_frequencies
        n_valid = f_bins.shape[0] - 1

        idx = jnp.searchsorted(f_bins, freqs, side="right") - 1
        idx = jnp.where(freqs == f_bins[-1], n_valid - 1, idx)          # close the last bin
        inside = (idx >= 0) & (idx < n_valid)
        idx = jnp.clip(idx, 0, n_valid - 1)

        centre = (f_bins[:-1] + f_bins[1:]) / 2
        shift = freqs - centre[idx]
        data_prod = jnp.where(inside, data * h_ref.conj() / psd, 0.0)
        self_prod = jnp.where(inside, (h_ref * h_ref.conj()) / psd, 0.0)

        def bin_sum(x):
            return jax.ops.segment_sum(x, idx, num_segments=n_valid, indices_are_sorted=True)

        summary_data = jnp.array([
            bin_sum(data_prod),            # A0
            bin_sum(data_prod * shift),    # A1
            bin_sum(self_prod),            # B0
            bin_sum(self_prod * shift),    # B1
        ])
        return 4 / detector.duration * summary_data
