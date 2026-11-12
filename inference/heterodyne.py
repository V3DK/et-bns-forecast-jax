"""Relative-binning likelihood whose summary data fit in memory at large bin counts.

jimgw's ``HeterodynedTransientLikelihoodFD._compute_coefficients`` builds dense
(n_bins x n_freq) arrays. For ET from 5 Hz (8192 s segment, ~12M frequencies) that is
~12 GB of complex temporaries per array at the default 62 bins, and ~100 GB at 512.
This subclass computes the same four summary arrays (A0, A1, B0, B1) with a segment
sum, which needs O(n_freq) memory for any bin count. Likelihood evaluation is unchanged.
"""
import jax
import jax.numpy as jnp

from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD


class LowMemHeterodynedTransientLikelihoodFD(HeterodynedTransientLikelihoodFD):
    """``HeterodynedTransientLikelihoodFD`` with O(n_freq)-memory summary data.

    Same binning convention as jimgw: bin k is [f_bins[k], f_bins[k+1]), the last bin
    is closed, and frequencies outside [f_bins[0], f_bins[-1]] contribute nothing.
    """

    @staticmethod
    def _compute_coefficients(detector, h_ref, f_bins):
        data = detector.sliced_fd_data
        psd = detector.sliced_psd
        freqs = detector.sliced_frequencies
        n_valid = f_bins.shape[0] - 1

        data_prod = data * h_ref.conj() / psd
        self_prod = (h_ref * h_ref.conj()) / psd

        # bin index of each frequency; the last edge belongs to the last bin
        idx = jnp.searchsorted(f_bins, freqs, side="right") - 1
        idx = jnp.where(freqs == f_bins[-1], n_valid - 1, idx)
        inside = (idx >= 0) & (idx < n_valid)
        idx = jnp.clip(idx, 0, n_valid - 1)

        centre = 0.5 * (f_bins[:-1] + f_bins[1:])
        shift = (freqs - centre[idx]) * inside

        def seg(x):
            return jax.ops.segment_sum(x, idx, num_segments=n_valid, indices_are_sorted=True)

        summary_data = jnp.array([
            seg(data_prod * inside),   # A0
            seg(data_prod * shift),    # A1
            seg(self_prod * inside),   # B0
            seg(self_prod * shift),    # B1
        ])
        return 4 / detector.duration * summary_data
