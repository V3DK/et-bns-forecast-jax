""" save / load everything needed to reproduce and inspect a run in runs/<RUN_NAME>/

Layout of a run directory:
    config.json          everything set up before sampling (injection, truths, data, prior, sampler, versions)
    <script>.py          copy of the script that produced the run
    results.json         scalar outcomes (timings, log Z, n evaluations)
    samples.npz          equal-weight posterior samples in prior space (+ log_likelihood)
    nested_samples.npz   weighted dead points (positions, logL, logL_birth) for evidence / reweighting
    diagnostics.npz      per-iteration sampler histories
    corner.png
"""
import json
import shutil
import sys
import importlib.metadata
from datetime import datetime
from pathlib import Path

import numpy as np

RUNS_DIR = Path(__file__).resolve().parent / "runs"

PACKAGES = ["jimgw", "jax", "jaxlib", "blackjax", "ripplegw", "bilby", "lalsuite", "anesthetic", "numpy"]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    arr = np.asarray(x)
    return arr.item() if arr.ndim == 0 else arr.tolist()


def package_versions():
    versions = {"python": sys.version.split()[0]}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def describe_prior(prior):
    """One entry per component of a CombinePrior: class, parameter(s), bounds / exponent."""
    components = []
    for p in getattr(prior, "base_prior", [prior]):
        entry = {"type": type(p).__name__, "parameter_names": list(p.parameter_names)}
        for attr in ("xmin", "xmax", "alpha"):
            if hasattr(p, attr):
                entry[attr] = float(getattr(p, attr))
        components.append(entry)
    return components


def prior_space_truths(injection_parameters, likelihood_transforms, parameter_names):
    """Map likelihood-space injection values back to the sampled (prior) parameters."""
    truths = dict(injection_parameters)
    for transform in reversed(likelihood_transforms):
        truths = transform.backward(truths)
    return {k: float(truths[k]) for k in parameter_names}


def save_config(out, script, config):
    """Write config.json and a copy of the generating script; call before sampling."""
    out = Path(out)
    shutil.copy(script, out / Path(script).name)
    config = {
        "run_name": out.name,
        "script": Path(script).name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "versions": package_versions(),
        **config,
    }
    (out / "config.json").write_text(json.dumps(_jsonable(config), indent=2))


def save_results(out, jim, results):
    """Write samples, nested samples, diagnostics and results.json after jim.sample()."""
    out = Path(out)
    chains = jim.get_samples()
    np.savez(out / "samples.npz", **chains)

    diagnostics = jim.get_diagnostics()
    arrays = {k: np.asarray(v) for k, v in diagnostics.items() if np.ndim(v) > 0}
    scalars = {k: v for k, v in diagnostics.items() if np.ndim(v) == 0}
    np.savez(out / "diagnostics.npz", **arrays)

    # weighted dead points; uses the NSS sampler's anesthetic NestedSamples (sampling space)
    nested = getattr(jim.sampler, "_nested_samples", None)
    if nested is not None:
        names = jim.prior.parameter_names
        np.savez(
            out / "nested_samples.npz",
            **{k: np.asarray(nested.iloc[:, i]) for i, k in enumerate(names)},
            logL=np.asarray(nested["logL"]),
            logL_birth=np.asarray(nested["logL_birth"]),
        )

    results = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "n_posterior_samples": len(chains["log_likelihood"]),
        **scalars,
        **results,
    }
    (out / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    return chains

