"""Save the compile-once sampler with `jax.export`, and load it without tracing (driver-plan.md, 3b).

The persistent compilation cache (T4) skips only the XLA compile. Tracing and lowering `ns_step` still
take ~30 s on CPU and ~100 s on GPU in every process. An export stores the lowered program (StableHLO).
Loading it skips tracing and lowering. Its compile still runs, but hits the persistent cache once one
process has compiled it.

Only flat lists of arrays cross the export boundary:
- the PRNG key goes in as `jax.random.key_data` and is rewrapped inside;
- `EventData`, the sampler state and the step's `info` are flattened to their leaves.
This avoids registering blackjax's NamedTuples for serialisation. Their treedefs are pickled next to the
program, so a loading process can rebuild the pytrees without tracing anything. Pickle stores classes by
import path, which is why the manifest pins the package versions. Load only exports you wrote yourself:
unpickling can run code.

Export directory layout:
    ns_init.bin, ns_step.bin   serialised `jax.export.Exported`
    treedefs.pkl               {"ev", "state", "info"} PyTreeDefs
    manifest.json              {"match": ..., "info": ...}; a load requires every "match" field to agree

The loaded `init(positions, ev)` and `step(key, state, ev)` take and return the same pytrees as
`kernel.make_sampler`'s, so `kernel.run_event` works with either.
"""
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import shutil
import socket
import time
from dataclasses import asdict
from pathlib import Path

import jax

# Bump when the export boundary below changes (flattening, key handling, file layout).
EXPORT_FORMAT = 1
_HERE = Path(__file__).resolve().parent
# Source files that define the compiled program. The kernel also depends on jimgw/blackjax/ripple
# internals; those are covered by the package versions and the pixi.lock hash.
PROGRAM_SOURCES = ("kernel.py", "event_model.py")
PIXI_LOCK = _HERE.parent / "pixi.lock"
VERSIONED_PACKAGES = ("jax", "jaxlib", "jimgw", "blackjax", "ripplegw")
FILES = ("ns_init.bin", "ns_step.bin", "treedefs.pkl", "manifest.json")


class ExportMismatch(RuntimeError):
    """The export was made for a different program or machine; `diffs` lists the fields that differ."""

    def __init__(self, diffs):
        self.diffs = diffs
        super().__init__("export manifest does not match:\n" + "\n".join(f"  {d}" for d in diffs))


def _spec(a):
    """ShapeDtypeStruct of an array or abstract value, keeping weak_type (part of the jit signature)."""
    return jax.ShapeDtypeStruct(a.shape, a.dtype, weak_type=bool(getattr(a, "weak_type", False)))


def _sha256(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).name.encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def manifest_match(model, settings, rebuild, ev):
    """Everything that must agree for an export to be valid in this process (JSON-able)."""
    s = asdict(settings)
    s.pop("termination_dlogz")      # host-side only, not part of the program
    waveform = model.waveform
    return {
        "export_format": EXPORT_FORMAT,
        "versions": {p: importlib.metadata.version(p) for p in VERSIONED_PACKAGES},
        "pixi_lock_sha256": _sha256([PIXI_LOCK]) if PIXI_LOCK.exists() else None,
        "program_sha256": _sha256(_HERE / f for f in PROGRAM_SOURCES),
        "settings": s,
        "n_delete": settings.n_delete,
        "intrinsic_only": model.intrinsic_only,
        "names": list(model.names),
        "rebuild_required_by_block": [[list(b), bool(r)] for b, r in rebuild.items()],
        "waveform": {"class": type(waveform).__name__, "f_ref": getattr(waveform, "f_ref", None)},
        "n_bins": int(ev.f_lo.shape[0]),
        "event_signature": {k: [list(v.shape), str(v.dtype), bool(getattr(v, "weak_type", False))]
                            for k, v in ev._asdict().items()},
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "prng_impl": str(jax.config.jax_default_prng_impl),
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "machine": platform.machine(),
    }


def _diff(want, got, prefix=""):
    if isinstance(want, dict) and isinstance(got, dict):
        out = []
        for k in sorted(set(want) | set(got)):
            out += _diff(want.get(k), got.get(k), f"{prefix}{k}.")
        return out
    return [] if want == got else [f"{prefix.rstrip('.')}: export has {got!r}, this process {want!r}"]


def export_sampler(out_dir, init, step, model, settings, rebuild, ev, positions, git_hash=None):
    """Export `init`/`step` from `kernel.make_sampler` to `out_dir`, which must not exist yet.

    `ev` and `positions` are any one event's (only their shapes and dtypes are used). Tracing reuses
    `init`'s and `step`'s trace caches when they have already run with the same shapes. Files are written
    to a temporary directory that is renamed at the end, so `out_dir` is either complete or absent.
    Returns {"t_export_init", "t_export_step", "bytes"}.
    """
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; remove it or choose another directory")
    ev_leaves, ev_td = jax.tree.flatten(ev)
    key_data = jax.random.key_data(jax.random.key(0))
    state_spec = jax.eval_shape(init, positions, ev)
    state_leaves, state_td = jax.tree.flatten(state_spec)
    state_out, info_spec = jax.eval_shape(step, jax.random.key(0), state_spec, ev)
    if jax.tree.structure(state_out) != state_td or jax.tree.map(_spec, state_out) != jax.tree.map(_spec, state_spec):
        raise RuntimeError("step's output state differs from its input state; cannot export a fixed signature")
    info_td = jax.tree.structure(info_spec)

    def ns_init_flat(positions, ev_leaves):
        return jax.tree.leaves(init(positions, jax.tree.unflatten(ev_td, ev_leaves)))

    def ns_step_flat(key_data, state_leaves, ev_leaves):
        state, info = step(jax.random.wrap_key_data(key_data),
                           jax.tree.unflatten(state_td, state_leaves), jax.tree.unflatten(ev_td, ev_leaves))
        return jax.tree.leaves(state), jax.tree.leaves(info)

    ev_specs = [_spec(a) for a in ev_leaves]
    t0 = time.perf_counter()
    exp_init = jax.export.export(jax.jit(ns_init_flat))(_spec(positions), ev_specs)
    t_init = time.perf_counter() - t0
    t0 = time.perf_counter()
    exp_step = jax.export.export(jax.jit(ns_step_flat))(_spec(key_data), [_spec(a) for a in state_leaves], ev_specs)
    t_step = time.perf_counter() - t0

    manifest = {
        "match": manifest_match(model, settings, rebuild, ev),
        "info": {"git_hash": git_hash, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "host": socket.gethostname(), "t_export_init": t_init, "t_export_step": t_step,
                 "platforms": list(exp_step.platforms),
                 "calling_convention_version": exp_step.calling_convention_version},
    }
    tmp = out_dir.with_name(f"{out_dir.name}.tmp-{os.getpid()}")
    tmp.mkdir(parents=True)
    try:
        (tmp / "ns_init.bin").write_bytes(exp_init.serialize())
        (tmp / "ns_step.bin").write_bytes(exp_step.serialize())
        with open(tmp / "treedefs.pkl", "wb") as f:
            pickle.dump({"ev": ev_td, "state": state_td, "info": info_td}, f)
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
        os.rename(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    n_bytes = sum((out_dir / f).stat().st_size for f in FILES)
    return {"t_export_init": t_init, "t_export_step": t_step, "bytes": n_bytes}


def load_sampler(export_dir, model, settings, rebuild, ev):
    """Load an export as `init(positions, ev)` and `step(key, state, ev)`, without tracing either.

    `ev` is any one event's `EventData`, used only to check the manifest. Raises `ExportMismatch` if the
    manifest disagrees with this process (a caller may then fall back to `kernel.make_sampler`). The
    returned functions carry their underlying jitted wrapper as `.jitted` (for `_cache_size()`); the
    wrappers are named `ns_init_exported` and `ns_step_exported` in JAX's logs.
    """
    export_dir = Path(export_dir)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    diffs = _diff(manifest_match(model, settings, rebuild, ev), manifest["match"])
    if diffs:
        raise ExportMismatch(diffs)
    with open(export_dir / "treedefs.pkl", "rb") as f:
        td = pickle.load(f)
    exp_init = jax.export.deserialize(bytearray((export_dir / "ns_init.bin").read_bytes()))
    exp_step = jax.export.deserialize(bytearray((export_dir / "ns_step.bin").read_bytes()))

    def ns_init_exported(positions, ev_leaves):
        return exp_init.call(positions, ev_leaves)

    def ns_step_exported(key_data, state_leaves, ev_leaves):
        return exp_step.call(key_data, state_leaves, ev_leaves)

    init_jit = jax.jit(ns_init_exported)
    step_jit = jax.jit(ns_step_exported, donate_argnums=1)

    def flatten(tree, treedef, what):
        leaves, got = jax.tree.flatten(tree)
        if got != treedef:
            raise TypeError(f"{what} structure {got} differs from the export's {treedef}")
        return leaves

    def init(positions, ev):
        return jax.tree.unflatten(td["state"], init_jit(positions, flatten(ev, td["ev"], "EventData")))

    def step(key, state, ev):
        state_leaves, info_leaves = step_jit(jax.random.key_data(key), flatten(state, td["state"], "state"),
                                             flatten(ev, td["ev"], "EventData"))
        return jax.tree.unflatten(td["state"], state_leaves), jax.tree.unflatten(td["info"], info_leaves)

    init.jitted, step.jitted = init_jit, step_jit
    init.manifest = step.manifest = manifest
    return init, step
