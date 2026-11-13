"""Test T6 (driver-plan.md, section 4): check a driver output directory after kills and restarts.

Passes if, for every event in --event-range:
- there is exactly one complete results file `event_NNNNN.npz` (no `.tmp` left anywhere, no `.failed`
  unless --allow-failed), and it loads, with consistent shapes and normalised weights;
- with --compare-dir (a clean, uninterrupted run of the same events): ln Z, iteration and evaluation
  counts, and hashes of the posterior and weighted samples are bit-identical;
- with --compare-summary (a check_kernel.py --summary-out JSON, e.g. t4_run1.json): the fields it
  records are identical, for the events both have.

Also prints which process (pid) wrote each event and whether that event paid the compile, so a
restart can be read off the table.

    python check_driver.py --out-dir runs_t6 --event-range 0 5 --compare-dir runs_t6_clean \
        --compare-summary t4_run1.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def load(path):
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    meta = json.loads(str(arrays.pop("meta")))
    return meta, arrays


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def summary(meta, arrays):
    """The fields check_kernel.event_summary records, plus the weighted samples' hash."""
    return {"log_Z": meta["log_Z"], "n_iterations": meta["n_iterations"],
            "n_likelihood_evaluations": meta["n_likelihood_evaluations"],
            "n_posterior": int(arrays["posterior_samples"].shape[0]),
            "posterior_sha256": sha(arrays["posterior_samples"]),
            "weighted_sha256": sha(arrays["weighted_samples"])}


def consistency(meta, arrays):
    """Problems with one results file (empty list if none)."""
    bad = []
    d = len(meta["names"])
    n, m = arrays["posterior_samples"].shape[0], arrays["weighted_samples"].shape[0]
    shapes = {"posterior_samples": (n, d), "posterior_log_likelihood": (n,), "posterior_log_prior": (n,),
              "weighted_samples": (m, d), "weighted_log_weight": (m,), "weighted_log_likelihood": (m,),
              "weighted_log_prior": (m,), "truths": (d,), "names": (d,)}
    for k, s in shapes.items():
        if arrays[k].shape != s:
            bad.append(f"{k} shape {arrays[k].shape}, expected {s}")
    if list(arrays["names"]) != meta["names"]:
        bad.append("names differ between meta and array")
    lw = arrays["weighted_log_weight"]
    total = float(np.log(np.sum(np.exp(lw - lw.max()))) + lw.max())
    if abs(total) > 1e-9:
        bad.append(f"weighted_log_weight logsumexp {total:.3g}, expected 0")
    for k in ("posterior_log_prior", "weighted_log_prior", "posterior_log_likelihood"):
        if not np.all(np.isfinite(arrays[k])):
            bad.append(f"non-finite {k}")
    if not (np.isfinite(meta["log_Z"]) and np.isfinite(meta["log_Z_error"])):
        bad.append("non-finite ln Z")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--event-range", type=int, nargs=2, required=True, metavar=("START", "STOP"))
    ap.add_argument("--compare-dir", help="clean run of the same events; results must be identical")
    ap.add_argument("--compare-summary", help="check_kernel.py --summary-out JSON to compare with")
    ap.add_argument("--allow-failed", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    ok = True
    tmps = sorted(p.name for p in out.glob("*.tmp"))
    if tmps:
        ok = False
        print(f"FAIL: leftover temporaries: {tmps}")

    ref_summary = {}
    if args.compare_summary:
        ref_summary = json.loads(Path(args.compare_summary).read_text())["events"]

    print(f"{'event':>5} {'pid':>8} {'task':>5} {'fresh':>5} {'mode':>6} {'ln Z':>10} {'N_iter':>6} "
          f"{'n_post':>6} {'n_w':>6} {'setup s':>7} {'total s':>7}  status")
    for eid in range(*args.event_range):
        npz, failed = out / f"event_{eid:05d}.npz", out / f"event_{eid:05d}.failed"
        if failed.exists():
            ok &= args.allow_failed
            print(f"{eid:>5} failed: {json.loads(failed.read_text())['error']}"
                  + ("" if args.allow_failed else "  FAIL"))
            continue
        if not npz.exists():
            ok = False
            print(f"{eid:>5} missing  FAIL")
            continue
        try:
            meta, arrays = load(npz)
        except Exception as e:      # a truncated file would land here
            ok = False
            print(f"{eid:>5} unreadable: {type(e).__name__}: {e}  FAIL")
            continue
        problems = consistency(meta, arrays)
        if meta["event_id"] != eid:
            problems.append(f"meta event_id {meta['event_id']}")
        s = summary(meta, arrays)
        if args.compare_dir:
            other = Path(args.compare_dir) / npz.name
            if other.exists():
                s2 = summary(*load(other))
                problems += [f"{k} differs from {args.compare_dir}" for k in s if s[k] != s2[k]]
            else:
                problems.append(f"not in {args.compare_dir}")
        if str(eid) in ref_summary:
            r = ref_summary[str(eid)]
            problems += [f"{k} differs from {args.compare_summary}" for k in r if s.get(k) != r[k]]
        ok &= not problems
        t = meta["timings"]
        tag = "ok" if not problems else "FAIL: " + "; ".join(problems)
        if not problems and (args.compare_dir or str(eid) in ref_summary):
            tag = "identical" + (f" (incl. {Path(args.compare_summary).name})" if str(eid) in ref_summary else "")
        if meta["exact_check"]:
            c = meta["exact_check"]
            tag += f"; exact check max|dlogL|={c['max_abs_dlogl']:.3g} {'ok' if c['passed'] else 'over tol'}"
        print(f"{eid:>5} {meta['environment']['pid']:>8} {meta['task'][0]:>2}/{meta['task'][1]:<2} "
              f"{str(meta['compiled_fresh'])[0]:>5} {meta['sampler_mode']:>6} {meta['log_Z']:10.3f} "
              f"{meta['n_iterations']:6d} {s['n_posterior']:6d} {arrays['weighted_samples'].shape[0]:6d} "
              f"{t['setup']:7.1f} {t['total_before_write']:7.1f}  {tag}")

    print(f"\nT6 {'PASSED' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
