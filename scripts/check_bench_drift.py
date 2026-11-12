#!/usr/bin/env python3
"""Check a runtime-bench run for residual clock drift across its timed samples.

Pairs the `*_start_s` perf_counter offsets recorded in the results block of a
rank0.log against the gpu_telemetry.csv written alongside it (--bench_log_clocks),
and reports, per phase, the SM clock / temperature / power the GPU was actually
running at during each sample.

A soak long enough for the workload shows a flat clock column and a sample
spread that no longer trends with it.  A clock still climbing (or decaying)
across the samples means --bench_soak_s is too short for this machine.

Usage:
    python scripts/check_bench_drift.py <run_dir>            # dir holding rank0.log
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import statistics
from datetime import datetime

RESULTS_RE = re.compile(r"Runtime benchmark results:\s*(\[.*\])\s*$", re.MULTILINE)
NP_SCALAR_RE = re.compile(r"np\.(?:float|int)\d*\(")
PHASES = ("prefill", "decode", "e2e")


def parse_results(log_path):
    with open(log_path, "r", errors="replace") as f:
        text = f.read()
    matches = RESULTS_RE.findall(text)
    if not matches:
        raise SystemExit(f"no benchmark results in {log_path}")
    return ast.literal_eval(NP_SCALAR_RE.sub("(", matches[-1]))


def parse_telemetry(path):
    """[(perf_counter_s, sm_mhz, temp_c, watts, throttle)] or [] if absent.

    nvidia-smi stamps wall-clock time; the samples are stamped with
    perf_counter. The header records both at the same instant, so each row maps
    over exactly rather than by assuming an even --loop-ms cadence.
    """
    if not os.path.isfile(path):
        return []
    rows, perf0, wall0 = [], None, None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(r"perf_counter at start:\s*([\d.]+)", line)
                if m:
                    perf0 = float(m.group(1))
                m = re.search(r"time at start:\s*([\d.]+)", line)
                if m:
                    wall0 = float(m.group(1))
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                stamp = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                sm = float(parts[1].split()[0])
                temp = float(parts[3])
                watt = float(parts[4].split()[0])
            except (ValueError, IndexError):
                continue
            rows.append((stamp, sm, temp, watt, parts[5]))
    if perf0 is None or not rows:
        return []
    if wall0 is None:
        # logs written before the header carried both clocks: anchor on the
        # first row instead, which costs only the sampler's startup latency
        wall0 = rows[0][0]
    return [(perf0 + (stamp - wall0), sm, temp, w, thr)
            for stamp, sm, temp, w, thr in rows]


def nearest(telemetry, t):
    if not telemetry:
        return None
    return min(telemetry, key=lambda r: abs(r[0] - t))


def report(run_dir):
    log = os.path.join(run_dir, "rank0.log")
    telemetry = parse_telemetry(os.path.join(run_dir, "gpu_telemetry.csv"))
    if not telemetry:
        print("!! no gpu_telemetry.csv -- rerun with --bench_log_clocks for the clock column")

    for res in parse_results(log):
        print(f"\n=== bsz={res['bsz']} prefill_len={res['prefill_len']} "
              f"decode_steps={res['decode_steps']}")
        for phase in PHASES:
            raw = res.get(f"{phase}_ms_raw")
            starts = res.get(f"{phase}_start_s")
            if not raw:
                continue
            med = statistics.median(raw)
            spread = (max(raw) / min(raw) - 1) * 100
            trend = (raw[-1] / raw[0] - 1) * 100
            print(f"\n  {phase}: median {med:.2f} ms  spread {spread:.2f}%  "
                  f"first->last {trend:+.2f}%")
            if not starts:
                print("    (no *_start_s in this log -- predates the soak change)")
                continue
            print(f"    {'#':>2} {'ms':>9} {'dev%':>7} {'SM MHz':>7} {'°C':>4} {'W':>6}  throttle")
            for i, (ms, t) in enumerate(zip(raw, starts)):
                tel = nearest(telemetry, t)
                if tel:
                    _, sm, temp, watt, thr = tel
                    print(f"    {i:>2} {ms:9.2f} {(ms/med-1)*100:+7.2f} {sm:7.0f} "
                          f"{temp:4.0f} {watt:6.1f}  {thr}")
                else:
                    print(f"    {i:>2} {ms:9.2f} {(ms/med-1)*100:+7.2f}")
            clocks = [nearest(telemetry, t)[1] for t in starts if nearest(telemetry, t)]
            if clocks:
                lo, hi = min(clocks), max(clocks)
                verdict = "FLAT" if hi - lo <= 15 else "STILL DRIFTING -- raise --bench_soak_s"
                print(f"    clock across samples: {lo:.0f}-{hi:.0f} MHz  -> {verdict}")


def find_runs(root):
    """[(config, round, run_dir)] for an experiments/<model>/<tag> tree."""
    out = []
    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        if not os.path.isdir(d) or entry == "qmodel":
            continue
        m = re.fullmatch(r"(.+)_r(\d+)", entry)
        config, rnd = (m.group(1), int(m.group(2))) if m else (entry, 1)
        for ts in sorted(os.listdir(d)):
            if os.path.isfile(os.path.join(d, ts, "rank0.log")):
                out.append((config, rnd, os.path.join(d, ts)))
    return out


def sweep(root):
    """One line per run: phase medians next to the clock they were measured at.

    A soak equalises temperature but cannot pin the boost bin, so a run that
    landed a bin lower reads slower across every phase at once. That signature
    is what separates residual clock drift from a real per-config difference.
    """
    print(f"{'config':<14} {'r':>2} {'prefill':>9} {'MHz':>5} {'decode':>10} {'MHz':>5} "
          f"{'e2e':>10} {'MHz':>5}")
    rows = []
    for config, rnd, run_dir in find_runs(root):
        telemetry = parse_telemetry(os.path.join(run_dir, "gpu_telemetry.csv"))
        try:
            results = parse_results(os.path.join(run_dir, "rank0.log"))
        except SystemExit:
            print(f"{config:<14} {rnd:>2}  (no results)")
            continue
        for res in results:
            cells, rec = [], {"config": config, "round": rnd}
            for phase in PHASES:
                raw = res.get(f"{phase}_ms_raw")
                starts = res.get(f"{phase}_start_s") or []
                if not raw:
                    cells += ["-", "-"]
                    continue
                med = statistics.median(raw)
                clocks = [nearest(telemetry, t)[1] for t in starts
                          if nearest(telemetry, t)]
                mhz = statistics.median(clocks) if clocks else None
                rec[phase] = med
                rec[f"{phase}_mhz"] = mhz
                cells += [f"{med:.2f}", f"{mhz:.0f}" if mhz else "-"]
            print(f"{config:<14} {rnd:>2} {cells[0]:>9} {cells[1]:>5} {cells[2]:>10} "
                  f"{cells[3]:>5} {cells[4]:>10} {cells[5]:>5}")
            rows.append(rec)

    print("\nround-to-round, per config:")
    for config in sorted({r["config"] for r in rows}):
        rs = sorted((r for r in rows if r["config"] == config), key=lambda r: r["round"])
        if len(rs) < 2:
            continue
        for phase in PHASES:
            a, b = rs[0].get(phase), rs[-1].get(phase)
            if a and b:
                ca, cb = rs[0].get(f"{phase}_mhz"), rs[-1].get(f"{phase}_mhz")
                clk = f"  ({ca:.0f} -> {cb:.0f} MHz)" if ca and cb else ""
                print(f"  {config:<14} {phase:<8} {(b/a-1)*100:+6.2f}%{clk}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a run dir (rank0.log + gpu_telemetry.csv), or a "
                                 "tag dir holding <config>_r<round>/ subdirs with --sweep")
    ap.add_argument("--sweep", action="store_true",
                    help="summarise every run under a tag instead of dumping one run")
    args = ap.parse_args()
    if args.sweep:
        sweep(args.path)
    else:
        report(args.path)


if __name__ == "__main__":
    main()
