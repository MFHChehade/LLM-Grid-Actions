#!/usr/bin/env python3
"""
eval_nn_voltage_and_j.py

Evaluate a trained NN policy the SAME way you evaluate LLMs in eval_voltage_and_j.py:
  - generate N candidate switching plans
  - score each with verify_plan_hybrid (voltage penalty V_pen == J_ac)
  - select best plan by MIN voltage penalty
  - then evaluate chosen plan with verify_plan (DC-OPF cost J + shed_MW)
  - also compute per-xi ground truth:
      run_ground_truth mode='mip_opt' and mode='baseline'

Reuses SAME SFT test index (for the chosen run):
  io/sft_test_index(_runX).json

Uses NN checkpoint:
  io/nn_model(_runX).pt

Outputs (io/):
  - eval_nn_vselect_rows.csv/json           (tall per-xi rows)
  - eval_nn_vselect_per_xi.csv             (wide per-xi summary)
  - eval_nn_vselect_summary.json           (aggregate stats)

Candidate generation (mirrors "temps"):
  - Get NN logits over lines.
  - Restrict to eligible lines (xi == 1).
  - Take top_L highest-scoring eligible lines.
  - For each temperature tau in TEMPS, sample without replacement K lines
    using softmax(logits / tau) over those top_L lines.
  - Total candidates = N_CANDS (default 3), split across temps like your LLM script.

Run from repo root:
  python python/eval_nn_voltage_and_j.py --run auto
  python python/eval_nn_voltage_and_j.py --run run0 --device cuda
  python python/eval_nn_voltage_and_j.py --run run0 --N_CANDS 3 --TEMPS "0.1,0.3,0.7" --top_L 30

Env (optional):
  CASE_NAME=case118
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import yaml
except Exception:
    yaml = None

ROOT       = Path(__file__).resolve().parents[1]
MATLAB_DIR = ROOT / "matlab"
CONFIG     = ROOT / "config"
IO_DIR     = ROOT / "io"
IO_DIR.mkdir(exist_ok=True)

CASE = os.getenv("CASE_NAME", "case118")

def matlab_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"

def _run_matlab_batch(cmd: str) -> int:
    """
    Runs: matlab -batch "<cmd>"
    where cmd should include addpath(...) and a function call.
    Using subprocess avoids Windows quoting issues.
    """
    proc = subprocess.run(["matlab", "-batch", cmd],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          text=True)
    if proc.returncode != 0:
        # print MATLAB output for debugging
        sys.stderr.write(proc.stdout + "\n")
    return proc.returncode

def call_make_summary(xi_path: Path, limits_yml: Path, out_summary: Path) -> Dict | None:
    if out_summary.exists():
        try:
            return json.loads(out_summary.read_text(encoding="utf-8"))
        except Exception:
            pass
    cmd = (
        f"addpath({matlab_quote(MATLAB_DIR.as_posix())});"
        f"make_summary({matlab_quote(CASE)},{matlab_quote(xi_path.as_posix())},"
        f"{matlab_quote(limits_yml.as_posix())},{matlab_quote(out_summary.as_posix())});"
    )
    rc = _run_matlab_batch(cmd)
    if rc != 0 or not out_summary.exists():
        return None
    try:
        return json.loads(out_summary.read_text(encoding="utf-8"))
    except Exception:
        return None

def call_verify_hybrid(plan_path: Path, xi_path: Path, limits_yml: Path, out_path: Path) -> Dict | None:
    cmd = (
        f"addpath({matlab_quote(MATLAB_DIR.as_posix())});"
        f"verify_plan_hybrid({matlab_quote(CASE)},{matlab_quote(plan_path.as_posix())},"
        f"{matlab_quote(xi_path.as_posix())},{matlab_quote(limits_yml.as_posix())},"
        f"{matlab_quote(out_path.as_posix())});"
    )
    rc = _run_matlab_batch(cmd)
    if rc != 0 or not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def call_verify_dcopf(plan_path: Path, xi_path: Path, limits_yml: Path, out_path: Path) -> Dict | None:
    cmd = (
        f"addpath({matlab_quote(MATLAB_DIR.as_posix())});"
        f"verify_plan({matlab_quote(CASE)},{matlab_quote(plan_path.as_posix())},"
        f"{matlab_quote(xi_path.as_posix())},{matlab_quote(limits_yml.as_posix())},"
        f"{matlab_quote(out_path.as_posix())});"
    )
    rc = _run_matlab_batch(cmd)
    if rc != 0 or not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def call_run_ground_truth(xi_path: Path, limits_yml: Path, out_path: Path, mode: str) -> Dict | None:
    cmd = (
        f"addpath({matlab_quote(MATLAB_DIR.as_posix())});"
        f"run_ground_truth({matlab_quote(CASE)},{matlab_quote(xi_path.as_posix())},"
        f"{matlab_quote(limits_yml.as_posix())},{matlab_quote(out_path.as_posix())},"
        f"{matlab_quote(mode)});"
    )
    rc = _run_matlab_batch(cmd)
    if rc != 0 or not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def v_eff(sc: Dict) -> float:
    """
    Effective voltage penalty:
      prefer V_pen if finite else J_ac if finite else inf
    """
    v = sc.get("V_pen")
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    j = sc.get("J_ac", float("inf"))
    if isinstance(j, (int, float)) and math.isfinite(j):
        return float(j)
    return float("inf")

def is_dc_ok(sc: Dict) -> bool:
    return bool(sc.get("feasible"))

# ---------------- NN model (same as your training script) ----------------

class MLPLineScorer(nn.Module):
    def __init__(self, n_line: int, hidden: int = 512, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_line, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_line),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_nn(ckpt_path: Path, device: str) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    n_line = int(ckpt["n_line"])
    hidden = int(ckpt.get("hidden", 512))
    dropout = float(ckpt.get("dropout", 0.10))
    model = MLPLineScorer(n_line=n_line, hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(x) / len(x)

def sample_k_without_replacement(items: List[int], probs: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    if k <= 0:
        return []
    k = min(k, len(items))
    # numpy choice without replacement
    chosen = rng.choice(np.array(items), size=k, replace=False, p=probs)
    return [int(x) for x in chosen.tolist()]

def build_plan_from_ids(line_ids_1based: List[int]) -> Dict:
    return {"corridor_actions": [{"action": "open", "line": int(lid)} for lid in line_ids_1based]}

def plan_to_text(line_ids_1based: List[int]) -> str:
    return "; ".join([f"open({int(lid)})" for lid in line_ids_1based])

def find_latest_run(io_dir: Path) -> str:
    cands = sorted(io_dir.glob("nn_model_run*.pt"))
    best = -1
    for p in cands:
        m = re.search(r"run(\d+)", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return f"run{best}" if best >= 0 else ""

def try_read_budget(repo_root: Path) -> int:
    limits = repo_root / "config" / "limits.yml"
    if yaml is None or not limits.exists():
        return 3
    obj = yaml.safe_load(limits.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "budget" in obj:
        try:
            return int(obj["budget"])
        except Exception:
            return 3
    return 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="auto", help="auto | run0 | run1 | ...")
    ap.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    ap.add_argument("--N_CANDS", type=int, default=3)
    ap.add_argument("--TEMPS", type=str, default="0.1,0.3,0.7")
    ap.add_argument("--top_L", type=int, default=30, help="sample candidates from top-L eligible lines")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    run_tag = args.run
    # Treat 'none'/'default' as the non-run (no suffix) variant.
    if run_tag.lower() in {"none", "default", "norun", "no_run"}:
        run_tag = ""

    if run_tag == "auto":
        run_tag = find_latest_run(IO_DIR)
        if not run_tag:
            print("ERROR: could not infer run (no nn_model_run*.pt found)", file=sys.stderr)
            sys.exit(1)

    suffix = f"_{run_tag}" if run_tag else ""
    ckpt_path = IO_DIR / f"nn_model{suffix}.pt"
    if not ckpt_path.exists():
        print(f"ERROR: missing checkpoint: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # Use SAME SFT test index for this run
    index_path = IO_DIR / f"sft_test_index{suffix}.json"
    if not index_path.exists():
        # fall back to non-run file
        index_path = IO_DIR / "sft_test_index.json"
    if not index_path.exists():
        print(f"ERROR: missing SFT test index: {index_path}", file=sys.stderr)
        sys.exit(1)

    limits_yml = CONFIG / "limits.yml"
    if not limits_yml.exists():
        print(f"ERROR: missing limits.yml: {limits_yml}", file=sys.stderr)
        sys.exit(1)

    # Load xi paths
    xi_paths_raw = json.loads(index_path.read_text(encoding="utf-8"))
    xi_paths = [Path(p) for p in xi_paths_raw]
    xi_paths = [p for p in xi_paths if p.exists()]
    if not xi_paths:
        print("No xi paths found after filtering.", file=sys.stderr)
        sys.exit(1)

    # NN
    model, ckpt = load_nn(ckpt_path, device=device)
    n_line = int(ckpt["n_line"])

    # candidate settings
    temps = [float(t) for t in args.TEMPS.split(",") if t.strip()]
    N_CANDS = int(args.N_CANDS)
    top_L = int(args.top_L)

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # caches for GT
    gt_opt_J: Dict[str, float] = {}
    gt_base_J: Dict[str, float] = {}
    gt_opt_shed: Dict[str, float] = {}
    gt_base_shed: Dict[str, float] = {}

    rows: List[Dict[str, Any]] = []

    for xi_path in xi_paths:
        xi_key = xi_path.as_posix()

        # ground truth
        if xi_key not in gt_opt_J:
            out = IO_DIR / f"gt_mip_opt_{xi_path.stem}.json"
            res = call_run_ground_truth(xi_path, limits_yml, out, mode="mip_opt")
            if res and isinstance(res.get("J"), (int,float)) and math.isfinite(res["J"]):
                gt_opt_J[xi_key] = float(res["J"])
                sv = res.get("shed_MW")
                gt_opt_shed[xi_key] = float(sv) if isinstance(sv,(int,float)) and math.isfinite(sv) else float("nan")
            else:
                gt_opt_J[xi_key] = float("nan")
                gt_opt_shed[xi_key] = float("nan")

        if xi_key not in gt_base_J:
            out = IO_DIR / f"gt_baseline_{xi_path.stem}.json"
            res = call_run_ground_truth(xi_path, limits_yml, out, mode="baseline")
            if res and isinstance(res.get("J"), (int,float)) and math.isfinite(res["J"]):
                gt_base_J[xi_key] = float(res["J"])
                sv = res.get("shed_MW")
                gt_base_shed[xi_key] = float(sv) if isinstance(sv,(int,float)) and math.isfinite(sv) else float("nan")
            else:
                gt_base_J[xi_key] = float("nan")
                gt_base_shed[xi_key] = float("nan")

        # summary (for toggle_budget)
        summ_path = IO_DIR / f"summary_{xi_path.stem}.json"
        summ = call_make_summary(xi_path, limits_yml, summ_path)
        toggle = int(summ.get("toggle_budget", try_read_budget(ROOT))) if summ else try_read_budget(ROOT)

        # load xi vector
        xi = json.loads(xi_path.read_text(encoding="utf-8"))
        xi = np.asarray(xi, dtype=np.float32).reshape(-1)
        if xi.size != n_line:
            print(f"[warn] {xi_path.name}: xi length {xi.size} != n_line {n_line}, skipping")
            continue

        # NN scores
        with torch.no_grad():
            x = torch.tensor(xi, dtype=torch.float32, device=device).unsqueeze(0)  # (1,n_line)
            logits = model(x).squeeze(0).detach().cpu().numpy()  # (n_line,)

        eligible_idx = np.where(xi > 0.5)[0].tolist()
        if not eligible_idx:
            continue

        # take top-L eligible
        eligible_scores = [(i, float(logits[i])) for i in eligible_idx]
        eligible_scores.sort(key=lambda t: t[1], reverse=True)
        top = [i for (i, _) in eligible_scores[:min(top_L, len(eligible_scores))]]

        # Build N_CANDS candidates split across temps (total N_CANDS)
        per = max(1, N_CANDS // len(temps))
        leftover = N_CANDS - per * len(temps)

        cand_sets: List[List[int]] = []
        for ti, tau in enumerate(temps):
            k_this = per + (1 if ti < leftover else 0)
            if k_this <= 0:
                continue
            # probabilities over top list
            z = np.array([logits[i] for i in top], dtype=np.float64) / max(1e-6, float(tau))
            probs = softmax_np(z)
            for _ in range(k_this):
                chosen0 = sample_k_without_replacement(top, probs, toggle, rng=rng)  # 0-based indices
                chosen1 = [c + 1 for c in chosen0]  # convert to 1-based line ids for MATLAB
                cand_sets.append(sorted(set(chosen1)))

        # score candidates by verify_plan_hybrid, pick best (min V penalty)
        best_ids = None
        best_sc = None
        best_v = float("inf")

        for cids in cand_sets:
            plan = build_plan_from_ids(cids)
            plan_path = IO_DIR / f"nn_eval_plan_{xi_path.stem}_{random.randint(0,999999)}.json"
            out_path  = IO_DIR / f"nn_eval_hybrid_{xi_path.stem}_{random.randint(0,999999)}.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            sc = call_verify_hybrid(plan_path, xi_path, limits_yml, out_path)

            try:
                plan_path.unlink(missing_ok=True)
                out_path.unlink(missing_ok=True)
            except Exception:
                pass

            if sc is None or (not is_dc_ok(sc)):
                continue
            v = v_eff(sc)
            if v < best_v:
                best_v = v
                best_sc = sc
                best_ids = cids

        if best_ids is None:
            # fallback: deterministic top-k (no sampling)
            best_ids = [i + 1 for i in top[:min(toggle, len(top))]]
            best_sc = {"feasible": False, "V_pen": float("inf"), "J_ac": float("inf")}

        # evaluate chosen plan with DC-OPF verifier
        plan = build_plan_from_ids(best_ids)
        plan_path = IO_DIR / f"nn_eval_plan_dc_{xi_path.stem}_{random.randint(0,999999)}.json"
        out_path  = IO_DIR / f"nn_eval_dc_{xi_path.stem}_{random.randint(0,999999)}.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        dc_sc = call_verify_dcopf(plan_path, xi_path, limits_yml, out_path)

        try:
            plan_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
        except Exception:
            pass

        if dc_sc and isinstance(dc_sc.get("J"), (int,float)) and math.isfinite(dc_sc["J"]):
            j_dc = float(dc_sc["J"])
            shed = dc_sc.get("shed_MW")
            shed_mw = float(shed) if isinstance(shed,(int,float)) and math.isfinite(shed) else float("nan")
        else:
            j_dc = float("nan")
            shed_mw = float("nan")

        rows.append({
            "xi": xi_key,
            "model": "nn",
            "V_pen_effective": float(best_v),
            "J_ac": float(best_v),
            "J_dc": j_dc,
            "J_dc_gt": gt_opt_J[xi_key],
            "J_dc_base": gt_base_J[xi_key],
            "shed_MW": shed_mw,
            "shed_MW_gt": gt_opt_shed[xi_key],
            "shed_MW_base": gt_base_shed[xi_key],
            "best_plan_text": plan_to_text(best_ids),
        })

    # Save
    out_json = IO_DIR / f"eval_nn_vselect_rows{suffix}.json"
    out_csv  = IO_DIR / f"eval_nn_vselect_rows{suffix}.csv"
    out_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows produced.")
        return

    # Wide per-xi
    wide = df[["xi","V_pen_effective","J_dc","J_ac","shed_MW","J_dc_gt","J_dc_base","shed_MW_gt","shed_MW_base","best_plan_text"]]
    wide.to_csv(IO_DIR / f"eval_nn_vselect_per_xi{suffix}.csv", index=False)

    # Simple summary stats
    def _mean(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return float(x.mean()) if len(x) else float("nan")

    summary = {
        "run_tag": run_tag,
        "checkpoint": str(ckpt_path),
        "index_file": str(index_path),
        "N_rows": int(len(df)),
        "V_mean": _mean(df["V_pen_effective"]),
        "J_dc_mean": _mean(df["J_dc"]),
        "shed_MW_mean": _mean(df["shed_MW"]),
        "J_dc_gt_mean": _mean(df["J_dc_gt"]),
        "J_dc_base_mean": _mean(df["J_dc_base"]),
        "outputs": {
            "rows_csv": str(out_csv),
            "rows_json": str(out_json),
            "per_xi_csv": str(IO_DIR / f"eval_nn_vselect_per_xi{suffix}.csv"),
        }
    }
    (IO_DIR / f"eval_nn_vselect_summary{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:",
          out_csv.as_posix(),
          (IO_DIR / f"eval_nn_vselect_per_xi{suffix}.csv").as_posix(),
          (IO_DIR / f"eval_nn_vselect_summary{suffix}.json").as_posix())

if __name__ == "__main__":
    main()
