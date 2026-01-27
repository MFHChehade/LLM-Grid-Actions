#!/usr/bin/env python
"""
psps_llm_pipeline.py

End-to-end PSPS LLM pipeline with:
  - SFT corpus build
  - SFT packing (train/test + OpenAI chat format)
  - SFT fine-tuning (OpenAI)
  - DPO pair generation (voltage-only)
  - DPO fine-tuning (OpenAI)
  - Evaluation with best-of-N selection by VOLTAGE (J_ac)
  - Multi-run aggregation (default 3 runs)
  - New metrics:
      * J_ac interpreted as L2-squared voltage in *raw volts*
      * RMSE per bus: sqrt(J_ac / (V_SCALE^2 * N_BUS))
      * J_ac_gt and RMSE_gt for ground truth
      * AC/DC penalties: 1e6 for AC infeasible, 5e4 for DC infeasible
      * J_dc gap = J_dc_method - J_dc_gt
  - Plots with confidence “shades” for CDFs where possible.
"""

import os
import sys
import json
import time
import math
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI

# -----------------------------------------------------------------------------
# Global paths / config
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MATLAB_DIR = ROOT / "matlab"
CONFIG = ROOT / "config"
IO_DIR = ROOT / "io"
IO_DIR.mkdir(exist_ok=True)

CASE = os.getenv("CASE_NAME", "case118")
LIMITS_YML = (CONFIG / "limits.yml").as_posix()
CORRIDORS_F = CONFIG / "corridor_map.json"
MATPOWER_DIR = os.getenv("MATPOWER_DIR", "")

# OpenAI config
API_KEY = os.getenv("OPENAI_API_KEY") or ""
if not API_KEY:
    print("WARNING: OPENAI_API_KEY is not set. Training/inference steps will fail.", file=sys.stderr)

BASE_MODEL = os.getenv("BASE_MODEL", "gpt-4.1-mini-2025-04-14")

# Multi-run configuration
N_RUNS = int(os.getenv("N_RUNS", "3"))
MASTER_SEED = int(os.getenv("MASTER_SEED", "123"))

# Scene sampling config (shared)
N_SCENES = int(os.getenv("N_SCENES", "200"))
K_OPEN_PER_SCEN = int(os.getenv("K_OPEN_PER_SCEN", "2"))
USE_EXISTING_PSPS = bool(int(os.getenv("USE_EXISTING_PSPS", "0")))
XI_OVERRIDE = os.getenv("XI", "")

# SFT splitting config (shared)
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.20"))  # 20% default
SPLIT_SEED_BASE = int(os.getenv("SPLIT_SEED", "13"))
KEEP_EMPTY_TARGETS = bool(int(os.getenv("KEEP_EMPTY_TARGETS", "1")))

# Voltage / cost penalties
AC_PENALTY = float(os.getenv("AC_PENALTY", "1e6"))   # AC infeasibility penalty for J_ac
DC_PENALTY = float(os.getenv("DC_PENALTY", "5e4"))   # DC infeasibility penalty for J_dc

# Voltage scaling for RMSE (raw volts -> per-unit if you want)
V_SCALE = float(os.getenv("VOLT_SCALE", "1.0"))      # if your J_ac was in raw volts, this is typically 1.0

# Candidate sampling config for eval
EVAL_N_CANDS = int(os.getenv("N_CANDS", "3"))        # forced to 3 later, per your original design
EVAL_TEMPS = [float(t) for t in os.getenv("TEMPS", "0.1,0.3,0.7").split(",")]
EVAL_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "120"))

# DPO config
DPO_N_EPOCHS = int(os.getenv("DPO_EPOCHS", "2"))
DPO_BETA = float(os.getenv("DPO_BETA", "0.1"))
DPO_BATCH_SIZE = int(os.getenv("DPO_BATCH_SIZE", "8"))

# SFT suffix base for OpenAI jobs
FT_SUFFIX_BASE = os.getenv("FT_SUFFIX", "psps-sft").strip() or "psps-sft"
DPO_SUFFIX_BASE = os.getenv("DPO_SUFFIX", "psps-dpo").strip() or "psps-dpo"

# System prompts (SFT & inference)
SYSTEM_MSG_SFT = (
    "You are a grid switching assistant. "
    "Given a JSON summary of a PSPS scenario, reply with a single line containing only OPEN actions. "
    "Grammar:\n"
    "- open(Sk:LINE) when the corridor Sk is known (e.g., open(S6:135))\n"
    "- open(LINE) when the corridor is unknown (e.g., open(131))\n"
    "Use at most the toggle_budget actions. No other text."
)

SYSTEM_MSG_INFER = (
    "You are a grid switching assistant.\n"
    "Input: a JSON summary of a PSPS scenario.\n"
    "Output: one line with at most 'toggle_budget' OPEN actions (no other text).\n"
    "Grammar:\n"
    "  - open(Sk:LINE)   e.g., open(S6:135)\n"
    "  - open(LINE)      e.g., open(131)\n"
    "Separate multiple actions with semicolons. No explanations."
)

# Regex for parsing open(...) tokens
_OPEN_RE = re.compile(r"open\s*\(\s*(?:([sS]\d+)\s*:\s*)?([0-9]+)\s*\)", re.IGNORECASE)

# -----------------------------------------------------------------------------
# MATLAB helpers
# -----------------------------------------------------------------------------

def _run_matlab_batch(batch: str) -> int:
    """Run a MATLAB -batch command with matlab helpers + MATPOWER on the path."""
    parts = [f"addpath('{MATLAB_DIR.as_posix()}');"]
    if MATPOWER_DIR:
        mp = MATPOWER_DIR.replace("\\", "/")
        parts.append(f"addpath(genpath('{mp}'));")
    prefix = " ".join(parts)
    cmd = f"matlab -batch \"{prefix} {batch}; exit;\""
    return os.system(cmd)


def get_case_counts(case_name: str) -> Dict[str, int]:
    """Return {'buses': ..., 'lines': ...} for the MATPOWER case."""
    out = IO_DIR / f"counts_{case_name}.json"
    if out.exists():
        try:
            return json.loads(out.read_text())
        except Exception:
            pass
    batch = (
        f"mpc=rundcpf('{case_name}'); "
        f"s=struct('case','{case_name}','buses',size(mpc.bus,1),'lines',size(mpc.branch,1)); "
        f"fid=fopen('{out.as_posix()}','w'); fwrite(fid,jsonencode(s)); fclose(fid)"
    )
    rc = _run_matlab_batch(batch)
    if rc != 0:
        raise RuntimeError("MATLAB failed while computing case counts.")
    return json.loads(out.read_text())


def call_make_summary(xi_path: Path, out_summary: Path) -> Optional[Dict[str, Any]]:
    if out_summary.exists():
        try:
            return json.loads(out_summary.read_text(encoding="utf-8"))
        except Exception:
            pass
    rc = _run_matlab_batch(
        f"make_summary('{CASE}','{xi_path.as_posix()}','{LIMITS_YML}','{out_summary.as_posix()}')"
    )
    if rc != 0 or not out_summary.exists():
        print(f"[make_summary FAIL] rc={rc} out_exists={out_summary.exists()} xi={xi_path}")
        return None
    try:
        return json.loads(out_summary.read_text(encoding="utf-8"))
    except Exception:
        return None


def call_run_ground_truth(xi_path: Path, out_path: Path, mode: str = "mip_opt") -> Optional[Dict[str, Any]]:
    """
    Ground-truth DC-OPF:
      mode = 'baseline' : PSPS only, no extra toggles
      mode = 'mip_opt'  : optimal open-only switching up to budget
    """
    rc = _run_matlab_batch(
        f"run_ground_truth('{CASE}','{xi_path.as_posix()}','{LIMITS_YML}','{out_path.as_posix()}','{mode}')"
    )
    if rc != 0 or not out_path.exists():
        print(f"[run_ground_truth FAIL] rc={rc} out_exists={out_path.exists()} xi={xi_path} mode={mode}")
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def call_verify_hybrid(plan_path: Path, xi_path: Path, out_path: Path) -> Optional[Dict[str, Any]]:
    """AC/DCPF hybrid verifier: returns V_pen, J_ac, feasible flags, etc."""
    rc = _run_matlab_batch(
        f"verify_plan_hybrid('{CASE}','{plan_path.as_posix()}','{xi_path.as_posix()}',"
        f"'{LIMITS_YML}','{out_path.as_posix()}')"
    )
    if rc != 0 or not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def call_verify_dcopf(plan_path: Path, xi_path: Path, out_path: Path) -> Optional[Dict[str, Any]]:
    """DC-OPF verifier: returns J (used as J_dc) and shed_MW, 'feasible' etc."""
    rc = _run_matlab_batch(
        f"verify_plan('{CASE}','{plan_path.as_posix()}','{xi_path.as_posix()}',"
        f"'{LIMITS_YML}','{out_path.as_posix()}')"
    )
    if rc != 0 or not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Corridor / PSPS helpers
# -----------------------------------------------------------------------------

def load_corridor_map() -> Dict[str, List[int]]:
    with open(CORRIDORS_F, "r") as f:
        return json.load(f)


def invert_corridors(corrmap: Dict[str, List[int]]) -> Dict[int, str]:
    inv: Dict[int, str] = {}
    for name, ids in corrmap.items():
        for i in ids:
            inv[int(i)] = name
    return inv


def list_existing_psps(io_dir: Path) -> List[Path]:
    files = list(io_dir.glob("psps_*.json"))
    return sorted(files, key=lambda p: p.stat().st_mtime)


def sample_psps_mask(n_line: int, corrmap: Dict[str, List[int]], k_open: int, rng: random.Random) -> List[int]:
    """Create a PSPS mask where all forced-open lines come from a single corridor."""
    xi = [1] * n_line
    eligible = [
        (name, [int(i) for i in ids])
        for name, ids in corrmap.items()
        if len(ids) >= k_open
    ]
    if not eligible:
        raise ValueError(f"No corridor has at least {k_open} lines for sampling.")
    corr_name, line_ids = rng.choice(eligible)
    chosen = rng.sample(line_ids, k_open)
    for idx in chosen:
        if 1 <= idx <= n_line:
            xi[idx - 1] = 0
        else:
            raise ValueError(f"Line index {idx} out of range 1..{n_line}")
    return xi


def save_json(path: Path, obj: Any):
    with open(path, "w") as f:
        json.dump(obj, f)


# -----------------------------------------------------------------------------
# Action formatting / parsing
# -----------------------------------------------------------------------------

def target_from_mip_switches(toggles: List[int], line2corr: Dict[int, str]) -> str:
    """
    Convert MILP-optimal open-switch line IDs into normalized open-only actions.
    Prefer corridor names if known; otherwise use L<id> fallback.
    """
    parts: List[str] = []
    for lid in sorted(int(x) for x in toggles):
        cname = line2corr.get(lid)
        if cname:
            parts.append(f"open({cname}:{lid})")
        else:
            parts.append(f"open({lid})")
    return "; ".join(parts) if parts else "do_nothing"


def _clamp_line(lid: int, n_line: int) -> int:
    return max(1, min(int(lid), n_line))


def parse_actions_to_plan(text: str, toggle_budget: int, n_line: int) -> Dict[str, Any]:
    """Parse LLM text into corridor_actions plan."""
    acts, seen = [], set()
    for (maybe_name, line_id) in _OPEN_RE.findall(text or ""):
        lid = _clamp_line(int(line_id), n_line)
        name = (maybe_name.upper() if maybe_name else "")
        key = (name, lid)
        if key in seen:
            continue
        seen.add(key)
        item = {"action": "open", "line": lid}
        if name:
            item["name"] = name
        acts.append(item)
        if len(acts) >= max(1, int(toggle_budget)):
            break
    return {"corridor_actions": acts}


def plan_to_text(plan: Dict[str, Any]) -> str:
    parts = []
    for a in plan.get("corridor_actions", []):
        lid = int(a["line"])
        name = (a.get("name") or "").strip()
        parts.append(f"open({name+':' if name else ''}{lid})")
    return "; ".join(parts)


def toggles_to_plan(toggles: List[int], line2corr: Dict[int, str]) -> Dict[str, Any]:
    acts = []
    for lid in sorted(int(t) for t in toggles):
        name = line2corr.get(lid)
        item = {"action": "open", "line": lid}
        if name:
            item["name"] = name
        acts.append(item)
    return {"corridor_actions": acts}


# -----------------------------------------------------------------------------
# AC/DC metrics normalization + RMSE
# -----------------------------------------------------------------------------

def normalize_ac_from_hybrid(sc: Optional[Dict[str, Any]]) -> Tuple[float, bool]:
    """
    Normalize AC metrics from verify_plan_hybrid.
    Returns (J_ac, ac_ok) with penalties (AC_PENALTY) if infeasible/missing.
    """
    if sc is None:
        return AC_PENALTY, False

    # ac_ok flag if present
    ac_ok = bool(sc.get("ac_ok")) if "ac_ok" in sc else None
    vpen = sc.get("V_pen")
    jac = sc.get("J_ac")

    val = None
    if isinstance(vpen, (int, float)) and math.isfinite(vpen):
        val = float(vpen)
    elif isinstance(jac, (int, float)) and math.isfinite(jac):
        val = float(jac)

    if val is None:
        # no usable AC metric
        return AC_PENALTY, False

    if ac_ok is False:
        return AC_PENALTY, False

    # treat finite val as J_ac
    return float(val), True


def normalize_jdc_from_dcopf(sc: Optional[Dict[str, Any]]) -> Tuple[float, bool]:
    """
    Normalize DC cost J from verify_plan / run_ground_truth.
    Returns (J_dc, feasible_dc) with penalty DC_PENALTY if infeasible/missing.
    """
    if sc is None:
        return DC_PENALTY, False

    feasible = bool(sc.get("feasible")) if "feasible" in sc else True
    J = sc.get("J")
    if feasible and isinstance(J, (int, float)) and math.isfinite(J):
        return float(J), True
    return DC_PENALTY, False


def compute_rmse_from_Jac(J_ac: float, n_bus: int, v_scale: float = 1.0) -> float:
    """
    Interpret J_ac as sum of squared voltage deviations in raw volts.
    Return RMSE per bus in volts: sqrt(J_ac / (N_BUS * V_SCALE^2)).
    """
    if not math.isfinite(J_ac):
        return float("nan")
    if J_ac <= 0 or n_bus <= 0:
        return 0.0
    return math.sqrt(J_ac / (n_bus * (v_scale ** 2)))


# -----------------------------------------------------------------------------
# OpenAI client helpers
# -----------------------------------------------------------------------------

def make_client() -> OpenAI:
    return OpenAI(api_key=API_KEY)


def sample_chat_batch(
    client: OpenAI,
    engine: str,
    prompt: str,
    n: int,
    temp: float,
    max_tokens: int,
) -> List[str]:
    if n <= 0:
        return []
    r = client.chat.completions.create(
        model=engine,
        messages=[
            {"role": "system", "content": SYSTEM_MSG_INFER},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
        max_tokens=max_tokens,
        n=n,
    )
    return [(c.message.content or "").strip() for c in r.choices]


# -----------------------------------------------------------------------------
# Part 1: SFT data build (per run)
# -----------------------------------------------------------------------------

def build_sft_from_gt_for_run(run_id: int, seed: int, n_bus: int, n_line: int,
                              corrmap: Dict[str, List[int]], line2corr: Dict[int, str]) -> None:
    """
    Build raw SFT corpus from MILP ground truth for a given run.
    Outputs: io/sft_raw_run{run_id}.jsonl and psps_*.json, summary_*, gt_mip_*.
    """
    print(f"[RUN {run_id}] Building SFT raw corpus with seed={seed}")
    rng = random.Random(seed)

    scenarios: List[Path] = []

    if XI_OVERRIDE:
        p = Path(XI_OVERRIDE)
        if not p.exists():
            raise FileNotFoundError(f"XI={p} does not exist")
        if not (p.name.startswith("psps_") and p.suffix == ".json"):
            raise ValueError("XI must point to a psps_*.json file")
        scenarios = [p]
    elif USE_EXISTING_PSPS:
        scenarios = list_existing_psps(IO_DIR)
        if not scenarios:
            raise FileNotFoundError("USE_EXISTING_PSPS=1 but no psps_*.json found in io/")
    else:
        for k in range(N_SCENES):
            xi = sample_psps_mask(n_line, corrmap, K_OPEN_PER_SCEN, rng)
            xi_path = IO_DIR / f"psps_r{run_id}_{k:05d}.json"
            save_json(xi_path, xi)
            scenarios.append(xi_path)

    out_jl = IO_DIR / f"sft_raw_run{run_id}.jsonl"
    wrote = 0
    with out_jl.open("w", encoding="utf-8") as out:
        for xi_path in scenarios:
            summary_path = IO_DIR / f"summary_{xi_path.stem}.json"
            summary = call_make_summary(xi_path, summary_path)
            if summary is None:
                continue

            gt_path = IO_DIR / f"gt_mip_{xi_path.stem}.json"
            gt = call_run_ground_truth(xi_path, gt_path, mode="mip_opt")
            if gt is None:
                continue

            toggles = gt.get("opt_switches", []) or gt.get("toggles", [])
            target = target_from_mip_switches(toggles, line2corr)

            rec = {
                "prompt": json.dumps(summary, indent=2),
                "target": target,
                "xi_file": xi_path.as_posix(),
                "summary_file": summary_path.as_posix(),
                "gt_file": gt_path.as_posix(),
            }
            out.write(json.dumps(rec) + "\n")
            wrote += 1

    print(f"[RUN {run_id}] Wrote {wrote} raw SFT examples to {out_jl.as_posix()}")


# -----------------------------------------------------------------------------
# Part 2: SFT pack (per run)
# -----------------------------------------------------------------------------

_OPEN_RE_SFT = re.compile(r"open\s*\(\s*(?:[sS]\d+\s*:\s*)?\d+\s*\)", re.IGNORECASE)

def _normalize_target_sft(t: str) -> str:
    t = (t or "").strip()
    if t.lower() == "do_nothing":
        return ""
    toks = _OPEN_RE_SFT.findall(t)
    return "; ".join(tok.strip() for tok in toks)


def _hash_str(s: str) -> int:
    return int(hashlib_md5(s), 16)


def hashlib_md5(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _key_for_record(rec: Dict[str, Any]) -> str:
    if rec.get("xi_file"):
        return str(rec["xi_file"])
    return "PROMPT#" + hashlib_md5(rec.get("prompt", ""))


def pack_sft_for_run(run_id: int) -> None:
    """Split raw SFT corpus into train/test and build OpenAI chat training file for this run."""
    raw_jl = IO_DIR / f"sft_raw_run{run_id}.jsonl"
    if not raw_jl.exists():
        raise FileNotFoundError(f"[RUN {run_id}] Missing input: {raw_jl.as_posix()}")

    train_raw = IO_DIR / f"sft_train_run{run_id}.jsonl"
    test_raw = IO_DIR / f"sft_test_run{run_id}.jsonl"
    train_chat = IO_DIR / f"sft_openai_chat_train_run{run_id}.jsonl"
    train_index = IO_DIR / f"sft_train_index_run{run_id}.json"
    test_index = IO_DIR / f"sft_test_index_run{run_id}.json"

    # Deduplicate by xi_file (last writer wins)
    dedup: Dict[str, Dict[str, Any]] = {}
    total_in = 0
    with raw_jl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total_in += 1
            rec["target"] = _normalize_target_sft(rec.get("target", ""))
            key = _key_for_record(rec)
            dedup[key] = rec

    pool = list(dedup.values())
    if not pool:
        raise RuntimeError(f"[RUN {run_id}] No records after de-duplication.")

    # Optional: drop empty targets for training
    train_candidates = pool if KEEP_EMPTY_TARGETS else [r for r in pool if r.get("target", "") != ""]

    # Deterministic split using hash of xi_file
    train, test = [], []
    for r in pool:
        key = _key_for_record(r)
        h = _hash_str(key) % 1000
        if h < int(TEST_RATIO * 1000):
            test.append(r)
        else:
            train.append(r)

    with train_raw.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")

    with test_raw.open("w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")

    # Chat training file
    with train_chat.open("w", encoding="utf-8") as out_f:
        n_written = 0
        for r in train_candidates:
            key = _key_for_record(r)
            h = _hash_str(key) % 1000
            if h < int(TEST_RATIO * 1000):
                continue
            obj = {
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG_SFT},
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["target"]},
                ]
            }
            out_f.write(json.dumps(obj) + "\n")
            n_written += 1

    with train_index.open("w", encoding="utf-8") as f:
        json.dump([r["xi_file"] for r in train if r.get("xi_file")], f, indent=2)

    with test_index.open("w", encoding="utf-8") as f:
        json.dump([r["xi_file"] for r in test if r.get("xi_file")], f, indent=2)

    print(f"[RUN {run_id}] SFT pack: total_in={total_in}, unique={len(pool)}, train={len(train)}, test={len(test)}, chat_train={n_written}")


# -----------------------------------------------------------------------------
# Part 3: Train SFT (per run)
# -----------------------------------------------------------------------------

def train_sft_for_run(run_id: int) -> str:
    """Fine-tune an OpenAI chat model for this run. Returns SFT model id."""
    train_jl = IO_DIR / f"sft_openai_chat_train_run{run_id}.jsonl"
    val_jl = IO_DIR / f"sft_openai_chat_val_run{run_id}.jsonl"  # optional
    out_sft = CONFIG / f"ft_model_sft_run{run_id}.txt"
    out_main = CONFIG / "ft_model_sft.txt"  # pointer to last SFT

    if not train_jl.exists():
        raise FileNotFoundError(f"[RUN {run_id}] Missing training data: {train_jl.as_posix()}")
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = make_client()

    with train_jl.open("rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    print(f"[RUN {run_id}] Uploaded SFT train file: {train_file.id}")

    job_kwargs: Dict[str, Any] = {
        "model": BASE_MODEL,
        "training_file": train_file.id,
        "suffix": f"{FT_SUFFIX_BASE}-run{run_id}",
    }

    if val_jl.exists():
        with val_jl.open("rb") as f:
            val_file = client.files.create(file=f, purpose="fine-tune")
        job_kwargs["validation_file"] = val_file.id
        print(f"[RUN {run_id}] Uploaded SFT val file: {val_file.id}")

    print(f"[RUN {run_id}] Creating SFT job on base: {BASE_MODEL}")
    job = client.fine_tuning.jobs.create(**job_kwargs)
    print(f"[RUN {run_id}] SFT job id: {job.id}")

    spinner = "|/-\\"
    k = 0
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        status = getattr(job, "status", "unknown")
        print(f"\r[RUN {run_id}] SFT status: {status} {spinner[k % len(spinner)]}", end="", flush=True)
        k += 1
        if status in ("succeeded", "failed", "cancelled"):
            print()
            break
        time.sleep(8)

    if job.status != "succeeded":
        raise RuntimeError(f"[RUN {run_id}] SFT finished with status={job.status}")

    model_id = job.fine_tuned_model
    if not model_id:
        raise RuntimeError(f"[RUN {run_id}] No fine_tuned_model returned on success.")

    print(f"[RUN {run_id}] SFT model id: {model_id}")
    CONFIG.mkdir(exist_ok=True)
    out_sft.write_text(model_id, encoding="utf-8")
    out_main.write_text(model_id, encoding="utf-8")  # pointer to latest
    return model_id


# -----------------------------------------------------------------------------
# Part 4: Build DPO pairs (voltage-only) per run
# -----------------------------------------------------------------------------
# (Logic mostly adapted from your build_dpo_pairs_vonly.py)
# -----------------------------------------------------------------------------

def to_openai_pref(prompt: str, chosen: str, rejected: str) -> Dict[str, Any]:
    """
    Construct a single DPO training line in the **OpenAI DPO format**:

      {
        "input": { "messages": [...] },
        "preferred_output":     [ { "role": "assistant", "content": "..." } ],
        "non_preferred_output": [ { "role": "assistant", "content": "..." } ]
      }
    """
    return {
        "input": {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG_INFER},
                {"role": "user", "content": prompt},
            ]
        },
        "preferred_output":     [{"role": "assistant", "content": chosen}],
        "non_preferred_output": [{"role": "assistant", "content": rejected}],
    }


def vpen_eff(sc: Dict[str, Any]) -> float:
    v = sc.get("V_pen")
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    j_ac = sc.get("J_ac", float("inf"))
    return float(j_ac) if isinstance(j_ac, (int, float)) and math.isfinite(j_ac) else float("inf")


def is_dc_ok(sc: Dict[str, Any]) -> bool:
    return bool(sc.get("feasible_dc")) if "feasible_dc" in sc else bool(sc.get("feasible"))


def is_ac_ok(sc: Dict[str, Any]) -> bool:
    if "ac_ok" in sc:
        return bool(sc.get("ac_ok"))
    v = sc.get("V_pen")
    if isinstance(v, (int, float)) and math.isfinite(v):
        return True
    j_ac = sc.get("J_ac")
    return isinstance(j_ac, (int, float)) and math.isfinite(j_ac)


def build_dpo_pairs_for_run(run_id: int, sft_model_id: str, n_line: int) -> None:
    """
    Voltage-only DPO pair builder (no Qmax scaling) for this run.
    Uses SFT model and optional base model, as in your original script.
    """
    print(f"[RUN {run_id}] Building DPO pairs (voltage-only)")

    index_file = IO_DIR / f"sft_train_index_run{run_id}.json"
    if not index_file.exists():
        raise FileNotFoundError(f"[RUN {run_id}] Missing INDEX_FILE {index_file}")

    with index_file.open("r", encoding="utf-8") as f:
        try:
            xi_paths = [Path(p) for p in json.load(f)]
        except Exception:
            f.seek(0)
            xi_paths = [Path(p.strip()) for p in f if p.strip()]
    xi_paths = [p for p in xi_paths if p.exists()]
    if not xi_paths:
        print(f"[RUN {run_id}] No xi paths for DPO.")
        return

    # DPO sampling config (reuse your env semantics but per-run)
    include_base = bool(int(os.getenv("INCLUDE_BASE", "1")))
    temps = [float(t) for t in os.getenv("TEMPS", "0.1,0.3,0.7,1.0").split(",")]
    n_total_cands = int(os.getenv("N_TOTAL_CANDS", "48"))
    max_tokens = int(os.getenv("MAX_TOKENS", "120"))
    sample_fraction = float(os.getenv("SAMPLE_FRACTION", "1.0"))
    shuffle_index = bool(int(os.getenv("SHUFFLE_INDEX", "1")))
    max_prompts = int(os.getenv("MAX_PROMPTS", "0"))

    V_ABS_MIN = float(os.getenv("V_PEN_ABS_MIN", "0"))
    V_REL_MIN = float(os.getenv("V_PEN_REL_MIN", "1.00"))
    RELAX_STEPS = int(os.getenv("RELAX_STEPS", "4"))
    RELAX_MULT = float(os.getenv("RELAX_MULT", "0.6"))
    REQUIRE_DC_OK_BOTH = bool(int(os.getenv("REQUIRE_DC_OK_BOTH", "1")))
    REQUIRE_AC_OK_BOTH = bool(int(os.getenv("REQUIRE_AC_OK_BOTH", "1")))
    TOPK_BEST = int(os.getenv("TOPK_BEST", "3"))
    TOPK_WORST = int(os.getenv("TOPK_WORST", "4"))
    MAX_PAIRS_PER_PROMPT = int(os.getenv("MAX_PAIRS_PER_PROMPT", "4"))
    ENABLE_BACKSTOP = bool(int(os.getenv("ENABLE_BACKSTOP", "1")))
    BACKSTOP_MIN_ABS = float(os.getenv("BACKSTOP_MIN_ABS", "200"))

    rng = random.Random(MASTER_SEED + 1000 * run_id + 17)

    if shuffle_index:
        rng.shuffle(xi_paths)

    total_available = len(xi_paths)
    if 0 < sample_fraction < 1.0 and total_available > 0:
        k_keep = max(1, int(round(total_available * sample_fraction)))
        xi_paths = xi_paths[:k_keep]
        print(f"[RUN {run_id}] Sampling ~{sample_fraction*100:.1f}% scenarios ({k_keep}/{total_available})")

    if max_prompts > 0:
        xi_paths = xi_paths[:max_prompts]

    engines = [sft_model_id] + ([BASE_MODEL] if include_base else [])
    grid = [(e, t) for e in engines for t in temps]

    client = make_client()
    run_tag = f"r{run_id}_{int(time.time())}"

    pairs_fp = IO_DIR / f"dpo_pairs_{run_tag}.jsonl"
    prefs_fp = IO_DIR / f"dpo_openai_prefs_run{run_id}.jsonl"
    qc_fp = IO_DIR / f"dpo_pairs_QC_{run_tag}.csv"
    eval_fp = IO_DIR / f"dpo_candidates_{run_tag}.csv"

    total_prompts = 0
    total_pairs = 0

    with pairs_fp.open("w", encoding="utf-8") as out_pairs, \
         prefs_fp.open("a", encoding="utf-8") as out_prefs, \
         qc_fp.open("w", newline="", encoding="utf-8") as qcf, \
         eval_fp.open("w", newline="", encoding="utf-8") as evf:

        qc_writer = csv.DictWriter(
            qcf,
            fieldnames=["xi", "chosen", "rejected", "best_V", "worst_V", "delta_V", "rel_V"],
        )
        qc_writer.writeheader()

        ev_writer = csv.DictWriter(
            evf,
            fieldnames=["xi", "plan", "V_pen", "J_ac", "feasible_dc", "ac_ok", "notes"],
        )
        ev_writer.writeheader()

        for xi in xi_paths:
            if max_prompts and total_prompts >= max_prompts:
                break

            summary_path = IO_DIR / f"summary_{xi.stem}.json"
            summary = call_make_summary(xi, summary_path)
            if summary is None:
                total_prompts += 1
                continue

            prompt = json.dumps(summary, indent=2)
            toggle_budget = int(summary.get("toggle_budget", 3))
            n_line_local = int(summary.get("lines") or summary.get("n_line") or n_line)

            per = max(1, n_total_cands // max(1, len(grid)))
            leftover = n_total_cands - per * len(grid)
            texts: List[str] = []
            for i, (eng, t) in enumerate(grid):
                k = per + (1 if i < leftover else 0)
                if k <= 0:
                    continue
                texts.extend(
                    sample_chat_batch(
                        client, eng, prompt, k, t, max_tokens
                    )
                )

            uniq: Dict[Tuple, Dict[str, Any]] = {}
            for txt in texts:
                pl = parse_actions_to_plan(txt, toggle_budget, n_line_local)
                key = tuple(
                    sorted((a.get("name", ""), int(a["line"])) for a in pl.get("corridor_actions", []))
                )
                if key and key not in uniq:
                    uniq[key] = pl

            if len(uniq) < 2:
                total_prompts += 1
                continue

            scored: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for j, pl in enumerate(uniq.values()):
                pfp = IO_DIR / f"dpo_plan_{xi.stem}_{run_tag}_{j}.json"
                rfp = IO_DIR / f"dpo_res_{xi.stem}_{run_tag}_{j}.json"
                pfp.write_text(json.dumps(pl), encoding="utf-8")
                sc = call_verify_hybrid(pfp, xi, rfp)
                try:
                    pfp.unlink(missing_ok=True)
                    rfp.unlink(missing_ok=True)
                except Exception:
                    pass
                if sc is None:
                    continue

                sc["ac_ok"] = is_ac_ok(sc)
                sc["feasible_dc"] = is_dc_ok(sc)

                ev_writer.writerow({
                    "xi": xi.as_posix(),
                    "plan": plan_to_text(pl),
                    "V_pen": sc.get("V_pen"),
                    "J_ac": sc.get("J_ac"),
                    "feasible_dc": int(sc["feasible_dc"]),
                    "ac_ok": int(sc["ac_ok"]),
                    "notes": sc.get("notes", ""),
                })
                scored.append((pl, sc))

            def keep(tup: Tuple[Dict[str, Any], Dict[str, Any]]) -> bool:
                sc = tup[1]
                if REQUIRE_DC_OK_BOTH and not is_dc_ok(sc):
                    return False
                if REQUIRE_AC_OK_BOTH and not is_ac_ok(sc):
                    return False
                return True

            pool = [t for t in scored if keep(t)]
            if len(pool) < 2:
                total_prompts += 1
                continue

            pool.sort(key=lambda t: vpen_eff(t[1]))
            best = pool[:max(1, TOPK_BEST)]
            worst = pool[-max(1, TOPK_WORST):][::-1]

            made = 0
            abs_thr = V_ABS_MIN
            rel_thr = V_REL_MIN

            for _ in range(RELAX_STEPS + 1):
                if made >= MAX_PAIRS_PER_PROMPT:
                    break
                for bp, bs in best:
                    if made >= MAX_PAIRS_PER_PROMPT:
                        break
                    bv = vpen_eff(bs)
                    for wp, ws in worst:
                        if made >= MAX_PAIRS_PER_PROMPT:
                            break
                        wv = vpen_eff(ws)
                        dv = wv - bv
                        rel = wv / max(bv, 1.0)
                        if dv >= abs_thr and rel >= rel_thr:
                            chosen = plan_to_text(bp).strip()
                            rejected = plan_to_text(wp).strip()
                            if not chosen or not rejected or chosen == rejected:
                                continue
                            rec = {
                                "prompt": prompt,
                                "xi_file": xi.as_posix(),
                                "run_tag": run_tag,
                                "chosen": chosen,
                                "rejected": rejected,
                                "best_V_pen_effective": bv,
                                "worst_V_pen_effective": wv,
                                "delta_V": dv,
                                "rel_V": rel,
                            }
                            out_pairs.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            out_prefs.write(
                                json.dumps(
                                    to_openai_pref(prompt, chosen, rejected),
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            qc_writer.writerow({
                                "xi": xi.as_posix(),
                                "chosen": chosen,
                                "rejected": rejected,
                                "best_V": bv,
                                "worst_V": wv,
                                "delta_V": dv,
                                "rel_V": rel,
                            })
                            made += 1
                            total_pairs += 1
                if made > 0:
                    break
                abs_thr *= RELAX_MULT
                rel_thr = 1.0 - (1.0 - rel_thr) * RELAX_MULT

            # backstop
            if ENABLE_BACKSTOP and made == 0 and pool:
                bp, bs = min(pool, key=lambda t: vpen_eff(t[1]))
                wp, ws = max(pool, key=lambda t: vpen_eff(t[1]))
                bv, wv = vpen_eff(bs), vpen_eff(ws)
                if wv > bv and (wv - bv) >= BACKSTOP_MIN_ABS:
                    chosen = plan_to_text(bp).strip()
                    rejected = plan_to_text(wp).strip()
                    if chosen and rejected and chosen != rejected:
                        rec = {
                            "prompt": prompt,
                            "xi_file": xi.as_posix(),
                            "run_tag": run_tag,
                            "chosen": chosen,
                            "rejected": rejected,
                            "best_V_pen_effective": bv,
                            "worst_V_pen_effective": wv,
                            "delta_V": wv - bv,
                            "rel_V": wv / max(bv, 1.0),
                        }
                        out_pairs.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_prefs.write(
                            json.dumps(
                                to_openai_pref(prompt, chosen, rejected),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        qc_writer.writerow({
                            "xi": xi.as_posix(),
                            "chosen": chosen,
                            "rejected": rejected,
                            "best_V": bv,
                            "worst_V": wv,
                            "delta_V": wv - bv,
                            "rel_V": wv / max(bv, 1.0),
                        })
                        total_pairs += 1

            total_prompts += 1

    print(f"[RUN {run_id}] DPO pairs: {pairs_fp} (pairs={total_pairs}, prompts={total_prompts})")
    print(f"[RUN {run_id}] Appended OpenAI prefs: {prefs_fp}")
    print(f"[RUN {run_id}] QC CSV: {qc_fp}")
    print(f"[RUN {run_id}] Candidate eval CSV: {eval_fp}")


# -----------------------------------------------------------------------------
# Part 5: Train DPO (per run)
# -----------------------------------------------------------------------------

def _normalize_pair_keys(obj: dict) -> dict:
    """
    Accept either:
      - preferred_output / non_preferred_output  (OpenAI canonical)
      - preferred / rejected                     (legacy local format)
    Return a dict with canonical keys: preferred, rejected
    """
    out = dict(obj)
    # Canonical OpenAI DPO format
    if "preferred_output" in obj and "non_preferred_output" in obj:
        out["preferred"] = obj["preferred_output"]
        out["rejected"] = obj["non_preferred_output"]
        return out
    # Legacy format (for backwards compatibility if needed)
    if "preferred" in obj and "rejected" in obj:
        return out
    raise ValueError(
        "Dataset line missing preference keys. Expected either "
        "('preferred_output','non_preferred_output') or ('preferred','rejected')."
    )


def _validate_messages(ms, field_name: str):
    if not isinstance(ms, list) or not ms:
        raise ValueError(f"`{field_name}` must be a non-empty list of messages.")
    first = ms[0]
    if first.get("role") != "assistant":
        raise ValueError(f"First message in `{field_name}` must have role='assistant'.")
    content = first.get("content")
    if content is None or (isinstance(content, str) and not content.strip()):
        raise ValueError(f"First message in `{field_name}` must have non-empty `content`.")


def _basic_schema_check(raw_obj: dict):
    if not isinstance(raw_obj, dict):
        raise ValueError("Dataset line is not a JSON object.")
    if "input" not in raw_obj or not isinstance(raw_obj["input"], dict):
        raise ValueError("Dataset line missing `input` object.")
    inp = raw_obj["input"]
    if "messages" not in inp or not isinstance(inp["messages"], list) or not inp["messages"]:
        raise ValueError("`input.messages` must be a non-empty list of chat messages.")
    obj = _normalize_pair_keys(raw_obj)
    _validate_messages(obj["preferred"], "preferred")
    _validate_messages(obj["rejected"], "rejected")
    return obj


def train_dpo_for_run(run_id: int, sft_model_id: str) -> str:
    """Launch a DPO fine-tune for this run, using the run-specific SFT model as parent."""
    if not sft_model_id.startswith("ft:"):
        raise ValueError(f"[RUN {run_id}] Parent SFT model must start with 'ft:'. Got: {sft_model_id}")

    data_path = IO_DIR / f"dpo_openai_prefs_run{run_id}.jsonl"
    if not data_path.exists() or data_path.stat().st_size == 0:
        raise FileNotFoundError(f"[RUN {run_id}] Missing or empty DPO dataset: {data_path.as_posix()}")

    # Light validation
    print(f"[RUN {run_id}] Validating DPO dataset: {data_path.as_posix()}")
    cnt = 0
    with data_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            _basic_schema_check(raw)
            cnt += 1
            if i <= 3:
                print(f"[RUN {run_id}]   sample[{i}]: OK")
    print(f"[RUN {run_id}] DPO lines: {cnt}")

    client = make_client()
    with data_path.open("rb") as f:
        up = client.files.create(file=f, purpose="fine-tune")
    print(f"[RUN {run_id}] Uploaded DPO file id: {up.id}")

    method = {
        "type": "dpo",
        "dpo": {
            "hyperparameters": {
                "n_epochs": DPO_N_EPOCHS,
                "beta": DPO_BETA,
                "batch_size": DPO_BATCH_SIZE,
            }
        }
    }

    print(f"[RUN {run_id}] Creating DPO job on parent SFT model: {sft_model_id}")
    job = client.fine_tuning.jobs.create(
        model=sft_model_id,
        training_file=up.id,
        method=method,
        suffix=f"{DPO_SUFFIX_BASE}-run{run_id}",
    )
    print(f"[RUN {run_id}] DPO job id: {job.id}")

    spinner = "|/-\\"
    k = 0
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        status = getattr(job, "status", "unknown")
        print(f"\r[RUN {run_id}] DPO status: {status} {spinner[k % len(spinner)]}", end="", flush=True)
        k += 1
        if status in ("succeeded", "failed", "cancelled"):
            print()
            break
        time.sleep(8)

    if job.status != "succeeded":
        raise RuntimeError(f"[RUN {run_id}] DPO finished with status={job.status}")

    model_id = job.fine_tuned_model
    if not model_id:
        raise RuntimeError(f"[RUN {run_id}] No DPO fine_tuned_model returned.")

    out_dpo = CONFIG / f"ft_model_dpo_run{run_id}.txt"
    out_main = CONFIG / "ft_model_dpo.txt"
    CONFIG.mkdir(exist_ok=True)
    out_dpo.write_text(model_id, encoding="utf-8")
    out_main.write_text(model_id, encoding="utf-8")
    print(f"[RUN {run_id}] DPO model id: {model_id}")
    return model_id


# -----------------------------------------------------------------------------
# Part 6: Evaluation per run (best-of-N by voltage) with RMSE + penalties
# -----------------------------------------------------------------------------

def eval_models_for_run(run_id: int, sft_model_id: str, dpo_model_id: Optional[str],
                        n_bus: int, n_line: int, line2corr: Dict[int, str]) -> None:
    """
    Evaluate zero-shot, SFT, and DPO for this run:
      - best-of-N candidates by J_ac (with AC_PENALTY for AC infeasibility)
      - DC cost J_dc via verify_plan (with DC_PENALTY for DC infeasibility)
      - J_dc_gt (mip_opt) and J_dc_base (baseline)
      - J_ac_gt and RMSE_gt via AC verification of ground truth plan
      - skip scenarios where GT AC is infeasible
    """
    print(f"[RUN {run_id}] Evaluating models (best-of-N by voltage)")
    index_file = IO_DIR / f"sft_test_index_run{run_id}.json"
    if not index_file.exists():
        raise FileNotFoundError(f"[RUN {run_id}] Missing test index: {index_file}")

    with index_file.open("r", encoding="utf-8") as f:
        try:
            xi_paths = [Path(p) for p in json.load(f)]
        except Exception:
            f.seek(0)
            xi_paths = [Path(p.strip()) for p in f.read().splitlines() if p.strip()]
    xi_paths = [p for p in xi_paths if p.exists()]

    if not xi_paths:
        print(f"[RUN {run_id}] No xi paths to evaluate.")
        return

    models: List[Tuple[str, str]] = [("zero_shot", BASE_MODEL)]
    if sft_model_id:
        models.append(("sft", sft_model_id))
    if dpo_model_id:
        models.append(("dpo", dpo_model_id))

    client = make_client()
    rows: List[Dict[str, Any]] = []

    # caches to avoid recomputation
    gt_opt_cache: Dict[str, float] = {}
    gt_base_cache: Dict[str, float] = {}
    gt_ac_cache: Dict[str, float] = {}   # J_ac_gt
    gt_rmse_cache: Dict[str, float] = {}

    # Force exactly 3 candidates as before
    N_CANDS = 3
    temps = EVAL_TEMPS
    max_tokens = EVAL_MAX_TOKENS

    rng = random.Random(MASTER_SEED + 2000 * run_id + 5)

    for xi in xi_paths:
        xi_key = xi.as_posix()

        # ---- Ground truth DC: mip_opt ----
        if xi_key not in gt_opt_cache:
            gt_opt_path = IO_DIR / f"gt_mip_opt_run{run_id}_{xi.stem}.json"
            gt_opt_res = call_run_ground_truth(xi, gt_opt_path, mode="mip_opt")
            j_dc_gt, _ = normalize_jdc_from_dcopf(gt_opt_res)
            gt_opt_cache[xi_key] = j_dc_gt

            # Build GT plan from toggles to get J_ac_gt
            toggles = []
            if gt_opt_res is not None:
                toggles = gt_opt_res.get("opt_switches", []) or gt_opt_res.get("toggles", [])
            if toggles:
                gt_plan = toggles_to_plan(toggles, line2corr)
                p_gt = IO_DIR / f"gt_plan_opt_run{run_id}_{xi.stem}.json"
                r_gt = IO_DIR / f"gt_res_opt_run{run_id}_{xi.stem}.json"
                p_gt.write_text(json.dumps(gt_plan), encoding="utf-8")
                sc_gt_ac = call_verify_hybrid(p_gt, xi, r_gt)
                try:
                    p_gt.unlink(missing_ok=True)
                    r_gt.unlink(missing_ok=True)
                except Exception:
                    pass
                J_ac_gt, ac_ok_gt = normalize_ac_from_hybrid(sc_gt_ac)
            else:
                J_ac_gt, ac_ok_gt = AC_PENALTY, False

            if not ac_ok_gt or J_ac_gt >= AC_PENALTY:
                # skip this xi altogether (GT AC infeasible)
                gt_ac_cache[xi_key] = AC_PENALTY
                gt_rmse_cache[xi_key] = compute_rmse_from_Jac(J_ac_gt, n_bus, V_SCALE)
                # mark DC costs anyway but continue
                continue
            else:
                gt_ac_cache[xi_key] = J_ac_gt
                gt_rmse_cache[xi_key] = compute_rmse_from_Jac(J_ac_gt, n_bus, V_SCALE)

        j_dc_gt = gt_opt_cache[xi_key]
        J_ac_gt = gt_ac_cache[xi_key]
        rmse_gt = gt_rmse_cache[xi_key]

        if J_ac_gt >= AC_PENALTY:
            # safeguard: skip this xi
            continue

        # ---- Baseline DC ----
        if xi_key not in gt_base_cache:
            gt_base_path = IO_DIR / f"gt_baseline_run{run_id}_{xi.stem}.json"
            gt_base_res = call_run_ground_truth(xi, gt_base_path, mode="baseline")
            j_dc_base, _ = normalize_jdc_from_dcopf(gt_base_res)
            gt_base_cache[xi_key] = j_dc_base

        j_dc_base = gt_base_cache[xi_key]

        # ---- Summary for prompt ----
        summ_path = IO_DIR / f"summary_{xi.stem}.json"
        summ = call_make_summary(xi, summ_path)
        if summ is None:
            continue

        prompt = json.dumps(summ, indent=2)
        toggle_budget = int(summ.get("toggle_budget", 3))
        n_line_local = int(summ.get("lines") or summ.get("n_line") or n_line)

        for label, model in models:
            # sample N_CANDS across temperatures
            per = max(1, N_CANDS // len(temps))
            leftover = N_CANDS - per * len(temps)
            texts: List[str] = []
            for i, t in enumerate(temps):
                k = per + (1 if i < leftover else 0)
                if k <= 0:
                    continue
                texts.extend(
                    sample_chat_batch(client, model, prompt, k, t, max_tokens)
                )

            best_pl = None
            best_J_ac = None
            best_txt = None

            for txt in texts:
                plan = parse_actions_to_plan(txt, toggle_budget, n_line_local)
                p = IO_DIR / f"eval_plan_{label}_run{run_id}_{xi.stem}_{rng.randint(0, 9999)}.json"
                r = IO_DIR / f"eval_res_{label}_run{run_id}_{xi.stem}_{rng.randint(0, 9999)}.json"
                p.write_text(json.dumps(plan), encoding="utf-8")
                sc = call_verify_hybrid(p, xi, r)
                try:
                    p.unlink(missing_ok=True)
                    r.unlink(missing_ok=True)
                except Exception:
                    pass

                J_ac, ac_ok = normalize_ac_from_hybrid(sc)
                # We allow AC infeasible with penalty AC_PENALTY, but they will be dominated
                if best_J_ac is None or J_ac < best_J_ac:
                    best_pl = plan
                    best_J_ac = J_ac
                    best_txt = txt

            if best_pl is None or best_J_ac is None:
                continue

            # compute RMSE for this J_ac
            rmse = compute_rmse_from_Jac(best_J_ac, n_bus, V_SCALE)

            # DC-OPF for chosen plan
            p_dc = IO_DIR / f"eval_plan_dc_{label}_run{run_id}_{xi.stem}_{rng.randint(0, 9999)}.json"
            r_dc = IO_DIR / f"eval_res_dc_{label}_run{run_id}_{xi.stem}_{rng.randint(0, 9999)}.json"
            p_dc.write_text(json.dumps(best_pl), encoding="utf-8")
            dc_sc = call_verify_dcopf(p_dc, xi, r_dc)
            try:
                p_dc.unlink(missing_ok=True)
                r_dc.unlink(missing_ok=True)
            except Exception:
                pass

            j_dc, feasible_dc = normalize_jdc_from_dcopf(dc_sc)

            rows.append({
                "run": run_id,
                "xi": xi.as_posix(),
                "model": label,
                "parent": model,
                "J_ac": float(best_J_ac),
                "J_ac_rmse": float(rmse),
                "J_dc": float(j_dc),
                "J_dc_gt": float(j_dc_gt),
                "J_dc_base": float(j_dc_base),
                "J_ac_gt": float(J_ac_gt),
                "J_ac_rmse_gt": float(rmse_gt),
                "feasible_dc": int(feasible_dc),
                "best_plan_text": plan_to_text(best_pl),
            })

    # Save tall rows
    out_json = IO_DIR / f"eval_vselect_rows_run{run_id}.json"
    out_csv = IO_DIR / f"eval_vselect_rows_run{run_id}.csv"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run", "xi", "model", "parent",
            "J_ac", "J_ac_rmse",
            "J_dc", "J_dc_gt", "J_dc_base",
            "J_ac_gt", "J_ac_rmse_gt",
            "feasible_dc", "best_plan_text",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Wide per-xi CSV (J_dc, J_ac, RMSE for each model + GT/base)
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[RUN {run_id}] No rows to analyze.")
        return

    df["J_ac"] = pd.to_numeric(df["J_ac"], errors="coerce")
    df["J_ac_rmse"] = pd.to_numeric(df["J_ac_rmse"], errors="coerce")
    df["J_dc"] = pd.to_numeric(df["J_dc"], errors="coerce")
    df["J_dc_gt"] = pd.to_numeric(df["J_dc_gt"], errors="coerce")
    df["J_dc_base"] = pd.to_numeric(df["J_dc_base"], errors="coerce")
    df["J_ac_gt"] = pd.to_numeric(df["J_ac_gt"], errors="coerce")
    df["J_ac_rmse_gt"] = pd.to_numeric(df["J_ac_rmse_gt"], errors="coerce")

    pivot_j = df.pivot_table(
        index="xi",
        columns="model",
        values=["J_dc", "J_ac", "J_ac_rmse"],
        aggfunc="first",
    )
    pivot_j.columns = [f"{metric}_{model}" for metric, model in pivot_j.columns]
    pivot_j = pivot_j.reset_index()

    gt_df = df[["xi", "J_dc_gt", "J_dc_base", "J_ac_gt", "J_ac_rmse_gt"]].drop_duplicates("xi")
    pivot_j = pivot_j.merge(gt_df, on="xi", how="left")

    wide_csv = IO_DIR / f"eval_vselect_jdc_jac_per_xi_run{run_id}.csv"
    pivot_j.to_csv(wide_csv, index=False)
    print(f"[RUN {run_id}] Saved per-xi wide CSV: {wide_csv.as_posix()}")


# -----------------------------------------------------------------------------
# Part 7: Multi-run aggregation + plots (confidence bands)
# -----------------------------------------------------------------------------

def detect_models(df: pd.DataFrame) -> List[str]:
    models = []
    for col in df.columns:
        if col.startswith("J_dc_") and col not in ("J_dc_gt", "J_dc_base"):
            name = col[len("J_dc_"):]
            models.append(name)
    return sorted(set(models))


def basic_global_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    out["n_scenarios"] = int(df["xi"].nunique())
    mask_gt = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan).notna()
    mask_base = df["J_dc_base"].replace([np.inf, -np.inf], np.nan).notna()
    mask_both = mask_gt & mask_base
    if mask_both.any():
        diff = df.loc[mask_both, "J_dc_base"] - df.loc[mask_both, "J_dc_gt"]
        ratio = df.loc[mask_both, "J_dc_base"] / df.loc[mask_both, "J_dc_gt"]
        out["baseline_minus_opt_mean"] = float(diff.mean())
        out["baseline_minus_opt_median"] = float(diff.median())
        out["baseline_minus_opt_std"] = float(diff.std(ddof=1))
        out["baseline_minus_opt_min"] = float(diff.min())
        out["baseline_minus_opt_max"] = float(diff.max())
        out["baseline_over_opt_mean"] = float(ratio.mean())
        out["baseline_over_opt_median"] = float(ratio.median())
    else:
        for k in [
            "baseline_minus_opt_mean", "baseline_minus_opt_median",
            "baseline_minus_opt_std", "baseline_minus_opt_min",
            "baseline_minus_opt_max", "baseline_over_opt_mean",
            "baseline_over_opt_median",
        ]:
            out[k] = np.nan
    return pd.DataFrame([out])


def per_model_stats(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        col_jdc = f"J_dc_{m}"
        col_jac = f"J_ac_{m}"
        col_jac_rmse = f"J_ac_rmse_{m}" if f"J_ac_rmse_{m}" in df.columns else None

        if col_jdc not in df.columns or col_jac not in df.columns:
            continue

        jdc = df[col_jdc].replace([np.inf, -np.inf], np.nan)
        jac = df[col_jac].replace([np.inf, -np.inf], np.nan)
        jgt = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)
        jbase = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)

        mask_jdc = jdc.notna()
        mask_jac = jac.notna()
        mask_gt = jgt.notna()
        mask_base = jbase.notna()
        mask_all = mask_jdc & mask_gt

        regret = (jdc - jgt).where(mask_all)
        ratio_to_gt = (jdc / jgt).where(mask_all)

        mask_valid_base = mask_jdc & mask_base
        better_than_base = (jdc <= jbase).where(mask_valid_base)
        better_or_equal_opt = (jdc <= jgt + 1e-6).where(mask_all)

        eps_list = [0.01, 0.05, 0.10]
        close_fracs = {}
        for eps in eps_list:
            thresh = jgt * (1.0 + eps)
            is_close = (jdc <= thresh).where(mask_all)
            close_fracs[f"frac_within_{int(eps*100)}pct_opt"] = float(is_close.mean()) if mask_all.any() else np.nan

        # RMSE stats
        if col_jac_rmse:
            jac_rmse = df[col_jac_rmse].replace([np.inf, -np.inf], np.nan)
            mask_jac_rmse = jac_rmse.notna()
            J_ac_rmse_mean = float(jac_rmse[mask_jac_rmse].mean()) if mask_jac_rmse.any() else np.nan
            J_ac_rmse_median = float(jac_rmse[mask_jac_rmse].median()) if mask_jac_rmse.any() else np.nan
        else:
            J_ac_rmse_mean = np.nan
            J_ac_rmse_median = np.nan

        row = {
            "model": m,
            "n_valid_J_dc": int(mask_jdc.sum()),
            "n_valid_J_ac": int(mask_jac.sum()),

            "J_dc_mean": float(jdc[mask_jdc].mean()) if mask_jdc.any() else np.nan,
            "J_dc_median": float(jdc[mask_jdc].median()) if mask_jdc.any() else np.nan,
            "J_dc_std": float(jdc[mask_jdc].std(ddof=1)) if mask_jdc.sum() > 1 else 0.0,
            "J_dc_min": float(jdc[mask_jdc].min()) if mask_jdc.any() else np.nan,
            "J_dc_max": float(jdc[mask_jdc].max()) if mask_jdc.any() else np.nan,

            "J_ac_mean": float(jac[mask_jac].mean()) if mask_jac.any() else np.nan,
            "J_ac_median": float(jac[mask_jac].median()) if mask_jac.any() else np.nan,
            "J_ac_std": float(jac[mask_jac].std(ddof=1)) if mask_jac.sum() > 1 else 0.0,
            "J_ac_min": float(jac[mask_jac].min()) if mask_jac.any() else np.nan,
            "J_ac_max": float(jac[mask_jac].max()) if mask_jac.any() else np.nan,

            "J_ac_rmse_mean": float(J_ac_rmse_mean),
            "J_ac_rmse_median": float(J_ac_rmse_median),

            "regret_mean": float(regret[mask_all].mean()) if mask_all.any() else np.nan,
            "regret_median": float(regret[mask_all].median()) if mask_all.any() else np.nan,
            "regret_std": float(regret[mask_all].std(ddof=1)) if mask_all.sum() > 1 else 0.0,
            "regret_min": float(regret[mask_all].min()) if mask_all.any() else np.nan,
            "regret_max": float(regret[mask_all].max()) if mask_all.any() else np.nan,

            "ratio_to_opt_mean": float(ratio_to_gt[mask_all].mean()) if mask_all.any() else np.nan,
            "ratio_to_opt_median": float(ratio_to_gt[mask_all].median()) if mask_all.any() else np.nan,

            "frac_better_than_baseline": float(better_than_base.mean()) if mask_valid_base.any() else np.nan,
            "frac_leq_opt": float(better_or_equal_opt.mean()) if mask_all.any() else np.nan,
        }
        row.update(close_fracs)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model")


def pairwise_stats(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            col_jdc_a = f"J_dc_{a}"
            col_jdc_b = f"J_dc_{b}"
            col_jac_a = f"J_ac_{a}"
            col_jac_b = f"J_ac_{b}"

            if not (col_jdc_a in df.columns and col_jdc_b in df.columns):
                continue

            ja = df[col_jdc_a].replace([np.inf, -np.inf], np.nan)
            jb = df[col_jdc_b].replace([np.inf, -np.inf], np.nan)
            mask_dc = ja.notna() & jb.notna()

            if mask_dc.any():
                diff_dc = jb - ja
                frac_a_better_dc = float((diff_dc > 0).mean())
                frac_b_better_dc = float((diff_dc < 0).mean())
                frac_tie_dc = float((diff_dc == 0).mean())
                delta_mean_dc = float(diff_dc.mean())
                delta_median_dc = float(diff_dc.median())
            else:
                frac_a_better_dc = frac_b_better_dc = frac_tie_dc = np.nan
                delta_mean_dc = delta_median_dc = np.nan

            if col_jac_a in df.columns and col_jac_b in df.columns:
                va = df[col_jac_a].replace([np.inf, -np.inf], np.nan)
                vb = df[col_jac_b].replace([np.inf, -np.inf], np.nan)
                mask_ac = va.notna() & vb.notna()
                if mask_ac.any():
                    diff_ac = vb - va
                    frac_a_better_ac = float((diff_ac > 0).mean())
                    frac_b_better_ac = float((diff_ac < 0).mean())
                    frac_tie_ac = float((diff_ac == 0).mean())
                    delta_mean_ac = float(diff_ac.mean())
                    delta_median_ac = float(diff_ac.median())
                else:
                    frac_a_better_ac = frac_b_better_ac = frac_tie_ac = np.nan
                    delta_mean_ac = delta_median_ac = np.nan
            else:
                frac_a_better_ac = frac_b_better_ac = frac_tie_ac = np.nan
                delta_mean_ac = delta_median_ac = np.nan
                mask_ac = pd.Series(False, index=df.index)

            rows.append({
                "pair": f"{a}_vs_{b}",
                "model_a": a,
                "model_b": b,
                "n_overlap_dc": int(mask_dc.sum()),
                "frac_a_better_dc": frac_a_better_dc,
                "frac_b_better_dc": frac_b_better_dc,
                "frac_tie_dc": frac_tie_dc,
                "delta_mean_dc(b-a)": delta_mean_dc,
                "delta_median_dc(b-a)": delta_median_dc,
                "n_overlap_ac": int(mask_ac.sum()),
                "frac_a_better_ac": frac_a_better_ac,
                "frac_b_better_ac": frac_b_better_ac,
                "frac_tie_ac": frac_tie_ac,
                "delta_mean_ac(b-a)": delta_mean_ac,
                "delta_median_ac(b-a)": delta_median_ac,
            })
    return pd.DataFrame(rows)


def _add_boxplot_mean_median_labels(bp, means, medians):
    handles, labels = [], []
    if "means" in bp and bp["means"]:
        handles.append(bp["means"][0])
        labels.append("mean")
    if "medians" in bp and bp["medians"]:
        handles.append(bp["medians"][0])
        labels.append("median")
    if handles:
        plt.legend(handles, labels)
    positions = np.arange(1, len(means) + 1)
    for x, mean, med in zip(positions, means, medians):
        if np.isfinite(mean):
            plt.text(x + 0.05, mean, f"{mean:.1f}", va="bottom", ha="left", fontsize=7, rotation=90)
        if np.isfinite(med):
            plt.text(x - 0.05, med, f"{med:.1f}", va="top", ha="right", fontsize=7, rotation=90)


def plot_Jdc_boxplots(df: pd.DataFrame, models: List[str], fig_dir: Path) -> None:
    """Two boxplots:
       - absolute J_dc (baseline, opt, models)
       - gap J_dc - J_dc_gt (baseline + models, no opt)
    """
    # --- Absolute J_dc ---
    names = ["baseline", "opt"] + models
    data, labels, means, medians = [], [], [], []
    for name in names:
        if name == "baseline":
            arr = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)
        elif name == "opt":
            arr = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)
        else:
            col = f"J_dc_{name}"
            if col not in df.columns:
                continue
            arr = df[col].replace([np.inf, -np.inf], np.nan)
        arr = arr.dropna()
        if arr.empty:
            continue
        data.append(arr.values)
        labels.append(name)
        means.append(float(arr.mean()))
        medians.append(float(arr.median()))

    if data:
        plt.figure(figsize=(7, 4))
        bp = plt.boxplot(data, tick_labels=labels, showmeans=True, showfliers=False)
        plt.ylabel("J_dc")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        _add_boxplot_mean_median_labels(bp, means, medians)
        plt.tight_layout()
        plt.savefig(fig_dir / "box_Jdc_absolute.png", dpi=300)
        plt.close()

    # --- Gap J_dc - J_dc_gt (no opt) ---
    names_gap = ["baseline"] + models
    data, labels, means, medians = [], [], [], []
    for name in names_gap:
        if name == "baseline":
            j = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)
        else:
            col = f"J_dc_{name}"
            if col not in df.columns:
                continue
            j = df[col].replace([np.inf, -np.inf], np.nan)
        jgt = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)
        mask = j.notna() & jgt.notna()
        if not mask.any():
            continue
        gap = (j - jgt)[mask]
        data.append(gap.values)
        labels.append(name)
        means.append(float(gap.mean()))
        medians.append(float(gap.median()))

    if data:
        plt.figure(figsize=(7, 4))
        bp = plt.boxplot(data, tick_labels=labels, showmeans=True, showfliers=False)
        plt.ylabel("J_dc - J_dc_gt")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        _add_boxplot_mean_median_labels(bp, means, medians)
        plt.tight_layout()
        plt.savefig(fig_dir / "box_Jdc_gap.png", dpi=300)
        plt.close()


def _cdf_with_conf_band(values: np.ndarray, label: str, color=None):
    """
    Plot empirical CDF with binomial 95% confidence band (normal approx).
    """
    arr = np.sort(values)
    n = len(arr)
    if n == 0:
        return
    y = np.arange(1, n + 1) / n
    p = y
    se = np.sqrt(p * (1 - p) / n)
    z = 1.96
    y_lo = np.clip(p - z * se, 0, 1)
    y_hi = np.clip(p + z * se, 0, 1)
    plt.plot(arr, y, label=label)
    plt.fill_between(arr, y_lo, y_hi, alpha=0.15)


def plot_regret_hist_and_cdf(df: pd.DataFrame, models: List[str], fig_dir: Path) -> None:
    methods = ["baseline"] + models

    # Histogram
    plt.figure(figsize=(7, 4))
    all_regrets = []
    linestyles = ["solid", "dashed", "dotted", "dashdot", (0, (5, 1))]
    for i, name in enumerate(methods):
        if name == "baseline":
            j = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)
        else:
            col = f"J_dc_{name}"
            if col not in df.columns:
                continue
            j = df[col].replace([np.inf, -np.inf], np.nan)
        jgt = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)
        mask = j.notna() & jgt.notna()
        if not mask.any():
            continue
        reg = (j - jgt)[mask].values
        all_regrets.append(reg)
    if all_regrets:
        concat = np.concatenate(all_regrets)
        bins = np.linspace(concat.min(), concat.max(), 30)
        for i, name in enumerate(methods):
            if name == "baseline":
                j = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)
            else:
                col = f"J_dc_{name}"
                if col not in df.columns:
                    continue
                j = df[col].replace([np.inf, -np.inf], np.nan)
            jgt = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)
            mask = j.notna() & jgt.notna()
            if not mask.any():
                continue
            reg = (j - jgt)[mask].values
            ls = linestyles[i % len(linestyles)]
            plt.hist(reg, bins=bins, density=True, histtype="step", alpha=0.9, linestyle=ls, label=name)
        plt.xlabel("J_dc - J_dc_gt")
        plt.ylabel("Density")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "hist_regret_Jdc_all_methods.png", dpi=300)
        plt.close()

    # CDF with confidence bands
    plt.figure(figsize=(7, 4))
    plotted = False
    for name in methods:
        if name == "baseline":
            j = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)
        else:
            col = f"J_dc_{name}"
            if col not in df.columns:
                continue
            j = df[col].replace([np.inf, -np.inf], np.nan)
        jgt = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)
        mask = j.notna() & jgt.notna()
        if not mask.any():
            continue
        reg = (j - jgt)[mask].values
        _cdf_with_conf_band(reg, name)
        plotted = True
    if plotted:
        plt.xlabel("J_dc - J_dc_gt")
        plt.ylabel("CDF")
        plt.grid(alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "cdf_regret_Jdc_all_methods_conf.png", dpi=300)
        plt.close()
    else:
        plt.close()


def plot_Jac_rmse_cdf(df: pd.DataFrame, models: List[str], fig_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    plotted = False
    for m in models:
        col = f"J_ac_rmse_{m}"
        if col not in df.columns:
            continue
        v = df[col].replace([np.inf, -np.inf], np.nan)
        mask = v.notna()
        if not mask.any():
            continue
        arr = v[mask].values
        _cdf_with_conf_band(arr, m)
        plotted = True
    if plotted:
        plt.xlabel("Voltage RMSE per bus (V)")
        plt.ylabel("CDF")
        plt.grid(alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "cdf_Jac_rmse_all_models_conf.png", dpi=300)
        plt.close()
    else:
        plt.close()


def analyze_across_runs(n_runs: int) -> None:
    """Aggregate eval_vselect_jdc_jac_per_xi_run*.csv across runs and produce stats/plots."""
    wide_files = []
    for r in range(n_runs):
        f = IO_DIR / f"eval_vselect_jdc_jac_per_xi_run{r}.csv"
        if f.exists():
            wide_files.append((r, f))
    if not wide_files:
        print("No eval_vselect_jdc_jac_per_xi_run*.csv found; skipping analysis.")
        return

    dfs = []
    for r, f in wide_files:
        df = pd.read_csv(f)
        df["run"] = r
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    models = detect_models(df_all)
    print("Detected models:", models)

    # Filter out rows where ground truth AC is clearly invalid (just in case)
    if "J_ac_gt" in df_all.columns:
        mask_bad_gt = df_all["J_ac_gt"] >= AC_PENALTY
        n_bad = int(mask_bad_gt.sum())
        if n_bad > 0:
            print(f"Filtering out {n_bad} rows due to J_ac_gt >= AC_PENALTY.")
            df_all = df_all[~mask_bad_gt].reset_index(drop=True)

    # Global stats, per-model stats, pairwise
    global_stats_df = basic_global_stats(df_all)
    per_model_df = per_model_stats(df_all, models)
    pairwise_df = pairwise_stats(df_all, models)

    fig_dir = IO_DIR / "figs_eval"
    fig_dir.mkdir(exist_ok=True)

    global_csv = IO_DIR / "eval_stats_global_all_runs.csv"
    per_model_csv = IO_DIR / "eval_stats_per_model_all_runs.csv"
    pairwise_csv = IO_DIR / "eval_stats_pairwise_all_runs.csv"

    global_stats_df.to_csv(global_csv, index=False)
    per_model_df.to_csv(per_model_csv, index=False)
    pairwise_df.to_csv(pairwise_csv, index=False)

    print("\n=== Global stats (all runs) ===")
    print(global_stats_df.to_string(index=False))
    print("\n=== Per-model stats (excerpt) ===")
    cols_show = [
        "model", "J_dc_mean", "J_ac_rmse_mean",
        "regret_mean", "ratio_to_opt_mean",
        "frac_better_than_baseline",
        "frac_within_1pct_opt", "frac_within_5pct_opt", "frac_within_10pct_opt",
    ]
    cols_show = [c for c in cols_show if c in per_model_df.columns]
    print(per_model_df[cols_show].to_string(index=False))

    # Plots
    plot_Jdc_boxplots(df_all, models, fig_dir)
    plot_regret_hist_and_cdf(df_all, models, fig_dir)
    plot_Jac_rmse_cdf(df_all, models, fig_dir)

    print(f"\nSaved stats to:\n  {global_csv}\n  {per_model_csv}\n  {pairwise_csv}")
    print(f"Saved figures to:\n  {fig_dir}")


# -----------------------------------------------------------------------------
# main(): orchestrate multi-run pipeline
# -----------------------------------------------------------------------------

def main():
    if not API_KEY:
        print("WARNING: OPENAI_API_KEY is not set. SFT/DPO/eval with OpenAI will fail.", file=sys.stderr)

    # Case counts + corridor info
    counts = get_case_counts(CASE)
    n_bus = int(counts["buses"])
    n_line = int(counts["lines"])
    corrmap = load_corridor_map()
    line2corr = invert_corridors(corrmap)

    print(f"Case={CASE}: buses={n_bus}, lines={n_line}")
    print(f"N_RUNS={N_RUNS}, MASTER_SEED={MASTER_SEED}")

    for run_id in range(N_RUNS):
        print("\n" + "=" * 70)
        print(f"=== RUN {run_id} ===")
        print("=" * 70)
        seed_run = MASTER_SEED + 9973 * run_id

        # 1) SFT corpus build
        build_sft_from_gt_for_run(run_id, seed_run, n_bus, n_line, corrmap, line2corr)

        # 2) SFT pack
        pack_sft_for_run(run_id)

        # 3) Train SFT
        sft_model_id = train_sft_for_run(run_id)

        # 4) DPO pairs
        build_dpo_pairs_for_run(run_id, sft_model_id, n_line)

        # 5) Train DPO
        dpo_model_id = train_dpo_for_run(run_id, sft_model_id)

        # 6) Evaluation
        eval_models_for_run(run_id, sft_model_id, dpo_model_id, n_bus, n_line, line2corr)

    # 7) Multi-run aggregation + plots
    print("\n" + "=" * 70)
    print("=== Multi-run analysis ===")
    print("=" * 70)
    analyze_across_runs(N_RUNS)


if __name__ == "__main__":
    main()
