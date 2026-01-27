#!/usr/bin/env python3
"""
nn_supervised_train_eval_from_sft.py

End-to-end supervised-learning pipeline that reuses the SAME SFT dataset + SAME SFT train/test split.

Goal:
  Train a neural net that maps PSPS outage mask xi -> corrective open actions (multi-hot / top-K).

Reads (from io/):
  - sft_raw(_runX).jsonl
  - sft_train_index(_runX).json
  - sft_test_index(_runX).json
  - psps_*.json files referenced by xi_file entries (expected to contain xi list)

Optionally reads:
  - config/limits.yml (for default budget K)
  - gt_*.json files referenced by gt_file entries (for logging ground-truth J + opt_switches)

Writes (to io/):
  - nn_model(_runX).pt                     (best model checkpoint)
  - nn_train_log(_runX).json               (training curve)
  - nn_eval_predictions(_runX).csv         (per-test predictions + metrics)
  - nn_eval_summary(_runX).json            (aggregate metrics)

Optional MATLAB evaluation (if matlab is installed and repo has matlab/verify_plan*.m):
  With --matlab_eval 1, the script will:
    - build plan JSON for each predicted action set,
    - call verify_plan and verify_plan_hybrid,
    - append J_dc / J_ac / feasibility fields to the eval CSV.

Usage examples (run from repo root):
  python python/nn_supervised_train_eval_from_sft.py --run auto --epochs 50
  python python/nn_supervised_train_eval_from_sft.py --run run0 --epochs 100 --device cuda
  python python/nn_supervised_train_eval_from_sft.py --run run0 --matlab_eval 1 --matlab_max 100

Notes:
  - Inputs are ONLY xi by default (matches your “outaged lines -> corrective action” request).
  - Decoding is top-K among eligible lines where xi==1, with K taken from prompt.toggle_budget when available,
    otherwise from config/limits.yml (budget), otherwise from --K.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

OPEN_RE = re.compile(r"open\(\s*(?:S\d+\s*:\s*)?(\d+)\s*\)", re.IGNORECASE)

# ---------------------------
# Helpers: IO
# ---------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def basename_any(p: str) -> str:
    return os.path.basename(p.replace("\\", "/"))

def find_latest_run(io_dir: Path) -> str:
    """Return "runX" for the max X found in io/sft_raw_runX.jsonl, else ""."""
    candidates = sorted(io_dir.glob("sft_raw_run*.jsonl"))
    best_r = -1
    for p in candidates:
        m = re.search(r"run(\d+)", p.name)
        if m:
            r = int(m.group(1))
            best_r = max(best_r, r)
    return f"run{best_r}" if best_r >= 0 else ""

def get_paths(io_dir: Path, run_tag: str) -> Tuple[Path, Path, Path]:
    suffix = f"_{run_tag}" if run_tag else ""
    raw = io_dir / f"sft_raw{suffix}.jsonl"
    tr  = io_dir / f"sft_train_index{suffix}.json"
    te  = io_dir / f"sft_test_index{suffix}.json"
    return raw, tr, te

def try_read_budget(repo_root: Path) -> Optional[int]:
    limits = repo_root / "config" / "limits.yml"
    if not limits.exists() or yaml is None:
        return None
    obj = yaml.safe_load(limits.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "budget" in obj:
        try:
            return int(obj["budget"])
        except Exception:
            return None
    return None

def parse_open_ids(target: str) -> List[int]:
    return [int(m.group(1)) for m in OPEN_RE.finditer(target or "")]

def infer_indexing(open_ids: List[int]) -> str:
    return "0-based" if any(i == 0 for i in open_ids) else "1-based"

def make_multi_hot(open_ids: List[int], n_line: int, indexing: str) -> np.ndarray:
    y = np.zeros((n_line,), dtype=np.int8)
    if indexing == "0-based":
        for lid in open_ids:
            if 0 <= lid < n_line:
                y[lid] = 1
    else:
        for lid in open_ids:
            if 1 <= lid <= n_line:
                y[lid - 1] = 1
    return y

def ids_from_multi_hot(y: np.ndarray, indexing: str) -> List[int]:
    idx = np.flatnonzero(y.astype(bool)).tolist()
    if indexing == "0-based":
        return idx
    return [i + 1 for i in idx]

# ---------------------------
# Dataset
# ---------------------------

@dataclass
class Example:
    sid: str                 # scenario id (psps filename)
    xi: np.ndarray           # (n_line,) int8
    y: np.ndarray            # (n_line,) int8
    k: int                   # budget K
    case_name: str           # e.g. "case118"
    target_text: str         # original SFT target text
    gt_file: Optional[str]   # basename of gt file if available

def build_examples(io_dir: Path, raw_rows: List[Dict], split_ids: List[str],
                   default_k: int) -> Tuple[List[Example], str, int]:
    """
    Returns (examples, indexing, n_line)
    """
    row_by_sid: Dict[str, Dict] = {}
    for r in raw_rows:
        sid = basename_any(r.get("xi_file", "")) or basename_any(r.get("summary_file", "")) or basename_any(r.get("gt_file", ""))
        if sid:
            row_by_sid[sid] = r

    # Keep split order stable (split_ids comes from the index files; order not guaranteed).
    # We'll sort for determinism.
    split_ids = sorted([sid for sid in split_ids if sid in row_by_sid])

    examples: List[Example] = []
    indexing = "1-based"
    n_line = None

    for sid in split_ids:
        r = row_by_sid[sid]
        xi_name = basename_any(r["xi_file"])
        xi_path = io_dir / xi_name
        if not xi_path.exists():
            raise FileNotFoundError(f"Missing xi file in io/: {xi_name}")

        xi_list = load_json(xi_path)
        if not isinstance(xi_list, list):
            raise ValueError(f"{xi_name} expected JSON list xi, got {type(xi_list)}")
        xi = np.asarray(xi_list, dtype=np.int8)

        if n_line is None:
            n_line = int(xi.shape[0])
        elif int(xi.shape[0]) != n_line:
            raise ValueError(f"Inconsistent n_line: expected {n_line}, got {xi.shape[0]} in {xi_name}")

        target_text = r.get("target", "")
        open_ids = parse_open_ids(target_text)
        indexing = infer_indexing(open_ids)

        y = make_multi_hot(open_ids, n_line, indexing=indexing)

        # K: from prompt JSON if possible
        k = default_k
        try:
            prompt_obj = json.loads(r.get("prompt", "{}"))
            if isinstance(prompt_obj, dict) and "toggle_budget" in prompt_obj:
                k = int(prompt_obj["toggle_budget"])
            case_name = str(prompt_obj.get("case", "case118"))
        except Exception:
            case_name = "case118"

        gt_file = basename_any(r.get("gt_file", "")) if r.get("gt_file") else None

        examples.append(Example(
            sid=sid,
            xi=xi,
            y=y,
            k=k,
            case_name=case_name,
            target_text=target_text,
            gt_file=gt_file,
        ))

    if n_line is None:
        n_line = 0

    return examples, indexing, n_line

class XiToOpenDataset(Dataset):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        x = torch.tensor(ex.xi.astype(np.float32), dtype=torch.float32)
        y = torch.tensor(ex.y.astype(np.float32), dtype=torch.float32)
        k = int(ex.k)
        return x, y, k, ex.sid

# ---------------------------
# Model
# ---------------------------

class MLPLineScorer(nn.Module):
    def __init__(self, n_line: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_line, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_line),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------------------------
# Metrics + decoding
# ---------------------------

def decode_topk(logits: torch.Tensor, xi: torch.Tensor, k: int, indexing: str) -> List[int]:
    """
    logits: (n_line,)
    xi:     (n_line,) float 0/1
    Select top-k among eligible lines where xi==1.
    Return list of line IDs in the chosen indexing convention (1-based typical).
    """
    with torch.no_grad():
        scores = logits.detach().clone()
        eligible = (xi > 0.5)
        scores[~eligible] = -1e30  # never pick forced-out lines
        k = max(0, int(k))
        if k == 0:
            return []
        k = min(k, scores.numel())
        topk = torch.topk(scores, k=k, largest=True).indices.cpu().numpy().tolist()
        if indexing == "0-based":
            return [int(i) for i in topk]
        return [int(i) + 1 for i in topk]

def set_metrics(pred: List[int], true: List[int]) -> Dict[str, float]:
    P, T = set(pred), set(true)
    inter = len(P & T)
    union = len(P | T) if (P | T) else 1
    prec = inter / (len(P) if len(P) else 1)
    rec  = inter / (len(T) if len(T) else 1)
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    jac  = inter / union
    exact = 1.0 if P == T else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "jaccard": jac, "exact": exact}

def aggregate(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {k: 0.0 for k in ["precision","recall","f1","jaccard","exact"]}
    keys = metrics[0].keys()
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys}

# ---------------------------
# Optional MATLAB verifier
# ---------------------------

def matlab_quote(s: str) -> str:
    """MATLAB single-quoted literal with escaping."""
    return "'" + s.replace("'", "''") + "'"

def run_matlab_verify(repo_root: Path, case_name: str, plan_json: Path, xi_json: Path,
                      limits_yml: Path, out_json: Path, hybrid: bool) -> Dict:
    """
    Calls either verify_plan or verify_plan_hybrid and returns parsed JSON output.
    Requires `matlab` on PATH.
    """
    matlab_dir = repo_root / "matlab"
    fn = "verify_plan_hybrid" if hybrid else "verify_plan"

    # Normalize paths for MATLAB (forward slashes) using Path.as_posix()
    matlab_dir_s = matlab_dir.as_posix()
    plan_s = plan_json.as_posix()
    xi_s = xi_json.as_posix()
    limits_s = limits_yml.as_posix()
    out_s = out_json.as_posix()

    cmd = (
        f"addpath({matlab_quote(matlab_dir_s)});"
        f"{fn}("
        f"{matlab_quote(case_name)},"
        f"{matlab_quote(plan_s)},"
        f"{matlab_quote(xi_s)},"
        f"{matlab_quote(limits_s)},"
        f"{matlab_quote(out_s)}"
        f");"
    )

    proc = subprocess.run(
        ["matlab", "-batch", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"MATLAB failed ({fn}) returncode={proc.returncode}\n{proc.stdout}")
    if not out_json.exists():
        raise FileNotFoundError(f"MATLAB did not produce output JSON: {out_json}")
    return load_json(out_json)


# ---------------------------
# Train / Eval
# ---------------------------

def compute_pos_weight(Y: np.ndarray, eps: float = 1.0) -> torch.Tensor:
    """
    pos_weight[j] = (neg_j + eps) / (pos_j + eps)
    for BCEWithLogitsLoss to counter class imbalance.
    """
    pos = Y.sum(axis=0).astype(np.float64)
    neg = (Y.shape[0] - pos).astype(np.float64)
    w = (neg + eps) / (pos + eps)
    return torch.tensor(w, dtype=torch.float32)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_model(model: nn.Module, loader: DataLoader, examples_by_sid: Dict[str, Example],
               device: str, indexing: str, max_batches: Optional[int] = None) -> Tuple[Dict[str,float], List[Dict]]:
    model.eval()
    per_ex_rows = []
    all_m = []
    batches = 0
    with torch.no_grad():
        for x, y, k, sid in loader:
            x = x.to(device)
            logits = model(x)  # (B, n_line)
            for i in range(x.shape[0]):
                sid_i = sid[i]
                ex = examples_by_sid[sid_i]
                k_i = int(k[i].item())
                pred_ids = decode_topk(logits[i], x[i], k_i, indexing=indexing)
                true_ids = ids_from_multi_hot(ex.y, indexing=indexing)
                m = set_metrics(pred_ids, true_ids)
                all_m.append(m)
                per_ex_rows.append({
                    "sid": sid_i,
                    "k": k_i,
                    "pred_ids": pred_ids,
                    "true_ids": true_ids,
                    **m
                })
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    return aggregate(all_m), per_ex_rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default=None, help="Repo root. Default: parent of this script (python/..).")
    ap.add_argument("--io", type=str, default=None, help="Path to io/. Default: <repo>/io.")
    ap.add_argument("--run", type=str, default="auto", help="auto | none | run0 | run1 | ...")

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    ap.add_argument("--K", type=int, default=None, help="Fallback budget if not in prompt/limits.yml")

    # MATLAB verifier
    ap.add_argument("--matlab_eval", type=int, default=0, help="0/1: run verify_plan + verify_plan_hybrid on predictions")
    ap.add_argument("--matlab_max", type=int, default=0, help="If >0, only run MATLAB eval for first N test cases")
    args = ap.parse_args()

    # Repo root inference
    if args.repo is None:
        here = Path(__file__).resolve()
        repo_root = here.parent.parent  # assume script lives in <repo>/python/
    else:
        repo_root = Path(args.repo).expanduser().resolve()

    io_dir = Path(args.io).expanduser().resolve() if args.io else (repo_root / "io")
    if not io_dir.exists():
        raise FileNotFoundError(f"io dir not found: {io_dir}")

    if args.run == "auto":
        run_tag = find_latest_run(io_dir)
    elif args.run in ("none", "", "norun"):
        run_tag = ""
    else:
        run_tag = args.run

    raw_path, train_idx_path, test_idx_path = get_paths(io_dir, run_tag)
    for p in (raw_path, train_idx_path, test_idx_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    set_seed(args.seed)

    # Budget default
    k_limits = try_read_budget(repo_root)
    default_k = int(args.K) if args.K is not None else (k_limits if k_limits is not None else 3)

    # Load SFT artifacts
    raw_rows = load_jsonl(raw_path)
    train_ids = [basename_any(p) for p in load_json(train_idx_path)]
    test_ids  = [basename_any(p) for p in load_json(test_idx_path)]

    # Build examples
    train_ex, indexing, n_line = build_examples(io_dir, raw_rows, train_ids, default_k)
    test_ex,  indexing2, n_line2 = build_examples(io_dir, raw_rows, test_ids, default_k)
    if indexing2 != indexing:
        print(f"[warn] indexing differs train={indexing} test={indexing2}; using train indexing.")
    if n_line2 != n_line:
        raise ValueError(f"n_line mismatch train={n_line} test={n_line2}")

    if n_line == 0:
        raise RuntimeError("No examples found. Check your io/sft_* files and paths.")

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # DataLoaders
    train_ds = XiToOpenDataset(train_ex)
    test_ds  = XiToOpenDataset(test_ex)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Model + loss
    model = MLPLineScorer(n_line=n_line, hidden=args.hidden, dropout=args.dropout).to(device)

    # pos_weight from training labels
    Ytr = np.stack([ex.y for ex in train_ex], axis=0).astype(np.int8)
    pos_weight = compute_pos_weight(Ytr).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # For eval mapping
    train_by_sid = {ex.sid: ex for ex in train_ex}
    test_by_sid  = {ex.sid: ex for ex in test_ex}

    # Output paths
    suffix = f"_{run_tag}" if run_tag else ""
    ckpt_path = io_dir / f"nn_model{suffix}.pt"
    log_path  = io_dir / f"nn_train_log{suffix}.json"
    pred_csv  = io_dir / f"nn_eval_predictions{suffix}.csv"
    sum_json  = io_dir / f"nn_eval_summary{suffix}.json"

    # Training loop
    best_key = -1.0
    train_log = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y, k, sid in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = bce(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        # Quick eval each epoch (set-based)
        train_metrics, _ = eval_model(model, train_loader, train_by_sid, device, indexing)
        test_metrics, _  = eval_model(model, test_loader,  test_by_sid,  device, indexing)

        row = {
            "epoch": epoch,
            "loss": epoch_loss,
            "train": train_metrics,
            "test": test_metrics,
            "time_sec": time.time() - t0,
        }
        train_log.append(row)

        # Select best by test jaccard (stable) then f1
        key = test_metrics.get("jaccard", 0.0) + 1e-3 * test_metrics.get("f1", 0.0)
        if key > best_key:
            best_key = key
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_line": n_line,
                "indexing": indexing,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "seed": args.seed,
                "default_k": default_k,
                "run_tag": run_tag,
            }, ckpt_path)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}/{args.epochs} | loss {epoch_loss:.4f} | "
                  f"test jacc {test_metrics['jaccard']:.3f} f1 {test_metrics['f1']:.3f} exact {test_metrics['exact']:.3f}")

    log_path.write_text(json.dumps(train_log, indent=2), encoding="utf-8")
    print(f"[OK] wrote training log: {log_path}")
    print(f"[OK] wrote best checkpoint: {ckpt_path}")

    # Load best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Final evaluation with per-example rows
    test_metrics, per_rows = eval_model(model, test_loader, test_by_sid, device, indexing)

    # Add richer per-example info: target text, gt J, etc + optional MATLAB metrics
    limits_yml = repo_root / "config" / "limits.yml"
    do_matlab = (args.matlab_eval == 1)
    matlab_cap = int(args.matlab_max) if args.matlab_max else 0
    tmp_dir = io_dir / f"nn_eval_tmp{suffix}"
    if do_matlab:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare CSV
    fieldnames = [
        "sid","case","k",
        "pred_text","true_text",
        "pred_ids","true_ids",
        "precision","recall","f1","jaccard","exact",
        "gt_J","gt_opt_switches",
        # optional MATLAB outputs:
        "pred_dc_feasible","pred_J_dc","pred_shed_MW",
        "pred_hybrid_feasible","pred_J_ac","pred_V_pen","pred_ac_ok","pred_notes",
    ]

    rows_out = []
    for idx, r in enumerate(per_rows):
        sid = r["sid"]
        ex = test_by_sid[sid]
        gt_J = ""
        gt_switches = ""
        if ex.gt_file:
            gt_path = io_dir / ex.gt_file
            if gt_path.exists():
                try:
                    gt = load_json(gt_path)
                    gt_J = gt.get("J", "")
                    gt_switches = gt.get("opt_switches", "")
                except Exception:
                    pass

        pred_ids = r["pred_ids"]
        true_ids = r["true_ids"]
        pred_text = "; ".join([f"open({lid})" for lid in pred_ids])
        true_text = ex.target_text

        out = {
            "sid": sid,
            "case": ex.case_name,
            "k": ex.k,
            "pred_text": pred_text,
            "true_text": true_text,
            "pred_ids": pred_ids,
            "true_ids": true_ids,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "jaccard": r["jaccard"],
            "exact": r["exact"],
            "gt_J": gt_J,
            "gt_opt_switches": gt_switches,
            "pred_dc_feasible": "",
            "pred_J_dc": "",
            "pred_shed_MW": "",
            "pred_hybrid_feasible": "",
            "pred_J_ac": "",
            "pred_V_pen": "",
            "pred_ac_ok": "",
            "pred_notes": "",
        }

        if do_matlab and limits_yml.exists():
            if matlab_cap <= 0 or idx < matlab_cap:
                # write plan JSON (corridor_actions list)
                plan = {"corridor_actions": [{"action":"open","line": int(lid)} for lid in pred_ids]}
                plan_path = tmp_dir / f"plan_{sid}.json"
                xi_path   = io_dir / basename_any(ex.sid)  # sid is xi basename in our split convention
                # NOTE: ex.sid is the *xi filename* (psps_*.json). So xi_path resolves to io/psps_*.json.
                plan_path.write_text(json.dumps(plan), encoding="utf-8")

                # DC verifier
                out_dc = tmp_dir / f"out_dc_{sid}.json"
                out_hy = tmp_dir / f"out_hybrid_{sid}.json"
                try:
                    dc = run_matlab_verify(repo_root, ex.case_name, plan_path, xi_path, limits_yml, out_dc, hybrid=False)
                    hy = run_matlab_verify(repo_root, ex.case_name, plan_path, xi_path, limits_yml, out_hy, hybrid=True)
                    out.update({
                        "pred_dc_feasible": dc.get("feasible",""),
                        "pred_J_dc": dc.get("J",""),
                        "pred_shed_MW": dc.get("shed_MW",""),
                        "pred_hybrid_feasible": hy.get("feasible",""),
                        "pred_J_ac": hy.get("J_ac",""),
                        "pred_V_pen": hy.get("V_pen",""),
                        "pred_ac_ok": hy.get("ac_ok",""),
                        "pred_notes": hy.get("notes",""),
                    })
                except Exception as e:
                    out["pred_notes"] = f"MATLAB eval failed: {e}"

        rows_out.append(out)

    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            # stringify lists for CSV
            r = dict(r)
            r["pred_ids"] = json.dumps(r["pred_ids"])
            r["true_ids"] = json.dumps(r["true_ids"])
            r["gt_opt_switches"] = json.dumps(r["gt_opt_switches"]) if isinstance(r["gt_opt_switches"], (list,dict)) else str(r["gt_opt_switches"])
            w.writerow(r)

    summary = {
        "run_tag": run_tag,
        "indexing": indexing,
        "n_line": n_line,
        "default_k": default_k,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "seed": args.seed,
        "test_metrics": test_metrics,
        "matlab_eval": bool(do_matlab),
        "matlab_max": matlab_cap,
        "outputs": {
            "checkpoint": str(ckpt_path),
            "train_log": str(log_path),
            "pred_csv": str(pred_csv),
        },
    }
    sum_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] wrote eval CSV: {pred_csv}")
    print(f"[OK] wrote summary: {sum_json}")
    print("Test metrics:", json.dumps(test_metrics, indent=2))

if __name__ == "__main__":
    main()
