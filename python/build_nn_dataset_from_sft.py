#!/usr/bin/env python3
"""
build_nn_dataset_from_sft.py

Build a *pure supervised-learning* dataset for a neural net that maps:
    PSPS outage mask xi  ->  corrective open actions (multi-hot vector)

It reuses your existing SFT artifacts:
  - io/sft_raw(_runX).jsonl            (prompt/target + file pointers)
  - io/sft_train_index(_runX).json     (train split, list of psps_*.json paths)
  - io/sft_test_index(_runX).json      (test split)

Outputs (npz):
  - io/nn_supervised_train_runX.npz
  - io/nn_supervised_test_runX.npz
and a small manifest:
  - io/nn_supervised_meta_runX.json

Each NPZ contains:
  X: int8 array shape (N, n_line)   (the outage/availability mask xi)
  Y: int8 array shape (N, n_line)   (multi-hot "open these lines" labels)
  ids: array of strings length N    (psps filename / scenario id)
  target_text: array of strings     (original SFT target string)

Notes:
  - The PSPS files io/psps_*.json in this repo store xi as a JSON list of length n_line.
  - The SFT target is parsed from text like: "open(41); open(S3:128); open(132)".
  - Line IDs are assumed 1-based unless a 0 appears in the targets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

OPEN_RE = re.compile(r"open\(\s*(?:S\d+\s*:\s*)?(\d+)\s*\)", re.IGNORECASE)

def _find_latest_run(io_dir: Path) -> str:
    """
    Returns "runX" where X is the max run number found in io/sft_raw_runX.jsonl.
    If none found, returns "" (meaning non-run files).
    """
    candidates = sorted(io_dir.glob("sft_raw_run*.jsonl"))
    if not candidates:
        return ""
    # pick max run number
    best = None
    best_r = -1
    for p in candidates:
        m = re.search(r"run(\d+)", p.name)
        if m:
            r = int(m.group(1))
            if r > best_r:
                best_r = r
                best = p
    return f"run{best_r}" if best_r >= 0 else ""

def _load_json_list(path: Path) -> List:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _basename(p: str) -> str:
    # Handle Windows paths inside JSON (C:\...) and POSIX; just take the leaf name.
    return os.path.basename(p.replace("\\", "/"))

def _parse_open_ids(target: str) -> List[int]:
    return [int(m.group(1)) for m in OPEN_RE.finditer(target or "")]

def _infer_indexing(open_ids: List[int], n_line: int) -> str:
    # If any 0 appears, treat as 0-based. Otherwise assume 1-based.
    if any(i == 0 for i in open_ids):
        return "0-based"
    return "1-based"

def _make_multi_hot(open_ids: List[int], n_line: int, indexing: str) -> np.ndarray:
    y = np.zeros((n_line,), dtype=np.int8)
    if not open_ids:
        return y

    if indexing == "0-based":
        for lid in open_ids:
            if 0 <= lid < n_line:
                y[lid] = 1
            else:
                raise ValueError(f"open({lid}) out of range for n_line={n_line} (0-based)")
    else:
        for lid in open_ids:
            if 1 <= lid <= n_line:
                y[lid - 1] = 1
            else:
                raise ValueError(f"open({lid}) out of range for n_line={n_line} (1-based)")
    return y

def build(io_dir: Path, run_tag: str) -> Tuple[Path, Path, Path]:
    """
    Build and save train/test npz, return their paths and meta path.
    run_tag is "" or "runX".
    """
    suffix = f"_{run_tag}" if run_tag else ""
    raw_path = io_dir / f"sft_raw{suffix}.jsonl"
    train_index_path = io_dir / f"sft_train_index{suffix}.json"
    test_index_path  = io_dir / f"sft_test_index{suffix}.json"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")
    if not train_index_path.exists():
        raise FileNotFoundError(f"Missing {train_index_path}")
    if not test_index_path.exists():
        raise FileNotFoundError(f"Missing {test_index_path}")

    raw_rows = _load_jsonl(raw_path)

    train_ids = set(_basename(p) for p in _load_json_list(train_index_path))
    test_ids  = set(_basename(p) for p in _load_json_list(test_index_path))

    # Map psps basename -> row
    row_by_id: Dict[str, Dict] = {}
    for r in raw_rows:
        psps_name = _basename(r.get("xi_file", ""))
        if not psps_name:
            # fallback: try to infer from summary/gt file
            psps_name = _basename(r.get("summary_file", "")) or _basename(r.get("gt_file", ""))
        if psps_name:
            row_by_id[psps_name] = r

    # Build ordered splits
    train_list = [pid for pid in train_ids if pid in row_by_id]
    test_list  = [pid for pid in test_ids  if pid in row_by_id]

    missing_train = sorted([pid for pid in train_ids if pid not in row_by_id])
    missing_test  = sorted([pid for pid in test_ids  if pid not in row_by_id])

    def make_split(psps_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        Xs = []
        Ys = []
        ids = []
        targets = []

        indexing = None
        n_line = None

        for pid in psps_names:
            row = row_by_id[pid]
            xi_name = _basename(row["xi_file"])
            xi_path = io_dir / xi_name
            if not xi_path.exists():
                raise FileNotFoundError(f"Cannot find PSPS xi file in io/: {xi_name}")

            xi = json.load(xi_path.open("r", encoding="utf-8"))
            if not isinstance(xi, list):
                raise ValueError(f"{xi_name} should be a JSON list, got {type(xi)}")
            xi = np.asarray(xi, dtype=np.int8)

            if n_line is None:
                n_line = int(xi.shape[0])
            elif int(xi.shape[0]) != n_line:
                raise ValueError(f"Inconsistent n_line: expected {n_line}, got {xi.shape[0]} in {xi_name}")

            open_ids = _parse_open_ids(row.get("target", ""))
            if indexing is None:
                indexing = _infer_indexing(open_ids, n_line)

            y = _make_multi_hot(open_ids, n_line, indexing=indexing)

            Xs.append(xi)
            Ys.append(y)
            ids.append(pid)
            targets.append(row.get("target", ""))

        if n_line is None:
            n_line = 0
            indexing = "1-based"

        X = np.stack(Xs, axis=0) if Xs else np.zeros((0, n_line), dtype=np.int8)
        Y = np.stack(Ys, axis=0) if Ys else np.zeros((0, n_line), dtype=np.int8)
        return X, Y, np.array(ids, dtype=object), np.array(targets, dtype=object), indexing

    Xtr, Ytr, id_tr, tgt_tr, indexing_tr = make_split(train_list)
    Xte, Yte, id_te, tgt_te, indexing_te = make_split(test_list)

    # sanity: expect same indexing
    indexing = indexing_tr if Xtr.shape[0] else indexing_te

    out_train = io_dir / f"nn_supervised_train{suffix}.npz"
    out_test  = io_dir / f"nn_supervised_test{suffix}.npz"
    out_meta  = io_dir / f"nn_supervised_meta{suffix}.json"

    np.savez_compressed(out_train, X=Xtr, Y=Ytr, ids=id_tr, target_text=tgt_tr)
    np.savez_compressed(out_test,  X=Xte, Y=Yte, ids=id_te, target_text=tgt_te)

    meta = {
        "run_tag": run_tag,
        "raw_path": str(raw_path),
        "train_index_path": str(train_index_path),
        "test_index_path": str(test_index_path),
        "n_line": int(Xtr.shape[1]) if Xtr.shape[0] else int(Xte.shape[1]) if Xte.shape[0] else None,
        "indexing": indexing,  # "1-based" or "0-based" interpretation of open(line_id)
        "train_N": int(Xtr.shape[0]),
        "test_N": int(Xte.shape[0]),
        "missing_train_ids_count": len(missing_train),
        "missing_test_ids_count": len(missing_test),
        "missing_train_ids_preview": missing_train[:10],
        "missing_test_ids_preview": missing_test[:10],
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_train, out_test, out_meta

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", type=str, default=None,
                    help="Path to io/ directory. Default: repo_root/io (repo_root inferred from this file location).")
    ap.add_argument("--run", type=str, default="auto",
                    help="Which run tag to use: 'auto' (default), 'run0', 'run1', ... or 'none' for non-run files.")
    args = ap.parse_args()

    if args.io is None:
        # infer repo root as parent of this file
        # If you place this under repo_root/python/, parent.parent is repo_root.
        # If you keep it elsewhere, pass --io explicitly.
        here = Path(__file__).resolve()
        guess_root = here.parent.parent
        io_dir = guess_root / "io"
    else:
        io_dir = Path(args.io).expanduser().resolve()

    if not io_dir.exists():
        raise FileNotFoundError(f"io dir not found: {io_dir}")

    if args.run == "auto":
        run_tag = _find_latest_run(io_dir)
    elif args.run in ("none", "", "norun"):
        run_tag = ""
    else:
        run_tag = args.run

    out_train, out_test, out_meta = build(io_dir, run_tag)
    print(f"[OK] wrote: {out_train}")
    print(f"[OK] wrote: {out_test}")
    print(f"[OK] wrote: {out_meta}")

if __name__ == "__main__":
    main()
