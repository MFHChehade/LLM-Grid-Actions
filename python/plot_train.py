# python/plot_train.py
"""
Generic, publishable training plots for OpenAI fine-tuning jobs (SFT + DPO),
styled to match black/gray paper figures (no titles, no legends).

Saves to:
  io/figs_train/

What it plots:
  --mode dpo : Loss + Error rate
  --mode sft : Log loss + Accuracy

It will pull metrics from the richest available source:
  events -> results csv -> checkpoints

Usage (PowerShell, one line):
  python python/plot_train.py --mode dpo --job ftjob-... --mode sft --job ftjob-... --window 11
  eg: python python/plot_train.py --mode dpo --job ftjob-yTDaCb2LYPa25ID9bAe0qrFb --mode sft --job ftjob-Gmjj8uGa5sBCMuEVT9udaPRZ --window 11
"""

from __future__ import annotations

import argparse
import io
import re
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from openai import OpenAI


# -------------------------
# Repo paths
# -------------------------
ROOT    = Path(__file__).resolve().parents[1]
IO_DIR  = ROOT / "io"
FIG_DIR = IO_DIR / "figs_train"

FIG_DPI = 800  # print-ready


# -------------------------
# Global publishable styling (VERY LARGE FONTS, THIN LINES)
# -------------------------
def set_pub_rcparams() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        # MUCH larger than before
        "font.size": 32,
        "axes.labelsize": 44,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,

        # thin + clean
        "axes.linewidth": 0.95,
        "lines.linewidth": 1.0,

        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "axes.unicode_minus": False,
    })


def _format_pub_ax(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.95)
        ax.spines[side].set_color("black")

    ax.tick_params(direction="out", length=7.0, width=0.95, colors="black")
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="out", length=3.5, width=0.7, colors="black")

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 4))
    ax.yaxis.set_major_formatter(fmt)


# -------------------------
# Smoothing (no edge artifacts)
# -------------------------
def smooth_centered(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with min_periods=1 (no boundary cliffs)."""
    y = pd.Series(np.asarray(y, dtype=float))
    if window <= 1:
        return y.to_numpy()
    return y.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def save_pub_plot(
    x: np.ndarray,
    y: np.ndarray,
    out_base: Path,
    xlabel: str,
    ylabel: str,
    window: int = 0,
    y01: bool = False,
    raw_as_scatter: bool = False,
) -> None:
    """
    Paper-like style: black/gray, dashed horizontal grid, thin spines.
    No title, no legend.
    Saves: PDF + SVG + high-DPI PNG.
    """
    set_pub_rcparams()

    x = np.asarray(x)
    y = np.asarray(y, dtype=float)

    # slightly larger canvas helps big fonts breathe
    fig, ax = plt.subplots(figsize=(8.4, 3.9), dpi=170)

    # raw in light gray (de-emphasized)
    if raw_as_scatter:
        ax.scatter(x, y, s=26, alpha=0.18, linewidths=0, color="0.72")
    else:
        ax.plot(x, y, linewidth=1.0, alpha=0.20, color="0.72")

    # smoothed in black (emphasized)
    if window and window > 1:
        ys = smooth_centered(y, window)
        ax.plot(x, ys, linewidth=2.6, color="black")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if y01:
        ax.set_ylim(-0.02, 1.02)

    _format_pub_ax(ax)
    fig.tight_layout(pad=0.25)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".svg"))
    fig.savefig(out_base.with_suffix(".png"), dpi=FIG_DPI)
    plt.close(fig)


# -------------------------
# Fetchers (events -> results CSV -> checkpoints)
# -------------------------
def _maybe_decode_base64_csv(raw: bytes) -> bytes:
    """Detect & decode base64 if it decodes to a CSV-like header."""
    try:
        txt = raw.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return raw

    looks_b64 = len(txt) > 50 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", txt) is not None
    if not looks_b64:
        return raw

    try:
        dec = base64.b64decode(txt)
        head = dec.decode("utf-8", errors="ignore").splitlines()[0].lower()
        if "step" in head and "," in head:
            return dec
    except Exception:
        pass

    return raw


def fetch_events_df(client: OpenAI, job_id: str, max_events: int = 5000) -> Optional[pd.DataFrame]:
    all_events = []
    after = None

    while True:
        page = client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id,
            limit=100,
            after=after
        )
        all_events.extend(page.data)

        if len(all_events) >= max_events or not page.has_more:
            break
        after = page.data[-1].id

    rows: List[Dict[str, Any]] = []
    metric_keys = {
        "train_loss", "valid_loss", "loss",
        "train_accuracy", "valid_accuracy", "accuracy",
        "train_mean_token_accuracy", "valid_mean_token_accuracy",
        "train_error_rate", "valid_error_rate", "error_rate",
    }

    for e in all_events:
        d = getattr(e, "data", None)
        if isinstance(d, dict) and "step" in d and any(k in d for k in metric_keys):
            rows.append(d)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    if "step" not in df.columns:
        return None

    df = df.dropna(how="all")
    df = df.sort_values("step").drop_duplicates("step", keep="last")
    return df


def fetch_results_csv_df(client: OpenAI, job_id: str) -> Optional[pd.DataFrame]:
    job = client.fine_tuning.jobs.retrieve(job_id)
    result_files = getattr(job, "result_files", None) or []
    if not result_files:
        return None

    file_id = result_files[0]
    raw = client.files.content(file_id).read()
    raw = _maybe_decode_base64_csv(raw)

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return None

    df.columns = [str(c).strip() for c in df.columns]
    if "step" not in df.columns:
        return None

    df = df.dropna(how="all")
    df = df.sort_values("step").drop_duplicates("step", keep="last")
    return df


def fetch_checkpoints_df(client: OpenAI, job_id: str, max_ckpts: int = 2000) -> Optional[pd.DataFrame]:
    all_ckpts = []
    after = None

    while True:
        page = client.fine_tuning.jobs.checkpoints.list(
            fine_tuning_job_id=job_id,
            limit=100,
            after=after
        )
        all_ckpts.extend(page.data)

        if len(all_ckpts) >= max_ckpts or not page.has_more:
            break
        after = page.data[-1].id

    if not all_ckpts:
        return None

    rows = []
    for c in all_ckpts:
        m = c.metrics
        rows.append({
            "step": getattr(c, "step_number", None),
            "train_loss": getattr(m, "train_loss", None),
            "valid_loss": getattr(m, "valid_loss", None),
            "train_accuracy": getattr(m, "train_accuracy", None),
            "valid_accuracy": getattr(m, "valid_accuracy", None),
            "train_mean_token_accuracy": getattr(m, "train_mean_token_accuracy", None),
            "valid_mean_token_accuracy": getattr(m, "valid_mean_token_accuracy", None),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["step"]).sort_values("step").drop_duplicates("step", keep="last")
    return df


def _score_df(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty or "step" not in df.columns:
        return 0
    step = pd.to_numeric(df["step"], errors="coerce").dropna()
    n = int(step.nunique())
    bonus = 0
    for col in [
        "train_loss", "valid_loss", "loss",
        "train_accuracy", "valid_accuracy", "accuracy",
        "train_mean_token_accuracy", "valid_mean_token_accuracy",
        "train_error_rate", "valid_error_rate", "error_rate",
    ]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            bonus += 50
    return n + bonus


def choose_best_df(events_df, results_df, ckpt_df) -> Tuple[pd.DataFrame, str]:
    cand = [("events", events_df), ("results", results_df), ("checkpoints", ckpt_df)]
    name, best = max(cand, key=lambda t: _score_df(t[1]))
    if best is None or best.empty:
        raise RuntimeError("No usable metrics found from events/results/checkpoints.")
    return best.copy(), name


# -------------------------
# Robust metric picking
# -------------------------
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pick_series(df: pd.DataFrame, keys: List[str]) -> Optional[pd.Series]:
    for k in keys:
        if k in df.columns and _num(df[k]).notna().any():
            return _num(df[k])
    return None


def pick_train_loss(df: pd.DataFrame) -> Optional[pd.Series]:
    return pick_series(df, ["train_loss", "loss"])


def pick_train_accuracy(df: pd.DataFrame) -> Optional[pd.Series]:
    return pick_series(df, ["train_mean_token_accuracy", "train_accuracy", "accuracy"])


def pick_train_error_rate(df: pd.DataFrame) -> Optional[pd.Series]:
    er = pick_series(df, ["train_error_rate", "error_rate"])
    if er is not None and er.notna().any():
        return er
    acc = pick_train_accuracy(df)
    if acc is not None and acc.notna().any():
        return 1.0 - acc
    return None


# -------------------------
# Plot per (mode, job)
# -------------------------
def plot_job(
    client: OpenAI,
    job_id: str,
    mode: str,
    window: int,
    max_events: int,
    max_ckpts: int,
) -> None:
    events_df  = fetch_events_df(client, job_id, max_events=max_events)
    results_df = fetch_results_csv_df(client, job_id)
    ckpt_df    = fetch_checkpoints_df(client, job_id, max_ckpts=max_ckpts)

    df, src = choose_best_df(events_df, results_df, ckpt_df)

    df.columns = [str(c).strip() for c in df.columns]
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"]).sort_values("step")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    used_csv = FIG_DIR / f"metrics_{mode}_{job_id}_{src}.csv"
    df.to_csv(used_csv, index=False)

    loss = pick_train_loss(df)
    acc  = pick_train_accuracy(df)
    er   = pick_train_error_rate(df)

    mode = mode.lower().strip()
    if mode not in ("dpo", "sft"):
        raise ValueError(f"Unknown mode '{mode}'. Use 'dpo' or 'sft'.")

    raw_scatter_loss = (mode == "sft")
    raw_scatter_acc  = True

    step = df["step"].to_numpy()

    if mode == "dpo":
        if loss is not None and loss.notna().any():
            save_pub_plot(
                step, loss.to_numpy(),
                FIG_DIR / f"dpo_{job_id}_loss",
                xlabel="Step",
                ylabel="Loss",
                window=window,
                y01=False,
                raw_as_scatter=False,
            )
        else:
            print(f"[dpo] No loss found for {job_id} (source={src}).")

        if er is not None and pd.to_numeric(er, errors="coerce").notna().any():
            save_pub_plot(
                step, pd.to_numeric(er, errors="coerce").to_numpy(),
                FIG_DIR / f"dpo_{job_id}_error_rate",
                xlabel="Step",
                ylabel="Error rate",
                window=window,
                y01=True,
                raw_as_scatter=False,
            )
        else:
            print(f"[dpo] No error-rate (or accuracy to derive it) found for {job_id} (source={src}).")

    else:  # sft
        if loss is not None and loss.notna().any():
            save_pub_plot(
                step, loss.to_numpy(),
                FIG_DIR / f"sft_{job_id}_log_loss",
                xlabel="Step",
                ylabel="Log loss",
                window=window,
                y01=False,
                raw_as_scatter=raw_scatter_loss,
            )
        else:
            print(f"[sft] No loss found for {job_id} (source={src}).")

        if acc is not None and acc.notna().any():
            save_pub_plot(
                step, acc.to_numpy(),
                FIG_DIR / f"sft_{job_id}_accuracy",
                xlabel="Step",
                ylabel="Accuracy",
                window=window,
                y01=True,
                raw_as_scatter=raw_scatter_acc,
            )
        else:
            print(f"[sft] No accuracy found for {job_id} (source={src}).")

    print(f"[{mode}] {job_id}: saved figures to {FIG_DIR}")
    print(f"[{mode}] {job_id}: saved metrics CSV used for plots: {used_csv}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", action="append", required=True, help="dpo or sft (repeatable)")
    p.add_argument("--job",  action="append", required=True, help="ftjob-... (repeatable)")
    p.add_argument("--window", type=int, default=11, help="Centered moving average window (0=no smoothing)")
    p.add_argument("--max-events", type=int, default=5000)
    p.add_argument("--max-ckpts", type=int, default=2000)
    args = p.parse_args()

    if len(args.mode) != len(args.job):
        raise SystemExit("You must pass the same number of --mode and --job arguments.")

    client = OpenAI()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for mode, job_id in zip(args.mode, args.job):
        plot_job(
            client=client,
            job_id=job_id,
            mode=mode,
            window=args.window,
            max_events=args.max_events,
            max_ckpts=args.max_ckpts,
        )

    print(f"Done. All figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
