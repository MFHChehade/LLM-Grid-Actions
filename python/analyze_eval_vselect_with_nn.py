# python/analyze_eval_vselect_with_nn.py
#
# Combine LLM v-select evaluation with NN v-select evaluation, then produce
# publishable stats + figures comparing models on J_dc, V_pen (J_ac), and reliability.

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT    = Path(__file__).resolve().parents[1]
IO_DIR  = ROOT / "io"
CONFIG  = ROOT / "config"
LIMITS_YML = CONFIG / "limits.yml"

CSV_LLM_DEFAULT = IO_DIR / "eval_vselect_jdc_jac_per_xi.csv"

CSV_MERGED_DEFAULT   = IO_DIR / "eval_vselect_jdc_jac_per_xi_with_nn.csv"
FIG_DIR_DEFAULT      = IO_DIR / "figs_eval_with_nn"
GLOBAL_SUMMARY_CSV   = IO_DIR / "eval_stats_global_with_nn.csv"
SUMMARY_CSV          = IO_DIR / "eval_stats_per_model_with_nn.csv"
PAIRWISE_CSV         = IO_DIR / "eval_stats_pairwise_with_nn.csv"

JAC_SCALE = 30000.0
AC_FAIL_SENTINEL_RAW = 1e6
AC_FAIL_SENTINEL     = AC_FAIL_SENTINEL_RAW / JAC_SCALE

FIG_DPI = 700


def set_pub_rcparams() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "font.size": 22,
        "axes.labelsize": 28,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,

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

    ax.tick_params(direction="out", length=6.0, width=0.95, colors="black")
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="out", length=3.0, width=0.7, colors="black")

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 4))
    ax.yaxis.set_major_formatter(fmt)


def disp_name(m: str) -> str:
    """Pretty display name for plots."""
    return "zero shot" if m == "zero_shot" else m


def _coerce_numeric(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def _savefig(fig_dir: Path, base_name: str, dpi: int = FIG_DPI) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{base_name}.png", dpi=dpi)
    plt.savefig(fig_dir / f"{base_name}.pdf")


def read_gamma_from_limits() -> float:
    try:
        txt = LIMITS_YML.read_text(encoding="utf-8")
    except Exception:
        return 100.0
    m = re.search(r"gamma:\s*([0-9\.\-eE]+)", txt)
    if not m:
        return 100.0
    try:
        return float(m.group(1))
    except Exception:
        return 100.0


def detect_models(df: pd.DataFrame) -> List[str]:
    models = []
    for col in df.columns:
        if col.startswith("J_dc_") and col not in ("J_dc_gt", "J_dc_base"):
            models.append(col[len("J_dc_"):])
    return sorted(set(models))


def order_models(models: List[str]) -> List[str]:
    preferred = ["zero_shot", "sft", "dpo", "nn"]
    out = [m for m in preferred if m in models]
    out += [m for m in models if m not in out]
    return out


def select_jac_models(models_all: List[str]) -> List[str]:
    wanted = [m for m in ["sft", "dpo", "nn"] if m in models_all]
    return wanted if wanted else [m for m in models_all if m != "zero_shot"]


def _ac_fail_rate(df: pd.DataFrame, col: str) -> float:
    v = _coerce_numeric(df[col])
    mask = v.notna()
    if not mask.any():
        return float("nan")
    return float(((v >= AC_FAIL_SENTINEL) & mask).mean())


def _boxplot_bw(ax, data, labels, ylabel: str, widths: float = 0.42):
    ax.boxplot(
        data,
        labels=labels,
        widths=widths,
        showfliers=False,
        patch_artist=False,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(color="black", linewidth=0.9),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black", linewidth=0.9),
    )
    ax.set_ylabel(ylabel)
    _format_pub_ax(ax)


def find_latest_nn_csv(io_dir: Path) -> Path | None:
    p0 = io_dir / "eval_nn_vselect_per_xi.csv"
    if p0.exists():
        return p0
    cands = sorted(io_dir.glob("eval_nn_vselect_per_xi_run*.csv"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def merge_llm_and_nn(llm_csv: Path, nn_csv: Path, out_csv: Path, how: str = "inner") -> pd.DataFrame:
    df_llm = pd.read_csv(llm_csv)
    df_nn  = pd.read_csv(nn_csv)

    rename = {}
    if "J_dc" in df_nn.columns: rename["J_dc"] = "J_dc_nn"
    if "J_ac" in df_nn.columns: rename["J_ac"] = "J_ac_nn"
    if "shed_MW" in df_nn.columns: rename["shed_MW"] = "shed_MW_nn"
    if "best_plan_text" in df_nn.columns: rename["best_plan_text"] = "best_plan_text_nn"
    if "V_pen_effective" in df_nn.columns and "J_ac" not in df_nn.columns:
        rename["V_pen_effective"] = "J_ac_nn"

    df_nn = df_nn.rename(columns=rename)

    drop_cols = [c for c in ["J_dc_gt","J_dc_base","shed_MW_gt","shed_MW_base"] if c in df_nn.columns]
    if drop_cols:
        df_nn = df_nn.drop(columns=drop_cols)

    if "xi" not in df_llm.columns or "xi" not in df_nn.columns:
        raise RuntimeError("Both CSVs must contain an 'xi' column for merging.")

    df = df_llm.merge(df_nn, on="xi", how=how, suffixes=("", "_dup"))
    dup_cols = [c for c in df.columns if c.endswith("_dup")]
    if dup_cols:
        df = df.drop(columns=dup_cols)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def filter_complete_cases_for_models(df: pd.DataFrame, jac_models: List[str]) -> pd.DataFrame:
    cols = [f"J_ac_{m}" for m in jac_models if f"J_ac_{m}" in df.columns]
    if not cols:
        return df
    mask = pd.Series(True, index=df.index)
    for c in cols:
        mask &= _coerce_numeric(df[c]).notna()
    return df.loc[mask].copy()


# ----------------- stats ----------------- #
def basic_global_stats(df: pd.DataFrame) -> pd.DataFrame:
    out: Dict[str, float] = {}
    out["n_scenarios"] = int(df["xi"].nunique())

    jac_cols = [c for c in df.columns if c.startswith("J_ac_")]
    if jac_cols:
        any_fail = (df[jac_cols] >= AC_FAIL_SENTINEL).any(axis=1)
        out["n_scenarios_any_ac_fail"] = int(any_fail.sum())
        out["frac_scenarios_any_ac_fail"] = float(any_fail.mean())
    else:
        out["n_scenarios_any_ac_fail"] = 0
        out["frac_scenarios_any_ac_fail"] = np.nan

    jgt   = _coerce_numeric(df.get("J_dc_gt", pd.Series(dtype=float)))
    jbase = _coerce_numeric(df.get("J_dc_base", pd.Series(dtype=float)))
    mask_both = jgt.notna() & jbase.notna()
    if mask_both.any():
        diff  = (jbase - jgt)[mask_both]
        ratio = (jbase / jgt)[mask_both]
        out["baseline_minus_opt_mean"]   = float(diff.mean())
        out["baseline_minus_opt_median"] = float(diff.median())
        out["baseline_minus_opt_std"]    = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
        out["baseline_over_opt_mean"]    = float(ratio.mean())
        out["baseline_over_opt_median"]  = float(ratio.median())
    else:
        out["baseline_minus_opt_mean"] = np.nan
        out["baseline_minus_opt_median"] = np.nan
        out["baseline_minus_opt_std"] = np.nan
        out["baseline_over_opt_mean"] = np.nan
        out["baseline_over_opt_median"] = np.nan

    return pd.DataFrame([out])


def per_model_stats(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    rows = []
    gamma_val = read_gamma_from_limits()

    jgt   = _coerce_numeric(df.get("J_dc_gt", pd.Series(dtype=float)))
    jbase = _coerce_numeric(df.get("J_dc_base", pd.Series(dtype=float)))

    for m in models:
        col_jdc = f"J_dc_{m}"
        col_jac = f"J_ac_{m}"
        if col_jdc not in df.columns or col_jac not in df.columns:
            continue

        jdc = _coerce_numeric(df[col_jdc])
        jac = _coerce_numeric(df[col_jac])

        mask_jac = jac.notna()
        mask_success = mask_jac & (jac < AC_FAIL_SENTINEL)
        mask_fail    = mask_jac & (jac >= AC_FAIL_SENTINEL)

        mask_all = jdc.notna() & jgt.notna()
        regret = (jdc - jgt).where(mask_all)
        ratio_to_gt = (jdc / jgt).where(mask_all)

        mask_base = jdc.notna() & jbase.notna()
        better_than_base = (jdc <= jbase).where(mask_base)

        close_fracs = {}
        for eps in [0.01, 0.05, 0.10]:
            thresh = jgt * (1.0 + eps)
            is_close = (jdc <= thresh).where(mask_all)
            close_fracs[f"frac_within_{int(eps*100)}pct_opt"] = float(is_close.mean()) if mask_all.any() else np.nan

        shed_col = f"shed_MW_{m}"
        shed = _coerce_numeric(df[shed_col]) if shed_col in df.columns else pd.Series(index=df.index, dtype=float)
        mask_shed = shed.notna()

        if shed_col in df.columns:
            shed_mean = float(shed[mask_shed].mean()) if mask_shed.any() else np.nan
            shed_median = float(shed[mask_shed].median()) if mask_shed.any() else np.nan
            shed_pen = shed * gamma_val
            gen_cost = jdc - shed_pen
            mask_ratio = jdc.notna() & mask_shed & gen_cost.notna() & (gen_cost > 1e-6)
            ratio_pen = (shed_pen / gen_cost).where(mask_ratio)
            ratio_mean = float(ratio_pen[mask_ratio].mean()) if mask_ratio.any() else np.nan
            ratio_median = float(ratio_pen[mask_ratio].median()) if mask_ratio.any() else np.nan
        else:
            shed_mean = shed_median = np.nan
            ratio_mean = ratio_median = np.nan

        jac_success = jac[mask_success]

        row = {
            "model": m,
            "n_scenarios": int(df["xi"].nunique()),
            "n_valid_J_dc": int(jdc.notna().sum()),
            "n_valid_J_ac": int(mask_jac.sum()),

            "ac_success_rate": float(mask_success.mean()) if mask_jac.any() else np.nan,
            "ac_fail_rate": float(mask_fail.mean()) if mask_jac.any() else np.nan,
            "n_ac_fail": int(mask_fail.sum()),

            "J_ac_mean_all": float(jac[mask_jac].mean()) if mask_jac.any() else np.nan,
            "J_ac_median_all": float(jac[mask_jac].median()) if mask_jac.any() else np.nan,

            "J_ac_mean_success": float(jac_success.mean()) if mask_success.any() else np.nan,
            "J_ac_median_success": float(jac_success.median()) if mask_success.any() else np.nan,

            "J_dc_mean": float(jdc[jdc.notna()].mean()) if jdc.notna().any() else np.nan,
            "J_dc_median": float(jdc[jdc.notna()].median()) if jdc.notna().any() else np.nan,

            "regret_mean": float(regret[mask_all].mean()) if mask_all.any() else np.nan,
            "regret_median": float(regret[mask_all].median()) if mask_all.any() else np.nan,
            "ratio_to_opt_mean": float(ratio_to_gt[mask_all].mean()) if mask_all.any() else np.nan,

            "frac_better_than_baseline": float(better_than_base.mean()) if mask_base.any() else np.nan,

            "shed_MW_mean": shed_mean,
            "shed_MW_median": shed_median,
            "ratio_pen_over_gen_mean": ratio_mean,
            "ratio_pen_over_gen_median": ratio_median,
        }
        row.update(close_fracs)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("model")


def pairwise_stats(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            col_jdc_a, col_jdc_b = f"J_dc_{a}", f"J_dc_{b}"
            col_jac_a, col_jac_b = f"J_ac_{a}", f"J_ac_{b}"

            if col_jdc_a not in df.columns or col_jdc_b not in df.columns:
                continue

            ja = _coerce_numeric(df[col_jdc_a])
            jb = _coerce_numeric(df[col_jdc_b])
            mask_dc = ja.notna() & jb.notna()
            if mask_dc.any():
                d = (jb - ja)[mask_dc]
                frac_a_better_dc = float((d > 0).mean())
                frac_b_better_dc = float((d < 0).mean())
                frac_tie_dc      = float((d == 0).mean())
                delta_mean_dc    = float(d.mean())
                delta_median_dc  = float(d.median())
            else:
                frac_a_better_dc = frac_b_better_dc = frac_tie_dc = np.nan
                delta_mean_dc = delta_median_dc = np.nan

            if col_jac_a in df.columns and col_jac_b in df.columns:
                va = _coerce_numeric(df[col_jac_a])
                vb = _coerce_numeric(df[col_jac_b])
                mask_ac = va.notna() & vb.notna()
                if mask_ac.any():
                    d = (vb - va)[mask_ac]
                    frac_a_better_ac = float((d > 0).mean())
                    frac_b_better_ac = float((d < 0).mean())
                    frac_tie_ac      = float((d == 0).mean())
                    delta_mean_ac    = float(d.mean())
                    delta_median_ac  = float(d.median())
                else:
                    frac_a_better_ac = frac_b_better_ac = frac_tie_ac = np.nan
                    delta_mean_ac = delta_median_ac = np.nan

                mask_succ = mask_ac & (va < AC_FAIL_SENTINEL) & (vb < AC_FAIL_SENTINEL)
                if mask_succ.any():
                    ds = (vb - va)[mask_succ]
                    delta_mean_ac_succ = float(ds.mean())
                    delta_median_ac_succ = float(ds.median())
                    n_overlap_ac_succ = int(mask_succ.sum())
                else:
                    delta_mean_ac_succ = np.nan
                    delta_median_ac_succ = np.nan
                    n_overlap_ac_succ = 0
            else:
                mask_ac = pd.Series(False, index=df.index)
                frac_a_better_ac = frac_b_better_ac = frac_tie_ac = np.nan
                delta_mean_ac = delta_median_ac = np.nan
                n_overlap_ac_succ = 0
                delta_mean_ac_succ = np.nan
                delta_median_ac_succ = np.nan

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

                "n_overlap_ac_success": n_overlap_ac_succ,
                "delta_mean_ac_success(b-a)": delta_mean_ac_succ,
                "delta_median_ac_success(b-a)": delta_median_ac_succ,
            })
    return pd.DataFrame(rows)


# ----------------- FIGURES ----------------- #
def plot_Jdc_boxplot_5(fig_dir: Path, df: pd.DataFrame, models: List[str]) -> None:
    names = ["baseline", "opt"] + models
    data, labels = [], []

    for name in names:
        if name == "baseline":
            arr = _coerce_numeric(df.get("J_dc_base", pd.Series(dtype=float))).dropna()
        elif name == "opt":
            arr = _coerce_numeric(df.get("J_dc_gt", pd.Series(dtype=float))).dropna()
        else:
            col = f"J_dc_{name}"
            if col not in df.columns:
                continue
            arr = _coerce_numeric(df[col]).dropna()

        if not arr.empty:
            data.append(arr.values)
            labels.append(disp_name(name) if name not in ("baseline","opt") else name)

    if not data:
        return

    set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=170)
    _boxplot_bw(ax, data, labels, ylabel=r"$J_{dc}$", widths=0.40)
    fig.tight_layout(pad=0.25)
    _savefig(fig_dir, "box_Jdc_5methods")
    plt.close(fig)


def plot_Jdc_boxplot_models_only(fig_dir: Path, df: pd.DataFrame, models: List[str]) -> None:
    data, labels = [], []
    for m in models:
        col = f"J_dc_{m}"
        if col not in df.columns:
            continue
        arr = _coerce_numeric(df[col]).dropna()
        if not arr.empty:
            data.append(arr.values)
            labels.append(disp_name(m))

    if not data:
        return

    set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=170)
    _boxplot_bw(ax, data, labels, ylabel=r"$J_{dc}$", widths=0.40)
    fig.tight_layout(pad=0.25)
    _savefig(fig_dir, "box_Jdc_models_only")
    plt.close(fig)


def plot_ac_fail_rate_bar(fig_dir: Path, df: pd.DataFrame, models: List[str]) -> None:
    rates = []
    for m in models:
        col = f"J_ac_{m}"
        rates.append(_ac_fail_rate(df, col) if col in df.columns else np.nan)

    set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=170)
    x = np.arange(len(models))
    ax.bar(x, rates, color="0.78", edgecolor="black", linewidth=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([disp_name(m) for m in models])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AC failure rate")
    _format_pub_ax(ax)

    for xi, r in zip(x, rates):
        if np.isfinite(r):
            ax.text(xi, min(1.0, r + 0.025), f"{100*r:.1f}%", ha="center", va="bottom", fontsize=20)

    fig.tight_layout(pad=0.25)
    _savefig(fig_dir, "bar_ac_fail_rate_models")
    plt.close(fig)


def plot_Jac_boxplot_models_success_only_common_success(fig_dir: Path, df: pd.DataFrame, jac_models: List[str]) -> None:
    cols = [f"J_ac_{m}" for m in jac_models if f"J_ac_{m}" in df.columns]
    if not cols:
        return

    mask = pd.Series(True, index=df.index)
    for c in cols:
        v = _coerce_numeric(df[c])
        mask &= v.notna() & (v < AC_FAIL_SENTINEL)

    if not mask.any():
        return

    data, labels = [], []
    for m in jac_models:
        col = f"J_ac_{m}"
        if col not in df.columns:
            continue
        v = _coerce_numeric(df[col])[mask]
        if v.empty:
            return
        data.append(v.values)
        labels.append(disp_name(m))

    if not data:
        return

    set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=170)
    # (2) y-axis: DO NOT mention success-only
    _boxplot_bw(ax, data, labels, ylabel=r"$V_{\mathrm{pen}}$", widths=0.38)
    fig.tight_layout(pad=0.25)
    _savefig(fig_dir, "box_Jac_models_common_success")
    plt.close(fig)


def plot_pairwise_delta_Jac_success_only(fig_dir: Path, df: pd.DataFrame, jac_models: List[str]) -> None:
    if len(jac_models) < 2:
        return

    pairs: List[Tuple[str, str]] = []
    for i in range(len(jac_models)):
        for j in range(i + 1, len(jac_models)):
            pairs.append((jac_models[i], jac_models[j]))

    data, labels = [], []
    for a, b in pairs:
        col_a = f"J_ac_{a}"
        col_b = f"J_ac_{b}"
        if col_a not in df.columns or col_b not in df.columns:
            continue

        va = _coerce_numeric(df[col_a])
        vb = _coerce_numeric(df[col_b])
        mask = va.notna() & vb.notna() & (va < AC_FAIL_SENTINEL) & (vb < AC_FAIL_SENTINEL)
        if not mask.any():
            continue

        delta = (vb - va)[mask].values
        data.append(delta)
        labels.append(f"{disp_name(b)} - {disp_name(a)}\n(n={int(mask.sum())})")

    if not data:
        return

    set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=170)
    _boxplot_bw(ax, data, labels, ylabel=r"$\Delta V_{\mathrm{pen}}$ (success overlap)", widths=0.36)
    ax.axhline(0.0, color="black", linewidth=0.95, linestyle="--", alpha=0.7)
    fig.tight_layout(pad=0.25)
    _savefig(fig_dir, "box_pairwise_delta_Jac_success_overlap")
    plt.close(fig)


def plot_tradeoff_successrate_vs_Jac(fig_dir: Path, df: pd.DataFrame, jac_models: List[str]) -> None:
    xs, ys, labs = [], [], []
    for m in jac_models:
        col = f"J_ac_{m}"
        if col not in df.columns:
            continue
        v = _coerce_numeric(df[col])
        mask = v.notna()
        if not mask.any():
            continue
        succ_mask = mask & (v < AC_FAIL_SENTINEL)
        succ = v[succ_mask]
        sr = float(succ_mask.mean())
        med = float(succ.median()) if succ.size else float("nan")
        xs.append(sr)
        ys.append(med)
        labs.append(m)

    if not xs:
        return

    set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=170)
    ax.scatter(xs, ys, marker="o", edgecolor="black", facecolor="0.78", linewidth=0.95, s=70)

    for x, y, lab in zip(xs, ys, labs):
        if np.isfinite(x) and np.isfinite(y):
            ax.text(x + 0.012, y, disp_name(lab), fontsize=22, va="center")

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("AC success rate")
    ax.set_ylabel(r"median $V_{\mathrm{pen}}$")
    _format_pub_ax(ax)

    fig.tight_layout(pad=0.25)
    _savefig(fig_dir, "scatter_tradeoff_successrate_vs_median_Jac")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_csv", type=str, default=str(CSV_LLM_DEFAULT))
    ap.add_argument("--nn_csv", type=str, default="")
    ap.add_argument("--out_csv", type=str, default=str(CSV_MERGED_DEFAULT))
    ap.add_argument("--fig_dir", type=str, default=str(FIG_DIR_DEFAULT))
    ap.add_argument("--join", type=str, default="inner", choices=["inner","outer"])
    ap.add_argument("--jac_complete_cases", action="store_true", default=True)
    args = ap.parse_args()

    llm_csv = Path(args.llm_csv)
    if not llm_csv.exists():
        raise FileNotFoundError(f"Missing LLM input CSV: {llm_csv}")

    nn_csv = Path(args.nn_csv) if args.nn_csv else (find_latest_nn_csv(IO_DIR) or Path(""))
    if not nn_csv or not nn_csv.exists():
        raise FileNotFoundError(
            "Could not find NN per-xi CSV. Expected io/eval_nn_vselect_per_xi.csv or io/eval_nn_vselect_per_xi_run*.csv.\n"
            "Pass --nn_csv path/to/file.csv if it lives elsewhere."
        )

    out_csv = Path(args.out_csv)
    fig_dir = Path(args.fig_dir)

    df = merge_llm_and_nn(llm_csv, nn_csv, out_csv, how=args.join)
    print(f"Merge join='{args.join}': n_merged={df.shape[0]}")
    print(f"Merged per-xi table -> {out_csv}")

    jac_cols = [c for c in df.columns if c.startswith("J_ac_")]
    if jac_cols:
        df[jac_cols] = df[jac_cols].apply(pd.to_numeric, errors="coerce")
        df[jac_cols] = df[jac_cols].replace([np.inf, -np.inf], np.nan).fillna(AC_FAIL_SENTINEL_RAW)
        df[jac_cols] = df[jac_cols] / JAC_SCALE
        total_fail_entries = int((df[jac_cols] >= AC_FAIL_SENTINEL).sum().sum())
        print(
            f"J_ac fill+scale done: divide by {JAC_SCALE:.0f}. "
            f"Scaled sentinel is {AC_FAIL_SENTINEL:.2f}. "
            f"Total entries >= sentinel: {total_fail_entries}"
        )

    models_all = order_models(detect_models(df))
    jac_models = select_jac_models(models_all)

    df_jac = filter_complete_cases_for_models(df, jac_models) if args.jac_complete_cases else df

    basic_global_stats(df).to_csv(GLOBAL_SUMMARY_CSV, index=False)
    per_model_stats(df, models_all).to_csv(SUMMARY_CSV, index=False)
    pairwise_stats(df, models_all).to_csv(PAIRWISE_CSV, index=False)

    plot_Jdc_boxplot_5(fig_dir, df, models_all)
    plot_Jdc_boxplot_models_only(fig_dir, df, models_all)
    plot_ac_fail_rate_bar(fig_dir, df, models_all)
    plot_Jac_boxplot_models_success_only_common_success(fig_dir, df_jac, jac_models)
    plot_pairwise_delta_Jac_success_only(fig_dir, df_jac, jac_models)
    plot_tradeoff_successrate_vs_Jac(fig_dir, df_jac, jac_models)

    print(f"\nSaved figures (PNG+PDF) to:\n  {fig_dir}")


if __name__ == "__main__":
    main()
