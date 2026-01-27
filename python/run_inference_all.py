# python/eval_voltage_and_j.py
# Evaluate models with best-of-N selection by VOLTAGE (V_pen == J_ac),
# and report:
#   - J_ac (AC voltage penalty from verify_plan_hybrid)
#   - J_dc (DC-OPF cost for the chosen LLM plan from verify_plan)
#   - J_dc_gt (ground-truth optimal DC-OPF with switching, run_ground_truth mode='mip_opt')
#   - J_dc_base (baseline DC-OPF with no extra switching, run_ground_truth mode='baseline')
#   - shed_MW (load shed for chosen LLM plan)
#   - shed_MW_gt (load shed for mip_opt)
#   - shed_MW_base (load shed for baseline)
#
# MATLAB functions used:
#   make_summary(case_name, xi_json, limits_yml, out_summary_json)
#   verify_plan_hybrid(case_name, plan_json, xi_json, limits_yml, out_json)
#   verify_plan(case_name, plan_json, xi_json, limits_yml, out_json)
#   run_ground_truth(case_name, xi_json, limits_yml, out_json, mode)
#
# Outputs:
#   io/eval_vselect_rows.csv/json
#       per (xi, model): J_dc, J_ac, J_dc_gt, J_dc_base,
#                       shed_MW, shed_MW_gt, shed_MW_base, best_plan_text
#   io/eval_vselect_jdc_jac_per_xi.csv
#       per xi: J_dc_*, J_ac_*, shed_MW_* per model + J_dc_gt + J_dc_base + shed_MW_gt + shed_MW_base
#   io/eval_vselect_per_model.csv
#       per-model stats on V/J_ac/J_dc
#   io/eval_vselect_pairwise_v.csv
#       pairwise head-to-head on V/J_ac
#   io/eval_vselect_pairwise_j.csv
#       pairwise head-to-head on J_dc (LLM plans)

import os, json, math, random, csv, re, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from openai import OpenAI

ROOT       = Path(__file__).resolve().parents[1]
MATLAB_DIR = ROOT / "matlab"
CONFIG     = ROOT / "config"
IO_DIR     = ROOT / "io"
IO_DIR.mkdir(exist_ok=True)

CASE        = os.getenv("CASE_NAME", "case118")
LIMITS_YML  = (CONFIG / "limits.yml").as_posix()
INDEX_FILE  = Path(os.getenv("INDEX_FILE", str(IO_DIR / "sft_test_index.json")))
API_KEY     = os.getenv("OPENAI_API_KEY") or ""
BASE_MODEL  = os.getenv("BASE_MODEL", "gpt-4.1-mini-2025-04-14")

FT_SFT_FILE = CONFIG / "ft_model_sft.txt"
FT_OLD_FILE = CONFIG / "ft_model.txt"
SFT_MODEL = FT_SFT_FILE.read_text(encoding="utf-8").strip() if FT_SFT_FILE.exists() else (
           FT_OLD_FILE.read_text(encoding="utf-8").strip() if FT_OLD_FILE.exists() else None)

FT_DPO_FILE = CONFIG / "ft_model_dpo.txt"
DPO_MODEL   = FT_DPO_FILE.read_text(encoding="utf-8").strip() if FT_DPO_FILE.exists() else None
if DPO_MODEL == "":
    DPO_MODEL = None

# --- Runtime knobs ---
N_CANDS         = int(os.getenv("N_CANDS", "3"))   # default 3
TEMPS           = [float(t) for t in os.getenv("TEMPS", "0.1,0.3,0.7").split(",")]
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "120"))
SEED            = int(os.getenv("SEED", "7"))
random.seed(SEED)

# Force exactly 3 candidates regardless of env overrides
N_CANDS = 3

SYSTEM_MSG = (
    "You are a grid switching assistant.\n"
    "Input: a JSON summary of a PSPS scenario.\n"
    "Output: one line with at most 'toggle_budget' OPEN actions (no other text).\n"
    "Grammar:\n"
    "  - open(Sk:LINE)   e.g., open(S6:135)\n"
    "  - open(LINE)      e.g., open(131)\n"
    "Separate multiple actions with semicolons. No explanations."
)

def _run_matlab_batch(batch: str) -> int:
    return os.system(f"matlab -batch \"addpath('{MATLAB_DIR.as_posix()}'); {batch}; exit;\"")

def call_make_summary(xi_path: Path, out_summary: Path):
    if out_summary.exists():
        try:
            return json.loads(out_summary.read_text(encoding="utf-8"))
        except Exception:
            pass
    rc = _run_matlab_batch(
        f"make_summary('{CASE}','{xi_path.as_posix()}','{LIMITS_YML}','{out_summary.as_posix()}')"
    )
    if rc != 0 or not out_summary.exists():
        return None
    try:
        return json.loads(out_summary.read_text(encoding="utf-8"))
    except Exception:
        return None

def call_verify_hybrid(plan_path: Path, xi_path: Path, out_path: Path):
    """AC/DCPF hybrid verifier: returns V_pen, J_ac, dummy J_dc, etc."""
    rc = _run_matlab_batch(
        f"verify_plan_hybrid('{CASE}','{plan_path.as_posix()}','{xi_path.as_posix()}','{LIMITS_YML}','{out_path.as_posix()}')"
    )
    if rc != 0 or not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def call_verify_dcopf(plan_path: Path, xi_path: Path, out_path: Path):
    """DC-OPF verifier: returns J (used as J_dc) and shed_MW."""
    rc = _run_matlab_batch(
        f"verify_plan('{CASE}','{plan_path.as_posix()}','{xi_path.as_posix()}','{LIMITS_YML}','{out_path.as_posix()}')"
    )
    if rc != 0 or not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def call_run_ground_truth(xi_path: Path, out_path: Path, mode: str):
    """
    Ground-truth DC-OPF:
      mode = 'baseline' : PSPS only, no extra toggles
      mode = 'mip_opt'  : optimal open-only switching up to budget
    """
    rc = _run_matlab_batch(
        f"run_ground_truth('{CASE}','{xi_path.as_posix()}','{LIMITS_YML}','{out_path.as_posix()}','{mode}')"
    )
    if rc != 0 or not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None

_OPEN_RE = re.compile(r"open\s*\(\s*(?:([sS]\d+)\s*:\s*)?([0-9]+)\s*\)", re.IGNORECASE)

def _clamp(lid: int, n: int) -> int:
    return max(1, min(int(lid), n))

def parse_actions_to_plan(text: str, toggle: int, nline: int) -> Dict:
    seen = set()
    acts = []
    for (nm, lid) in _OPEN_RE.findall(text or ""):
        lid = _clamp(int(lid), nline)
        key = ((nm.upper() if nm else ""), lid)
        if key in seen:
            continue
        seen.add(key)
        item = {"action": "open", "line": lid}
        if nm:
            item["name"] = nm.upper()
        acts.append(item)
        if len(acts) >= max(1, int(toggle)):
            break
    return {"corridor_actions": acts}

def plan_to_text(plan: Dict) -> str:
    parts = []
    for a in plan.get("corridor_actions", []):
        lid = int(a["line"])
        name = (a.get("name") or "").strip()
        parts.append(f"open({name + ':' if name else ''}{lid})")
    return "; ".join(parts) if parts else ""

def v_eff(sc: Dict) -> float:
    """
    Effective voltage penalty.
    By your setup, V_pen == J_ac, so this is the J_ac scalar.
    """
    v = sc.get("V_pen")
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    j = sc.get("J_ac", float("inf"))
    return float(j) if isinstance(j, (int, float)) and math.isfinite(j) else float("inf")

def is_dc(sc: Dict) -> bool:
    # 'feasible' is set in verify_plan_hybrid output
    return bool(sc.get("feasible"))

def is_ac(sc: Dict) -> bool:
    return math.isfinite(v_eff(sc))

def make_client():
    if not API_KEY:
        print("ERROR: set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=API_KEY)

def sample(client, model, prompt, n, temp):
    if n <= 0:
        return []
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        temperature=temp,
        max_tokens=MAX_TOKENS,
        n=n
    )
    return [(c.message.content or "").strip() for c in r.choices]

def main():
    if not INDEX_FILE.exists():
        print("ERROR: missing INDEX_FILE", file=sys.stderr)
        sys.exit(1)

    with INDEX_FILE.open("r", encoding="utf-8") as f:
        try:
            xi_paths = [Path(p) for p in json.load(f)]
        except Exception:
            f.seek(0)
            xi_paths = [Path(p.strip()) for p in f.read().splitlines() if p.strip()]

    xi_paths = [p for p in xi_paths if p.exists()]

    if not xi_paths:
        print("No xi paths found after filtering.")
        return

    models = [("zero_shot", BASE_MODEL)]
    if SFT_MODEL:
        models.append(("sft", SFT_MODEL))
    if DPO_MODEL:
        models.append(("dpo", DPO_MODEL))

    client = make_client()
    rows: List[Dict[str, Any]] = []

    # caches so we don't recompute ground truth for each model
    gt_opt_J_cache: Dict[str, float]     = {}
    gt_base_J_cache: Dict[str, float]    = {}
    gt_opt_shed_cache: Dict[str, float]  = {}
    gt_base_shed_cache: Dict[str, float] = {}

    for xi in xi_paths:
        xi_key = xi.as_posix()

        # --- per-xi ground truth: optimal (mip_opt) ---
        if xi_key not in gt_opt_J_cache:
            gt_opt_path = IO_DIR / f"gt_mip_opt_{xi.stem}.json"
            gt_opt_res  = call_run_ground_truth(xi, gt_opt_path, mode="mip_opt")
            if (
                gt_opt_res is not None
                and isinstance(gt_opt_res.get("J"), (int, float))
                and math.isfinite(gt_opt_res["J"])
            ):
                gt_opt_J_cache[xi_key] = float(gt_opt_res["J"])
                shed_val = gt_opt_res.get("shed_MW")
                if isinstance(shed_val, (int, float)) and math.isfinite(shed_val):
                    gt_opt_shed_cache[xi_key] = float(shed_val)
                else:
                    gt_opt_shed_cache[xi_key] = float("nan")
            else:
                gt_opt_J_cache[xi_key]    = float("nan")
                gt_opt_shed_cache[xi_key] = float("nan")

        # --- per-xi ground truth: baseline (do nothing beyond PSPS) ---
        if xi_key not in gt_base_J_cache:
            gt_base_path = IO_DIR / f"gt_baseline_{xi.stem}.json"
            gt_base_res  = call_run_ground_truth(xi, gt_base_path, mode="baseline")
            if (
                gt_base_res is not None
                and isinstance(gt_base_res.get("J"), (int, float))
                and math.isfinite(gt_base_res["J"])
            ):
                gt_base_J_cache[xi_key] = float(gt_base_res["J"])
                shed_val = gt_base_res.get("shed_MW")
                if isinstance(shed_val, (int, float)) and math.isfinite(shed_val):
                    gt_base_shed_cache[xi_key] = float(shed_val)
                else:
                    gt_base_shed_cache[xi_key] = float("nan")
            else:
                gt_base_J_cache[xi_key]    = float("nan")
                gt_base_shed_cache[xi_key] = float("nan")

        j_dc_gt    = gt_opt_J_cache[xi_key]
        j_dc_base  = gt_base_J_cache[xi_key]
        shed_gt    = gt_opt_shed_cache[xi_key]
        shed_base  = gt_base_shed_cache[xi_key]

        # --- PSPS summary for the LLM prompt ---
        summ_path = IO_DIR / f"summary_{xi.stem}.json"
        summ = call_make_summary(xi, summ_path)
        if summ is None:
            continue

        prompt = json.dumps(summ, indent=2)
        toggle = int(summ.get("toggle_budget", 3))
        n_line = int(summ.get("lines") or summ.get("n_line") or 999999)

        for label, model in models:
            # diversify: split N_CANDS across temps (total of 3)
            per = max(1, N_CANDS // len(TEMPS))
            leftover = N_CANDS - per * len(TEMPS)
            texts: List[str] = []
            for i, t in enumerate(TEMPS):
                k = per + (1 if i < leftover else 0)
                if k <= 0:
                    continue
                texts += sample(client, model, prompt, k, t)

            # score, keep only DC-feasible & AC-OK, pick MIN v_eff (J_ac)
            best_pl = None
            best_sc = None
            best_txt = ""
            for txt in texts:
                plan = parse_actions_to_plan(txt, toggle, n_line)
                p = IO_DIR / f"eval_plan_{label}_{xi.stem}_{random.randint(0, 9999)}.json"
                r = IO_DIR / f"eval_res_{label}_{xi.stem}_{random.randint(0, 9999)}.json"
                p.write_text(json.dumps(plan), encoding="utf-8")
                sc = call_verify_hybrid(p, xi, r)
                try:
                    p.unlink(missing_ok=True)
                    r.unlink(missing_ok=True)
                except Exception:
                    pass

                if sc is None or (not is_dc(sc)) or (not is_ac(sc)):
                    continue

                if (best_sc is None) or (v_eff(sc) < v_eff(best_sc)):
                    best_pl, best_sc, best_txt = plan, sc, txt

            if best_pl is None:
                continue

            # AC side: v_pen == J_ac
            jac_value = v_eff(best_sc)

            # DC-OPF verifier for the chosen plan: J_dc and shed_MW
            p_dc = IO_DIR / f"eval_plan_dc_{label}_{xi.stem}_{random.randint(0, 9999)}.json"
            r_dc = IO_DIR / f"eval_res_dc_{label}_{xi.stem}_{random.randint(0, 9999)}.json"
            p_dc.write_text(json.dumps(best_pl), encoding="utf-8")

            dc_sc = call_verify_dcopf(p_dc, xi, r_dc)
            try:
                p_dc.unlink(missing_ok=True)
                r_dc.unlink(missing_ok=True)
            except Exception:
                pass

            if (
                dc_sc is not None
                and isinstance(dc_sc.get("J"), (int, float))
                and math.isfinite(dc_sc["J"])
            ):
                j_dc_value = float(dc_sc["J"])   # DC-OPF cost of LLM plan
                shed_val = dc_sc.get("shed_MW")
                if isinstance(shed_val, (int, float)) and math.isfinite(shed_val):
                    shed_llm = float(shed_val)
                else:
                    shed_llm = float("nan")
            else:
                j_dc_value = float("nan")
                shed_llm   = float("nan")

            rows.append({
                "xi": xi.as_posix(),
                "model": label,
                "parent": model,
                "V_pen_effective": jac_value,   # AC side, same as J_ac
                "J_dc": j_dc_value,             # DC-OPF cost of chosen plan
                "J_ac": float(jac_value),       # AC voltage penalty
                "J_dc_gt": j_dc_gt,             # ground truth (mip_opt)
                "J_dc_base": j_dc_base,         # baseline (do nothing)
                "best_plan_text": plan_to_text(best_pl),
                "shed_MW": shed_llm,            # load shed for LLM plan
                "shed_MW_gt": shed_gt,          # load shed for mip_opt
                "shed_MW_base": shed_base,      # load shed for baseline
            })

    # Save rows (tall format)
    out_json = IO_DIR / "eval_vselect_rows.json"
    out_csv  = IO_DIR / "eval_vselect_rows.csv"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "xi",
                "model",
                "parent",
                "V_pen_effective",
                "J_dc",
                "J_ac",
                "J_dc_gt",
                "J_dc_base",
                "shed_MW",
                "shed_MW_gt",
                "shed_MW_base",
                "best_plan_text",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows to analyze.")
        return

    # Ensure numeric
    df["V_pen_effective"] = pd.to_numeric(df["V_pen_effective"], errors="coerce")
    df["J_dc"]            = pd.to_numeric(df["J_dc"], errors="coerce")
    df["J_ac"]            = pd.to_numeric(df["J_ac"], errors="coerce")
    df["J_dc_gt"]         = pd.to_numeric(df["J_dc_gt"], errors="coerce")
    df["J_dc_base"]       = pd.to_numeric(df["J_dc_base"], errors="coerce")
    df["shed_MW"]         = pd.to_numeric(df["shed_MW"], errors="coerce")
    df["shed_MW_gt"]      = pd.to_numeric(df["shed_MW_gt"], errors="coerce")
    df["shed_MW_base"]    = pd.to_numeric(df["shed_MW_base"], errors="coerce")

    # ---- Per-xi J_dc, J_ac, shed_MW in wide format ----
    pivot_j = df.pivot_table(
        index="xi",
        columns="model",
        values=["J_dc", "J_ac", "shed_MW"],
        aggfunc="first",
    )
    pivot_j.columns = [f"{metric}_{model}" for metric, model in pivot_j.columns]
    pivot_j = pivot_j.reset_index()

    # Attach per-xi ground truth J_dc_gt / baseline and shedMW as well
    gt_df = df[["xi", "J_dc_gt", "J_dc_base", "shed_MW_gt", "shed_MW_base"]].drop_duplicates("xi")
    pivot_j = pivot_j.merge(gt_df, on="xi", how="left")

    pivot_j.to_csv(IO_DIR / "eval_vselect_jdc_jac_per_xi.csv", index=False)

    # ---- Per-model stats on V(=J_ac), J_dc, and J_ac ----
    per = []
    for m, g in df.groupby("model"):
        v    = pd.to_numeric(g["V_pen_effective"], errors="coerce")
        j_dc = pd.to_numeric(g["J_dc"], errors="coerce")
        j_ac = pd.to_numeric(g["J_ac"], errors="coerce")

        v_valid    = v.dropna()
        j_dc_valid = j_dc.dropna()
        j_ac_valid = j_ac.dropna()

        # "Good" subset: J_ac finite and not the 1e6 sentinel (AC failure penalty)
        mask_good = (j_ac_valid < 1e6)
        j_ac_good = j_ac_valid[mask_good]
        # align J_dc_good to same indices as J_ac_good
        j_dc_good = j_dc_valid.reindex(j_ac_good.index).dropna()

        per.append({
            "model": m,
            "count": int(len(g)),

            # V stats (same numbers as J_ac since v_pen == J_ac)
            "V_mean": float(v_valid.mean())      if len(v_valid)    else float("nan"),
            "V_median": float(v_valid.median())  if len(v_valid)    else float("nan"),
            "V_std": float(v_valid.std(ddof=1))  if len(v_valid) > 1 else 0.0,
            "V_min": float(v_valid.min())        if len(v_valid)    else float("nan"),
            "V_max": float(v_valid.max())        if len(v_valid)    else float("nan"),

            # Unfiltered J_dc and J_ac
            "J_dc_mean": float(j_dc_valid.mean())      if len(j_dc_valid)    else float("nan"),
            "J_dc_median": float(j_dc_valid.median())  if len(j_dc_valid)    else float("nan"),
            "J_dc_std": float(j_dc_valid.std(ddof=1))  if len(j_dc_valid) > 1 else float("nan"),
            "J_dc_min": float(j_dc_valid.min())        if len(j_dc_valid)    else float("nan"),
            "J_dc_max": float(j_dc_valid.max())        if len(j_dc_valid)    else float("nan"),

            "J_ac_mean": float(j_ac_valid.mean())      if len(j_ac_valid)    else float("nan"),
            "J_ac_median": float(j_ac_valid.median())  if len(j_ac_valid)    else float("nan"),
            "J_ac_std": float(j_ac_valid.std(ddof=1))  if len(j_ac_valid) > 1 else float("nan"),
            "J_ac_min": float(j_ac_valid.min())        if len(j_ac_valid)    else float("nan"),
            "J_ac_max": float(j_ac_valid.max())        if len(j_ac_valid)    else float("nan"),

            # Filtered (drop J_ac == 1e6 or worse)
            "J_dc_mean_filtered": float(j_dc_good.mean()) if len(j_dc_good) else float("nan"),
            "J_ac_mean_filtered": float(j_ac_good.mean()) if len(j_ac_good) else float("nan"),
            "n_filtered": int(len(j_ac_good)),
        })

    pd.DataFrame(per).sort_values("model").to_csv(
        IO_DIR / "eval_vselect_per_model.csv", index=False
    )

    # ---- Pairwise head-to-head on V (primary, same as J_ac) ----
    piv = df.pivot_table(
        index="xi",
        columns="model",
        values="V_pen_effective",
        aggfunc="first",
    )
    pairs = [("zero_shot", "sft"), ("sft", "dpo"), ("zero_shot", "dpo")]
    out_v = []
    for a, b in pairs:
        if a not in piv or b not in piv:
            out_v.append({"pair": f"{a}_vs_{b}", "n_overlap": 0})
            continue
        sub = piv[[a, b]].dropna()
        if sub.empty:
            out_v.append({"pair": f"{a}_vs_{b}", "n_overlap": 0})
            continue
        d = sub[b] - sub[a]  # negative => b better on V/J_ac (lower is better)
        out_v.append({
            "pair": f"{a}_vs_{b}",
            "n_overlap": int(len(d)),
            f"frac_{a}_better": float((d > 0).mean()),
            f"frac_{b}_better": float((d < 0).mean()),
            "frac_tie": float((d == 0).mean()),
            "delta_mean(b-a)": float(d.mean()),
            "delta_median(b-a)": float(d.median()),
        })
    pd.DataFrame(out_v).to_csv(IO_DIR / "eval_vselect_pairwise_v.csv", index=False)

    # ---- Pairwise head-to-head on J_dc using SAME V-selected picks ----
    pivj = df.pivot_table(
        index="xi",
        columns="model",
        values="J_dc",
        aggfunc="first",
    )
    out_j = []
    for a, b in pairs:
        if a not in pivj or b not in pivj:
            out_j.append({"pair": f"{a}_vs_{b}", "n_overlap": 0})
            continue
        sub = pivj[[a, b]].dropna()
        if sub.empty:
            out_j.append({"pair": f"{a}_vs_{b}", "n_overlap": 0})
            continue
        d = sub[b] - sub[a]  # negative => b better on J (lower cost)
        out_j.append({
            "pair": f"{a}_vs_{b}",
            "n_overlap": int(len(d)),
            f"frac_{a}_better": float((d > 0).mean()),
            f"frac_{b}_better": float((d < 0).mean()),
            "frac_tie": float((d == 0).mean()),
            "delta_mean(b-a)": float(d.mean()),
            "delta_median(b-a)": float(d.median()),
        })
    pd.DataFrame(out_j).to_csv(IO_DIR / "eval_vselect_pairwise_j.csv", index=False)

    print(
        "Saved:",
        (IO_DIR / "eval_vselect_rows.csv").as_posix(),
        (IO_DIR / "eval_vselect_jdc_jac_per_xi.csv").as_posix(),
        (IO_DIR / "eval_vselect_per_model.csv").as_posix(),
        (IO_DIR / "eval_vselect_pairwise_v.csv").as_posix(),
        (IO_DIR / "eval_vselect_pairwise_j.csv").as_posix(),
    )

if __name__ == "__main__":
    main()
