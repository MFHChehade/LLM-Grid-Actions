# python/build_dpo_pairs_vonly.py
# Voltage-only DPO pair builder (no Qmax scaling).
# - Teaches: "smaller V_pen is better."
# - Requires DC-feasible + AC-OK for both sides (configurable via env).
# - Samples only a fraction of scenarios, default 1% (see SAMPLE_FRACTION).
# - Saves: pairs.jsonl, OpenAI prefs, QC CSV, and a candidate-eval CSV.

import os
import sys
import json
import time
import subprocess
import re
import random
import math
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from openai import OpenAI
so here 
ROOT       = Path(__file__).resolve().parents[1]
MATLAB_DIR = ROOT / "matlab"
CONFIG     = ROOT / "config"
IO_DIR     = ROOT / "io"
IO_DIR.mkdir(exist_ok=True)

MATPOWER_DIR = os.getenv("MATPOWER_DIR", "")

CASE        = os.getenv("CASE_NAME", "case118")
LIMITS_YML  = (CONFIG / "limits.yml").as_posix()
INDEX_FILE  = Path(os.getenv("INDEX_FILE", str(IO_DIR / "sft_train_index.json")))

FT_SFT_FILE = Path(os.getenv("FT_SFT_FILE", str(CONFIG / "ft_model_sft.txt")))
if not FT_SFT_FILE.exists():
    FT_SFT_FILE = Path(os.getenv("FT_FILE", str(CONFIG / "ft_model.txt")))  # fallback

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_OPENAI_API_KEY") or ""
if not API_KEY:
    print("ERROR: set OPENAI_API_KEY", file=sys.stderr)
    sys.exit(1)

BASE_MODEL      = os.getenv("BASE_MODEL", "gpt-4.1-mini-2025-04-14")
INCLUDE_BASE    = bool(int(os.getenv("INCLUDE_BASE", "1")))
TEMPS           = [float(t) for t in os.getenv("TEMPS", "0.1,0.3,0.7,1.0").split(",")]
N_TOTAL_CANDS   = int(os.getenv("N_TOTAL_CANDS", "48"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "120"))
SEED            = int(os.getenv("SEED", "7"))
random.seed(SEED)

# Pairing thresholds (progressive relaxation)
V_ABS_MIN   = float(os.getenv("V_PEN_ABS_MIN", "0"))
V_REL_MIN   = float(os.getenv("V_PEN_REL_MIN", "1.00"))  # should not be < 1.0
RELAX_STEPS = int(os.getenv("RELAX_STEPS", "4"))
RELAX_MULT  = float(os.getenv("RELAX_MULT", "0.6"))

# Pools / backstop
REQUIRE_DC_OK = bool(int(os.getenv("REQUIRE_DC_OK_BOTH", "1")))
REQUIRE_AC_OK = bool(int(os.getenv("REQUIRE_AC_OK_BOTH", "1")))
TOPK_BEST     = int(os.getenv("TOPK_BEST", "3"))
TOPK_WORST    = int(os.getenv("TOPK_WORST", "4"))
MAX_PAIRS_PER = int(os.getenv("MAX_PAIRS_PER_PROMPT", "4"))
ENABLE_BACKSTOP  = bool(int(os.getenv("ENABLE_BACKSTOP", "1")))
BACKSTOP_MIN_ABS = float(os.getenv("BACKSTOP_MIN_ABS", "200"))

SHUFFLE_INDEX     = bool(int(os.getenv("SHUFFLE_INDEX", "1")))
MAX_PROMPTS       = int(os.getenv("MAX_PROMPTS", "0"))

# Fraction of scenarios to sample (default 1%).
SAMPLE_FRACTION   = float(os.getenv("SAMPLE_FRACTION", "1"))

OPENAI_PREFS_PATH = Path(os.getenv("OPENAI_PREFS_PATH", str(IO_DIR / "dpo_openai_prefs.jsonl")))

SYSTEM_MSG = (
    "You are a grid switching assistant.\n"
    "Input: a JSON summary of a PSPS scenario.\n"
    "Output: one line with at most 'toggle_budget' OPEN actions (no other text).\n"
    "Grammar: open(Sk:LINE) or open(LINE); semicolon-separated; no explanations."
)

# ---------- MATLAB bridges ----------
def _run_matlab_batch(batch: str) -> int:
    """
    Run a MATLAB -batch command with our matlab helpers and MATPOWER on the path.
    """
    parts = [f"addpath('{MATLAB_DIR.as_posix()}');"]
    if MATPOWER_DIR:
        mp = MATPOWER_DIR.replace("\\", "/")
        # Add MATPOWER and its subfolders (lib, data, etc.)
        parts.append(f"addpath(genpath('{mp}'));")
        # If you have a MATPOWER startup, you can also uncomment:
        # parts.append(\"if exist('startup_matpower','file'), startup_matpower; end;\")
    prefix = " ".join(parts)
    cmd = f"matlab -batch \"{prefix} {batch}; exit;\""
    return subprocess.run(cmd, shell=True).returncode


def call_make_summary(xi_path: Path, out_summary: Path) -> Optional[Dict]:
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


def call_verify(plan_path: Path, xi_path: Path, out_path: Path) -> Optional[Dict]:
    # MATLAB signature: verify_plan_hybrid(case_name, plan_json, xi_json, limits_yml, out_json)
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

# ---------- Parsing ----------
_OPEN_RE = re.compile(r"open\s*\(\s*(?:([sS]\d+)\s*:\s*)?([0-9]+)\s*\)", re.IGNORECASE)


def _clamp(lid: int, n_line: int) -> int:
    return max(1, min(int(lid), n_line))


def parse_actions_to_plan(text: str, toggle_budget: int, n_line: int) -> Dict:
    acts, seen = [], set()
    for (maybe_name, line_id) in _OPEN_RE.findall(text or ""):
        lid = _clamp(int(line_id), n_line)
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


def plan_to_text(plan: Dict) -> str:
    parts = []
    for a in plan.get("corridor_actions", []):
        lid = int(a["line"])
        name = (a.get("name") or "").strip()
        parts.append(f"open({name+':' if name else ''}{lid})")
    return "; ".join(parts)

# ---------- Scoring helpers ----------
def vpen_eff(sc: Dict) -> float:
    v = sc.get("V_pen")
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    j_ac = sc.get("J_ac", float("inf"))
    return float(j_ac) if isinstance(j_ac, (int, float)) and math.isfinite(j_ac) else float("inf")


def is_dc_ok(sc: Dict) -> bool:
    # Prefer explicit feasible_dc if present, else 'feasible'
    return bool(sc.get("feasible_dc")) if "feasible_dc" in sc else bool(sc.get("feasible"))


def is_ac_ok(sc: Dict) -> bool:
    # Prefer explicit ac_ok if present, else infer from finiteness of V_pen/J_ac
    if "ac_ok" in sc:
        return bool(sc.get("ac_ok"))
    return math.isfinite(vpen_eff(sc))

# ---------- LLM ----------
def make_client() -> OpenAI:
    return OpenAI(api_key=API_KEY)


def sample_batch(
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
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        temperature=temp,
        max_tokens=max_tokens,
        n=n,
    )
    return [(c.message.content or "").strip() for c in r.choices]


def to_openai_pref(prompt: str, chosen: str, rejected: str) -> Dict[str, Any]:
    return {
        "input": {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": prompt},
            ]
        },
        "preferred_output":     [{"role": "assistant", "content": chosen}],
        "non_preferred_output": [{"role": "assistant", "content": rejected}],
    }

# ---------- Main ----------
def main():
    if not FT_SFT_FILE.exists():
        print(f"ERROR: missing SFT pointer: {FT_SFT_FILE}", file=sys.stderr)
        sys.exit(1)
    ENGINE_SFT = FT_SFT_FILE.read_text(encoding="utf-8").strip()
    if not ENGINE_SFT:
        print("ERROR: SFT pointer empty.", file=sys.stderr)
        sys.exit(1)

    if not INDEX_FILE.exists():
        print(f"ERROR: missing INDEX_FILE {INDEX_FILE}", file=sys.stderr)
        sys.exit(1)
    with INDEX_FILE.open("r", encoding="utf-8") as f:
        try:
            xi_paths = [Path(p) for p in json.load(f)]
        except Exception:
            f.seek(0)
            xi_paths = [Path(p.strip()) for p in f if p.strip()]
    xi_paths = [p for p in xi_paths if p.exists()]

    if SHUFFLE_INDEX:
        random.shuffle(xi_paths)

    # Sample only a fraction of scenarios.
    total_available = len(xi_paths)
    if 0 < SAMPLE_FRACTION < 1.0 and total_available > 0:
        k_keep = max(1, int(round(total_available * SAMPLE_FRACTION)))
        xi_paths = xi_paths[:k_keep]
        print(
            f"Sampling ~{SAMPLE_FRACTION*100:.1f}% of scenarios "
            f"({k_keep}/{total_available})"
        )

    # Optional hard cap by count
    if MAX_PROMPTS > 0:
        xi_paths = xi_paths[:MAX_PROMPTS]

    engines = [ENGINE_SFT] + ([BASE_MODEL] if INCLUDE_BASE else [])
    grid = [(e, t) for e in engines for t in TEMPS]

    client = make_client()
    run_id = f"{int(time.time())}"
    pairs_fp = IO_DIR / f"dpo_pairs_{run_id}.jsonl"
    prefs_fp = OPENAI_PREFS_PATH
    qc_fp    = IO_DIR / f"dpo_pairs_QC_{run_id}.csv"
    eval_fp  = IO_DIR / f"dpo_candidates_{run_id}.csv"

    total_prompts = 0
    total_pairs   = 0

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
            fieldnames=["xi", "plan", "J_dc", "V_pen", "J_ac", "feasible_dc", "ac_ok", "notes"],
        )
        ev_writer.writeheader()

        for xi in xi_paths:
            if MAX_PROMPTS and total_prompts >= MAX_PROMPTS:
                break

            summary_path = IO_DIR / f"summary_{xi.stem}.json"
            summary = call_make_summary(xi, summary_path)
            if summary is None:
                total_prompts += 1
                continue

            prompt        = json.dumps(summary, indent=2)
            toggle_budget = int(summary.get("toggle_budget", 3))
            n_line        = int(summary.get("lines") or summary.get("n_line") or 999999)

            # Sample candidates from (engine, temp) grid
            per = max(1, N_TOTAL_CANDS // max(1, len(grid)))
            leftover = N_TOTAL_CANDS - per * len(grid)
            texts: List[str] = []
            for i, (eng, t) in enumerate(grid):
                k = per + (1 if i < leftover else 0)
                texts.extend(sample_batch(client, eng, prompt, k, t, MAX_TOKENS))

            # Deduplicate plans
            uniq: Dict[Tuple, Dict[str, Any]] = {}
            for txt in texts:
                pl = parse_actions_to_plan(txt, toggle_budget, n_line)
                key = tuple(
                    sorted(
                        (a.get("name", ""), int(a["line"]))
                        for a in pl.get("corridor_actions", [])
                    )
                )
                if key and key not in uniq:
                    uniq[key] = pl

            if len(uniq) < 2:
                total_prompts += 1
                continue

            scored: List[Tuple[Dict, Dict]] = []
            for j, pl in enumerate(uniq.values()):
                pfp = IO_DIR / f"dpo_plan_{xi.stem}_{run_id}_{j}.json"
                rfp = IO_DIR / f"dpo_res_{xi.stem}_{run_id}_{j}.json"
                pfp.write_text(json.dumps(pl), encoding="utf-8")
                sc = call_verify(pfp, xi, rfp)
                try:
                    pfp.unlink(missing_ok=True)
                    rfp.unlink(missing_ok=True)
                except Exception:
                    pass
                if sc is None:
                    continue

                # Fill ac_ok if missing
                sc["ac_ok"] = is_ac_ok(sc)

                ev_writer.writerow({
                    "xi": xi.as_posix(),
                    "plan": plan_to_text(pl),
                    "J_dc": sc.get("J_dc"),
                    "V_pen": sc.get("V_pen"),
                    "J_ac": sc.get("J_ac"),
                    "feasible_dc": int(is_dc_ok(sc)),
                    "ac_ok": int(sc.get("ac_ok", 0)),
                    "notes": sc.get("notes", ""),
                })
                scored.append((pl, sc))

            def keep(tup: Tuple[Dict, Dict]) -> bool:
                sc = tup[1]
                if REQUIRE_DC_OK and not is_dc_ok(sc):
                    return False
                if REQUIRE_AC_OK and not is_ac_ok(sc):
                    return False
                return True

            pool = [t for t in scored if keep(t)]
            if len(pool) < 2:
                total_prompts += 1
                continue

            pool.sort(key=lambda t: vpen_eff(t[1]))
            best  = pool[:max(1, TOPK_BEST)]
            worst = pool[-max(1, TOPK_WORST):][::-1]

            made = 0
            abs_thr = V_ABS_MIN
            rel_thr = V_REL_MIN

            for _ in range(RELAX_STEPS + 1):
                if made >= MAX_PAIRS_PER:
                    break
                for bp, bs in best:
                    if made >= MAX_PAIRS_PER:
                        break
                    bv = vpen_eff(bs)
                    for wp, ws in worst:
                        if made >= MAX_PAIRS_PER:
                            break
                        wv = vpen_eff(ws)
                        dv  = wv - bv
                        rel = wv / max(bv, 1.0)
                        if dv >= abs_thr and rel >= rel_thr:
                            chosen   = plan_to_text(bp).strip()
                            rejected = plan_to_text(wp).strip()
                            if not chosen or not rejected or chosen == rejected:
                                continue
                            rec = {
                                "prompt": prompt,
                                "xi_file": xi.as_posix(),
                                "run_id": run_id,
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

            # Backstop: at least one pair if pool has diversity
            if ENABLE_BACKSTOP and made == 0 and pool:
                bp, bs = min(pool, key=lambda t: vpen_eff(t[1]))
                wp, ws = max(pool, key=lambda t: vpen_eff(t[1]))
                bv, wv = vpen_eff(bs), vpen_eff(ws)
                if wv > bv and (wv - bv) >= BACKSTOP_MIN_ABS:
                    chosen   = plan_to_text(bp).strip()
                    rejected = plan_to_text(wp).strip()
                    if chosen and rejected and chosen != rejected:
                        rec = {
                            "prompt": prompt,
                            "xi_file": xi.as_posix(),
                            "run_id": run_id,
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

    # Pointer to latest pairs file
    pointer = IO_DIR / "dpo_pairs_pointer.jsonl"
    try:
        if pointer.exists():
            pointer.unlink()
        try:
            pointer.symlink_to(pairs_fp.name)
        except Exception:
            # Fallback: copy contents
            pointer.write_text(pairs_fp.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    print(f"Wrote pairs: {pairs_fp} (pairs={total_pairs}, prompts={total_prompts})")
    print(f"Appended OpenAI prefs: {prefs_fp}")
    print(f"QC CSV: {qc_fp}")
    print(f"Candidate eval CSV: {eval_fp}")


if __name__ == "__main__":
    main()
