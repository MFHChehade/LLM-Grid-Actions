# python/build_sft_from_gt.py
# Build a raw SFT corpus from MILP ground truth.
# Each record includes:
#  - prompt  : full JSON summary (pretty-printed)
#  - target  : "open(...)" actions from mip_opt
#  - xi_file : PSPS mask path (so we can reproduce/infer later)
#  - gt_file / summary_file : cached artifacts for audit
#
# Outputs:
#   io/sft_raw.jsonl         (append-only corpus; one record per scenario)
#   io/psps_*.json           (if sampling)
#   io/summary_*.json        (cached summaries)
#   io/gt_mip_*.json         (cached MILP results)

import os, json, time, random, subprocess
from pathlib import Path
from typing import Dict, List, Any

# -----------------------------
# Paths / constants
# -----------------------------
ROOT       = Path(__file__).resolve().parents[1]
MATLAB_DIR = ROOT / "matlab"
CONFIG     = ROOT / "config"
IO_DIR     = ROOT / "io"
IO_DIR.mkdir(exist_ok=True)

CASE        = os.getenv("CASE_NAME", "case118")
LIMITS_YML  = (CONFIG / "limits.yml").as_posix()
CORRIDORS_F = CONFIG / "corridor_map.json"
OUT_JL      = IO_DIR / "sft_raw.jsonl"

SEED              = int(os.getenv("SEED", "7"))
N_SCENES          = int(os.getenv("N_SCENES", "200"))
USE_EXISTING      = bool(int(os.getenv("USE_EXISTING_PSPS", "0")))
K_OPEN_PER_SCEN   = int(os.getenv("K_OPEN_PER_SCEN", "2"))
XI_OVERRIDE       = os.getenv("XI", "")

random.seed(SEED)

# -----------------------------
# MATLAB helpers
# -----------------------------
def _run_matlab_batch(batch: str) -> int:
    cmd = f"matlab -batch \"addpath('{MATLAB_DIR.as_posix()}'); {batch}; exit;\""
    return subprocess.run(cmd, shell=True).returncode

def get_case_counts() -> Dict[str, int]:
    out = (IO_DIR / f"counts_{int(time.time())}.json").as_posix()
    batch = (
        f"mpc=rundcpf('{CASE}'); "
        f"s=struct('case','{CASE}','buses',size(mpc.bus,1),'lines',size(mpc.branch,1)); "
        f"fid=fopen('{out}','w'); fwrite(fid,jsonencode(s)); fclose(fid)"
    )
    rc = _run_matlab_batch(batch)
    if rc != 0:
        raise RuntimeError("MATLAB failed while computing case counts.")
    with open(out, "r") as f:
        return json.load(f)

def call_make_summary(xi_path: Path, out_summary: Path) -> Dict:
    batch = f"make_summary('{CASE}','{xi_path.as_posix()}','{LIMITS_YML}','{out_summary.as_posix()}')"
    rc = _run_matlab_batch(batch)
    if rc != 0:
        raise RuntimeError("MATLAB make_summary failed")
    with open(out_summary, "r") as f:
        return json.load(f)

def call_run_ground_truth_mip(xi_path: Path, out_path: Path) -> Dict:
    batch = f"run_ground_truth('{CASE}','{xi_path.as_posix()}','{LIMITS_YML}','{out_path.as_posix()}','mip_opt')"
    rc = _run_matlab_batch(batch)
    if rc != 0:
        raise RuntimeError("MATLAB run_ground_truth mip_opt failed")
    with open(out_path, "r") as f:
        return json.load(f)

# -----------------------------
# Corridors & PSPS helpers
# -----------------------------
def load_corridor_map() -> Dict[str, List[int]]:
    with open(CORRIDORS_F, "r") as f:
        return json.load(f)

def invert_corridors(corrmap: Dict[str, List[int]]) -> Dict[int, str]:
    inv = {}
    for name, ids in corrmap.items():
        for i in ids:
            inv[int(i)] = name
    return inv

def list_existing_psps(io_dir: Path) -> List[Path]:
    files = list(io_dir.glob("psps_*.json"))
    return sorted(files, key=lambda p: p.stat().st_mtime)

def sample_psps_mask(n_line: int, corrmap: Dict[str, List[int]], k_open: int) -> List[int]:
    """Create a PSPS mask where all forced-open lines come from a single corridor."""
    xi = [1] * n_line

    # Only corridors that have at least k_open lines are eligible
    eligible = [
        (name, [int(i) for i in ids])
        for name, ids in corrmap.items()
        if len(ids) >= k_open
    ]
    if not eligible:
        raise ValueError(f"No corridor has at least {k_open} lines for sampling.")

    # Pick a random corridor
    corr_name, line_ids = random.choice(eligible)

    # Pick k_open distinct lines from this corridor
    chosen = random.sample(line_ids, k_open)

    for idx in chosen:
        if 1 <= idx <= n_line:
            xi[idx - 1] = 0
        else:
            raise ValueError(f"Line index {idx} out of range 1..{n_line}")

    return xi


def save_json(path: Path, obj: Any):
    with open(path, "w") as f:
        json.dump(obj, f)

# -----------------------------
# Target formatting
# -----------------------------
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

# -----------------------------
# Main
# -----------------------------
def main():
    counts = get_case_counts()
    n_line = int(counts["lines"])
    corrmap   = load_corridor_map()
    line2corr = invert_corridors(corrmap)

    # Build the PSPS scenario list
    scenarios: List[Path] = []

    if XI_OVERRIDE:
        p = Path(XI_OVERRIDE)
        if not p.exists():
            raise FileNotFoundError(f"XI={p} does not exist")
        if not (p.name.startswith("psps_") and p.suffix == ".json"):
            raise ValueError("XI must point to a psps_*.json file")
        scenarios = [p]
    elif USE_EXISTING:
        scenarios = list_existing_psps(IO_DIR)
        if not scenarios:
            raise FileNotFoundError("USE_EXISTING_PSPS=1 but no psps_*.json found in io/")
    else:
        for _ in range(N_SCENES):
            xi = sample_psps_mask(n_line, corrmap, K_OPEN_PER_SCEN)
            xi_path = IO_DIR / f"psps_{int(time.time())}_{random.randint(0,9999)}.json"
            save_json(xi_path, xi)
            scenarios.append(xi_path)

    wrote = 0
    with open(OUT_JL, "a", encoding="utf-8") as out:
        for xi_path in scenarios:
            # Summary (full JSON becomes the prompt)
            summary_path = IO_DIR / f"summary_{xi_path.stem}.json"
            summary = call_make_summary(xi_path, summary_path)

            # MILP ground truth (label)
            gt_path = IO_DIR / f"gt_mip_{xi_path.stem}.json"
            gt = call_run_ground_truth_mip(xi_path, gt_path)

            # Extract toggles and format target
            toggles = gt.get("opt_switches", []) or gt.get("toggles", [])
            target  = target_from_mip_switches(toggles, line2corr)

            rec = {
                "prompt": json.dumps(summary, indent=2),
                "target": target,
                "xi_file": xi_path.as_posix(),
                "summary_file": summary_path.as_posix(),
                "gt_file": gt_path.as_posix(),
            }
            out.write(json.dumps(rec) + "\n")
            wrote += 1

    print(f"Wrote {wrote} raw SFT examples to {OUT_JL.as_posix()}")

if __name__ == "__main__":
    main()
