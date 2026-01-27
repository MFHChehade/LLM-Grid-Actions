# python/inspect_line_limits.py

import os, json, time, subprocess, csv
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT = Path(__file__).resolve().parents[1]
MATLAB_DIR = ROOT / "matlab"
CONFIG = ROOT / "config"
IO_DIR = ROOT / "io"

CASE = os.getenv("CASE_NAME", "case118")
LIMITS_YML = (CONFIG / "limits.yml").as_posix()
MATPOWER_DIR = os.getenv("MATPOWER_DIR", "")

SFT_RAW = Path(os.getenv("SFT_RAW", (IO_DIR / "sft_raw.jsonl").as_posix()))
MAX_SCENES = int(os.getenv("MAX_SCENES", "999999"))

def run_matlab(batch: str) -> int:
    parts = [f"addpath('{MATLAB_DIR.as_posix()}');"]
    if MATPOWER_DIR:
        mp = MATPOWER_DIR.replace("\\", "/")
        parts.append(f"addpath(genpath('{mp}'));")
    prefix = " ".join(parts)
    cmd = f"matlab -batch \"{prefix} {batch}; exit;\""
    return subprocess.run(cmd, shell=True).returncode

def write_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj), encoding="utf-8")

def call_dc_flows(plan_path: Path, xi_path: Path, out_path: Path) -> Optional[Dict[str, Any]]:
    rc = run_matlab(
        f"dc_flows_report('{CASE}','{plan_path.as_posix()}','{xi_path.as_posix()}','{LIMITS_YML}','{out_path.as_posix()}')"
    )
    if rc != 0 or not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def load_corridor_map() -> Dict[int, str]:
    cm = json.loads((CONFIG / "corridor_map.json").read_text(encoding="utf-8"))
    inv = {}
    for name, ids in cm.items():
        for i in ids:
            inv[int(i)] = name
    return inv

def top_lines_str(rep: Dict[str, Any], line2corr: Dict[int, str], k: int = 5) -> str:
    tops = rep.get("top_lines", []) or []
    out = []
    for t in tops[:k]:
        lid = int(t["line"])
        corr = line2corr.get(lid, "UNK")
        out.append(f"{lid}({corr}):{t['ratio']:.3f}")
    return ";".join(out)

def main():
    if not SFT_RAW.exists():
        raise FileNotFoundError(f"Missing SFT_RAW file: {SFT_RAW}")

    line2corr = load_corridor_map()

    stamp = int(time.time())
    out_csv = IO_DIR / f"line_limits_audit_{stamp}.csv"

    rows = []
    n = 0

    with SFT_RAW.open("r", encoding="utf-8") as f:
        for line in f:
            if n >= MAX_SCENES:
                break
            rec = json.loads(line)

            xi_file = Path(rec["xi_file"])
            gt_file = Path(rec["gt_file"])

            if not xi_file.exists() or not gt_file.exists():
                continue

            gt = json.loads(gt_file.read_text(encoding="utf-8"))
            gt_plan = gt.get("opt_plan", {"corridor_actions": []})
            if not isinstance(gt_plan, dict):
                gt_plan = {"corridor_actions": []}

            # Make plan files
            stem = xi_file.stem
            plan_dn = IO_DIR / f"plan_dn_{stem}.json"
            plan_gt = IO_DIR / f"plan_gt_{stem}.json"
            write_json(plan_dn, {"corridor_actions": []})
            write_json(plan_gt, gt_plan)

            # Run MATLAB reports
            rep_dn_path = IO_DIR / f"flows_dn_{stem}.json"
            rep_gt_path = IO_DIR / f"flows_gt_{stem}.json"
            rep_dn = call_dc_flows(plan_dn, xi_file, rep_dn_path)
            rep_gt = call_dc_flows(plan_gt, xi_file, rep_gt_path)

            if rep_dn is None or rep_gt is None:
                continue

            row = {
                "xi_file": xi_file.as_posix(),
                "gt_file": gt_file.as_posix(),
                "dn_feasible": rep_dn.get("feasible", False),
                "gt_feasible": rep_gt.get("feasible", False),
                "dn_J": rep_dn.get("J", None),
                "gt_J": rep_gt.get("J", None),
                "dn_shed_MW": rep_dn.get("shed_MW", None),
                "gt_shed_MW": rep_gt.get("shed_MW", None),
                "dn_max_loading": rep_dn.get("max_loading", None),
                "gt_max_loading": rep_gt.get("max_loading", None),
                "dn_n_ge_95": rep_dn.get("n_ge_95", None),
                "gt_n_ge_95": rep_gt.get("n_ge_95", None),
                "dn_n_ge_99": rep_dn.get("n_ge_99", None),
                "gt_n_ge_99": rep_gt.get("n_ge_99", None),
                "dn_top5": top_lines_str(rep_dn, line2corr, 5),
                "gt_top5": top_lines_str(rep_gt, line2corr, 5),
            }
            rows.append(row)
            n += 1

    # Write CSV
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Print summary
    total = len(rows)
    if total == 0:
        print("No rows written. Check MATLAB/MATPOWER paths and SFT_RAW content.")
        return

    dn_no_cong = sum((r["dn_n_ge_95"] or 0) == 0 for r in rows)
    gt_no_cong = sum((r["gt_n_ge_95"] or 0) == 0 for r in rows)
    both_no_cong = sum(((r["dn_n_ge_95"] or 0) == 0 and (r["gt_n_ge_95"] or 0) == 0) for r in rows)

    print(f"Wrote: {out_csv}")
    print(f"Total scenarios: {total}")
    print(f"Do-nothing: {dn_no_cong}/{total} have NO lines >=95%")
    print(f"Ground-truth: {gt_no_cong}/{total} have NO lines >=95%")
    print(f"Both: {both_no_cong}/{total} have NO lines >=95%  (=> often ED-like)")

if __name__ == "__main__":
    main()
