# python/clean_io.py
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Clean generated IO files by prefix.")
    parser.add_argument(
        "--io",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "io",
        help="Path to the io/ directory (default: project/io)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without deleting them.",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra prefixes to delete (e.g., --extra tmp scratch).",
    )
    args = parser.parse_args()

    io_dir: Path = args.io
    if not io_dir.exists() or not io_dir.is_dir():
        raise SystemExit(f"IO directory not found: {io_dir}")

    # Built-in prefixes + optional extras; case-insensitive match
    prefixes = {"inspect", "eval", "res", "run_all", "report", "gt", "counts", "chosen", "case", "psps", "result", "resullt", "plan", "summary","nb", "test", "qcap", "prefs", "xi", "plot", "sft", "tmp", "best"}
    # prefixes = {"dpo"}
    prefixes |= {p.lower() for p in args.extra}

    deleted = 0
    for p in io_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(name.startswith(pref) for pref in prefixes):
            if args.dry_run:
                print(f"[DRY RUN] would delete: {p}")
            else:
                try:
                    p.unlink()
                    print(f"deleted: {p}")
                    deleted += 1
                except Exception as e:
                    print(f"failed to delete {p}: {e}")

    if args.dry_run:
        print("Done (dry run).")
    else:
        print(f"Done. Deleted {deleted} file(s).")

if __name__ == "__main__":
    main()
