"""Copy or check the cross-SDK fixtures shared with the TypeScript SDK.

The TypeScript repo (raglite) is the source of truth: it generates
tests/fixtures/shared/*.json with scripts/generate-shared-fixtures.ts. This
repo keeps an identical copy so both SDKs are tested against the same
expected outputs.

    python scripts/sync_shared_fixtures.py            # copy from ../raglite
    python scripts/sync_shared_fixtures.py --check    # exit 1 if out of date
    python scripts/sync_shared_fixtures.py --source /path/to/raglite
"""
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "tests" / "fixtures" / "shared"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=REPO_ROOT.parent / "raglite")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = args.source / "tests" / "fixtures" / "shared"
    files = sorted(source.glob("*.json"))
    if not files:
        print(f"No shared fixtures found in {source}", file=sys.stderr)
        return 1

    stale = [f.name for f in files if not (TARGET / f.name).exists() or not filecmp.cmp(f, TARGET / f.name, shallow=False)]
    extra = sorted({p.name for p in TARGET.glob("*.json")} - {f.name for f in files})

    if args.check:
        for name in stale:
            print(f"out of date: {name}", file=sys.stderr)
        for name in extra:
            print(f"not in source: {name}", file=sys.stderr)
        return 1 if stale or extra else 0

    TARGET.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(f, TARGET / f.name)
    for name in extra:
        (TARGET / name).unlink()
    print(f"Synced {len(files)} fixture(s) from {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
