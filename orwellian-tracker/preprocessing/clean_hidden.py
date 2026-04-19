from __future__ import annotations

import argparse
from pathlib import Path


def list_hidden_garbage(root: Path) -> list[Path]:
    garbage = []
    for path in root.rglob("*"):
        name = path.name
        if name.startswith("._"):
            garbage.append(path)
        if path.is_dir() and name == "__MACOSX":
            garbage.append(path)
            garbage.extend(list(path.rglob("*")))
    return garbage


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove hidden macOS garbage files from dataset trees.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = list_hidden_garbage(args.root)
    print(f"Found {len(targets)} hidden artifacts")

    if args.dry_run:
        for p in targets[:50]:
            print(p)
        return

    removed = 0
    for p in sorted(targets, key=lambda x: len(x.parts), reverse=True):
        try:
            if p.is_dir():
                p.rmdir()
            else:
                p.unlink()
            removed += 1
        except OSError:
            continue

    print(f"Removed {removed} hidden artifacts")


if __name__ == "__main__":
    main()
