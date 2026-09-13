"""Download one explicitly selected public asset and verify its SHA256."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from prepare_common import MANIFEST, verified_asset


def main():
    assets = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=sorted(assets))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = assets[args.asset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / spec["filename"]
    if path.exists():
        verified_asset(args.asset, path)
        print(f"Already verified: {path.name}")
        return
    temporary = path.with_name(path.name + ".partial")
    existing = temporary.stat().st_size if temporary.exists() else 0
    if shutil.disk_usage(args.output_dir).free < spec["bytes"] - existing + 1024**3:
        raise ValueError("Insufficient disk space for the selected asset")
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "6",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "-C",
            "-",
            "-o",
            str(temporary),
            spec["url"],
        ],
        check=True,
    )
    verified_asset(args.asset, temporary)
    temporary.rename(path)
    print(f"Verified: {path.name}")


if __name__ == "__main__":
    main()
