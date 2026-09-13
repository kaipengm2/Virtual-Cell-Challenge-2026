"""Command-line entry points; prediction never submits or logs into a service."""

import argparse
import json
from pathlib import Path

from . import __version__
from .io import digest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vcc-atlas")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("predict", help="Build counts from prepared source statistics")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify-inputs", help="Check historical prepared-input checksums")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--data-dir", type=Path, required=True)
    demo = sub.add_parser("demo", help="Generate synthetic data and run the complete pipeline")
    demo.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            from .pipeline import predict

            report = predict(args.config, args.data_dir, args.output)
            print(
                json.dumps({k: report[k] for k in ["status", "shape", "nnz", "seconds"]}, indent=2)
            )
        elif args.command == "verify-inputs":
            results = {}
            for name, expected in json.loads(args.manifest.read_text())["files"].items():
                path = args.data_dir / name
                results[name] = (
                    "ok" if path.is_file() and digest(path) == expected else "missing or mismatch"
                )
            print(json.dumps(results, indent=2))
            if any(x != "ok" for x in results.values()):
                raise SystemExit(1)
        else:
            from .demo import run_demo

            run_demo(args.output)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
