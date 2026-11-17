"""Create primary-only datasets separately for each base model."""

import argparse
import json
from pathlib import Path

from .build import MODEL_NAMES, VITAL, flatten, plan, rewrite, assemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    flat = commands.add_parser("flatten", help="Turn the twenty-answer pools into extractor input JSONL")
    flat.add_argument("--samples-dir", type=Path, default=VITAL / "samples")
    flat.add_argument("--metadata-dir", type=Path, default=VITAL / "metadata")
    flat.add_argument("--output-dir", type=Path, required=True)
    flat.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    flat.add_argument("--expected-samples", type=int, default=20)
    for command in ("plan", "run", "rewrite", "assemble"):
        child = commands.add_parser(command)
        child.add_argument("--work-dir", type=Path, required=True)
        if command in {"plan", "run"}:
            child.add_argument("--processed-dir", type=Path, required=True)
            child.add_argument("--metadata-dir", type=Path, default=VITAL / "metadata")
            child.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
            child.add_argument("--threshold", type=float, default=.8)
            child.add_argument("--expected-samples", type=int, default=20)
            child.add_argument("--dev-fraction", type=float, default=1/9)
            child.add_argument("--seed", type=int, default=42)
            child.add_argument("--include-idk-negatives", action="store_true")
        if command in {"run", "assemble"}:
            child.add_argument("--output-dir", type=Path, required=True)
        if command in {"run", "rewrite"}:
            child.add_argument("--batch-size", type=int, default=16)
            child.add_argument("--device", default="cuda")
            child.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
            child.add_argument("--attention-implementation", default="sdpa")
    args = parser.parse_args()
    if getattr(args, "expected_samples", 1) < 1 or getattr(args, "batch_size", 1) < 1:
        parser.error("Sample and batch sizes must be positive")
    if not 0 < getattr(args, "threshold", .8) <= 1 or not 0 < getattr(args, "dev_fraction", 1/9) < 1:
        parser.error("Invalid consensus threshold or dev fraction")
    if getattr(args, "seed", 42) < 0 or len(set(getattr(args, "models", []))) != len(getattr(args, "models", [])):
        parser.error("Seed must be nonnegative and models must be unique")
    if args.command == "flatten":
        flatten(args)
        print(f"Extractor inputs written to {args.output_dir}")
        return
    if args.command in {"plan", "run"}:
        manifest = plan(args)
        print(f"Frozen plan: {manifest['unique_rewrite_tasks']} unique rewrite tasks")
    if args.command in {"run", "rewrite", "assemble"}:
        # Plan files are immutable; guard only the mutable rewrite cache/output.
        import fcntl
        if not args.work_dir.is_dir():
            parser.error("Work directory does not contain a frozen plan")
        with (args.work_dir / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("Another process is using this work directory")
            if args.command in {"run", "rewrite"}:
                rewrite(args)
            if args.command in {"run", "assemble"}:
                print(json.dumps(assemble(args), indent=2))


if __name__ == "__main__":
    main()
