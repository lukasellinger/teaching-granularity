"""JSONL plumbing for the filter and rewriter commands."""

import argparse
import json
from pathlib import Path


def run(model_class, output_field):
    parser = argparse.ArgumentParser(description=model_class.__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--adapter-path", type=Path, help="Local trained final/ adapter; default is the pinned released adapter")
    args = parser.parse_args()
    if not args.input.is_file() or args.output.exists() or args.batch_size < 1:
        parser.error("Input must exist, output must be new and batch size must be positive")
    model = model_class(device=args.device, dtype=args.dtype, batch_size=args.batch_size,
                        attention_implementation=args.attention_implementation, local_files_only=args.local_files_only,
                        adapter_path=args.adapter_path)
    from tqdm.auto import tqdm
    with args.input.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as target, tqdm(desc=output_field, unit="answer") as progress:
        def save(rows):
            for row, result in zip(rows, model.process_many(rows), strict=True):
                target.write(json.dumps({**row, output_field: result}, ensure_ascii=False) + "\n")
            target.flush()
            progress.update(len(rows))
        batch = []
        for line in source:
            if line.strip():
                row = json.loads(line)
                if output_field in row:
                    raise ValueError(f"Input already contains {output_field}")
                batch.append(row)
                if len(batch) == args.batch_size:
                    save(batch)
                    batch = []
        if batch:
            save(batch)
