"""Extract one answer or a JSONL file with question/answer fields."""

import argparse
import json
from pathlib import Path

from .inference import LocationExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answer", nargs="?")
    parser.add_argument("--question")
    parser.add_argument("--input", type=Path, help="JSONL records with question and answer fields")
    parser.add_argument("--output", type=Path, help="New JSONL file; input fields are preserved with an extraction field")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--adapter-path", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.input:
        if args.answer is not None or args.question or not args.output:
            parser.error("Use --input and --output together, without an answer/--question")
        if args.output.exists():
            parser.error(f"Output already exists: {args.output}")
    elif args.answer is None or not args.question or args.output:
        parser.error("Provide an answer and --question, or --input and --output")

    extractor = LocationExtractor(
        device=args.device, dtype=args.dtype, batch_size=args.batch_size,
        attention_implementation=args.attention_implementation, local_files_only=args.local_files_only,
        adapter_path=args.adapter_path,
    )
    if not args.input:
        print(json.dumps(extractor.extract(args.question, args.answer), indent=2, ensure_ascii=False))
        return
    from tqdm.auto import tqdm
    with args.input.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as destination:
        batch = []
        with tqdm(desc="Location extraction", unit="answer") as progress:
            def save(rows):
                results = extractor.extract_many(rows)
                for row, result in zip(rows, results, strict=True):
                    destination.write(json.dumps({**row, "extraction": result}, ensure_ascii=False) + "\n")
                destination.flush()
                progress.update(len(rows))
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    if "extraction" in row:
                        raise ValueError("Input already contains an extraction field")
                    batch.append(row)
                    if len(batch) == args.batch_size:
                        save(batch)
                        batch = []
            if batch:
                save(batch)


if __name__ == "__main__":
    main()
