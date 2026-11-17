"""Normalize extractor JSONL, or safely snapshot a SQLite cache."""

import argparse
import json
from pathlib import Path

from .cache import snapshot_cache
from .service import LocationNormalizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    normalize = commands.add_parser("normalize", help="Resolve extracted components")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--cache-db", type=Path, required=True)
    normalize.add_argument("--workers", type=int, default=4)
    normalize.add_argument("--allow-network", action="store_true", help="Default: cached evidence only")
    normalize.add_argument("--user-agent", help="Wikidata User-Agent with contact information; alternatively WIKI_USER_AGENT")
    normalize.add_argument("--refresh-evidence", action="store_true", help="Use separately versioned entity-profile evidence")
    snapshot = commands.add_parser("snapshot", help="Create a standalone database safe to copy")
    snapshot.add_argument("--cache-db", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot_cache(args.cache_db, args.output)
        print(f"Cache snapshot written to {args.output}")
        return
    if args.workers < 1 or not args.input.is_file() or args.output.exists():
        parser.error("Input must exist, output must be new, and workers must be positive")
    from tqdm.auto import tqdm
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with LocationNormalizer(args.cache_db, offline=not args.allow_network,
                            user_agent=args.user_agent, refresh_evidence=args.refresh_evidence) as model:
        with args.input.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as target:
            with tqdm(desc="Location normalization", unit="answer") as progress:
                def save(batch):
                    for row, result in zip(batch, model.process_many(batch, workers=args.workers), strict=True):
                        target.write(json.dumps({**row, "normalization": result}, ensure_ascii=False) + "\n")
                    target.flush()
                    progress.update(len(batch))

                batch = []
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "normalization" in row:
                        raise ValueError("Input already contains normalization; use the original extracted answers")
                    batch.append(row)
                    if len(batch) == args.workers * 4:
                        save(batch)
                        batch = []
                if batch:
                    save(batch)
    print(f"Normalization written to {args.output}")


if __name__ == "__main__":
    main()
