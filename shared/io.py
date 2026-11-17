"""Read JSONL records and atomically save JSON/JSONL outputs."""
import json
import hashlib
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value, *, rows=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    text = (''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in value)
            if rows else json.dumps(value, indent=2)+'\n')
    temporary.write_text(text)
    temporary.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def iter_rows(path):
    """Stream JSONL for preparation jobs without materializing the full file."""
    with Path(path).open(encoding='utf-8') as source:
        for line in source:
            if line.strip():
                yield json.loads(line)

