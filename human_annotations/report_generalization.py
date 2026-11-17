"""Recompute human-study results using public labels, without benchmark downloads."""
import argparse
from importlib import import_module
import json
from pathlib import Path
import tempfile

from shared.io import read_rows, file_hash

ROOT = Path(__file__).resolve().parent/'generalization'


def report(root, output):
    manifest = json.loads((root/'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if file_hash(root/name) != expected:
            raise ValueError('Public annotation file changed: '+name)
    pairs = read_rows(root/'pairs.jsonl')
    labels = read_rows(root/'granularity.jsonl')
    correctness = read_rows(root/'correctness.jsonl')
    if len(pairs) != manifest['pairs'] or len({r['pair_id'] for r in pairs}) != len(pairs):
        raise ValueError('Invalid public pair coverage')
    index = {r['pair_id']: r for r in labels}
    if len(index) != len(labels) or index.keys() != {r['pair_id'] for r in pairs}:
        raise ValueError('Granularity labels must cover exactly the public pairs')
    answers = {(r['pair_id'], r['variant']) for r in correctness}
    expected = {(r['pair_id'], v) for r in pairs if r['dataset']=='granola' for v in ('base', 'finetuned')}
    if answers != expected or len(answers) != len(correctness):
        raise ValueError('Correctness labels must cover both answers of every Granola study pair')
    # The ordinal has the same ordering as original normalized question strings.
    # It preserves the exact seeded fold membership, not an alternative split.
    rows = [{**r, 'question': f"question-group-{r['question_group']:06d}"} for r in pairs]
    final = {k: r['final'] for k, r in index.items()}
    independent = {(k, a): r[a] for k, r in index.items() for a in ('annotator_1', 'annotator_2')}
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results/summary.json').exists():
        raise ValueError('Choose a new report output directory')
    study = import_module('generalization.human-study.human_study')
    study.evaluate(output, rows, final, independent, 5, 42)
    study.evaluate_correctness(output, correctness)
    return json.loads((output/'results/summary.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, help='New directory for reports; defaults to a temporary directory')
    args = parser.parse_args()
    if args.output_dir:
        report(args.data_dir, args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='human-study-report-') as tmp:
            report(args.data_dir, Path(tmp))


if __name__ == '__main__':
    main()
