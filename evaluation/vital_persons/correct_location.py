"""Apply explicit corrections to saved evaluation normalizations, without a normalizer or cache."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from shared.io import digest, file_hash, read_rows, write
from parsing.location.normalization.schemas import ComponentResolution, NormalizedLocation


def input_key(components, answer):
    if (not isinstance(components, (list, tuple)) or not components
            or any(not isinstance(c, str) for c in components)
            or not any(c.strip() for c in components)
            or not isinstance(answer, str) or not answer.strip()):
        raise ValueError('Correction input requires components and the full answer')
    return digest({'components': list(components), 'answer': answer})


def snapshot(normalization):
    return deepcopy({key: normalization.get(key) for key in ('status', 'value')})


def load_corrections(path):
    records = {}
    for record in read_rows(path):
        key = input_key(record['components'], record['answer'])
        if key in records or not record.get('note') or not record.get('reviewer'):
            raise ValueError('Corrections require unique exact inputs, reviewer and note')
        after = record['after']
        value = NormalizedLocation.model_validate(after['value']) if after['value'] else None
        ComponentResolution(status=after['status'], components=record['components'], value=value,
                            consensus_value=value if after['status'] == 'normalized' else None)
        if record['before'] == after:
            raise ValueError('A correction must change the saved normalization')
        records[key] = record
    return records


def correct_row(row, records, corrections_hash):
    components = row['extraction'].get('components') or []
    if not components:
        return row, None
    key = input_key(components, row['answer'])
    correction = records.get(key)
    if correction is None:
        return row, None
    original = row.get('normalization') or {}
    if snapshot(original) != correction['before']:
        raise ValueError(f'Correction no longer matches saved normalization: {key}')
    after = correction['after']
    value = NormalizedLocation.model_validate(after['value']) if after['value'] else None
    resolution = ComponentResolution(
        status=after['status'], components=components, value=value,
        consensus_value=value if after['status'] == 'normalized' else None,
        disambiguation={'note': f'Evaluation output correction: {correction["note"]}'})
    result = deepcopy(row)
    result['normalization_before_correction'] = original
    result['normalization'] = resolution.model_dump(mode='json')
    result['normalization_correction'] = {'input_key': key, 'file_sha256': corrections_hash,
                                         'reviewer': correction['reviewer'], 'note': correction['note']}
    return result, key


def apply(results_dir, corrections_path, output_dir):
    if output_dir.exists():
        raise FileExistsError('Use a new output directory; original evaluation files are never overwritten')
    if output_dir.resolve().is_relative_to(results_dir.resolve()):
        raise ValueError('Corrected output must be outside the original results directory')
    records = load_corrections(corrections_path)
    corrections_hash = file_hash(corrections_path)
    sources = sorted(results_dir.glob('*/*/location_extractions/*.jsonl'))
    if not sources:
        raise ValueError('No saved location evaluation outputs found')
    # Preflight every correction before writing any output.
    matched, changed, total = set(), 0, 0
    hashes = {}
    for path in sources:
        hashes[str(path.relative_to(results_dir))] = file_hash(path)
        for row in read_rows(path):
            _, key = correct_row(row, records, corrections_hash)
            total += 1
            if key:
                matched.add(key)
                changed += 1
    if matched != records.keys():
        raise ValueError('Some correction inputs do not occur in the selected evaluation outputs')
    for path in sources:
        if file_hash(path) != hashes[str(path.relative_to(results_dir))]:
            raise ValueError('Evaluation input changed after preflight')
        rows = [correct_row(row, records, corrections_hash)[0] for row in read_rows(path)]
        write(output_dir/path.relative_to(results_dir), rows, rows=True)
    summary = {'evaluated_rows': total, 'corrected_rows': changed, 'corrected_unique_inputs': len(matched),
               'corrections_sha256': corrections_hash, 'source_sha256': hashes,
               'note': 'Saved evaluation normalization outputs only. Original answers, extraction components, gold, training data and normalization cache are unchanged. Outcomes must be rescored from these corrected records.'}
    write(output_dir/'corrections-summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'source_sha256'}, indent=2))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=root/'results')
    parser.add_argument('--corrections', type=Path, default=root/'corrections.jsonl')
    parser.add_argument('--output-dir', type=Path, default=root/'results-corrected')
    args = parser.parse_args()
    apply(args.results_dir, args.corrections, args.output_dir)


if __name__ == '__main__':
    main()
