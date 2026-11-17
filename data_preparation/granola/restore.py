"""Reconstruct local Granola training files from released IDs and selected levels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from generalization.benchmark_data import dataset_settings
from shared.constants import MODEL_NAMES, GRANOLA_TRAIN_RELATIONS
from shared.io import digest, read_rows
from training.schemas import (GranolaGroup, GranolaLevel, GranolaTrainingExample,
                              canonical_answer_key, realize_candidate)
from .recognition import prepare_rows

ROOT = Path(__file__).resolve().parent
SPLITS = ('train', 'dev', 'test')


def development_qids(qids, seed=42, fraction=0.1):
    """Original entity-grouped split: first 10% under a seeded SHA-256 ordering."""
    ordered = sorted(qids, key=lambda q: hashlib.sha256(f'{seed}:{q}'.encode()).digest())
    count = min(len(ordered)-1, max(1, round(len(ordered)*fraction))) if len(ordered) > 1 else 0
    return set(ordered[:count])


def export_assignments(directory, model):
    """Extract identifiers, decisions and verification hashes; no benchmark text."""
    rows = [r for split in SPLITS for r in read_rows(directory/f'{split}.jsonl')]
    return {'schema_version': 1, 'model': model, 'source_model': MODEL_NAMES[model],
            'dataset': dataset_settings('granola'),
            'assignments': [
                {'original_row_id': r['original_row_id'], 'qid': r['qid'],
                 'relation': r['relation'], 'split': r['split'],
                 'selected_level': float(r['primary_target_level'].removeprefix('GRANOLA_').replace('_', '.')),
                 'example_sha256': digest(r)} for r in rows]}


def build_example(assignment, source, source_model):
    if assignment['qid'] != source['question_entity_qid'] or assignment['relation'] != source['relation']:
        raise ValueError('Source identity differs for row '+assignment['original_row_id'])
    hierarchy = {GranolaLevel.from_number(level): answer.strip()
                 for level, answer in zip(source['normalized_levels'], source['granola_answers'], strict=True)}
    target = GranolaLevel.from_number(assignment['selected_level'])
    if target not in hierarchy:
        raise ValueError('Selected level is absent from source hierarchy')
    entity = re.sub(r'\s+', ' ', source['question_entity']).strip()
    candidates, seen = [], set()
    for level in [target, *(level for level in GranolaLevel if level != target and level in hierarchy)]:
        candidate = realize_candidate(entity, source['relation'], level, hierarchy[level])
        key = canonical_answer_key(candidate.answer)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    group = GranolaGroup(target_level=target, target_index=0, candidates=candidates)
    return GranolaTrainingExample(
        case_id=f"granola:{source_model}:{assignment['qid']}:{assignment['relation']}:{assignment['original_row_id']}",
        qid=assignment['qid'], entity=entity, original_row_id=assignment['original_row_id'],
        split=assignment['split'], source_model=source_model, relation=assignment['relation'],
        prompt=source['question'], source_answer=hierarchy[target],
        positive_answer=candidates[0].answer, primary_target_level=target,
        contrastive_groups=[group]).model_dump(mode='json')


def reconstruct(directory, raw, model):
    """Validate frozen decisions and reconstruct all splits before writing anything."""
    public = json.loads((directory/'split_assignments.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    if (public['schema_version'] != 1 or public['model'] != model
            or public['source_model'] != MODEL_NAMES[model]
            or manifest['source_model'] != MODEL_NAMES[model]
            or public['dataset'] != dataset_settings('granola')):
        raise ValueError('Assignment model or dataset version mismatch')
    raw = [{**r, 'Unnamed: 0': r.get('Unnamed: 0', r.get('id'))} for r in raw]
    if any(r['Unnamed: 0'] is None for r in raw):
        raise ValueError('Local Granola rows must retain their original source IDs')
    sources = {str(r['id']): r for r in prepare_rows(raw)}
    rows = {split: [] for split in SPLITS}
    ids, entities = set(), {}
    held_out = set(manifest['held_out_entity_qids'])
    for assignment in public['assignments']:
        key, split = assignment['original_row_id'], assignment['split']
        if key in ids or split not in rows:
            raise ValueError('Duplicate source ID or invalid split')
        ids.add(key)
        qid, relation = assignment['qid'], assignment['relation']
        if entities.setdefault(qid, split) != split:
            raise ValueError('Entity occurs in more than one split')
        if ((split == 'test') != (relation in manifest['held_out_relations'])
                or (split == 'test') != (qid in held_out)
                or split != 'test' and relation not in GRANOLA_TRAIN_RELATIONS):
            raise ValueError('Assignment violates held-out relation/entity policy')
        if key not in sources:
            raise ValueError('Missing cleaned source row: '+key)
        example = build_example(assignment, sources[key], public['source_model'])
        if digest(example) != assignment['example_sha256']:
            raise ValueError('Reconstructed example differs from the recorded training data: '+key)
        rows[split].append(example)
    train_qids = {q for q, split in entities.items() if split != 'test'}
    expected_dev = development_qids(train_qids, manifest['seed'], manifest['dev_fraction'])
    if {q for q, split in entities.items() if split == 'dev'} != expected_dev:
        raise ValueError('Assignments differ from the original entity-grouped development split')
    if dict(Counter(r['split'] for group in rows.values() for r in group)) != manifest['accepted_by_split']:
        raise ValueError('Assignment counts differ from the recorded manifest')
    outputs = {}
    for split, group in rows.items():
        payload = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in group).encode('utf-8')
        expected = next(item['sha256'] for item in manifest['output_files'] if item['path'] == f'{split}.jsonl')
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Reconstructed split checksum mismatch: '+split)
        outputs[f'{split}.jsonl'] = payload
    return outputs


def restore(directory, output_dir, raw, model):
    outputs = reconstruct(directory, raw, model)
    # Allow the default directory's public metadata and identical existing splits.
    # Reject conflicting outputs before creating or writing any file.
    for name, payload in outputs.items():
        path = output_dir/name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError('Refusing to overwrite different local data: '+str(path))
    manifest = (directory/'manifest.json').read_bytes()
    target = output_dir/'manifest.json'
    if target.exists() and target.read_bytes() != manifest:
        raise ValueError('Refusing to overwrite a different manifest')
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (output_dir/name).write_bytes(payload)
    target.write_bytes(manifest)
    return {name: len(payload.splitlines()) for name, payload in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_NAMES, required=True)
    parser.add_argument('--granola-jsonl', type=Path,
                        help='Locally reconstructed benchmark before the notebook split experiment; otherwise download the pinned upstream revision')
    parser.add_argument('--output-dir', type=Path, help='Defaults to data_preparation/granola/datasets/<model>')
    args = parser.parse_args()
    directory = ROOT/'datasets'/args.model
    if args.granola_jsonl:
        raw = read_rows(args.granola_jsonl)
    else:
        from datasets import load_dataset
        settings = dataset_settings('granola')
        raw = load_dataset(settings['dataset_id'], revision=settings['revision'], split=settings['split'])
    counts = restore(directory, args.output_dir or directory, raw, args.model)
    print(json.dumps({'model': args.model, 'reconstructed': counts}, indent=2))


if __name__ == '__main__':
    main()
