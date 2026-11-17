"""Refresh source-ablation test scoring with reviewed gold and the no-answer judge.

uv run --locked python -m ablations.source_evaluation prepare
uv run --locked python -m evaluation.vital_persons.answer_status.judge run \
  --items ablations/lfm/sources/results/answer-status/items.jsonl \
  --output-dir ablations/lfm/sources/results/answer-status --allow-api
uv run --locked python -m ablations.source_evaluation score
uv run --locked python -m ablations.source_table
"""
import argparse
from pathlib import Path

from shared.io import read_rows, write, digest, file_hash
from evaluation.vital_persons.data import load_cases
from evaluation.vital_persons.temporal import score_temporal_generations
from evaluation.vital_persons.pipeline import SavedLocationPipeline
from evaluation.vital_persons.correct_location import load_corrections, correct_row
from evaluation.vital_persons.answer_status.judge import needs_judge
from evaluation.vital_persons.report import score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'ablations/lfm/sources/results'
STATUS = OUT / 'answer-status'


def sources():
    for condition in ('base', 'consistency-only', 'recognition-only', 'mixed'):
        original = (ROOT / 'evaluation/vital_persons/results/lfm/test'
                    if condition in {'base', 'mixed'} else
                    ROOT / 'ablations/lfm/sources' / condition / 'vital_persons/test')
        yield condition, original, 'base_plain' if condition == 'base' else 'finetuned'


def refresh(stage, split, entries, output):
    cases = load_cases('lfm', split)
    status = output / 'answer-status'
    corrections_path = ROOT / 'evaluation/vital_persons/corrections.jsonl'
    corrections = load_corrections(corrections_path)
    items = []
    for condition, original, variant in entries:
        destination = output / 'rescored' / condition
        location_dir = destination / 'location_extractions'
        if stage == 'prepare':
            rows = read_rows(original / 'location_extractions' / f'{variant}.jsonl')
            corrected = []
            for row in rows:
                # Always start from the original saved resolution on reruns.
                if row.get('normalization_before_correction') is not None:
                    row = {**row, 'normalization': row['normalization_before_correction']}
                    row.pop('normalization_before_correction', None)
                    row.pop('normalization_correction', None)
                corrected.append(correct_row(row, corrections, file_hash(corrections_path))[0])
            write(location_dir / f'{variant}.jsonl', corrected, rows=True)
            for modality, cs in cases.items():
                generations = read_rows(original / 'generations' / modality / f'{variant}.jsonl')
                _, outcomes = (score_temporal_generations(cs, generations) if modality == 'temporal'
                               else SavedLocationPipeline(location_dir).score(cs, generations))
                questions = {c.case_id: c.prompt for c in cs}
                for row in outcomes:
                    routing = {**row, 'parse_status': row.get('answer_status_routing_parse_status', row['parse_status'])}
                    if needs_judge(routing):
                        item = {k: row[k] for k in ('case_id', 'qid', 'answer')}
                        item.update(question=questions[row['case_id']], model='lfm', split=split,
                                    modality=modality, variant=variant, condition=condition,
                                    parse_status=routing['parse_status'])
                        item['item_id'] = digest(item)
                        items.append(item)
        else:
            score(cases, [variant], original / 'generations', destination,
                  pipeline=SavedLocationPipeline(location_dir), num_resamples=0,
                  answer_status_predictions=status / 'predictions.jsonl')
    if stage == 'prepare':
        write(status / 'items.jsonl', items, rows=True)
        print(f'{len(items)} routed items; the shared judge cache is reused.')
    else:
        print(f'Updated scoring: {output / "rescored"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'score'))
    args = parser.parse_args()
    refresh(args.stage, 'test', sources(), OUT)


if __name__ == '__main__':
    main()
