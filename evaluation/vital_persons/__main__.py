"""Generate and/or score Vital Persons answers on model-specific dev/test cases."""
import argparse
import json
from pathlib import Path

from .cases import REPO, SOURCE_MODELS
from .data import load_cases
from .generate import VARIANTS, settings, drag_samples, run
from .pipeline import LocationPipeline, SavedLocationPipeline
from .report import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=SOURCE_MODELS, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), default='test')
    parser.add_argument('--stage', choices=('generate', 'score', 'both'), default='generate')
    parser.add_argument('--modality', choices=('temporal', 'location', 'both'), default='both')
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    parser.add_argument('--data-dir', type=Path, help='Prepared model directory; defaults to the included dataset')
    parser.add_argument('--metadata', type=Path, help='Gold metadata JSONL; defaults to the reviewed gold_adapted.jsonl')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--generation-dir', type=Path, help='Use another generation directory; default: OUTPUT/generations')
    parser.add_argument('--answer-status-predictions', type=Path,
                        help='Opt-in audited full-answer status judgments; all requested answers must be covered')
    parser.add_argument('--baseline-dir', type=Path, help='Reuse baseline generations from this directory; only generate finetuned answers')
    parser.add_argument('--adapter', type=Path, help='Local PEFT adapter or training run; otherwise use released model settings')
    parser.add_argument('--adapter-ref', type=Path, help='Pinned model.json for a released ablation instead of the main adapter')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--drag-batch-size', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--drag-samples-dir', type=Path, help='Directory with the selected model\'s four *_samples.jsonl files')
    parser.add_argument('--normalization-cache', type=Path, help='Existing SQLite cache when computing location normalizations; unnecessary with --saved-location-dir')
    parser.add_argument('--saved-location-dir', type=Path,
                        help='Rescore saved location_extractions/ files, including reviewed output corrections; no normalizer/cache needed')
    parser.add_argument('--extraction-cache', type=Path, default=REPO/'data/cache/vital-location-extraction')
    parser.add_argument('--extractor-device', default='cpu')
    parser.add_argument('--extractor-dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    parser.add_argument('--extractor-batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--online', action='store_true', help='Permit missing Wikidata requests; default is cache-only/offline')
    parser.add_argument('--user-agent', help='Contact-bearing Wikidata User-Agent for --online; or set WIKI_USER_AGENT')
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--confidence-level', type=float, default=.95)
    parser.add_argument('--seed', type=int, default=42, help='Person-cluster bootstrap seed; generation seed is fixed at 42')
    parser.add_argument('--limit', type=int, help='Smoke test: first N cases per modality; use a separate output directory')
    parser.add_argument('--dry-run', action='store_true', help='Show case counts/settings without loading weights or writing outputs')
    args = parser.parse_args()
    if args.adapter and args.adapter_ref:
        parser.error('Use either --adapter or --adapter-ref')
    if len(set(args.variants)) != len(args.variants):
        parser.error('Select each variant once')
    for value in (args.batch_size, args.drag_batch_size, args.max_new_tokens, args.extractor_batch_size, args.workers, args.limit):
        if value is not None and value < 1:
            parser.error('Batch sizes, workers, token limits and case limits must be positive')
    if args.bootstrap_samples < 0 or not 0 < args.confidence_level < 1:
        parser.error('Use a nonnegative bootstrap count and confidence level in (0,1)')
    cases = load_cases(args.model, args.split, data_dir=args.data_dir, metadata=args.metadata)
    cases = {key: rows[:args.limit] if args.limit else rows for key, rows in cases.items() if args.modality in {'both', key}}
    samples = {key: drag_samples(args.model, rows, args.drag_samples_dir) for key, rows in cases.items()} if 'base_drag' in args.variants else {}
    # Score-only deliberately does not resolve/check a local or remote adapter.
    config = settings(args.model, batch_size=args.batch_size, drag_batch_size=args.drag_batch_size,
                      max_new_tokens=args.max_new_tokens, adapter=args.adapter) if args.stage != 'score' else None
    if config is not None and args.adapter_ref:
        from shared.models import adapter_reference
        config = adapter_reference(config, args.adapter_ref)
    print(json.dumps({'model': args.model, 'split': args.split, 'cases': {key: len(rows) for key, rows in cases.items()},
                      'variants': args.variants, 'generation': config, 'smoke_test': args.limit is not None}, indent=2))
    if args.dry_run:
        return
    if args.stage in {'score', 'both'} and 'location' in cases:
        if not args.saved_location_dir and (args.normalization_cache is None or not args.normalization_cache.is_file()):
            parser.error('Location scoring requires --normalization-cache pointing to an existing SQLite cache')
    directory = args.generation_dir or args.output_dir/'generations'
    if args.stage in {'generate', 'both'}:
        variants = [v for v in args.variants if v == 'finetuned'] if args.baseline_dir else args.variants
        if variants:
            run(cases, variants, directory, config, device=args.device, samples_by_modality=samples)
    if args.stage in {'score', 'both'}:
        pipeline = (SavedLocationPipeline(args.saved_location_dir) if args.saved_location_dir else LocationPipeline(args.extraction_cache, args.normalization_cache,
            device=args.extractor_device, dtype=args.extractor_dtype, batch_size=args.extractor_batch_size,
            workers=args.workers, offline=not args.online, user_agent=args.user_agent)) if 'location' in cases else None
        try:
            score(cases, args.variants, directory, args.output_dir, pipeline=pipeline, samples_by_modality=samples,
                  num_resamples=args.bootstrap_samples, confidence_level=args.confidence_level, seed=args.seed,
                  baseline_dir=args.baseline_dir, answer_status_predictions=args.answer_status_predictions)
        finally:
            if pipeline:
                pipeline.close()
        print(f'Evaluation written to {args.output_dir}')


if __name__ == '__main__':
    main()
