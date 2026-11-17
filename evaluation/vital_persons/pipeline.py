"""Cached full-answer extraction and strict normalized-location scoring."""
import json
from pathlib import Path

from shared.io import write, digest, read_rows
from parsing.temporal import is_idk_response
from .answer_status.integration import generation_abstains
from parsing.location.extractor.inference import SETTINGS
from .location import ParsedLocationResponse, score_location_prediction, aggregate_location_outcomes


def parse_component_response(answer, extraction, resolution=None, *, abstains=None):
    # Extractor IDK is not the same as the answer model explicitly abstaining.
    if (abstains if abstains is not None else is_idk_response(answer)):
        return ParsedLocationResponse('idk', None, (), ())
    if extraction['status'] != 'ok':
        return ParsedLocationResponse('extraction_failed', None, (), (), 'extraction',
                                      (extraction['status'],), (extraction.get('error', extraction['status']),))
    if not extraction.get('components'):
        return ParsedLocationResponse('unparsed', None, (), ())
    if resolution is None:
        raise ValueError('Stated extraction is missing its normalization result')
    if resolution.status != 'normalized':
        return ParsedLocationResponse('normalization_failed', None, (), (), 'normalization',
            (resolution.status,), (resolution.error or 'Incomplete or conflicting component normalization',))
    value = resolution.value.to_value()
    return ParsedLocationResponse('parsed', value, tuple(resolution.value.explicit_levels), (value,))


class LocationPipeline:
    """Load the extractor only on a cache miss; reuse SQLite Wikidata evidence."""
    def __init__(self, cache_dir, cache_db, *, device='cpu', dtype='float32', batch_size=4,
                 workers=8, offline=True, user_agent=None, extractor=None, normalizer=None):
        self.cache_dir = Path(cache_dir)
        self.cache_db = Path(cache_db)
        self.device, self.dtype, self.batch_size = device, dtype, batch_size
        self.workers, self.offline, self.user_agent = workers, offline, user_agent
        self.extractor, self.normalizer = extractor, normalizer
        self.contract = {'version': 'event-components-v1', 'extractor': SETTINGS,
                         'device': device, 'dtype': dtype, 'batch_size': batch_size, 'attention': 'sdpa'}

    def score(self, cases, generations):
        by_case = {row['case_id']: row for row in generations}
        if len(by_case) != len(generations) or set(by_case) != {case.case_id for case in cases}:
            raise ValueError('Location scoring requires unique matching case IDs')
        tasks, keys, abstentions = {}, {}, {}
        for case in cases:
            answer = by_case[case.case_id]['answer']
            identity = {'contract': self.contract, 'question': case.prompt, 'answer': answer}
            key = digest(identity)
            tasks.setdefault(key, identity)
            keys[case.case_id] = key
            abstains = generation_abstains(by_case[case.case_id])
            if key in abstentions and abstentions[key] != abstains:
                raise ValueError('Conflicting answer-status judgments for identical inputs')
            abstentions[key] = abstains
        extractions, pending = {}, []
        for key, identity in tasks.items():
            path = self.cache_dir/key[:2]/f'{key}.json'
            if abstentions[key]:
                extractions[key] = {'status': 'ok', 'location': 'IDK', 'components': [], 'local_abstention': True}
            elif path.exists():
                record = json.loads(path.read_text())
                if record['identity'] != identity:
                    raise ValueError('Extraction cache identity mismatch')
                extractions[key] = record['extraction']
            elif not identity['answer'].strip():
                extractions[key] = {'status': 'empty_answer', 'components': [], 'location': None}
            else:
                pending.append(key)
        if pending and self.extractor is None:
            from parsing.location.extractor import LocationExtractor
            self.extractor = LocationExtractor(device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        if pending:
            from tqdm.auto import tqdm
            for offset in tqdm(range(0, len(pending), self.batch_size), desc='Location extraction', unit='batch'):
                batch = pending[offset:offset+self.batch_size]
                extracted = self.extractor.extract_many([tasks[key] for key in batch])
                if len(extracted) != len(batch):
                    raise ValueError('Extractor result count mismatch')
                for key, extraction in zip(batch, extracted, strict=True):
                    extractions[key] = extraction
                    write(self.cache_dir/key[:2]/f'{key}.json', {'identity': tasks[key], 'extraction': extraction})
        rows = [{**identity, 'extraction': extractions[key]} for key, identity in tasks.items()]
        # No normalizer/model is needed for a set of abstentions or extraction failures.
        needed = [i for i, row in enumerate(rows) if row['extraction']['status'] == 'ok' and row['extraction'].get('components')]
        resolutions = {}
        if needed:
            if self.normalizer is None:
                from parsing.location.normalization import LocationNormalizer
                self.normalizer = LocationNormalizer(self.cache_db, offline=self.offline, user_agent=self.user_agent)
            values = self.normalizer.process_many([rows[i] for i in needed], workers=self.workers)
            if len(values) != len(needed):
                raise ValueError('Normalizer result count mismatch')
            from parsing.location.normalization.schemas import ComponentResolution
            resolutions = {i: ComponentResolution.model_validate(value) for i, value in zip(needed, values, strict=True)}
        processed = {}
        for i, (key, identity) in enumerate(tasks.items()):
            resolution = resolutions.get(i)
            parsed = parse_component_response(identity['answer'], extractions[key], resolution,
                                              abstains=abstentions[key])
            processed[key] = parsed, {
                'extraction_cache_key': key, 'extraction': extractions[key],
                'normalization': resolution.model_dump(mode='json') if resolution else None,
            }
        outcomes = []
        for case in cases:
            parsed, audit = processed[keys[case.case_id]]
            outcome = score_location_prediction(case, by_case[case.case_id], parsed)
            outcome.update(location_pipeline='component-location-evaluation-v1', location_audit=audit)
            outcomes.append(outcome)
        return aggregate_location_outcomes(outcomes), outcomes

    def close(self):
        if self.normalizer:
            self.normalizer.close()


class SavedLocationPipeline:
    """Rescore saved (optionally corrected) outputs without extraction or normalization."""
    def __init__(self, directory):
        self.directory = Path(directory)

    def score(self, cases, generations):
        from parsing.location.normalization.schemas import ComponentResolution
        variants = {row['variant'] for row in generations}
        if len(variants) != 1:
            raise ValueError('Saved location scoring requires one variant')
        path = self.directory/f'{next(iter(variants))}.jsonl'
        saved = read_rows(path)
        by_id = {row['case_id']: row for row in saved}
        by_generation = {row['case_id']: row for row in generations}
        ids = {case.case_id for case in cases}
        if len(by_id) != len(saved) or len(by_generation) != len(generations) or set(by_id) != ids or set(by_generation) != ids:
            raise ValueError('Saved location scoring requires unique matching case IDs')
        outcomes = []
        for case in cases:
            row, generation = by_id[case.case_id], by_generation[case.case_id]
            if row['answer'] != generation['answer'] or row['question'] != case.prompt:
                raise ValueError('Saved location input differs from the evaluation answer/question')
            resolution = ComponentResolution.model_validate(row['normalization']) if row.get('normalization') else None
            parsed = parse_component_response(generation['answer'], row['extraction'], resolution,
                                              abstains=generation_abstains(generation))
            outcome = score_location_prediction(case, generation, parsed)
            outcome.update(location_pipeline='saved-location-evaluation-v1',
                           location_audit={key: row.get(key) for key in (
                               'extraction', 'normalization', 'normalization_before_correction', 'normalization_correction')},
                           saved_location_source=str(path))
            if row.get('normalization_before_correction') is not None:
                original = ComponentResolution.model_validate(row['normalization_before_correction'])
                original_parsed = parse_component_response(generation['answer'], row['extraction'], original,
                                                            abstains=generation_abstains(generation))
                outcome['answer_status_routing_parse_status'] = original_parsed.status
            outcomes.append(outcome)
        return aggregate_location_outcomes(outcomes), outcomes

    def close(self):
        pass
