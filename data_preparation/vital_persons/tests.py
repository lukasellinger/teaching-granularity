"""Offline collection, direct gold construction and generation-pool tests."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shared.io import read_rows, write
from . import collect, generate, metadata, split


def entity(name='', **properties):
    return {'labels': {'en': {'value': name}} if name else {}, 'claims': {
        p: [{'rank': 'normal', 'mainsnak': {'datavalue': {'value': {'id': q}}}} for q in qids]
        for p, qids in properties.items()}}


def binding(**fields):
    return {key: {'type': 'uri' if text.startswith('Q') and text[1:].isdigit() else 'literal',
                  'value': 'http://www.wikidata.org/entity/'+text if text.startswith('Q') and text[1:].isdigit() else text}
            for key, text in fields.items()}


def location_fixture(cache):
    cache.put('hierarchy', 'Q11299', [binding(place='Q11299', placeLabel='Manhattan', admin=q,
        adminLabel=name, adminDescription=description) for q,name,description in [
            ('Q11299', 'Manhattan', 'borough of New York City'),
            ('Q60', 'New York City', 'city in the United States'), ('Q1384', 'New York', 'state of the United States')]])
    for q in ('Q11299', 'Q60', 'Q1384'):
        cache.put('node_metadata', q, [binding(node=q, country='Q30', countryLabel='United States',
                                              continent='Q49', continentLabel='North America')])
    cache.put('search', 'Manhattan', [{'description': 'borough of New York City'}])
    entities = {'Q11299': entity('Manhattan', P31=['Q408804'], P131=['Q60']),
                'Q60': entity('New York City', P31=['Q515']), 'Q1384': entity('New York')}
    for q, payload in entities.items():
        cache.put('entity', q, payload)
    return entities


class VitalTests(unittest.TestCase):
    def test_initial_and_adapted_gold_preserve_cohort_dates_and_event_identity(self):
        initial = read_rows(metadata.ROOT/'metadata/gold_initial.jsonl')
        adapted = read_rows(metadata.ROOT/'metadata/gold_adapted.jsonl')
        self.assertEqual(len(initial), 1640)
        self.assertEqual([p['qid'] for p in initial], [p['qid'] for p in adapted])
        for before, after in zip(initial, adapted, strict=True):
            for key in ('qid', 'name', 'wiki_title', 'birth_date', 'birth_precision',
                        'death_date', 'death_precision', 'birth_location_qid', 'death_location_qid'):
                self.assertEqual(before.get(key), after.get(key))
        self.assertNotEqual(initial, adapted)

    def test_multilingual_label_fallback_order_and_opaque_id_rejection(self):
        for index, language in enumerate(collect.LABEL_LANGUAGES):
            labels = {lang: {'value': lang+' name'} for lang in collect.LABEL_LANGUAGES[index:]}
            self.assertEqual(metadata.label({'labels': labels}), language+' name')
        self.assertEqual(metadata.label({'labels': {'en': {'value': 'Q123'},
                                                    'fr': {'value': 'Nom français'}}}), 'Nom français')
        self.assertIsNone(metadata.label({'labels': {'ja': {'value': '日本語'},
                                                     'en': {'value': 'Q123'}}}))

    def test_targeted_label_fetch_is_cached_without_overwriting_claims(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            cache = collect.Cache(tmp/'cache.sqlite3', allow_network=True, user_agent='Test project')
            original = entity('', P131=['Q3'])
            cache.put('entity', 'Q2', original)
            write(tmp/'issues.json', [{'entity_qid': 'Q2', 'reason': 'Missing cached English/language-neutral label'},
                                     {'entity_qid': 'Q2', 'reason': 'Missing cached label'},
                                     {'entity_qid': 'Q3', 'reason': 'Ambiguous district chain'}])
            def request(endpoint, params):
                self.assertEqual(params['ids'], 'Q2')
                self.assertEqual(params['props'], 'labels')
                self.assertEqual(params['languages'], 'en|mul|fr|es|de|pt')
                return {'entities': {'Q2': {'labels': {'fr': {'value': 'Province'}}}}}
            try:
                with patch.object(cache, 'request', side_effect=request) as fetch:
                    collect.refresh_missing_labels(cache, tmp/'issues.json')
                    collect.refresh_missing_labels(cache, tmp/'issues.json')
                    self.assertEqual(fetch.call_count, 1)
                self.assertEqual(cache.get('entity', 'Q2'), original)
                self.assertEqual(cache.get('entity_labels', 'Q2')['labels']['fr']['value'], 'Province')
            finally:
                cache.close()

    def test_gold_build_uses_supplementary_labels_offline(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            cache = collect.Cache(tmp/'cache.sqlite3')
            cache.put('person', 'Q9', [binding(birthDate='1900-01-01T00:00:00Z', birthPrecision='11', birthPlace='Q1')])
            region = binding(place='Q1', placeLabel='Town', admin='Q2', adminDescription='province')
            region['adminLabel'] = {'type': 'literal', 'value': 'Q2'}
            cache.put('hierarchy', 'Q1', [binding(place='Q1', placeLabel='Town', admin='Q1', adminLabel='Town', adminDescription='city'),
                                        region])
            for q in ('Q1', 'Q2'):
                cache.put('node_metadata', q, [binding(country='Q3', countryLabel='Country', continent='Q4', continentLabel='Continent')])
            cache.put('entity', 'Q1', entity('Town', P31=['Q515']))
            cache.put('entity', 'Q2', entity(''))
            cache.put('entity_labels', 'Q2', {'labels': {'fr': {'value': 'Province française'}}})
            cache.put('search', 'Town', [])
            cache.close()
            write(tmp/'cohort.jsonl', [{'qid': 'Q9', 'name': 'Test', 'wiki_title': 'Test'}], rows=True)
            rows = metadata.prepare(tmp/'cohort.jsonl', tmp/'cache.sqlite3', tmp/'gold')
            self.assertIn('Province française', rows[0]['birth_region'])
            self.assertIn('Q2', rows[0]['birth_entity_ids']['region'])
            self.assertEqual(json.loads((tmp/'gold/unresolved.json').read_text()), [])

    def test_fixed_cohort_and_splits(self):
        root = metadata.ROOT
        source = read_rows(root/'metadata/cohort.jsonl')
        all_rows = read_rows(root/'metadata/gold_adapted.jsonl')
        development = read_rows(root/'metadata/development.jsonl')
        test = read_rows(root/'metadata/test.jsonl')
        self.assertEqual((len(all_rows), len(development), len(test)), (1640, 1476, 164))
        self.assertEqual([r['qid'] for r in all_rows], [r['qid'] for r in source])
        self.assertFalse({r['qid'] for r in development} & {r['qid'] for r in test})
        self.assertEqual({r['qid'] for r in development+test}, {r['qid'] for r in all_rows})
        full = {r['qid']: r for r in all_rows}
        self.assertTrue(all(r == {k:full[r['qid']][k] for k in ('qid','name','wiki_title')} for r in development+test))
        for before, after in zip(source, all_rows, strict=True):
            for key in ('qid', 'name', 'wiki_title'):
                self.assertEqual(before[key], after[key])

    def test_automatic_gold_rebuild_is_offline_and_does_not_write_adapted_gold(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            out = Path(tmp)
            rows = metadata.prepare(metadata.ROOT/'metadata/cohort.jsonl', metadata.ROOT/'cache/wikidata.sqlite3', out)
            adapted = read_rows(metadata.ROOT/'metadata/gold_adapted.jsonl')
            self.assertEqual([p['qid'] for p in rows], [p['qid'] for p in adapted])
            self.assertEqual(rows, read_rows(out/'gold_initial.jsonl'))
            self.assertFalse((out/'gold_adapted.jsonl').exists())
            for before, after in zip(adapted, rows, strict=True):
                for event in ('birth', 'death'):
                    for field in ('date', 'precision', 'location_qid'):
                        self.assertEqual(before.get(event+'_'+field), after.get(event+'_'+field))

    def test_region_policy_is_shared_with_prediction_normalization(self):
        nodes = [{'qid': f'Q{i}', 'label': name, 'description': description} for i,name,description in [
            (1, 'Town', 'city in Country'), (2, 'Province', 'province in Country'),
            (3, 'Old Province', 'former province of Country'), (4, 'Country', 'sovereign state'),
            (5, 'Archive', 'state archive in Country'), (6, 'Unknown', None),
            (7, 'Baghdad eyalet', 'administrative division'),
            (8, 'Quarter', 'quarter of Town'), (9, 'Island', 'island in Country'),
            (10, 'Q10', 'province in Country')]]
        entities = {'Q10': {'labels': {'de': {'value': 'Provinz'}}}}
        self.assertEqual(metadata.region_ancestors(nodes, entities), {'Q2', 'Q9', 'Q10'})

    def test_region_policy_preserves_dual_role_administrative_parent(self):
        nodes = [{'qid': 'Q1', 'label': 'Town', 'description': 'city in Country'},
                 {'qid': 'Q2', 'label': 'Territory', 'description': 'city and special administrative region'}]
        self.assertEqual(metadata.region_ancestors(nodes, {}), {'Q2'})

    def test_region_filter_keeps_names_and_ids_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = collect.Cache(Path(tmp)/'cache.sqlite3')
            raw = {'place': None, 'city': 'Town', 'country': 'Country', 'continent': 'Continent',
                   'region': ['Province', 'Old Province', 'Archive'], 'city_ids': ['Q1'],
                   'nodes': [{'qid': q, 'label': name, 'description': desc} for q,name,desc in [
                       ('Q1', 'Town', 'city in Country'), ('Q2', 'Province', 'province in Country'),
                       ('Q3', 'Old Province', 'former province'), ('Q4', 'Archive', 'state archive'),
                       ('Q5', 'Province', 'former province')]]}
            try:
                with patch.object(metadata, 'place_fields', return_value=raw), patch.object(metadata, 'ambiguity', return_value=False):
                    result, issues = metadata.build_location(cache, 'Q1', {})
                self.assertEqual(result['region'], ['Province'])
                self.assertEqual(result['entity_ids']['region'], ['Q2'])
                self.assertEqual(result['city'], 'Town')
                self.assertEqual(issues, [])
            finally:
                cache.close()

    def test_collection_replays_cached_requests_without_network(self):
        cache = collect.Cache(metadata.ROOT/'cache/wikidata.sqlite3')
        try:
            with patch.object(cache, 'request', side_effect=AssertionError('Unexpected network')), redirect_stdout(io.StringIO()):
                collect.collect(cache, read_rows(metadata.ROOT/'metadata/cohort.jsonl'))
            self.assertEqual(cache.requests, 0)
        finally:
            cache.close()

    def test_fresh_collection_populates_all_required_evidence(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            cache = collect.Cache(Path(tmp)/'cache.sqlite3', allow_network=True, user_agent='Test project')
            try:
                def request(endpoint, params):
                    if endpoint == collect.API:
                        if params['action'] == 'wbgetentities':
                            entities = {'Q1': entity('Riverton', P31=['Q515']), 'Q515': entity('city')}
                            return {'entities': {q: entities[q] for q in params['ids'].split('|')}}
                        return {'search': [{'id':'Q1', 'label':'Riverton', 'description':'city in Country'}]}
                    query = params['query']
                    if '?birthDate' in query:
                        rows = [binding(person='Q9', birthDate='1900-01-01T00:00:00Z', birthPrecision='11', birthPlace='Q1')]
                    elif '?adminDescription' in query:
                        rows = [binding(place='Q1',placeLabel='Riverton', admin='Q1',adminLabel='Riverton',adminDescription='city in Country')]
                    else:
                        rows = [binding(node='Q1', country='Q2',countryLabel='Country', continent='Q3',continentLabel='Continent')]
                    return {'results': {'bindings': rows}}
                with patch.object(cache, 'request', side_effect=request):
                    collect.collect(cache, [{'qid':'Q9','name':'Test','wiki_title':'Test','split':'test'}])
                self.assertTrue(cache.has('person', 'Q9'))
                self.assertTrue(cache.has('hierarchy', 'Q1'))
                self.assertTrue(cache.has('entity', 'Q515'))
                self.assertTrue(cache.has('search', 'Riverton'))
                with patch.object(cache, 'request', side_effect=AssertionError('Unexpected repeat request')):
                    collect.collect(cache, [{'qid':'Q9','name':'Test','wiki_title':'Test','split':'test'}])
            finally:
                cache.close()

    def test_fresh_wikipedia_title_resolution_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = collect.Cache(Path(tmp)/'cache.sqlite3', allow_network=True, user_agent='Test project')
            try:
                html = '<div><a href="/wiki/Outside">Outside</a></div><div id="mw-content-text"><a href="/wiki/Test_Person">Test Person</a><a href="/wiki/Help:Ignored">Ignore</a></div>'
                response = {'query': {'pages': {'1': {'title':'Test Person','pageprops': {'wikibase_item':'Q9'}}}}}
                with patch.object(cache, 'request', side_effect=[html, response]):
                    people = collect.discover(cache)
                self.assertEqual(people, [{'qid':'Q9','name':'Test Person','wiki_title':'Test_Person'}])
                with patch.object(cache, 'request', side_effect=AssertionError('Unexpected repeat request')):
                    self.assertEqual(collect.discover(cache), people)
            finally:
                cache.close()

    def test_cached_wikipedia_page_and_title_resolution(self):
        cache = collect.Cache(metadata.ROOT/'cache/wikidata.sqlite3')
        try:
            with patch.object(cache, 'request', side_effect=AssertionError('Unexpected network')):
                people = collect.discover(cache)
            self.assertEqual(len(people), 1929)
            self.assertLessEqual({r['qid'] for r in read_rows(metadata.ROOT/'metadata/cohort.jsonl')}, {r['qid'] for r in people})
        finally:
            cache.close()

    def test_subcity_moves_to_own_field_without_changing_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = collect.Cache(Path(tmp)/'cache.sqlite3')
            try:
                row, issues = metadata.build_location(cache, 'Q11299', location_fixture(cache))
                self.assertEqual(row['subcity'], ['Manhattan'])
                self.assertEqual(row['city'], 'New York City')
                self.assertEqual(row['region'], ['New York'])
                self.assertEqual(issues, [])
            finally:
                cache.close()

    def test_subcity_uncertain_containment_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = collect.Cache(Path(tmp)/'cache.sqlite3')
            try:
                entities = location_fixture(cache)
                entities['Q11299'] = entity('Manhattan', P31=['Q408804'], P131=['Q60','Q2'])
                entities['Q2'] = entity('Other city', P31=['Q515'])
                row, issues = metadata.build_location(cache, 'Q11299', entities)
                self.assertIsNone(row['subcity'])
                self.assertTrue(any('no unique' in i['reason'] for i in issues))
            finally:
                cache.close()

    def test_conflicting_containing_cities_are_not_guessed(self):
        entities = {'Q1': entity('District', P31=['Q408804'], P131=['Q2','Q3']),
                    'Q2': entity('City A', P31=['Q515']), 'Q3': entity('City B', P31=['Q515'])}
        self.assertIsNone(metadata.nearest_city('Q1', entities))

    def test_sibling_districts_are_not_aliases(self):
        entities = {'Q1': entity('District', P31=['Q2983893'], P131=['Q2','Q3']),
                    'Q2': entity('A', P31=['Q2983893']), 'Q3': entity('B', P31=['Q2983893'])}
        self.assertEqual(metadata.subcity_ancestors('Q1', entities), (['Q1'], {'Q1','Q2','Q3'}))

    def test_city_type_beats_indirect_neighborhood_role(self):
        entities = {'Q1': entity('City', P31=['Q515','Q2']), 'Q2': entity('Role', P279=['Q123705'])}
        self.assertEqual(metadata.classification('Q1', entities), 'CITY')

    def test_missing_evidence_fails_and_cache_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = collect.Cache(Path(tmp)/'cache.sqlite3')
            try:
                with self.assertRaises(ValueError):
                    cache.ensure('person', ['Q1'])
                cache.put('person', 'Q1', [1])
                cache.put('person', 'Q1', [2])
                self.assertEqual(cache.get('person', 'Q1'), [1])
                with self.assertRaises(ValueError):
                    cache.search('Unknown city')
            finally:
                cache.close()

    def test_country_fallback_does_not_use_unrelated_cached_countries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = collect.Cache(Path(tmp)/'cache.sqlite3')
            try:
                cache.put('hierarchy', 'Q1', [binding(place='Q1', placeLabel='Town', admin='Q1', adminLabel='Town', adminDescription='city'),
                    binding(place='Q1', placeLabel='Town', admin='Q2', adminLabel='Province', adminDescription='province')])
                cache.put('node_metadata', 'Q1', [binding(node='Q1', country='Q3', countryLabel='Country')])
                cache.put('node_metadata', 'Q2', [binding(node='Q2', country='Q3', countryLabel='Country', continent='Q4', continentLabel='Continent')])
                cache.put('country_continent', 'Q3', [binding(country='Q3', continent='Q5', continentLabel='Unrelated fallback')])
                self.assertIsNone(metadata.hierarchy(cache,'Q1')[0]['continent'])
            finally:
                cache.close()

    def test_filters_blank_node_ids_and_low_date_precision(self):
        p = metadata.person_properties([{'birthPlace': {'type': 'bnode', 'value': 'abc'}}])
        self.assertIsNone(p['birth_location_qid'])
        self.assertFalse(metadata.eligible(p))
        p = metadata.person_properties([binding(birthDate='1900-01-01T00:00:00Z', birthPrecision='6', birthPlace='Q1')])
        self.assertFalse(metadata.eligible(p))
        p['birth_precision'] = 7
        self.assertTrue(metadata.eligible(p))

    def test_direct_build_needs_no_existing_gold_and_protects_outputs(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            cache = collect.Cache(tmp/'cache.sqlite3')
            location_fixture(cache)
            cache.put('person','Q9',[binding(person='Q9', birthDate='1900-01-01T00:00:00Z', birthPrecision='11', birthPlace='Q11299')])
            cache.close()
            write(tmp/'cohort.jsonl', [{'qid':'Q9','name':'Test','wiki_title':'Test'}], rows=True)
            rows = metadata.prepare(tmp/'cohort.jsonl', tmp/'cache.sqlite3', tmp/'gold')
            self.assertEqual(rows[0]['birth_city'], 'New York City')
            self.assertEqual(rows[0]['birth_date'], '1900-01-01T00:00:00Z')
            write(tmp/'gold/gold_initial.jsonl', [{'wrong':True}], rows=True)
            with self.assertRaises(ValueError):
                metadata.prepare(tmp/'cohort.jsonl', tmp/'cache.sqlite3', tmp/'gold')
            self.assertEqual(read_rows(tmp/'gold/gold_initial.jsonl'), [{'wrong':True}])

    def test_fixed_people_are_retained_even_when_fresh_metadata_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            cache = collect.Cache(tmp/'cache.sqlite3')
            cache.put('person', 'Q9', [])
            cache.close()
            write(tmp/'cohort.jsonl', [{'qid':'Q9','name':'Test','wiki_title':'Test'}], rows=True)
            rows = metadata.prepare(tmp/'cohort.jsonl', tmp/'cache.sqlite3', tmp/'gold')
            self.assertEqual([r['qid'] for r in rows], ['Q9'])
            self.assertIsNone(rows[0]['birth_date'])
            self.assertFalse((tmp/'gold/test.jsonl').exists())

    def test_wikipedia_to_gold_collection_matches_the_original_people(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            cache = collect.Cache(metadata.ROOT/'cache/wikidata.sqlite3')
            try:
                with patch.object(cache, 'request', side_effect=AssertionError('Unexpected network')):
                    candidates = collect.discover(cache)
                    collect.collect(cache, candidates, filter_eligible=True)
            finally:
                cache.close()
            write(tmp/'candidates.jsonl', candidates, rows=True)
            rows = metadata.prepare(tmp/'candidates.jsonl', metadata.ROOT/'cache/wikidata.sqlite3', tmp/'gold', filter_eligible=True)
            cohort = read_rows(metadata.ROOT/'metadata/cohort.jsonl')
            self.assertEqual([(r['qid'],r['name'],r['wiki_title']) for r in rows],
                             [(r['qid'],r['name'],r['wiki_title']) for r in cohort])
            self.assertFalse((tmp/'gold/test.jsonl').exists())

    def test_input_cohort_has_no_preassigned_splits(self):
        for p in read_rows(metadata.ROOT/'metadata/cohort.jsonl'):
            self.assertEqual(set(p), {'qid','name','wiki_title'})

    def test_executable_split_is_disjoint_complete_and_seeded_within_process(self):
        people = [{'qid': f'Q{i}', 'name':str(i), 'wiki_title':str(i), 'birth_date':'1900',
                   'birth_precision':11 if i%2 else 9} for i in range(1,41)]
        selected = split.test_people(people, fraction=.2, trials=25, seed=42)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected, split.test_people(people, fraction=.2, trials=25, seed=42))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            write(tmp/'persons.jsonl', people, rows=True)
            outputs = split.prepare(tmp/'persons.jsonl', tmp/'split', fraction=.2, trials=25, seed=42)
            self.assertEqual(set(outputs), {'development.jsonl', 'test.jsonl'})
            self.assertFalse((tmp/'split/cohort.jsonl').exists())
            self.assertEqual(len(outputs['development.jsonl']),32)
            self.assertFalse({p['qid'] for p in outputs['development.jsonl']} & selected)
            self.assertEqual({p['qid'] for p in outputs['test.jsonl']}, selected)
            write(tmp/'split/test.jsonl', [{'old':True}], rows=True)
            with self.assertRaises(ValueError):
                split.prepare(tmp/'persons.jsonl', tmp/'split', fraction=.2, trials=25)
            self.assertEqual(read_rows(tmp/'split/test.jsonl'), [{'old':True}])

    def test_split_strata_use_gold_availability_precision_and_ambiguity(self):
        person = {'birth_date':'1900','birth_precision':11,'birth_city':'City', 'birth_needs_disambiguation':True}
        self.assertEqual(split.strata(person), ['birth|precision=11','birthplace|needs_disambiguation=true',
                                               'death|not_present','deathplace|not_present'])

    def test_preferred_and_deprecated_statements(self):
        value = entity(P131=['Q1','Q2','Q3'])
        value['claims']['P131'][0]['rank'] = 'deprecated'
        value['claims']['P131'][1]['rank'] = 'preferred'
        self.assertEqual(metadata.ids(value, 'P131'), ['Q2'])

    def test_all_original_pools_cover_exactly_the_same_people(self):
        people = read_rows(metadata.ROOT/'metadata/gold_adapted.jsonl')
        for model in generate.MODELS:
            for relation in generate.PROMPTS:
                rows = read_rows(generate.ROOT/'samples'/model/f'{relation}_samples.jsonl')
                self.assertEqual(generate.validate_pool(rows, people, relation, 20), {r['qid'] for r in people})

    def test_pool_validation_rejects_missing_samples_and_duplicates(self):
        people = [{'qid': 'Q1', 'name': 'Person'}]
        row = {'qid': 'Q1', 'samples': ['Text']*20}
        self.assertEqual(generate.validate_pool([row], people, 'birth', 20), {'Q1'})
        with self.assertRaises(ValueError):
            generate.validate_pool([row, row], people, 'birth', 20)
        with self.assertRaises(ValueError):
            generate.validate_pool([{'qid': 'Q1', 'samples': ['Text']*19}], people, 'birth', 20)
        with self.assertRaises(ValueError):
            generate.validate_pool([{**row, 'question': 'Wrong question'}], people, 'birth', 20)

    def test_seed_is_stable_and_question_specific(self):
        self.assertEqual(generate.person_seed(42, 'birth', 'Q1'), generate.person_seed(42, 'birth', 'Q1'))
        self.assertNotEqual(generate.person_seed(42, 'birth', 'Q1'), generate.person_seed(42, 'death', 'Q1'))

    def test_decode_excludes_prompt_eos_and_padding_but_keeps_truncation(self):
        class Tokenizer:
            def decode(self, ids, **kwargs):
                return ' '.join(str(i) for i in ids)
        texts, reasons = generate.decode_samples(Tokenizer(), [[9,9,1,0,0], [9,9,1,2,3]], 2, {0}, 3)
        self.assertEqual(texts, ['1', '1 2 3'])
        self.assertEqual(reasons, ['stop', 'length'])
        with self.assertRaises(ValueError):
            generate.decode_samples(Tokenizer(), [[9,9,0]], 2, {0}, 3)


if __name__ == '__main__':
    unittest.main()
