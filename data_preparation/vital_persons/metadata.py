"""Construct gold metadata directly from cached Wikipedia/Wikidata responses."""
import argparse
from collections import deque
import json
from pathlib import Path
import re

from shared.io import read_rows, write
from .collect import Cache, value, qid, LABEL_LANGUAGES

ROOT = Path(__file__).resolve().parent
QID = re.compile(r"Q[1-9][0-9]*\Z")
EVENTS = ("birth", "death")
FIELDS = ("place", "subcity", "city", "region", "country", "continent")
SUBCITY_TYPES = {"Q408804", "Q123705", "Q2983893", "Q702842", "Q211690", "Q4286337"}
CITY_TYPES = {"Q515", "Q3957", "Q532", "Q7930989"}
CITY_PATTERN = re.compile(r"\b(city|megacity|capital|town|metropolis)\b", re.I)


def ids(entity, prop):
    statements = [s for s in entity.get("claims", {}).get(prop, []) if s.get("rank") != "deprecated"]
    preferred = [s for s in statements if s.get("rank") == "preferred"]
    result = []
    for statement in preferred or statements:
        value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and QID.fullmatch(value.get("id", "")):
            result.append(value["id"])
    return list(dict.fromkeys(result))


def label(entity):
    for language in LABEL_LANGUAGES:
        value = entity.get("labels", {}).get(language, {}).get("value")
        if value and not QID.fullmatch(value):
            return value
    return None


def classification(qid, entities):
    direct = set(ids(entities.get(qid, {}), "P31"))
    branches = []
    for start in direct:
        queue, reached = deque([start]), set()
        while queue:
            q = queue.popleft()
            if q not in reached:
                reached.add(q)
                reached.add(entities.get(q, {}).get("id", q))
                queue.extend(ids(entities.get(q, {}), "P279"))
        branches.append(reached)
    # Wikidata role taxonomies overlap: college town -> city, but also
    # academic enclave -> neighborhood. That branch is not subcity evidence.
    if direct & SUBCITY_TYPES:
        return "SUBCITY"
    if any(b & CITY_TYPES for b in branches):
        return "CITY"
    if any(b & SUBCITY_TYPES for b in branches):
        return "SUBCITY"
    return None


def nearest_city(qid, entities):
    frontier, visited, cities = [qid], {qid}, set()
    for _ in range(8):
        parents = sorted({p for q in frontier for p in ids(entities.get(q, {}), "P131")} - visited)
        if not parents:
            break
        found = {p for p in parents if classification(p, entities) == "CITY" and label(entities.get(p, {}))}
        cities.update(found)
        visited.update(parents)
        frontier = [p for p in parents if p not in found]
    return next(iter(cities)) if len(cities) == 1 else None


def subcity_ancestors(qid, entities):
    """Keep a unique containment chain; do not treat sibling districts as aliases."""
    certified, current, seen = [qid], qid, {qid}
    for _ in range(8):
        parents = ids(entities.get(current, {}), "P131")
        if len(parents) != 1 or parents[0] in seen:
            break
        current = parents[0]
        seen.add(current)
        if classification(current, entities) != "SUBCITY":
            break
        certified.append(current)
    all_subcities, frontier, seen = {qid}, [qid], {qid}
    for _ in range(8):
        parents = {p for q in frontier for p in ids(entities.get(q, {}), "P131")} - seen
        all_subcities.update(p for p in parents if classification(p, entities) == "SUBCITY")
        seen.update(parents)
        frontier = [p for p in parents if classification(p, entities) != "CITY"]
        if not frontier:
            break
    return certified, all_subcities


def person_properties(bindings):
    first = bindings[0] if bindings else {}
    result = {}
    for event in EVENTS:
        result[event+'_date'] = value(first, event+'Date')
        precision = value(first, event+'Precision')
        result[event+'_precision'] = int(precision) if precision is not None else None
        result[event+'_location_qid'] = next((qid(row, event+'Place') for row in bindings if qid(row, event+'Place')), None)
    return result


def eligible(properties):
    return (any(properties[e+'_date'] and properties[e+'_location_qid'] for e in EVENTS)
            and all(not properties[e+'_date'] or (properties[e+'_precision'] or 0) >= 7 for e in EVENTS))


def raw_hierarchy(cache, location):
    nodes = {}
    for row in cache.get('hierarchy', location):
        identity = qid(row, 'admin')
        if identity and identity not in nodes:
            node = {'qid': identity, 'label': value(row, 'adminLabel'),
                    'description': value(row, 'adminDescription')}
            raw = cache.get('node_metadata', identity)
            metadata = raw[-1] if raw else {}
            node.update(country=value(metadata, 'countryLabel'), continent=value(metadata, 'continentLabel'))
            node['country_qid'] = qid(metadata, 'country')
            nodes[identity] = node
    return list(nodes.values())


def fallback_countries(cache, places):
    # Limit fallback to countries requested for this collection, not every
    # country that happens to be in the cache.
    countries = set()
    for place in places:
        nodes = raw_hierarchy(cache, place)
        if not any(n['continent'] for n in nodes):
            countries.update(n['country_qid'] for n in nodes if n['country_qid'])
    return countries


def hierarchy(cache, location):
    nodes = raw_hierarchy(cache, location)
    countries = getattr(cache, 'fallback_countries', None)
    if countries is None:
        countries = {n['country_qid'] for n in nodes if n['country_qid']} if not any(n['continent'] for n in nodes) else set()
    for node in nodes:
        country = node['country_qid']
        if node['continent'] is None and country in countries:
            fallback = cache.get('country_continent', country)
            node['continent'] = next((value(b, 'continentLabel') for b in fallback if qid(b, 'continent')), None)
    return nodes


def candidate_city(location, nodes):
    cities = {node['label'] for node in nodes if node['label'] and CITY_PATTERN.search(node['description'] or '')}
    head = next((node for node in nodes if node['qid'] == location), None)
    name = next(iter(cities)) if len(cities) == 1 else head['label'] if head and head['label'] in cities else None
    if not name:
        return None
    candidates = [node['qid'] for node in nodes if node['label'] == name]
    return {'label': name, 'qids': candidates, 'qid': candidates[0] if len(candidates) == 1 else None}


def place_fields(cache, location):
    if location is None:
        return None
    bindings = cache.get('hierarchy', location)
    nodes = hierarchy(cache, location)
    city = candidate_city(location, nodes)
    head = next((n for n in nodes if n['qid'] == location), None)
    if not city or not head:
        return None
    countries = {n['country'] for n in nodes if n['country']}
    continents = {n['continent'] for n in nodes if n['continent']}
    country = next(iter(countries)) if len(countries) == 1 else head['country']
    continent = next(iter(continents)) if len(continents) == 1 else head['continent']
    if not country or not continent:
        return None
    place = value(bindings[0], 'placeLabel') if bindings else None
    excluded = {place, city['label'], country, continent}
    region = [n['label'] for n in nodes if n['label'] not in excluded]
    return {'place': None if place == city['label'] else place, 'city': city['label'],
            'region': region, 'country': country, 'continent': continent,
            'city_ids': city['qids'], 'nodes': nodes}


def ambiguity(cache, name):
    city_keywords = {'city', 'megacity', 'capital', 'town', 'metropolis'}
    count = 0
    for result in cache.search(name):
        description = result.get('description', '').lower().split(' in ', 1)[0]
        count += any(re.search(rf'\b{re.escape(word)}\b', description) for word in city_keywords)
    return count > 1


def region_ancestors(nodes, entities):
    """Use prediction normalization's exact inferred-region policy, offline."""
    from parsing.location.normalization.resolver import AuditableWikidataNormalizer
    bindings = []
    for node in nodes:
        text = node['label']
        if text and QID.fullmatch(text):
            text = label(entities.get(node['qid'], {}))
        row = {}
        if text:
            row['adminLabel'] = {'value': text}
        if node.get('description'):
            row['adminDescription'] = {'value': node['description']}
        bindings.append(row)
    allowed = set(AuditableWikidataNormalizer._metadata_from_bindings(bindings)['region'] or [])
    context = [{'adminDescription': row['adminDescription']} for row in bindings if 'adminDescription' in row]
    selected = set()
    for node, row in zip(nodes, bindings, strict=True):
        text = row.get('adminLabel', {}).get('value')
        if text not in allowed:
            continue
        # Preserve type context but isolate this label, so a historical homonym
        # cannot inherit another entity's accepted name and enter the gold IDs.
        own_regions = AuditableWikidataNormalizer._metadata_from_bindings([row, *context])['region'] or []
        if text in own_regions:
            selected.add(node['qid'])
    return selected


def build_location(cache, location, entities):
    result = {field: None for field in FIELDS}
    result.update(location_qid=location, entity_ids={field: [] for field in FIELDS if field != 'subcity'},
                  needs_disambiguation=False)
    raw = place_fields(cache, location)
    issues = []
    if raw is None:
        return result, issues
    result['needs_disambiguation'] = ambiguity(cache, raw['city'])
    nodes = raw['nodes']
    allowed_regions = region_ancestors(nodes, entities)
    # Filter inferred ancestors, not the factual event place or its city.
    # Keep source order and align readable labels with their entity IDs.
    region_names = {n['label'] for n in nodes if n['qid'] in allowed_regions}
    raw = {**raw, 'region': [name for name in raw['region'] or [] if name in region_names]}
    for field in ('place', 'city', 'region', 'country', 'continent'):
        original = raw[field]
        labels = original if isinstance(original, list) else [original] if original is not None else []
        result['entity_ids'][field] = list(dict.fromkeys(n['qid'] for text in labels for n in nodes if n['label'] == text))
        names = []
        for text in labels:
            if QID.fullmatch(text):
                resolved = label(entities.get(text, {}))
                if not resolved and field == 'country':
                    parents = ids(entities.get(text, {}), 'P17')
                    if len(parents) == 1:
                        resolved = label(entities.get(parents[0], {}))
                        result['entity_ids'][field] = parents
                if resolved:
                    names.append(resolved)
                else:
                    issues.append({'field': field, 'entity_qid': text,
                                   'reason': 'Missing cached label in fallback languages (en, mul, fr, es, de, pt)'})
            else:
                names.append(text)
        names = list(dict.fromkeys(names)) if any(QID.fullmatch(t) for t in labels) else names
        result[field] = names if isinstance(original, list) else (names[0] if names else None)
    result['entity_ids']['region'] = [q for q in result['entity_ids']['region'] if q in allowed_regions]
    if result['city'] and len(raw['city_ids']) == 1:
        source = raw['city_ids'][0]
        if classification(source, entities) == 'SUBCITY':
            city = nearest_city(source, entities)
            if city:
                chain, all_subcities = subcity_ancestors(source, entities)
                result['subcity'] = list(dict.fromkeys(label(entities[q]) for q in chain if label(entities[q]))) or None
                result['city'] = label(entities[city])
                result['entity_ids']['subcity'], result['entity_ids']['city'] = chain, [city]
                removed = {result['city'], *(label(entities.get(q, {})) for q in all_subcities)}
                result['region'] = [name for name in result['region'] or [] if name not in removed] or None
                result['entity_ids']['region'] = [q for q in result['entity_ids']['region'] if q not in {*all_subcities, city}]
                if all_subcities-set(chain):
                    issues.append({'field': 'subcity', 'entity_qid': source, 'reason': 'Ambiguous district chain; sibling districts omitted'})
            else:
                issues.append({'field': 'city', 'entity_qid': source, 'reason': 'Subcity has no unique verified containing city'})
    return result, issues


def prepare(cohort_path, cache_path, output_dir, *, filter_eligible=False, overwrite=False):
    people = read_rows(cohort_path)
    if not people or len({p['qid'] for p in people}) != len(people):
        raise ValueError('Invalid fixed cohort')
    for p in people:
        if set(p) != {'qid', 'name', 'wiki_title'}:
            raise ValueError('Cohort must contain only qid, name and wiki_title; run split separately')
        if not QID.fullmatch(p['qid']):
            raise ValueError(f'Invalid cohort identity: {p}')
    cache = Cache(cache_path)
    try:
        properties = {p['qid']: person_properties(cache.get('person', p['qid'])) for p in people}
        if filter_eligible:
            people = [p for p in people if eligible(properties[p['qid']])]
            properties = {p['qid']: properties[p['qid']] for p in people}
        places = {p[e+'_location_qid'] for p in properties.values() for e in EVENTS if p[e+'_location_qid']}
        cache.fallback_countries = fallback_countries(cache, places)
        entities = {key: json.loads(payload) for key, payload in cache.db.execute("SELECT key,payload FROM evidence WHERE kind='entity'")}
        for key, payload in cache.db.execute("SELECT key,payload FROM evidence WHERE kind='entity_labels'"):
            evidence = json.loads(payload)
            original = entities.setdefault(key, {})
            # Supplement missing labels only; keep original labels and all claims.
            labels = original.setdefault('labels', {})
            for language, entry in evidence.get('labels', {}).items():
                if not label({'labels': {language: labels.get(language, {})}}):
                    labels[language] = entry
        rows, issues = [], []
        for person in people:
            person_values = properties[person['qid']]
            row = {key: person[key] for key in ('qid', 'name', 'wiki_title')}
            row_issues = []
            for event in EVENTS:
                for field in ('date', 'precision'):
                    row[event+'_'+field] = person_values[event+'_'+field]
                location, flags = build_location(cache, person_values[event+'_location_qid'], entities)
                row.update({event+'_'+key: v for key, v in location.items()})
                row_issues.extend({'qid': person['qid'], 'event': event, **flag} for flag in flags)
            if not any(row[e+'_date'] and row[e+'_continent'] for e in EVENTS):
                if filter_eligible:
                    continue
                row_issues.append({'qid': person['qid'], 'reason': 'No complete dated geographic event; fixed cohort membership retained'})
            rows.append(row)
            issues.extend(row_issues)
        if not rows:
            raise ValueError('No eligible people remain; no outputs written')
        outputs = {'gold_initial.jsonl': rows}
        for name, payload in outputs.items():
            path = output_dir/name
            if path.exists() and not overwrite and read_rows(path) != payload:
                raise ValueError(f'{path} differs from the rebuild; choose a new --output-dir or explicitly --overwrite')
        for name, payload in outputs.items():
            write(output_dir/name, payload, rows=True)
        write(output_dir/'unresolved.json', issues)
        print(json.dumps({'people': len(rows), 'unresolved': len(issues)}, indent=2))
        return rows
    finally:
        cache.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=ROOT/'metadata/cohort.jsonl')
    parser.add_argument('--cache', type=Path, default=ROOT/'cache/wikidata.sqlite3')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'metadata')
    parser.add_argument('--filter-eligible', action='store_true', help='Filter a discovered list by date precision and complete event metadata')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    prepare(args.cohort, args.cache, args.output_dir, filter_eligible=args.filter_eligible, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
