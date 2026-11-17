"""Collect Wikipedia identities and Wikidata evidence using a reusable request cache."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import re
from html.parser import HTMLParser
from urllib.parse import urlencode, unquote
from urllib.request import Request, urlopen

from shared.io import read_rows, write

ROOT = Path(__file__).resolve().parent
SPARQL = 'https://query.wikidata.org/sparql'
API = 'https://www.wikidata.org/w/api.php'
WIKIPEDIA_API = 'https://en.wikipedia.org/w/api.php'
PEOPLE_URL = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level_4/People'
EVENTS = ('birth', 'death')
SUFFIXES = {'city', 'megacity', 'capital', 'town', 'village', 'metropolis', 'region', 'state',
            'province', 'county', 'district', 'prefecture', 'continent', 'island', 'suburb', 'neighborhood'}
QUERIES = {
    'person': '''SELECT ?person ?birthDate ?birthPrecision ?deathDate ?deathPrecision ?birthPlace ?birthPlaceLabel ?deathPlace ?deathPlaceLabel WHERE {
      VALUES ?person { VALUES_LIST }
      OPTIONAL { ?person p:P569/psv:P569 ?birthValue . ?birthValue wikibase:timeValue ?birthDate ; wikibase:timePrecision ?birthPrecision . }
      OPTIONAL { ?person p:P570/psv:P570 ?deathValue . ?deathValue wikibase:timeValue ?deathDate ; wikibase:timePrecision ?deathPrecision . }
      OPTIONAL { ?person wdt:P19 ?birthPlace . } OPTIONAL { ?person wdt:P20 ?deathPlace . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }''',
    'hierarchy': '''SELECT ?place ?placeLabel ?admin ?adminLabel ?adminDescription WHERE {
      VALUES ?place { VALUES_LIST } ?place wdt:P131* ?admin .
      OPTIONAL { ?admin schema:description ?adminDescription . FILTER(LANG(?adminDescription) = "en") }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }''',
    'node_metadata': '''SELECT ?node ?country ?countryLabel ?continent ?continentLabel WHERE {
      VALUES ?node { VALUES_LIST } OPTIONAL { ?node wdt:P17 ?country . } OPTIONAL { ?node wdt:P30 ?continent . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }''',
    'country_continent': '''SELECT ?country ?continent ?continentLabel WHERE {
      VALUES ?country { VALUES_LIST } OPTIONAL { ?country wdt:P30 ?continent . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }''',
}
IDENTITY = {'person': 'person', 'hierarchy': 'place', 'node_metadata': 'node', 'country_continent': 'country'}
LABEL_LANGUAGES = ('en', 'mul', 'fr', 'es', 'de', 'pt')


def value(binding, key):
    return binding.get(key, {}).get('value')


def qid(binding, key):
    text = value(binding, key)
    identity = text.rsplit('/', 1)[-1] if text and text.startswith('http://www.wikidata.org/entity/') else ''
    return identity if re.fullmatch(r'Q[1-9][0-9]*', identity) else None


def search_query(name):
    words = name.split()
    while words and words[-1].lower() in SUFFIXES:
        words.pop()
    return ' '.join(words)


class Cache:
    """One logical response per entity/query; batching order does not change keys."""
    def __init__(self, path, *, allow_network=False, user_agent=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS evidence (kind TEXT, key TEXT, payload TEXT NOT NULL, collected_at TEXT NOT NULL, PRIMARY KEY(kind,key))')
        self.allow_network, self.user_agent, self.requests = allow_network, user_agent, 0

    def has(self, kind, key):
        return self.db.execute('SELECT 1 FROM evidence WHERE kind=? AND key=?', (kind, key)).fetchone() is not None

    def get(self, kind, key):
        record = self.db.execute('SELECT payload FROM evidence WHERE kind=? AND key=?', (kind, key)).fetchone()
        if record is None:
            raise ValueError(f'Missing cached {kind}: {key}; run collect with --allow-network to fetch missing evidence')
        return json.loads(record[0])

    def put(self, kind, key, payload, collected_at=None):
        self.db.execute('INSERT OR IGNORE INTO evidence VALUES (?,?,?,?)',
            (kind, key, json.dumps(payload, ensure_ascii=False), collected_at or datetime.now(timezone.utc).isoformat()))
        self.db.commit()

    def request(self, endpoint, params=None, *, text=False):
        if not self.allow_network:
            raise ValueError('Evidence missing in offline cache; explicitly enable --allow-network to collect it')
        if not self.user_agent:
            raise ValueError('Online collection requires --user-agent with project/contact information')
        import time
        from urllib.error import HTTPError, URLError
        request = Request(endpoint+('?' + urlencode(params) if params else ''),
                          headers={'User-Agent': self.user_agent, 'Accept': 'text/html' if text else 'application/json'})
        for attempt in range(6):
            try:
                with urlopen(request, timeout=60) as response:
                    result = response.read().decode('utf-8') if text else json.load(response)
                if not text and 'error' in result:
                    raise ValueError(f'Wikidata error: {result["error"]}')
                self.requests += 1
                time.sleep(.2)
                return result
            except (HTTPError, URLError, TimeoutError) as error:
                if isinstance(error, HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}:
                    raise
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 30))

    def ensure(self, kind, keys):
        keys = set(keys)
        missing = sorted(set(keys)-{key for key in keys if self.has(kind, key)})
        if missing and not self.allow_network:
            raise ValueError(f'Missing cached {kind}: {", ".join(missing[:10])} ({len(missing)} total); run collect with --allow-network')
        for start in range(0, len(missing), 10 if kind == 'hierarchy' else 50):
            batch = missing[start:start+(10 if kind == 'hierarchy' else 50)]
            if kind in {'entity', 'entity_labels'}:
                result = self.request(API, {'action': 'wbgetentities', 'ids': '|'.join(batch),
                    'props': 'labels' if kind == 'entity_labels' else 'labels|claims',
                    'languages': '|'.join(LABEL_LANGUAGES), 'languagefallback': 1,
                    'format': 'json'})['entities']
                for key in batch:
                    if key not in result:
                        raise ValueError(f'Missing returned entity: {key}')
                    self.put(kind, key, result[key])
            else:
                query = QUERIES[kind].replace('VALUES_LIST', ' '.join('wd:'+q for q in batch))
                bindings = self.request(SPARQL, {'query': query, 'format': 'json'})['results']['bindings']
                groups = {key: [] for key in batch}
                for row in bindings:
                    key = qid(row, IDENTITY[kind])
                    if key not in groups:
                        raise ValueError('Unexpected query response identity')
                    groups[key].append(row)
                for key, group in groups.items():
                    self.put(kind, key, group)

    def search(self, name):
        query = search_query(name)
        if not query:
            return []
        if not self.has('search', query):
            if not self.allow_network:
                raise ValueError(f'Missing cached city search: {query}')
            result = self.request(API, {'action': 'wbsearchentities', 'search': query,
                'language': 'en', 'format': 'json', 'limit': 5})
            self.put('search', query, result['search'])
        return self.get('search', query)

    def close(self):
        self.db.close()


class PeoplePage(HTMLParser):
    """Extract /wiki/ links inside #mw-content-text without a browser."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth, self.inside, self.current, self.names, self.people = 0, None, None, [], {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'div':
            self.depth += 1
            if attrs.get('id') == 'mw-content-text':
                self.inside = self.depth
        if self.inside is not None and tag == 'a':
            href = attrs.get('href', '')
            if href.startswith('/wiki/') and ':' not in href:
                self.current, self.names = unquote(href[len('/wiki/'):]), []

    def handle_data(self, data):
        if self.current is not None:
            self.names.append(data)

    def handle_endtag(self, tag):
        if tag == 'a' and self.current is not None:
            self.people[self.current] = {'name': ''.join(self.names).strip(), 'wiki_title': self.current}
            self.current = None
        if tag == 'div':
            if self.inside == self.depth:
                self.inside = None
            self.depth -= 1


def discover(cache):
    """Collect the person list and resolve Wikipedia titles to Wikidata QIDs."""
    if not cache.has('wikipedia', PEOPLE_URL):
        cache.put('wikipedia', PEOPLE_URL, cache.request(PEOPLE_URL, text=True))
    page = PeoplePage()
    page.feed(cache.get('wikipedia', PEOPLE_URL))
    people = list(page.people.values())
    if not people:
        raise ValueError('No people found in the cached Wikipedia page')
    missing = [p['wiki_title'].replace('_', ' ') for p in people
               if not cache.has('wikipedia_title', p['wiki_title'].replace('_', ' '))]
    for start in range(0, len(missing), 50):
        batch = missing[start:start+50]
        response = cache.request(WIKIPEDIA_API, {'action': 'query', 'prop': 'pageprops',
                                 'titles': '|'.join(batch), 'format': 'json'})
        pages = response['query']['pages'].values()
        resolved = {p['title']: p.get('pageprops', {}).get('wikibase_item') for p in pages}
        normalized = {p['from']: p['to'] for p in response['query'].get('normalized', [])}
        for title in batch:
            cache.put('wikipedia_title', title, resolved.get(normalized.get(title, title)))
    result, seen = [], set()
    for person in people:
        identity = cache.get('wikipedia_title', person['wiki_title'].replace('_', ' '))
        if identity and identity not in seen:
            result.append({'qid': identity, **person})
            seen.add(identity)
    return result


def collect(cache, people, *, filter_eligible=False):
    from .metadata import person_properties, eligible, place_fields, fallback_countries, ids
    cache.ensure('person', [r['qid'] for r in people])
    properties = {r['qid']: person_properties(cache.get('person', r['qid'])) for r in people}
    if filter_eligible:
        properties = {q: p for q, p in properties.items() if eligible(p)}
    places = {v[event+'_location_qid'] for v in properties.values() for event in EVENTS if v[event+'_location_qid']}
    cache.ensure('hierarchy', places)
    nodes = {qid(b, 'admin') for place in places for b in cache.get('hierarchy', place) if qid(b, 'admin')}
    cache.ensure('node_metadata', nodes)
    countries = fallback_countries(cache, places)
    cache.fallback_countries = countries
    cache.ensure('country_continent', countries)
    # Collect containment and type evidence for the subcity-aware gold builder.
    fields = [raw for place in sorted(places) if (raw := place_fields(cache, place)) is not None]
    seeds = {q for raw in fields for q in raw['city_ids']}
    seeds.update(text for raw in fields for field in ('place', 'city', 'region', 'country', 'continent')
                 for text in (raw[field] if isinstance(raw[field], list) else [raw[field]])
                 if isinstance(text, str) and text.startswith('Q') and text[1:].isdigit())
    frontier, visited = set(seeds), set(seeds)
    cache.ensure('entity', frontier)
    for _ in range(8):
        parents = {p for q in frontier for prop in ('P131', 'P17') for p in ids(cache.get('entity', q), prop)}-visited
        cache.ensure('entity', parents)
        visited.update(parents)
        frontier = parents
    frontier = {t for q in visited for t in ids(cache.get('entity', q), 'P31')}
    cache.ensure('entity', frontier)
    visited = set(frontier)
    for _ in range(6):
        parents = {p for q in frontier for p in ids(cache.get('entity', q), 'P279')}-visited
        cache.ensure('entity', parents)
        visited.update(parents)
        frontier = parents
    for raw in fields:
        cache.search(raw['city'])
    print(f'{len(people)} people; {len(places)} locations; {cache.requests} network requests')


def refresh_missing_labels(cache, issues_path):
    """Append label-only evidence; never overwrite archived entity/claim records."""
    issues = json.loads(issues_path.read_text())
    identities = {r['entity_qid'] for r in issues if r.get('reason', '').startswith('Missing cached')}
    if any(not re.fullmatch(r'Q[1-9][0-9]*', q) for q in identities):
        raise ValueError('Invalid entity identity in label issues')
    cache.ensure('entity_labels', identities)
    print(f'{len(identities)} entities; {cache.requests} label-only network requests')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=ROOT/'metadata/cohort.jsonl')
    parser.add_argument('--discover', type=Path, metavar='OUTPUT_JSONL',
                        help='Collect the Wikipedia candidate list instead of using the fixed cohort')
    parser.add_argument('--cache', type=Path, default=ROOT/'cache/wikidata.sqlite3')
    parser.add_argument('--allow-network', action='store_true')
    parser.add_argument('--user-agent')
    parser.add_argument('--refresh-labels-from', type=Path, metavar='UNRESOLVED_JSON',
                        help='Only collect multilingual label evidence for missing-label review flags')
    args = parser.parse_args()
    cache = Cache(args.cache, allow_network=args.allow_network, user_agent=args.user_agent)
    try:
        if args.refresh_labels_from:
            if args.discover:
                parser.error('--refresh-labels-from cannot be combined with --discover')
            refresh_missing_labels(cache, args.refresh_labels_from)
            return
        people = discover(cache) if args.discover else read_rows(args.cohort)
        collect(cache, people, filter_eligible=bool(args.discover))
        if args.discover:
            write(args.discover, people, rows=True)
    finally:
        cache.close()


if __name__ == '__main__':
    main()
