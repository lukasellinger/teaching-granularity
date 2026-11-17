"""SQLite persistence for raw requests, lookups and answer-aware resolutions."""
from __future__ import annotations
from collections import Counter
from enum import Enum
import hashlib
import json
import os
import re
from pathlib import Path
import sqlite3
import threading
import tempfile
from typing import Callable, TypeVar
from .resolver import AuditableWikidataNormalizer, ComponentResolution, canonical_component
from .versions import lookup_cache_identity, profile_cache_identity
from .values import LocationLevel
T = TypeVar("T")

def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))

def _resolution_cache_key(identity: dict) -> str:
    """Match the legacy resolution-key encoding exactly.

    The original file cache used json.dumps' default ensure_ascii=True. Keep
    that byte representation so Unicode identities retain their existing keys.
    """
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()

def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value

class SQLiteNormalizationCache:
    """Thread-safe persistent cache backed by one SQLite database in WAL mode."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections = []
        self._connections_lock = threading.Lock()
        self._closed = False
        connection = sqlite3.connect(self.path, timeout=60, isolation_level=None,
                                     check_same_thread=False)
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS primitive_cache (
                normalizer TEXT NOT NULL,
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                request_json TEXT NOT NULL,
                value_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (normalizer, namespace, cache_key)
            );
            CREATE TABLE IF NOT EXISTS resolution_cache (
                cache_key TEXT PRIMARY KEY,
                identity_json TEXT NOT NULL,
                resolution_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS primitive_cache_namespace
                ON primitive_cache(normalizer, namespace);
            """
        )
        connection.close()

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60, isolation_level=None,
                                     check_same_thread=False)
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA synchronous=NORMAL")
        with self._connections_lock:
            self._connections.append(connection)
        return connection

    def _connection(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("Cache is closed")
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._new_connection()
            self._local.connection = connection
        return connection

    def close(self):
        """Close after workers have completed; do not race active requests."""
        with self._connections_lock:
            self._closed = True
            for connection in self._connections:
                connection.close()
            self._connections.clear()

    def load_primitive(self, normalizer: str, namespace: str, cache_key: str,
                       request: object) -> tuple[bool, object | None]:
        row = self._connection().execute(
            """SELECT request_json, value_json FROM primitive_cache
               WHERE normalizer = ? AND namespace = ? AND cache_key = ?""",
            (normalizer, namespace, cache_key),
        ).fetchone()
        if row is None:
            return False, None
        request_json = _canonical_json(request)
        if row[0] != request_json:
            raise ValueError(
                f"primitive cache identity mismatch: {normalizer}/{namespace}/{cache_key}"
            )
        return True, json.loads(row[1])

    def store_primitive(self, normalizer: str, namespace: str, cache_key: str,
                        request: object, value: object) -> bool:
        request_json = _canonical_json(request)
        value_json = _canonical_json(_jsonable(value))
        cursor = self._connection().execute(
            """INSERT OR IGNORE INTO primitive_cache
               (normalizer, namespace, cache_key, request_json, value_json)
               VALUES (?, ?, ?, ?, ?)""",
            (normalizer, namespace, cache_key, request_json, value_json),
        )
        found, _ = self.load_primitive(normalizer, namespace, cache_key, request)
        assert found
        return cursor.rowcount == 1

    def resolution_keys(self) -> set[str]:
        return {
            row[0] for row in self._connection().execute(
                "SELECT cache_key FROM resolution_cache"
            )
        }

    def load_resolution(self, key: str, identity: dict) -> ComponentResolution | None:
        row = self._connection().execute(
            """SELECT identity_json, resolution_json FROM resolution_cache
               WHERE cache_key = ?""",
            (key,),
        ).fetchone()
        if row is None:
            return None
        identity_json = _canonical_json(identity)
        if row[0] != identity_json:
            raise ValueError(f"normalization cache identity mismatch: {key}")
        return ComponentResolution.model_validate(json.loads(row[1]))

    def store_resolution(self, key: str, identity: dict,
                         resolution: ComponentResolution) -> bool:
        identity_json = _canonical_json(identity)
        resolution_json = _canonical_json(resolution.model_dump(mode="json"))
        cursor = self._connection().execute(
            """INSERT OR IGNORE INTO resolution_cache
               (cache_key, identity_json, resolution_json) VALUES (?, ?, ?)""",
            (key, identity_json, resolution_json),
        )
        cached = self.load_resolution(key, identity)
        assert cached is not None
        return cursor.rowcount == 1

    def counts(self) -> dict[str, int]:
        connection = self._connection()
        return {
            "primitive": connection.execute(
                "SELECT COUNT(*) FROM primitive_cache"
            ).fetchone()[0],
            "resolutions": connection.execute(
                "SELECT COUNT(*) FROM resolution_cache"
            ).fetchone()[0],
        }

def snapshot_cache(source, destination):
    """Create a standalone, consistent backup, even while the source uses WAL.

    Never overwrites an existing destination or modifies source records.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or source == destination:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".sqlite-snapshot-", dir=destination.parent)
    os.close(descriptor)
    try:
        reader = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=60)
        writer = sqlite3.connect(temporary)
        try:
            reader.backup(writer, pages=1024, sleep=0.1)
            if writer.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("Snapshot failed SQLite integrity check")
            writer.execute("PRAGMA journal_mode=DELETE")
        finally:
            writer.close()
            reader.close()
        # Same-filesystem, atomic publication without overwriting another file.
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


class PersistentLookupNormalizer(AuditableWikidataNormalizer):
    """The frozen normalizer with persistent SQLite request-level caching."""

    def __init__(self, cache: SQLiteNormalizationCache, normalizer_id: str,
                 *, lookup_id: str | None = None, refresh_evidence=False, offline=False):
        super().__init__()
        self.cache = cache
        self.normalizer_id = normalizer_id
        self.lookup_id = lookup_id or lookup_cache_identity()
        self.metadata_id = profile_cache_identity() if refresh_evidence else self.lookup_id
        self.offline = offline
        self._raw_coordinator = RawRequestCoordinator()
        self._inflight: dict[tuple[str, str], threading.Event] = {}
        self._inflight_lock = threading.Lock()
        self.cache_stats: Counter[str] = Counter()

    def _session(self):
        session = getattr(self._local, "raw_cached_session", None)
        if session is None:
            session = CachedWikidataSession(super()._session(), self.cache,
                                           coordinator=self._raw_coordinator,
                                           count=self._count_cache, offline=self.offline)
            self._local.raw_cached_session = session
        return session

    def _count_cache(self, outcome: str) -> None:
        with self._inflight_lock:
            self.cache_stats[outcome] += 1

    @staticmethod
    def _token(payload: object) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    def _memoized(self, namespace: str, payload: object, loader: Callable[[], T],
                  restore: Callable[[object], T]) -> T:
        token = self._token(payload)
        cache_key = hashlib.sha256(token.encode()).hexdigest()
        while True:
            found, cached = self.cache.load_primitive(
                self.lookup_id, namespace, cache_key, payload
            )
            if found:
                self._count_cache("hits")
                return restore(cached)
            inflight_key = (namespace, cache_key)
            with self._inflight_lock:
                event = self._inflight.get(inflight_key)
                owner = event is None
                if owner:
                    event = threading.Event()
                    self._inflight[inflight_key] = event
            assert event is not None
            if owner:
                break
            event.wait()

        try:
            self._count_cache("misses")
            value = loader()
            self.cache.store_primitive(
                self.lookup_id, namespace, cache_key, payload, value
            )
            return value
        finally:
            with self._inflight_lock:
                self._inflight.pop(inflight_key, None)
                event.set()

    @staticmethod
    def _restore_candidates(value: object) -> list[dict[str, object]]:
        rows = []
        for raw in value:
            row = dict(raw)
            row["level"] = LocationLevel(row["level"])
            rows.append(row)
        return rows

    def _search(self, component: str) -> list[dict[str, object]]:
        request = {"component": canonical_component(component)}
        return self._memoized(
            "search", request, lambda: super(PersistentLookupNormalizer, self)._search(component),
            self._restore_candidates,
        )

    def _context_search(self, component: str, context: str) -> list[dict[str, object]]:
        request = {"component": canonical_component(component),
                   "context": canonical_component(context)}
        return self._memoized(
            "context-search", request,
            lambda: super(PersistentLookupNormalizer, self)._context_search(component, context),
            self._restore_candidates,
        )

    def _search_place_candidates(self, component: str) -> list[dict[str, object]]:
        request = {"component": canonical_component(component)}
        return self._memoized(
            "place-search", request,
            lambda: super(PersistentLookupNormalizer, self)._search_place_candidates(component),
            self._restore_candidates,
        )

    def _fetch_entity_profile(self, qid: str) -> tuple[set[str], dict[str, object]]:
        """Fetch the two existing QID views in one semantics-preserving query."""
        from . import wikidata as base

        query = f"""
        SELECT DISTINCT ?kind ?ancestor ?admin ?adminLabel ?adminDescription
               ?country ?countryLabel ?countryDescription
               ?directCountryDescription ?continent ?continentLabel
               ?countryContinent ?countryContinentLabel
               ?directCountry ?directCountryLabel
               ?directCountryContinent ?directCountryContinentLabel
               ?directContinent ?directContinentLabel WHERE {{
          {{
            BIND("ancestor" AS ?kind)
            wd:{qid} (wdt:P131|wdt:P276|wdt:P706)* ?ancestor .
          }}
          UNION
          {{
            BIND("metadata" AS ?kind)
            wd:{qid} (wdt:P131|wdt:P276)* ?admin .
            OPTIONAL {{
              ?admin schema:description ?adminDescription .
              FILTER(LANG(?adminDescription) = "en")
            }}
            OPTIONAL {{
              ?admin p:P17 ?countryStatement .
              ?countryStatement a wikibase:BestRank .
              ?countryStatement ps:P17 ?country .
              OPTIONAL {{
                ?country schema:description ?countryDescription .
                FILTER(LANG(?countryDescription) = "en")
              }}
              FILTER NOT EXISTS {{ ?countryStatement pq:P582 ?countryEnd . }}
              OPTIONAL {{ ?country wdt:P30 ?countryContinent . }}
            }}
            OPTIONAL {{ ?admin wdt:P30 ?continent . }}
            OPTIONAL {{
              wd:{qid} p:P17 ?directCountryStatement .
              ?directCountryStatement a wikibase:BestRank .
              ?directCountryStatement ps:P17 ?directCountry .
              OPTIONAL {{
                ?directCountry schema:description ?directCountryDescription .
                FILTER(LANG(?directCountryDescription) = "en")
              }}
              FILTER NOT EXISTS {{ ?directCountryStatement pq:P582 ?directCountryEnd . }}
              OPTIONAL {{ ?directCountry wdt:P30 ?directCountryContinent . }}
            }}
            OPTIONAL {{ wd:{qid} wdt:P30 ?directContinent . }}
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        response = self._session().get(
            base.WIKIDATA_SPARQL_URL,
            params={"query": query, "format": "json"}, timeout=120,
        )
        response.raise_for_status()
        rows = response.json()["results"]["bindings"]
        ancestors = {
            row["ancestor"]["value"].rsplit("/", 1)[-1]
            for row in rows
            if "ancestor" in row
        }
        metadata_rows = [row for row in rows if "admin" in row]
        # SPARQL's label service can return an opaque QID. Fetch human labels
        # in cheap API batches only for those unresolved labels, retaining the
        # original SPARQL/API responses in the raw request cache.
        missing = set()
        for row in metadata_rows:
            for field in ("admin", "country", "directCountry", "continent", "countryContinent",
                          "directCountryContinent", "directContinent"):
                label = row.get(field + "Label", {}).get("value", "")
                if field in row and (not label or re.fullmatch(r"Q\d+", label)):
                    missing.add(row[field]["value"].rsplit("/", 1)[-1])
        labels = {}
        ordered = sorted(missing)
        for offset in range(0, len(ordered), 50):
            label_response = self._session().get(
                base.WIKIDATA_API_URL, params={"action": "wbgetentities", "ids": "|".join(ordered[offset:offset + 50]),
                    "props": "labels|descriptions", "languages": "en|mul|fr|es|de|pt", "languagefallback": 1,
                    "format": "json"}, timeout=60)
            label_response.raise_for_status()
            for entity_id, entity in label_response.json().get("entities", {}).items():
                names = entity.get("labels", {})
                name = next((names[lang]["value"] for lang in ("en", "mul", "fr", "es", "de", "pt")
                             if lang in names), None)
                if name:
                    labels[entity_id] = name
        for row in metadata_rows:
            for field in ("admin", "country", "directCountry", "continent", "countryContinent",
                          "directCountryContinent", "directContinent"):
                if field not in row:
                    continue
                entity_id = row[field]["value"].rsplit("/", 1)[-1]
                old = row.get(field + "Label", {}).get("value", "")
                if entity_id in labels and (not old or re.fullmatch(r"Q\d+", old)):
                    row[field + "Label"] = {"type": "literal", "value": labels[entity_id]}
        return ancestors, self._metadata_from_bindings(metadata_rows)

    def _entity_profile(self, qid: str) -> tuple[set[str], dict[str, object]]:
        """Single-flight a combined request and populate the legacy namespaces."""
        request = {"qid": qid}
        token = self._token(request)
        cache_key = hashlib.sha256(token.encode()).hexdigest()
        while True:
            ancestors_found, ancestors_raw = self.cache.load_primitive(
                self.metadata_id, "ancestors", cache_key, request
            )
            metadata_found, metadata_raw = self.cache.load_primitive(
                self.metadata_id, "metadata", cache_key, request
            )
            if ancestors_found and metadata_found:
                return set(ancestors_raw), dict(metadata_raw)
            inflight_key = ("entity-profile", cache_key)
            with self._inflight_lock:
                event = self._inflight.get(inflight_key)
                owner = event is None
                if owner:
                    event = threading.Event()
                    self._inflight[inflight_key] = event
            assert event is not None
            if owner:
                break
            event.wait()

        try:
            self._count_cache("profile_requests")
            ancestors, metadata = self._fetch_entity_profile(qid)
            self.cache.store_primitive(
                self.metadata_id, "ancestors", cache_key, request, ancestors
            )
            self.cache.store_primitive(
                self.metadata_id, "metadata", cache_key, request, metadata
            )
            return ancestors, metadata
        finally:
            with self._inflight_lock:
                self._inflight.pop(inflight_key, None)
                event.set()

    def _ancestor_qids(self, child_qid: str) -> set[str]:
        request = {"qid": child_qid}
        cache_key = hashlib.sha256(self._token(request).encode()).hexdigest()
        found, cached = self.cache.load_primitive(
            self.metadata_id, "ancestors", cache_key, request
        )
        if found:
            self._count_cache("hits")
            return set(cached)
        self._count_cache("misses")
        ancestors, _ = self._entity_profile(child_qid)
        return ancestors

    def _metadata(self, qid: str) -> dict[str, object]:
        request = {"qid": qid}
        cache_key = hashlib.sha256(self._token(request).encode()).hexdigest()
        found, cached = self.cache.load_primitive(
            self.metadata_id, "metadata", cache_key, request
        )
        if found:
            self._count_cache("hits")
            return dict(cached)
        self._count_cache("misses")
        _, metadata = self._entity_profile(qid)
        return metadata

RAW_CACHE_ID = "wikidata-http-json-v1"

class RawRequestCoordinator:
    def __init__(self):
        self.lock = threading.Lock()
        self.inflight = {}

class CachedWikidataSession:
    def __init__(self, transport, cache, *, coordinator=None, count=None, offline=False):
        self.transport = transport
        self.cache = cache
        self.coordinator = coordinator or RawRequestCoordinator()
        self.count = count or (lambda outcome: None)
        self.offline = offline

    @staticmethod
    def response(record):
        import requests
        response = requests.Response()
        response.status_code = record["status_code"]
        response.url = record["url"]
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.encoding = "utf-8"
        response._content = json.dumps(record["json"], ensure_ascii=False).encode()
        return response

    def get(self, url, *, params=None, **kwargs):
        request = {"method": "GET", "url": url, "params": params or {}}
        # These internal calls are public Wikidata requests only. Never cache
        # authenticated/personalized requests or unrelated services.
        if url not in {"https://www.wikidata.org/w/api.php", "https://query.wikidata.org/sparql"}:
            raise ValueError("Raw Wikidata cache received an unexpected endpoint")
        if kwargs.get("auth") or kwargs.get("cookies") or kwargs.get("headers"):
            raise ValueError("Raw Wikidata cache does not accept personalized request headers")
        key = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        while True:
            found, cached = self.cache.load_primitive(RAW_CACHE_ID, "http-get", key, request)
            if found:
                self.count("raw_hits")
                return self.response(cached)
            if self.offline:
                raise RuntimeError(f"Missing raw Wikidata request in offline mode: {key}")
            with self.coordinator.lock:
                event = self.coordinator.inflight.get(key)
                owner = event is None
                if owner:
                    event = threading.Event()
                    self.coordinator.inflight[key] = event
            if owner:
                break
            event.wait()
        try:
            self.count("raw_requests")
            response = self.transport.get(url, params=params, **kwargs)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(f"Wikidata API error: {payload['error'].get('code', 'unknown')}")
            self.cache.store_primitive(RAW_CACHE_ID, "http-get", key, request,
                                       {"status_code": response.status_code, "url": response.url or url,
                                        "json": payload})
            return response
        finally:
            with self.coordinator.lock:
                self.coordinator.inflight.pop(key, None)
                event.set()
