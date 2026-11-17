import hashlib
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from parsing.location.normalization.cache import (
    SQLiteNormalizationCache, CachedWikidataSession, PersistentLookupNormalizer,
    _resolution_cache_key, snapshot_cache,
)
from parsing.location.normalization.schemas import ComponentResolution
from parsing.location.normalization.service import LocationNormalizer
from parsing.location.normalization.versions import (
    lookup_cache_identity, profile_cache_identity, resolution_cache_identity,
)

URL = "https://www.wikidata.org/w/api.php"


class Transport:
    def __init__(self, payload=None, status=200):
        self.calls = 0
        self.payload = payload if payload is not None else {"search": [{"id": "Q1", "label": "Example"}]}
        self.status = status

    def get(self, url, **kwargs):
        self.calls += 1
        response = requests.Response()
        response.status_code = self.status
        response.url = url
        response._content = json.dumps(self.payload).encode()
        return response


def test_raw_reuse_and_offline_miss(tmp_path):
    cache = SQLiteNormalizationCache(tmp_path / "cache.sqlite3")
    transport = Transport()
    expected = CachedWikidataSession(transport, cache).get(URL, params={"search": "Paris", "language": "en"}).json()
    offline = CachedWikidataSession(Transport(status=500), cache, offline=True)
    assert offline.get(URL, params={"language": "en", "search": "Paris"}).json() == expected
    assert transport.calls == 1
    with pytest.raises(RuntimeError, match="Missing raw"):
        offline.get(URL, params={"search": "Paris", "language": "fr"})
    cache.close()


def test_errors_not_cached(tmp_path):
    cache = SQLiteNormalizationCache(tmp_path / "cache.sqlite3")
    for transport in (Transport(status=429), Transport({"error": {"code": "maxlag"}})):
        with pytest.raises((requests.HTTPError, RuntimeError)):
            CachedWikidataSession(transport, cache).get(URL, params={"action": "test"})
    assert cache.counts()["primitive"] == 0
    cache.close()


def test_concurrent_raw_requests_and_close(tmp_path):
    cache = SQLiteNormalizationCache(tmp_path / "cache.sqlite3")
    transport = Transport()
    session = CachedWikidataSession(transport, cache)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda _: session.get(URL, params={"ids": "Q1"}).json(), range(8)))
    assert rows == [transport.payload] * 8
    assert transport.calls == 1
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        cache.counts()


def test_snapshot_includes_live_wal_and_is_standalone(tmp_path):
    cache = SQLiteNormalizationCache(tmp_path / "live.sqlite3")
    cache.store_primitive("version", "search", "key", {"q": "München"}, ["Q1726"])
    destination = tmp_path / "snapshot.sqlite3"
    snapshot_cache(cache.path, destination)
    with sqlite3.connect(destination) as reader:
        assert reader.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert reader.execute("SELECT COUNT(*) FROM primitive_cache").fetchone() == (1,)
    with pytest.raises(FileExistsError):
        snapshot_cache(cache.path, destination)
    assert cache.counts()["primitive"] == 1
    cache.close()
    reopened = SQLiteNormalizationCache(destination)
    assert reopened.load_primitive("version", "search", "key", {"q": "München"}) == (True, ["Q1726"])
    reopened.close()


def test_unicode_cache_key_matches_existing_encoding():
    identity = {"components": ["Cumaná"], "answer": "Geboren in Cumaná"}
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert _resolution_cache_key(identity) == expected


def test_lookup_namespace_reuse(tmp_path, monkeypatch):
    cache = SQLiteNormalizationCache(tmp_path / "cache.sqlite3")
    request = {"component": "paris"}
    key = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache.store_primitive(lookup_cache_identity(), "search", key, request, [])
    resolver = PersistentLookupNormalizer(cache, "fixture", refresh_evidence=True, offline=True)
    monkeypatch.setattr(resolver, "_session", lambda: pytest.fail("Unexpected HTTP request"))
    assert resolver._search("paris") == []
    assert resolver.metadata_id == profile_cache_identity() != lookup_cache_identity()
    assert resolution_cache_identity(refresh_evidence=True) != resolution_cache_identity()
    cache.close()


def test_entity_profile_and_missing_label_fallback_are_persisted(tmp_path, monkeypatch):
    cache = SQLiteNormalizationCache(tmp_path / "cache.sqlite3")
    resolver = PersistentLookupNormalizer(cache, "fixture", offline=True)
    rows = [
        {"ancestor": {"value": "http://www.wikidata.org/entity/Q1"}},
        {"admin": {"value": "http://www.wikidata.org/entity/Q1"},
         "adminLabel": {"value": "Q1"}, "adminDescription": {"value": "city in France"},
         "directCountry": {"value": "http://www.wikidata.org/entity/Q142"},
         "directCountryLabel": {"value": "France"},
         "directCountryContinentLabel": {"value": "Europe"}},
    ]
    transport = Transport()
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        transport.payload = ({"results": {"bindings": rows}} if url.endswith("sparql") else
                             {"entities": {"Q1": {"labels": {"en": {"value": "Paris"}}}}})
        return transport.get(url, **kwargs)
    class Session:
        pass
    session = Session()
    session.get = get
    monkeypatch.setattr(resolver, "_session", lambda: session)
    assert resolver._ancestor_qids("Q1") == {"Q1"}
    assert resolver._metadata("Q1")["city"] == "Paris"
    assert resolver._metadata("Q1")["country"] == "France"
    assert len(calls) == 2
    cache.close()
    cache = SQLiteNormalizationCache(tmp_path / "cache.sqlite3")
    resolver = PersistentLookupNormalizer(cache, "fixture", offline=True)
    assert resolver._ancestor_qids("Q1") == {"Q1"}
    assert resolver._metadata("Q1")["city"] == "Paris"
    cache.close()


def test_answer_aware_reuse_and_failure_retry(tmp_path, monkeypatch):
    calls = []
    with LocationNormalizer(tmp_path / "cache.sqlite3") as normalizer:
        def resolve(components, *, answer):
            calls.append(answer)
            return ComponentResolution(status="unresolved", components=components)
        monkeypatch.setattr(normalizer.resolver, "resolve", resolve)
        rows = [{"answer": "Born in Paris", "extraction": {"components": ["Paris"]}, "model": str(i)} for i in range(8)]
        assert len(normalizer.process_many(rows)) == 8
        assert calls == ["Born in Paris"]
        normalizer.normalize(["Paris"], answer="Died in Paris")
        assert len(calls) == 2
    with LocationNormalizer(tmp_path / "cache.sqlite3") as normalizer:
        monkeypatch.setattr(normalizer.resolver, "resolve", lambda *a, **k: pytest.fail("Cache not reused"))
        normalizer.normalize(["Paris"], answer="Born in Paris")
        def broken(*a, **k):
            raise RuntimeError("offline miss")
        monkeypatch.setattr(normalizer.resolver, "resolve", broken)
        assert normalizer.normalize(["Paris"], answer="Another answer").status == "failed"
        monkeypatch.setattr(normalizer.resolver, "resolve", resolve)
        assert normalizer.normalize(["Paris"], answer="Another answer").status == "unresolved"


def test_upstream_status_and_idk(tmp_path):
    with LocationNormalizer(tmp_path / "cache.sqlite3") as normalizer:
        assert normalizer.process({"extraction": {"status": "generation_truncated"}})["status"] == "skipped"
        assert normalizer.process({"extraction": {"location": "IDK"}})["status"] == "idk"
        with pytest.raises(ValueError, match="full generated answer"):
            normalizer.normalize(["Paris"], answer="")
        assert normalizer.normalize([], answer="Test").status == "failed"


def test_workers_reused_between_batches(tmp_path):
    with LocationNormalizer(tmp_path / "cache.sqlite3") as normalizer:
        row = {"extraction": {"location": "IDK"}}
        for _ in range(20):
            assert normalizer.process_many([row] * 16, workers=4)[0]["status"] == "idk"
        assert len(normalizer._pools) == 1
        assert len(normalizer._pools[4]._threads) <= 4


def test_offline_miss_does_not_become_idk(tmp_path):
    with LocationNormalizer(tmp_path / "cache.sqlite3") as normalizer:
        result = normalizer.normalize(["Uncached City"], answer="Born in Uncached City.")
        assert result.status == "failed"
        assert normalizer.cache.counts()["resolutions"] == 0


def test_compatible_version_ids():
    assert lookup_cache_identity() == "source-order-components-v1-7fde69e4c029c068"
    assert resolution_cache_identity() == "source-order-components-v1-9489bb2d473927d9"
    assert profile_cache_identity() == "wikidata-entity-profile-v2-2173be093f024300"


def test_existing_sqlite_resolution_encoding(tmp_path, monkeypatch):
    from parsing.location.normalization.resolver import _cache_identity
    case = json.loads((Path(__file__).parent / "fixtures/regressions.jsonl").read_text().splitlines()[0])
    identity = _cache_identity(case["components"], answer=case["answer"],
                               normalizer_id="source-order-components-v1-9489bb2d473927d9")
    key = _resolution_cache_key(identity)
    db = tmp_path / "existing.sqlite3"
    cache = SQLiteNormalizationCache(db)
    cache.close()
    with sqlite3.connect(db) as connection:
        connection.execute("INSERT INTO resolution_cache (cache_key, identity_json, resolution_json) VALUES (?, ?, ?)",
                           (key, json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                            json.dumps(case["expected_resolution"], ensure_ascii=False)))
    with LocationNormalizer(db) as normalizer:
        monkeypatch.setattr(normalizer.resolver, "resolve", lambda *a, **k: pytest.fail("Existing resolution was not reused"))
        assert normalizer.normalize(case["components"], answer=case["answer"]).model_dump(mode="json") == case["expected_resolution"]


def test_command_line_offline_and_snapshot(tmp_path):
    source, target, cache, snapshot = (tmp_path / name for name in ("input.jsonl", "output.jsonl", "cache.sqlite3", "snapshot.sqlite3"))
    row = {"question": "Where was A born?", "answer": "Unknown.", "extraction": {"status": "ok", "location": "IDK", "components": []}}
    source.write_text(json.dumps(row) + "\n")
    command = [sys.executable, "-m", "parsing.location.normalization"]
    subprocess.run([*command, "normalize", "--input", str(source), "--output", str(target),
                    "--cache-db", str(cache)], check=True, capture_output=True, text=True)
    result = json.loads(target.read_text())
    assert result["normalization"]["status"] == "idk"
    assert result["answer"] == row["answer"]
    subprocess.run([*command, "snapshot", "--cache-db", str(cache), "--output", str(snapshot)],
                   check=True, capture_output=True, text=True)
    assert snapshot.is_file()
