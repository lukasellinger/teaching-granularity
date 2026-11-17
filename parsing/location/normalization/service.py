"""Answer-aware cached normalization, without filtering or person factual gold."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading

from .cache import SQLiteNormalizationCache, PersistentLookupNormalizer, _resolution_cache_key
from .resolver import _cache_identity, prepare_resolution_components
from .schemas import ComponentResolution
from .versions import resolution_cache_identity


class LocationNormalizer:
    """Resolve extracted components and reuse the portable SQLite cache."""

    def __init__(self, cache_db, *, offline=True, user_agent=None, refresh_evidence=False):
        user_agent = user_agent or os.getenv("WIKI_USER_AGENT")
        if not offline and not user_agent:
            raise ValueError("Online normalization requires --user-agent or WIKI_USER_AGENT with contact information")
        self.cache = SQLiteNormalizationCache(Path(cache_db))
        self.identity = resolution_cache_identity(refresh_evidence=refresh_evidence)
        self.resolver = PersistentLookupNormalizer(self.cache, self.identity, refresh_evidence=refresh_evidence, offline=offline)
        if user_agent:
            self.resolver.user_agent = user_agent
        self._lock = threading.Lock()
        self._inflight = {}
        self._pools = {}

    def normalize(self, components, *, answer):
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Provide the full generated answer used for normalization")
        components, error = prepare_resolution_components(components)
        if error:
            return ComponentResolution(status="failed", components=components, error=error)
        identity = _cache_identity(components, answer=answer, normalizer_id=self.identity)
        key = _resolution_cache_key(identity)
        while True:
            cached = self.cache.load_resolution(key, identity)
            if cached is not None and cached.status != "failed":
                return cached
            with self._lock:
                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    break
            event.wait()
        try:
            # A previous leader may have finished between our read and lock.
            cached = self.cache.load_resolution(key, identity)
            if cached is not None and cached.status != "failed":
                return cached
            try:
                result = self.resolver.resolve(components, answer=answer)
            except Exception as exc:
                return ComponentResolution(status="failed", components=components, error=f"{type(exc).__name__}: {str(exc)[:4000]}")
            if result.status != "failed":
                self.cache.store_resolution(key, identity, result)
            return result
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                event.set()

    def process(self, row):
        extraction = row.get("extraction", row)
        status = extraction.get("status", "ok")
        if status != "ok":
            return {"status": "skipped", "reason": f"extraction_{status}"}
        if extraction.get("location") == "IDK":
            return ComponentResolution(status="idk", components=[]).model_dump(mode="json")
        components = extraction.get("components")
        return self.normalize(components, answer=row["answer"]).model_dump(mode="json")

    def process_many(self, rows, *, workers=4):
        if workers < 1:
            raise ValueError("workers must be positive")
        with self._lock:
            if workers not in self._pools:
                self._pools[workers] = ThreadPoolExecutor(max_workers=workers)
            pool = self._pools[workers]
        return list(pool.map(self.process, rows))

    def close(self):
        for pool in self._pools.values():
            pool.shutdown(wait=True)
        self._pools.clear()
        self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
