"""Indexed, transactional storage for correctness judgments."""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sqlite3


def content_key(*, model, question, reference_answers, response, prompt_sha256, cache_namespace):
    payload = dict(model=model, question=question, reference_answers=reference_answers,
                   response=response, prompt_sha256=prompt_sha256, cache_namespace=cache_namespace)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class SQLiteJudgmentCache(Mapping):
    """One immutable record per content key; conflicting labels are rejected."""

    def __init__(self, path, identity):
        self.path = Path(path)
        self.identity = identity
        self.connection = None
        if self.path.exists():
            self._open(create=False)

    def _open(self, create):
        if self.connection is not None:
            return self.connection
        if not create and not self.path.exists():
            return None
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path.resolve().as_uri() + ("?mode=rwc" if create else "?mode=rw"), uri=True, timeout=30)
        try:
            if create:
                connection.execute("PRAGMA journal_mode=DELETE")
                connection.execute("PRAGMA synchronous=FULL")
                with connection:
                    connection.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
                    connection.execute("CREATE TABLE IF NOT EXISTS judgments (key TEXT PRIMARY KEY, label TEXT NOT NULL CHECK(label IN ('Correct','Incorrect','Abstain','Cannot determine')), record_json TEXT NOT NULL)")
                    connection.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (json.dumps(self.identity, sort_keys=True),))
                    connection.execute("PRAGMA user_version=1")
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("Unsupported SQLite cache schema")
            row = connection.execute("SELECT value FROM metadata WHERE name='identity'").fetchone()
            if row is None or json.loads(row[0]) != self.identity:
                raise ValueError("SQLite cache model/prompt/backend mismatch")
        except BaseException:
            connection.close()
            raise
        self.connection = connection
        return connection

    def validate(self, record):
        if any(record.get(k) != v for k, v in self.identity.items()):
            raise ValueError("Cache record model/prompt/backend mismatch")
        if record.get("label") not in {"Correct", "Incorrect", "Abstain", "Cannot determine"} or not isinstance(record.get("raw_output"), str):
            raise ValueError("Invalid cached label or raw output")
        expected = content_key(**self.identity, question=record["question"],
                               reference_answers=record["reference_answers"], response=record["response"])
        if record.get("key") != expected:
            raise ValueError("Cache content key mismatch")

    def __len__(self):
        connection = self._open(create=False)
        return connection.execute("SELECT count(*) FROM judgments").fetchone()[0] if connection else 0

    def __iter__(self):
        connection = self._open(create=False)
        if connection:
            yield from (r[0] for r in connection.execute("SELECT key FROM judgments ORDER BY key"))

    def __contains__(self, key):
        connection = self._open(create=False)
        return bool(connection and connection.execute("SELECT 1 FROM judgments WHERE key=?", (key,)).fetchone())

    def __getitem__(self, key):
        connection = self._open(create=False)
        row = connection.execute("SELECT label, record_json FROM judgments WHERE key=?", (key,)).fetchone() if connection else None
        if row is None:
            raise KeyError(key)
        record = json.loads(row[1])
        self.validate(record)
        if record["key"] != key or record["label"] != row[0]:
            raise ValueError("SQLite cache index/payload mismatch")
        return record

    def put_many(self, records):
        """Atomically insert new records, leaving existing judgments unchanged."""
        records = list(records)
        for record in records:
            self.validate(record)
        if not records:
            return 0
        connection = self._open(create=True)
        inserted = 0
        with connection:
            for record in records:
                previous = connection.execute("SELECT label FROM judgments WHERE key=?", (record["key"],)).fetchone()
                if previous:
                    if previous[0] != record["label"]:
                        raise ValueError("Conflicting cached label; transaction rolled back")
                    continue
                connection.execute("INSERT INTO judgments VALUES (?, ?, ?)",
                    (record["key"], record["label"], json.dumps(record, ensure_ascii=False, sort_keys=True)))
                inserted += 1
        return inserted

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()

