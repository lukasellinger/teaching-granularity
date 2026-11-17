"""Replay recorded Wikidata evidence without network calls."""

import json
from pathlib import Path

import pytest

from parsing.location.normalization.resolver import AuditableWikidataNormalizer
from parsing.location.normalization.values import LocationLevel


def key(*parts):
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


class FixtureNormalizer(AuditableWikidataNormalizer):
    def __init__(self, evidence):
        self.evidence = evidence

    def required(self, section, *parts):
        return self.evidence[section][key(*parts)]

    @staticmethod
    def candidates(rows):
        return [{**row, "level": LocationLevel(row["level"])} for row in rows]

    def _search(self, component):
        return self.candidates(self.required("search", component))

    def _context_search(self, component, context):
        return self.candidates(self.required("context_search", component, context))

    def _search_place_candidates(self, component):
        return self.candidates(self.required("place_search", component))

    def _ancestor_qids(self, qid):
        return set(self.required("ancestors", qid))

    def _metadata(self, qid):
        return dict(self.required("metadata", qid))


CASES = [json.loads(line) for line in (Path(__file__).parent / "fixtures/regressions.jsonl").read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("case", CASES, ids=lambda row: row["case_id"])
def test_recorded_resolution(case):
    result = FixtureNormalizer(case["evidence"]).resolve(case["components"], answer=case["answer"])
    assert result.model_dump(mode="json") == case["expected_resolution"]
