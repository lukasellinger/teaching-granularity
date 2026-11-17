"""Wikidata search, entity typing and geographic hierarchy lookup."""
from __future__ import annotations
import json
import os
import re
import threading
import unicodedata
from .values import LocationLevel, LOCATION_PRECISION

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

WIKIDATA_LANGUAGE = "en"

WIKIDATA_SEARCH_LIMIT = 5

_LEVEL_PRIORITY = (
    LocationLevel.COUNTRY,
    LocationLevel.CITY,
    LocationLevel.SUBCITY,
    LocationLevel.REGION,
    LocationLevel.CONTINENT,
    LocationLevel.PLACE,
)

_DESCRIPTION_KEYWORDS = {
    LocationLevel.PLACE: {
        "airport",
        "building",
        "castle",
        "cemetery",
        "church",
        "estate",
        "facility",
        "hospital",
        "hotel",
        "house",
        "monument",
        "museum",
        "palace",
        "residence",
        "site",
        "station",
        "university",
        "venue",
        "square",
        "plaza",
    },
    LocationLevel.CITY: {
        "borough",
        "capital",
        "city",
        "commune",
        "human settlement",
        "locality",
        "megacity",
        "municipality",
        "town",
        "village",
    },
    LocationLevel.SUBCITY: {"neighborhood", "neighbourhood", "city district"},
    LocationLevel.REGION: {
        "area",
        "archipelago",
        "county",
        "department",
        "district",
        "island",
        "oblast",
        "prefecture",
        "province",
        "region",
        "state",
        "territory",
    },
    LocationLevel.COUNTRY: {
        "constituent country",
        "country",
        "sovereign state",
    },
    LocationLevel.CONTINENT: {"continent"},
}

_COMPONENT_SUFFIXES = {
    "area",
    "borough",
    "capital",
    "city",
    "commune",
    "continent",
    "county",
    "district",
    "island",
    "megacity",
    "municipality",
    "oblast",
    "prefecture",
    "province",
    "region",
    "state",
    "town",
    "village",
}

_SUFFIX_LEVELS = {
    "borough": LocationLevel.CITY,
    "capital": LocationLevel.CITY,
    "city": LocationLevel.CITY,
    "commune": LocationLevel.CITY,
    "megacity": LocationLevel.CITY,
    "municipality": LocationLevel.CITY,
    "town": LocationLevel.CITY,
    "village": LocationLevel.CITY,
    "archipelago": LocationLevel.REGION,
    "county": LocationLevel.REGION,
    "department": LocationLevel.REGION,
    "district": LocationLevel.REGION,
    "island": LocationLevel.REGION,
    "oblast": LocationLevel.REGION,
    "prefecture": LocationLevel.REGION,
    "province": LocationLevel.REGION,
    "region": LocationLevel.REGION,
    "state": LocationLevel.REGION,
    "territory": LocationLevel.REGION,
    "continent": LocationLevel.CONTINENT,
}

_HISTORICAL_POLITY = re.compile(
    r"\b(?:empire|historical (?:country|state|region|territory|area)|"
    r"former (?:country|state|colony|region|territory|kingdom|duchy)|"
    r"historical (?:kingdom|duchy|monarchy)|composite monarchy|"
    r"colonial rule|former colony)\b|"
    r"\b(?:state|country|colony|territory)\b.{0,60}\b(?:between|from)\s+"
    r"\d{3,4}\s+(?:and|to)\s+\d{3,4}\b|"
    r"\bfrom\s+\d{3,4}\s+to\s+\d{3,4}\s+(?:while\s+)?under\b|"
    r"\b(?:rule|administration|colony|republic|country|state)\b.{0,100}"
    r"\b\d{3,4}\s*[–—-]\s*\d{2,4}\b|"
    r"(?<!united )\bkingdom\b(?=.{0,120}\b\d{3,4}\s*[–—-]\s*\d{2,4}\b)",
    flags=re.IGNORECASE,
)


class WikidataLocationNormalizer:
    """Resolve explicit location components and infer only broader hierarchy."""

    def __init__(self):
        self.user_agent = os.getenv("WIKI_USER_AGENT") or "teaching-granularity/offline"
        self._local = threading.local()

    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            try:
                import requests
                from requests.adapters import HTTPAdapter
                from urllib3.util import Retry
            except ImportError as error:
                raise RuntimeError(
                    "location normalization requires requests"
                ) from error
            session = requests.Session()
            retry = Retry(
                total=8,
                connect=8,
                read=8,
                status=8,
                backoff_factor=1,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
                respect_retry_after_header=True,
            )
            session.mount("https://", HTTPAdapter(max_retries=retry))
            session.headers.update({"User-Agent": self.user_agent})
            self._local.session = session
        return session

    @staticmethod
    def _level_from_description(description: str | None) -> LocationLevel | None:
        if not description:
            return None
        description_cf = description.casefold()
        entity_type = description_cf.split(" in ", 1)[0]
        if re.match(r"one of (?:the )?.*\bautonomous countries\b", description_cf):
            return LocationLevel.COUNTRY
        if re.match(r"(?:nationality and )?autonomous community\b", description_cf):
            return LocationLevel.REGION
        if (re.search(r"\bconstituent country\b", description_cf)
                or re.search(r"^country\b.*\bpart of (?:the )?united kingdom\b", description_cf)):
            return LocationLevel.REGION
        if re.match(r"(?:federal|capital) district\b", description_cf):
            return LocationLevel.REGION
        # Country descriptions such as "island country" must win over generic
        # regional words such as "island". Requiring country to be the noun at
        # the end avoids misclassifying descriptions such as "country house".
        if re.search(r"\bsovereign state\b", entity_type) or re.search(
            r"(?:^|\s)country$", entity_type
        ):
            return LocationLevel.COUNTRY
        # London boroughs and urban neighborhoods occupy the SUBCITY level.
        # Do not infer village/city types from aliases such as Hampstead Village.
        if re.match(r"borough in the london region\b", description_cf):
            return LocationLevel.SUBCITY
        if re.match(r"(?:borough of (?:new york|.*city)|neighbou?rhood|city district|area of .*city|area of .*london)\b", description_cf):
            return LocationLevel.SUBCITY
        # A venue/institution remains a PLACE even when an adjective such as
        # "state" precedes its head noun (for example, "state public
        # university").  A leading geographic type is also authoritative:
        # "region ... containing the City of London" describes the region.
        # Otherwise retain level priority; descriptions such as "seat of Los
        # Angeles County, and largest city ..." mention a county before naming
        # the entity's own city type.
        if any(
            re.search(rf"\b{re.escape(keyword)}\b", entity_type)
            for keyword in _DESCRIPTION_KEYWORDS[LocationLevel.PLACE]
        ):
            return LocationLevel.PLACE
        for level in _LEVEL_PRIORITY:
            if level is LocationLevel.PLACE:
                continue
            for keyword in _DESCRIPTION_KEYWORDS[level]:
                if re.match(rf"{re.escape(keyword)}\b", entity_type):
                    return level
        for level in _LEVEL_PRIORITY:
            if level is LocationLevel.PLACE:
                continue
            if any(
                re.search(rf"\b{re.escape(keyword)}\b", entity_type)
                for keyword in _DESCRIPTION_KEYWORDS[level]
            ):
                return level
        return None

    @staticmethod
    def _normalize_query(component: str) -> str:
        words = component.split()
        # "area" qualifies the complete preceding name.  After removing it,
        # do not also strip a meaningful final token such as "City" from
        # "Century City area" or "New York City area".
        if words and words[-1].casefold() == "area":
            return " ".join(words[:-1])
        while words and words[-1].casefold() in _COMPONENT_SUFFIXES:
            words.pop()
        return " ".join(words)

    def _search(self, component: str) -> list[dict[str, object]]:
        # Search the written name first. Stripping "City" from "Mexico City"
        # before search makes the country Mexico look like an exact hit.
        for query in dict.fromkeys((component, self._normalize_query(component))):
            if not query:
                continue
            response = self._session().get(
                WIKIDATA_API_URL,
                params={
                    "action": "wbsearchentities",
                    "search": query,
                    "language": WIKIDATA_LANGUAGE,
                    "format": "json",
                    "limit": WIKIDATA_SEARCH_LIMIT,
                },
                timeout=60,
            )
            response.raise_for_status()
            candidates = []
            for result in response.json().get("search", []):
                description = result.get("description")
                historical = bool(
                    description and _HISTORICAL_POLITY.search(description)
                )
                level = self._level_from_description(description)
                if level is None and historical:
                    level = LocationLevel.COUNTRY
                if level is not None:
                    candidates.append(
                        {
                            "qid": result["id"],
                            "label": result["label"],
                            "level": level,
                            "historical": historical,
                        }
                    )
            if self._valid_candidates(component, None, candidates):
                return candidates
        return []


    def _valid_candidates(
        self,
        component: str,
        parent: dict[str, object] | None,
        candidates: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        normalized = self._normalize_query(component).casefold()
        component_cf = component.casefold()
        suffix = component_cf.rsplit(" ", 1)[-1]
        expected_level = (
            _SUFFIX_LEVELS.get(suffix) if normalized != component_cf else None
        )

        if expected_level is not None:
            candidates = [
                candidate
                for candidate in candidates
                if str(candidate["label"]).casefold() == component_cf
                or candidate["level"] is expected_level
            ]

        def match_score(candidate: dict[str, object]) -> tuple[int, int]:
            label = str(candidate["label"]).casefold()
            contains_full = bool(re.search(rf"\b{re.escape(component_cf)}\b", label))
            contains_stripped = bool(
                normalized and re.search(rf"\b{re.escape(normalized)}\b", label)
            )
            return (
                (
                    0
                    if label == component_cf
                    else (
                        1
                        if contains_full
                        else (
                            2
                            if normalized and label == normalized
                            else 3 if contains_stripped else 4
                        )
                    )
                ),
                _LEVEL_PRIORITY.index(candidate["level"]),
            )

        if parent is not None:
            parent_level = parent["level"]
            parent_qid = str(parent["qid"])
            candidates = [
                candidate
                for candidate in candidates
                if LOCATION_PRECISION[candidate["level"]]
                >= LOCATION_PRECISION[parent_level]
                and parent_qid in self._ancestor_qids(str(candidate["qid"]))
            ]
        return sorted(candidates, key=match_score)


