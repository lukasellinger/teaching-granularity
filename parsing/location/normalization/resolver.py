"""Resolve extracted components with answer-local type and alias evidence."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Sequence
from .wikidata import WikidataLocationNormalizer
from .schemas import NormalizedLocation, ResolvedEntity, CandidateEntity, CandidateHierarchy, DisambiguationTrace, ComponentTrace, ComponentResolution
from .values import LOCATION_CONTEXT_FIELDS, LOCATION_FIELDS, LOCATION_PRECISION, LocationLevel, LocationValue, location_consensus

NORMALIZER_VERSION = "source-order-components-v1"

AUDIT_FORMAT = "location-component-normalization-audit-v1"

PLACE_SEARCH_LIMIT = 20

COMPONENT_SEARCH_LIMIT = 10

_PLACE_WORDS = re.compile(
    r"\b(?:address|apartment|building|care center|care centre|castle|clinic|estate|"
    r"cell|facility|farmhouse|fortress|hall|home|homestead|hospital|hotel|house|palace|"
    r"prison|residence|road|street|thoroughfare|townhouse|villa)\b",
    flags=re.IGNORECASE,
)

_GENERIC_PLACE = re.compile(
    r"^(?:(?:his|her|their|the|a|an)\s+)?(?:family(?:'s)?\s+)?"
    r"(?:(?:private|country)\s+)?"
    r"(?:home|house|townhouse|residence|estate|hospital|prison|cell|"
    r"care (?:center|centre)|facility|town(?:'s)? (?:center|centre))$",
    flags=re.IGNORECASE,
)

_COMPONENT_ALIASES = {
    "u.s.": "united states", "u.s.a.": "united states",
    "us": "united states", "usa": "united states",
    "uk": "united kingdom", "u.k.": "united kingdom",
}

_LEXICAL_ALIASES = {
    "china": "people's republic of china", "prc": "people's republic of china",
    "kiev": "kyiv",
}

_CONTEXT_QUALIFIER = re.compile(
    r"^(?P<qualifier>near|close to|in the vicinity of|within the vicinity of|outside|around)\s+"
    r"(?P<name>.+)$",
    flags=re.IGNORECASE,
)

_CONTEXT_RELATIONS = {
    "near": "near", "close to": "near", "in the vicinity of": "near",
    "within the vicinity of": "near", "outside": "outside", "around": "around",
}

_DISTANCE_CONTEXT = re.compile(
    r"^(?:about\s+|approximately\s+)?\d+(?:\.\d+)?\s*"
    r"(?:kilometers?|kilometres?|km|miles?)\s+"
    r"(?:north|south|east|west|northeast|northwest|southeast|southwest)(?:ern)?\s+of\s+(.+)$",
    re.IGNORECASE,
)

_DIRECTIONAL_AREA = re.compile(
    r"^(?:the\s+)?(?:(?:coast|region|area)\s+of\s+(?:the\s+)?)?"
    r"(?:(?:northern|southern|eastern|western|northeastern|northwestern|"
    r"southeastern|southwestern)(?:\s+part\s+of)?\s+|"
    r"(?:northeast|northwest|southeast|southwest)\s+|"
    r"(?:north|south|east|west)\s+of\s+)"
    r"(?P<name>.+)$", re.IGNORECASE,
)

_TYPE_PREFIX_LEVELS = {
    "borough": LocationLevel.SUBCITY,
    "city": LocationLevel.CITY,
    "municipality": LocationLevel.CITY,
    "town": LocationLevel.CITY,
    "village": LocationLevel.CITY,
    "autonomous community": LocationLevel.REGION,
    "county": LocationLevel.REGION,
    "department": LocationLevel.REGION,
    "district": LocationLevel.REGION,
    "island": LocationLevel.REGION,
    "province": LocationLevel.REGION,
    "region": LocationLevel.REGION,
    "state": LocationLevel.REGION,
    "territory": LocationLevel.REGION,
}

_TYPE_PREFIX = re.compile(
    r"^(?:the\s+)?(?P<type>autonomous community|borough|city|county|department|district|"
    r"island|municipality|province|region|state|territory|town|village)\s+of\s+"
    r"(?P<name>.+)$",
    re.IGNORECASE,
)

_SOURCE_TYPES = {
    "city": LocationLevel.CITY, "town": LocationLevel.CITY,
    "village": LocationLevel.CITY, "commune": LocationLevel.CITY,
    "region": LocationLevel.REGION, "province": LocationLevel.REGION,
    "county": LocationLevel.REGION, "department": LocationLevel.REGION,
    "country": LocationLevel.COUNTRY, "continent": LocationLevel.CONTINENT,
    "neighborhood": LocationLevel.SUBCITY, "neighbourhood": LocationLevel.SUBCITY,
    "hospital": LocationLevel.PLACE,
}

def source_component_hint(component: str, answer: str | None) -> dict:
    """Only directly attached type descriptions, never person-level gold facts.

    Conflicting attached types do not produce a hint. Cross-sentence context
    cannot accidentally turn a later city into the type of an earlier region.
    """
    if not answer or contextual_component(component):
        return {}
    text = re.sub(r"[*_`]", "", answer)
    name = re.escape(component)
    hits = []
    for word, level in _SOURCE_TYPES.items():
        patterns = (
            # Consume a clause/preposition boundary. Without this guard,
            # "Yogyakarta Special Region of Indonesia" incorrectly attaches
            # "Region" to the separately extracted component "Indonesia".
            rf"(?:^|[,;(.:]\s*|\b(?:in|at|from|within|inside|with)\s+(?:the\s+)?)"
            rf"(?P<type>(?:historical\s+|Roman\s+)?{word})\s+(?:of\s+)?{name}(?!\w)",
            rf"(?<!\w){name}\s*,\s*(?:a|an|the)\s+(?P<type>(?:historical\s+|Roman\s+)?{word})\b",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                hits.append((level, bool(re.search(r"historical|Roman", match.group("type"), re.I)),
                             match.group(0)))
    levels = {hit[0] for hit in hits}
    if len(levels) != 1:
        return {}
    return {"level": hits[0][0], "historical": any(hit[1] for hit in hits),
            "evidence": hits[0][2]}

def source_alias_targets(components: Sequence[str], answer: str | None) -> dict[int, int]:
    """Map explicitly stated aliases to the component naming their entity."""
    if not answer:
        return {}
    text = re.sub(r"[*_`]", "", answer)
    aliases: dict[int, int] = {}
    marker = r"(?:also\s+known\s+as|also\s+called|formerly\s+(?:known\s+as|called)|now\s+known\s+as)"
    for target_index, target in enumerate(components):
        for alias_index, alias in enumerate(components):
            if alias_index == target_index:
                continue
            pattern = (
                rf"(?<!\w){re.escape(target)}\s*\(\s*{marker}\s+(?:the\s+)?"
                rf"[\"'“”]?{re.escape(alias)}[\"'“”]?\s*\)"
            )
            if re.search(pattern, text, re.IGNORECASE):
                aliases[alias_index] = target_index
    return aliases

def canonical_component(value: str) -> str:
    value = re.sub(r"[*_`]", "", value)
    value = " ".join(value.split()).strip(" ,").casefold()
    return _COMPONENT_ALIASES.get(value, value)

def contextual_component(value: str) -> tuple[str, str] | None:
    """Return (relation, lookup name) for non-containment location context."""
    normalized = " ".join(value.split()).strip(" ,")
    distance = _DISTANCE_CONTEXT.fullmatch(normalized)
    if distance:
        return ("near", distance.group(1))
    match = _CONTEXT_QUALIFIER.fullmatch(normalized)
    if match is None:
        return None
    qualifier = match.group("qualifier").casefold()
    name = re.sub(r"^(?:modern-day|present-day)\s+", "", match.group("name"),
                  flags=re.IGNORECASE).strip()
    return (_CONTEXT_RELATIONS[qualifier], name) if name else None

def explicit_component_level(value: str) -> LocationLevel | None:
    """Return a level explicitly named by a component suffix.

    Wikidata search descriptions sometimes contain several geographic type
    words (for example, a province whose description mentions its capital).
    In that case the extractor's literal suffix is stronger evidence than the
    first keyword found in the search description.
    """
    from . import wikidata as base

    lookup = contextual_component(value)
    raw = lookup[1] if lookup else value
    if match := _TYPE_PREFIX.fullmatch(" ".join(raw.split()).strip(" ,")):
        return _TYPE_PREFIX_LEVELS[match.group("type").casefold()]
    normalized = canonical_component(raw)
    words = normalized.split()
    return base._SUFFIX_LEVELS.get(words[-1]) if words else None

def is_place_expression(component: str) -> bool:
    """Identify explicit venue/address evidence without outside knowledge."""
    if contextual_component(component):
        return False
    return bool(_PLACE_WORDS.search(component) or re.match(r"^\d+[A-Za-z]?\s+\S", component))

def is_generic_place(component: str) -> bool:
    return bool(_GENERIC_PLACE.fullmatch(" ".join(component.split())))

@dataclass(frozen=True)
class _Option:
    component_index: int
    component: str
    qid: str
    label: str
    level: LocationLevel
    historical: bool
    search_rank: int
    lexical_score: int
    administrative_region_role: bool = False
    context_supported: bool = False

class AuditableWikidataNormalizer(WikidataLocationNormalizer):
    """Resolve unordered extracted components while retaining an audit trace."""

    @staticmethod
    def _historical_description(description: str | None) -> bool:
        from . import wikidata as base
        return bool(description and (
            base._HISTORICAL_POLITY.search(description)
            or re.search(r"\bformer\s+(?:federation|polity|realm)\b", description, re.I)
            or re.search(r"\b(?:Roman|ancient|historical|former)\s+province\b", description, re.I)
            or re.search(r"\bkingdom\b.{0,100}\b(?:between|from)\s+\d{3,4}\s+(?:and|to)\s+\d{3,4}", description, re.I)
        ))

    @classmethod
    def _historical_candidate(cls, label: str, description: str | None) -> bool:
        """Recognize dated historical polity names omitted by terse descriptions.

        Wikidata sometimes describes a historical entity as merely an
        "official name ... during the period of 1929–1941".  The description
        therefore lacks words such as "former" even though the label and closed
        date interval make the historical role explicit.
        """
        if cls._historical_description(description):
            return True
        if description and (
            re.search(r"\bmedieval\s+(?:country|state|polity|realm|federation)\b",
                      description, re.IGNORECASE)
            or (
                re.search(r"\b(?:country|state|polity|realm|monarchy|empire|kingdom)\b",
                          description, re.IGNORECASE)
                and re.search(r"\b\d{3,4}\s*[–—-]\s*\d{2,4}\b", description)
            )
        ):
            return True
        return bool(
            description
            and re.match(
                r"(?:kingdom|empire|duchy|republic|state|colony|territory)\b",
                label,
                re.IGNORECASE,
            )
            and re.search(r"\b\d{3,4}\s*[–—-]\s*\d{2,4}\b", description)
        )

    @staticmethod
    def _level_from_description(description: str | None) -> LocationLevel | None:
        text = (description or "").casefold()
        # Entity type, not a geographic word appearing in an institution name.
        if re.search(r"\b(?:fire and rescue service|fire department|archive|archival)\b", text):
            return None
        if re.match(r"(?:court\s+)?prison\b|(?:road|street|thoroughfare)\b", text):
            return LocationLevel.PLACE
        if re.match(r"(?:historical\s+)?(?:low\s+)?(?:valley|mountain range)\b", text):
            return LocationLevel.REGION
        if re.match(r"peninsula\b", text):
            return LocationLevel.REGION
        if re.match(r"(?:exclave|federal subject|oblast)\b", text):
            return LocationLevel.REGION
        if re.match(r"hamlet\b", text):
            return LocationLevel.CITY
        if re.match(r"(?:subdivision|quarter|urban district|district) of\b", text):
            return LocationLevel.SUBCITY
        return WikidataLocationNormalizer._level_from_description(description)

    @staticmethod
    def _normalize_query(component: str) -> str:
        # The legacy suffix list omitted "department". Search the literal
        # component first, then the geographic name, never the institution.
        if re.search(r"\s+department$", component, re.I):
            return component.rsplit(" ", 1)[0]
        if match := _TYPE_PREFIX.fullmatch(component.strip()):
            return match.group("name").strip()
        return WikidataLocationNormalizer._normalize_query(component)

    def _ancestor_qids(self, child_qid: str) -> set[str]:
        """Administrative AND positively stated physical-location containment.

        P706 (located in/on physical feature) admits valleys/ranges without
        assuming every feature sharing a country contains every city there.
        """
        from . import wikidata as base
        query = f"""SELECT DISTINCT ?ancestor WHERE {{
          wd:{child_qid} (wdt:P131|wdt:P276|wdt:P706)* ?ancestor .
        }}"""
        response = self._session().get(base.WIKIDATA_SPARQL_URL,
                                       params={"query": query, "format": "json"}, timeout=120)
        response.raise_for_status()
        return {row["ancestor"]["value"].rsplit("/", 1)[-1]
                for row in response.json()["results"]["bindings"]}

    @staticmethod
    def _has_administrative_region_role(description: str | None) -> bool:
        """Administrative type evidence, not merely being inside a region."""
        entity_type = (description or "").casefold().split(" in ", 1)[0]
        return bool(re.search(
            r"\b(?:special administrative region|autonomous region|federal state)\b",
            entity_type,
        ))

    def _search(self, component: str) -> list[dict[str, object]]:
        """Search deeply enough to retain administrative hierarchy aliases.

        Wikidata ranks the city item for London (Q84) first, while the London
        administrative entity occurring in Hampstead's P131 chain (Q23939248)
        is sixth. The generic five-result search therefore cannot resolve the
        otherwise explicit Hampstead -> London chain.
        """
        from . import wikidata as base

        cache = getattr(self, "_audit_search_cache", None)
        if cache is None:
            cache = {}
            self._audit_search_cache = cache
        cache_key = canonical_component(component)
        if cache_key in cache:
            return list(cache[cache_key])

        combined: dict[str, dict[str, object]] = {}
        for query in dict.fromkeys((component, self._normalize_query(component))):
            if not query:
                continue
            response = self._session().get(
                base.WIKIDATA_API_URL,
                params={"action": "wbsearchentities", "search": query,
                        "language": base.WIKIDATA_LANGUAGE, "format": "json",
                        "limit": COMPONENT_SEARCH_LIMIT},
                timeout=60,
            )
            response.raise_for_status()
            candidates = []
            for result in response.json().get("search", []):
                description = result.get("description")
                historical = self._historical_candidate(result["label"], description)
                level = self._level_from_description(description)
                if level is None and historical:
                    level = LocationLevel.COUNTRY
                if level is not None:
                    candidates.append({
                        "qid": result["id"], "label": result["label"],
                        "matched_name": result.get("match", {}).get("text"),
                        "level": level, "historical": historical,
                        "administrative_region_role": self._has_administrative_region_role(description),
                    })
            for candidate in self._valid_candidates(component, None, candidates):
                combined.setdefault(str(candidate["qid"]), candidate)
        cache[cache_key] = tuple(combined.values())
        return list(combined.values())

    @staticmethod
    def _lexical_score(component: str, label: str) -> int:
        context = contextual_component(component)
        lookup = context[1] if context else component
        component_cf, label_cf = canonical_component(lookup), canonical_component(label)
        component_cf = _LEXICAL_ALIASES.get(component_cf, component_cf)
        label_cf = _LEXICAL_ALIASES.get(label_cf, label_cf)
        stripped = canonical_component(AuditableWikidataNormalizer._normalize_query(component_cf))
        if label_cf == component_cf or (stripped and label_cf == stripped):
            return 2
        if (component_cf in label_cf or label_cf in component_cf or
                (stripped and (stripped in label_cf or label_cf in stripped))):
            return 1
        return 0

    def _context_search(self, component: str, context: str) -> list[dict[str, object]]:
        """Search a name with explicit context without changing the extracted name."""
        from . import wikidata as base
        response = self._session().get(
            base.WIKIDATA_API_URL,
            params={"action": "wbsearchentities", "search": f"{component}, {context}",
                    "language": base.WIKIDATA_LANGUAGE, "format": "json",
                    "limit": base.WIKIDATA_SEARCH_LIMIT}, timeout=60,
        )
        response.raise_for_status()
        candidates = []
        results = response.json().get("search", [])
        context_terms = [
            token for token in re.findall(r"[\w'-]+", context.casefold())
            if token not in {"the", "of", "province", "region", "state", "country"}
        ]
        context_evidenced = any(
            any(term in " ".join((str(result.get("label", "")),
                                   str(result.get("description", "")))).casefold()
                for term in context_terms)
            for result in results
        ) if context_terms else bool(results)
        if not results or not context_evidenced:
            # Entity-name search does not generally search descriptions. Text
            # search can recover a venue whose city occurs only in its
            # description (e.g. U.S. Naval Hospital in Washington, D.C.) and a
            # context-supported longer administrative name (e.g. Fengyang
            # County for the extracted component Fengyang in Anhui).
            fallback = self._session().get(
                base.WIKIDATA_API_URL,
                params={"action": "query", "list": "search",
                        "srsearch": f"{component} {context}", "srnamespace": 0,
                        "srlimit": base.WIKIDATA_SEARCH_LIMIT, "format": "json"},
                timeout=60,
            )
            fallback.raise_for_status()
            ids = [row["title"] for row in fallback.json().get("query", {}).get("search", [])
                   if re.fullmatch(r"Q\d+", row["title"])]
            if ids:
                entities_response = self._session().get(
                    base.WIKIDATA_API_URL,
                    params={"action": "wbgetentities", "ids": "|".join(ids),
                            "props": "labels|descriptions", "languages": "en", "format": "json"},
                    timeout=60,
                )
                entities_response.raise_for_status()
                entities = entities_response.json().get("entities", {})
                fallback_results = [
                    {"id": qid, "label": entities[qid]["labels"]["en"]["value"],
                     "description": entities[qid].get("descriptions", {}).get("en", {}).get("value")}
                    for qid in ids if qid in entities and "en" in entities[qid].get("labels", {})
                ]
                seen = {str(result["id"]) for result in results}
                results.extend(result for result in fallback_results
                               if str(result["id"]) not in seen)
        for result in results:
            description = result.get("description")
            historical = self._historical_candidate(result["label"], description)
            level = self._level_from_description(description)
            if level is None and historical:
                level = LocationLevel.COUNTRY
            if level is not None:
                candidates.append({"qid": result["id"], "label": result["label"],
                                   "matched_name": result.get("match", {}).get("text"),
                                   "level": level, "historical": historical,
                                   "administrative_region_role": self._has_administrative_region_role(description)})
        return list({str(candidate["qid"]): candidate for candidate in candidates}.values())

    def _search_place_candidates(self, component: str) -> list[dict[str, object]]:
        """Search deeper for ambiguous venue names without expanding every query."""
        from . import wikidata as base
        response = self._session().get(
            base.WIKIDATA_API_URL,
            params={"action": "wbsearchentities", "search": component,
                    "language": base.WIKIDATA_LANGUAGE, "format": "json",
                    "limit": PLACE_SEARCH_LIMIT}, timeout=60,
        )
        response.raise_for_status()
        candidates = []
        for result in response.json().get("search", []):
            description = result.get("description")
            historical = self._historical_candidate(result["label"], description)
            level = self._level_from_description(description)
            if level is not None:
                candidates.append({"qid": result["id"], "label": result["label"],
                                   "matched_name": result.get("match", {}).get("text"),
                                   "level": level, "historical": historical,
                                   "administrative_region_role": self._has_administrative_region_role(description)})
        return candidates

    def _options(self, components: Sequence[str], *, answer: str | None = None) -> dict[int, list[_Option]]:
        alias_targets = source_alias_targets(components, answer)
        independent: dict[int, list[dict[str, object]]] = {}
        for index, component in enumerate(components):
            context = contextual_component(component)
            lookup = context[1] if context else component
            directional = _DIRECTIONAL_AREA.fullmatch(lookup)
            if directional:
                lookup = directional.group("name")
            independent[index] = ([] if is_generic_place(component)
                                  else self._search(canonical_component(lookup)))
            independent[index] = [dict(candidate, _context_supported=False)
                                  for candidate in independent[index]]
            # Venue names are unusually ambiguous.  Search deeper only when the
            # text itself looks like a place or the ordinary search already
            # surfaced a PLACE candidate.
            if (not is_generic_place(component)
                    and (is_place_expression(component) or any(
                        candidate["level"] is LocationLevel.PLACE
                        for candidate in independent[index]
                    ))):
                independent[index].extend(
                    dict(candidate, _context_supported=False)
                    for candidate in self._search_place_candidates(canonical_component(lookup))
                )
        # The answer itself explicitly identifies these names as aliases. Reuse
        # the target's candidates instead of independently resolving a generic
        # alias such as "Golden House" to an unrelated hotel.  If only the
        # modern alias resolves (for example a garbled former hospital name
        # followed by "now known as ..."), retain that evidence rather than
        # replacing it with the target's empty candidate list.
        for alias_index, target_index in alias_targets.items():
            if independent[target_index]:
                independent[alias_index] = list(independent[target_index])
        broad_context = []
        for index, candidates in independent.items():
            if any(candidate["level"] in {LocationLevel.SUBCITY, LocationLevel.CITY,
                                           LocationLevel.REGION, LocationLevel.COUNTRY}
                   and self._lexical_score(components[index], str(candidate["label"])) == 2
                   for candidate in candidates):
                broad_context.append((index, components[index]))
        corroborated_qids = {
            str(candidate["qid"])
            for index, candidates in independent.items()
            for candidate in candidates
            if self._lexical_score(components[index], str(candidate["label"])) == 2
        }
        options: dict[int, list[_Option]] = {}
        for index, component in enumerate(components):
            if is_generic_place(component):
                options[index] = []
                continue
            candidates = list(independent[index])
            context_component = contextual_component(component)
            lookup = context_component[1] if context_component else component
            directional = _DIRECTIONAL_AREA.fullmatch(lookup)
            if directional:
                lookup = directional.group("name")
            for context_index, context in broad_context[:3]:
                if context_index != index:
                    candidates.extend(
                        dict(candidate, _context_supported=True)
                        for candidate in self._context_search(lookup, context)
                    )
            deduplicated = {}
            for rank, candidate in enumerate(candidates):
                label_lexical = self._lexical_score(lookup, str(candidate["label"]))
                lexical = label_lexical
                matched_name = candidate.get("matched_name")
                if matched_name and self._lexical_score(lookup, str(matched_name)) == 2:
                    lexical = 2
                # Search may qualify a settlement alias with its country:
                # "Frankfurt am Main, Germany" is still an exact city-name
                # match, not less exact than a cemetery's misleading match.
                qualified_match = str(matched_name or "").split(",", 1)
                if (candidate["level"] is LocationLevel.CITY and len(qualified_match) == 2
                        and label_lexical > 0
                        and self._lexical_score(lookup, qualified_match[0]) == 2
                        and any(row["level"] is LocationLevel.COUNTRY
                                and not row.get("historical")
                                and self._lexical_score(qualified_match[1].strip(), str(row["label"])) == 2
                                for rows in independent.values() for row in rows)):
                    lexical = 2
                if index in alias_targets:
                    lexical = 2
                # A context search hit must match this component. Otherwise a
                # country retrieved for a city can masquerade as the city's alias.
                independent_alias = (
                    str(candidate["qid"]) in {str(row["qid"]) for row in independent[index]}
                    and str(candidate["qid"]) in corroborated_qids
                )
                if not lexical and not independent_alias:
                    continue
                # Literal suffixes select types; they cannot turn a service or
                # station into a department or settlement.
                level = candidate["level"]
                hint = source_component_hint(component, answer)
                # The extracted component is primary. Its own suffix can repair
                # a non-PLACE type. A directly attached source type is still a
                # valid constraint, but source_component_hint must first prove
                # that the wording belongs to this exact component.
                expected = explicit_component_level(lookup) or hint.get("level")
                if expected is not None and level != expected:
                    # A literal province suffix can repair a geographic
                    # description that mentions its capital. Never use it to
                    # retype a venue/institution, nor to override a source hint.
                    if (hint or level is LocationLevel.PLACE):
                        continue
                    if explicit_component_level(lookup) is not None:
                        level = expected
                if hint.get("historical") and not candidate.get("historical"):
                    continue
                # A bare settlement name is not evidence for a longer venue
                # name. Explicit hospital/address names can still be fuzzy.
                if (level is LocationLevel.PLACE and lexical < 2
                        and not is_place_expression(component)
                        and not re.match(r"^st\.\s", component, re.I)):
                    continue
                qid = str(candidate["qid"])
                option = _Option(
                    index, component, str(candidate["qid"]), str(candidate["label"]),
                    level, bool(candidate.get("historical")), rank, lexical,
                    bool(candidate.get("administrative_region_role")),
                    bool(candidate.get("_context_supported")),
                )
                previous = deduplicated.get(qid)
                if previous is None:
                    deduplicated[qid] = option
                elif option.context_supported and not previous.context_supported:
                    # The same QID may occur in the ordinary search before a
                    # context query rediscovers it. Preserve the better search
                    # rank, but not at the cost of discarding corroboration by
                    # another extracted component. Otherwise a context-correct
                    # longer name such as Fengyang County is removed merely
                    # because an exact wrong-city homonym also exists.
                    deduplicated[qid] = _Option(
                        previous.component_index, previous.component,
                        previous.qid, previous.label, previous.level,
                        previous.historical, previous.search_rank,
                        previous.lexical_score,
                        previous.administrative_region_role,
                        context_supported=True,
                    )
            retained = list(deduplicated.values())
            # A fuzzy independent hit cannot displace an exact written name.
            # A longer CONTEXT-search match remains eligible because the other
            # extracted components directly support it (Fengyang County in
            # Anhui versus an exact-label Fengyang homonym in Guangxi).
            if (not is_place_expression(component)
                    and any(option.lexical_score == 2 for option in retained)):
                retained = [option for option in retained
                            if option.lexical_score == 2 or option.context_supported]
            options[index] = retained
        return options

    @classmethod
    def _metadata_from_bindings(cls, rows: Sequence[dict]) -> dict[str, object]:
        """Complete hierarchy from direct facts and typed administrative ancestors."""
        from . import wikidata as base
        cities, subcities, regions, countries, continents = set(), set(), set(), set(), set()
        direct_countries, direct_continents, direct_country_continents = set(), set(), set()
        physical_continents = set()
        # The query returns the head and its ancestors, not descendants. A
        # distinct city in this chain supports a dual-role administrative parent
        # as REGION. Standalone Hong Kong has no such child in its chain.
        has_child_city = any(
            cls._level_from_description(row.get("adminDescription", {}).get("value"))
            is LocationLevel.CITY
            and not cls._has_administrative_region_role(
                row.get("adminDescription", {}).get("value")
            )
            for row in rows
        )
        for row in rows:
            label = row.get("adminLabel", {}).get("value")
            description = row.get("adminDescription", {}).get("value")
            # Ottoman administrative units are historical even when terse
            # Wikidata descriptions omit "former" or an explicit end date.
            historical = (cls._historical_description(description)
                          or bool(re.search(r"\b(?:eyalet|vilayet)\b", label or "", re.I)))
            if label and re.fullmatch(r"Q\d+", label):
                label = None  # IDs are identities, never geographic labels.
            level = None if historical else cls._level_from_description(description)
            if (level is LocationLevel.CITY and has_child_city
                    and cls._has_administrative_region_role(description)):
                level = LocationLevel.REGION
            if label and level is LocationLevel.CITY:
                cities.add(label)
            elif label and level is LocationLevel.SUBCITY:
                subcities.add(label)
            elif label and level is LocationLevel.REGION:
                regions.add(label)
            elif label and level is LocationLevel.COUNTRY:
                # Some administrative entities, including Styria, reach their
                # country through P131 rather than carrying a direct P17 claim.
                countries.add(label)
            country_description = row.get("countryDescription", {}).get("value", "")
            direct_country_description = row.get("directCountryDescription", {}).get("value", "")
            if (value := row.get("countryLabel", {}).get("value")) and not (
                historical or cls._historical_description(country_description)
            ):
                countries.add(value)
            if value := row.get("continentLabel", {}).get("value"):
                if not re.fullmatch(r"Q\d+", value) and not historical:
                    physical_continents.add(value)
            if value := row.get("countryContinentLabel", {}).get("value"):
                continents.add(value)
            if (value := row.get("directCountryLabel", {}).get("value")) and not (
                cls._historical_description(direct_country_description)
            ):
                direct_countries.add(value)
            if value := row.get("directContinentLabel", {}).get("value"):
                direct_continents.add(value)
            if value := row.get("directCountryContinentLabel", {}).get("value"):
                direct_country_continents.add(value)

        def unique_or_none(values: set[str]):
            return next(iter(values)) if len(values) == 1 else None

        return {"city": unique_or_none(cities),
                "subcity": sorted(subcities) or None,
                "region": sorted(regions) or None,
                "country": unique_or_none(direct_countries) or unique_or_none(countries),
                "country_candidates": sorted(direct_countries or countries),
                # Physical facts from the head/admin chain outrank the
                # sovereign country's continent (overseas territories). If
                # physical evidence is conflicting, do not guess a continent.
                "continent": (unique_or_none(direct_continents) if direct_continents
                              else unique_or_none(physical_continents) if physical_continents
                              else unique_or_none(direct_country_continents)
                              or unique_or_none(continents))}

    def _metadata(self, qid: str) -> dict[str, object]:
        """Prefer direct country/continent facts; never choose alphabetically."""
        from . import wikidata as base
        query = f"""
        SELECT ?admin ?adminLabel ?adminDescription ?country ?countryLabel
               ?countryDescription ?directCountryDescription
               ?continent ?continentLabel ?countryContinent ?countryContinentLabel
               ?directCountry ?directCountryLabel
               ?directCountryContinent ?directCountryContinentLabel
               ?directContinent ?directContinentLabel WHERE {{
          wd:{qid} (wdt:P131|wdt:P276)* ?admin .
          OPTIONAL {{ ?admin schema:description ?adminDescription . FILTER(LANG(?adminDescription) = "en") }}
          OPTIONAL {{
            ?admin p:P17 ?countryStatement .
            ?countryStatement a wikibase:BestRank .
            ?countryStatement ps:P17 ?country .
            OPTIONAL {{ ?country schema:description ?countryDescription . FILTER(LANG(?countryDescription) = "en") }}
            FILTER NOT EXISTS {{ ?countryStatement pq:P582 ?countryEnd . }}
            OPTIONAL {{ ?country wdt:P30 ?countryContinent . }}
          }}
          OPTIONAL {{ ?admin wdt:P30 ?continent . }}
          OPTIONAL {{
            wd:{qid} p:P17 ?directCountryStatement .
            ?directCountryStatement a wikibase:BestRank .
            ?directCountryStatement ps:P17 ?directCountry .
            OPTIONAL {{ ?directCountry schema:description ?directCountryDescription . FILTER(LANG(?directCountryDescription) = "en") }}
            FILTER NOT EXISTS {{ ?directCountryStatement pq:P582 ?directCountryEnd . }}
            OPTIONAL {{ ?directCountry wdt:P30 ?directCountryContinent . }}
          }}
          OPTIONAL {{ wd:{qid} wdt:P30 ?directContinent . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        response = self._session().get(base.WIKIDATA_SPARQL_URL,
                                       params={"query": query, "format": "json"}, timeout=120)
        response.raise_for_status()
        return self._metadata_from_bindings(response.json()["results"]["bindings"])

    def resolve(self, components: Sequence[str], *, answer: str | None = None) -> ComponentResolution:
        raw = [" ".join(component.split()).strip(" ,") for component in components]
        if not raw or any(not component for component in raw):
            raise ValueError("normalization requires nonblank extraction components")
        options = self._options(raw, answer=answer)
        alias_targets = source_alias_targets(raw, answer)

        # PLACE is expressed-answer evidence, not a member of the administrative
        # CITY -> REGION -> COUNTRY -> CONTINENT chain.  A bare name counts as a
        # place only when its best lexical Wikidata candidates are all PLACE.
        place_options: dict[int, list[_Option]] = {}
        candidate_place_options: dict[int, list[_Option]] = {}
        place_indices: set[int] = set()
        for index, values in options.items():
            typed_places = [option for option in values if option.level is LocationLevel.PLACE]
            candidate_place_options[index] = typed_places
            if is_place_expression(raw[index]) or is_generic_place(raw[index]):
                place_indices.add(index)
                place_options[index] = typed_places
                continue
            if values:
                best_lexical = max(option.lexical_score for option in values)
                best = [option for option in values if option.lexical_score == best_lexical]
                if best and all(option.level is LocationLevel.PLACE for option in best):
                    place_indices.add(index)
                    place_options[index] = [option for option in best if option.level is LocationLevel.PLACE]

        hierarchy_indices = {index for index, component in enumerate(raw)
                             if contextual_component(component) is None
                             and index not in place_indices}
        all_options = [option for index, values in options.items() if index in hierarchy_indices
                       for option in values if option.level is not LocationLevel.PLACE
                       and not option.historical]

        ancestor_cache = getattr(self, "_audit_ancestor_cache", None)
        if ancestor_cache is None:
            ancestor_cache = {}
            self._audit_ancestor_cache = ancestor_cache
        metadata_cache = getattr(self, "_audit_metadata_cache", None)
        if metadata_cache is None:
            metadata_cache = {}
            self._audit_metadata_cache = metadata_cache

        def ancestors(qid: str) -> set[str]:
            if qid not in ancestor_cache:
                ancestor_cache[qid] = self._ancestor_qids(qid)
            return ancestor_cache[qid]

        def metadata(qid: str) -> dict[str, object]:
            if qid not in metadata_cache:
                metadata_cache[qid] = self._metadata(qid)
            result = metadata_cache[qid]
            # Normalize a physical subcontinental label only in derived values;
            # retain the original primitive Wikidata response unchanged.
            if result.get("continent") == "Insular Oceania":
                return {**result, "continent": "Oceania"}
            return result

        def place_decision(index: int, hierarchy_head_qid: str | None,
                           geography: dict[str, object]) -> tuple[str, _Option | None, str]:
            component = raw[index]
            if is_generic_place(component):
                return ("generic", None,
                        "generic PLACE evidence; no Wikidata entity is expected")
            candidates = place_options.get(index, [])
            if not candidates:
                return ("explicit", None,
                        "explicit venue/address evidence; no Wikidata PLACE entity was selected")

            ranked = []
            for option in candidates:
                option_ancestors = ancestors(option.qid)
                compatible = int(
                    hierarchy_head_qid is not None
                    and hierarchy_head_qid in option_ancestors | {option.qid}
                )
                ranked.append(((compatible, option.lexical_score, -option.search_rank), option))
            _, chosen = max(ranked, key=lambda row: row[0])
            if hierarchy_head_qid is not None and max(row[0][0] for row in ranked) > 0:
                return ("wikidata_compatible", chosen,
                        "Wikidata PLACE entity contains the selected geographic head")

            # Only positive contradictory evidence is a conflict.  Missing or
            # incomplete hierarchy links remain recognized-but-unverified.
            comparable = []
            for _, option in ranked:
                option_metadata = metadata(option.qid)
                contradictions = [
                    field for field in ("city", "country", "continent")
                    if geography.get(field) is not None and option_metadata.get(field) is not None
                    and geography[field] != option_metadata[field]
                ]
                has_comparison = any(
                    geography.get(field) is not None and option_metadata.get(field) is not None
                    for field in ("city", "country", "continent")
                )
                if has_comparison:
                    comparable.append((option, contradictions))
            if comparable and all(contradictions for _, contradictions in comparable):
                _, contradictions = max(
                    comparable,
                    key=lambda row: (row[0].lexical_score, -row[0].search_rank),
                )
                # PLACE is literal answer evidence outside the administrative
                # hierarchy. Search hits in incompatible geography are rejected
                # as homonyms; their QIDs remain auditable, but none is selected
                # and the coherent written hierarchy is not penalized.
                return (
                    "explicit", None,
                    "explicit PLACE retained; incompatible Wikidata candidates "
                    "rejected on " + ", ".join(contradictions),
                )
            return ("wikidata_recognized", chosen,
                    "Wikidata identifies a PLACE, but compatibility is not verifiable")

        def place_trace(index: int, hierarchy_head_qid: str | None,
                        geography: dict[str, object]) -> ComponentTrace:
            place_case, option, note = place_decision(index, hierarchy_head_qid, geography)
            selected = None if option is None else ResolvedEntity(
                component=raw[index], qid=option.qid, label=option.label,
                level=LocationLevel.PLACE, role="place", historical=option.historical,
            )
            return ComponentTrace(
                component=raw[index], status="place_expression", selected=selected,
                candidate_qids=list(dict.fromkeys(
                    option.qid for option in place_options.get(index, [])
                )), place_case=place_case, note=note,
            )

        def as_normalized(value: LocationValue) -> NormalizedLocation:
            return NormalizedLocation(
                level=value.level, explicit_levels=[value.level],
                place=value.place, city=value.city,
                subcity=list(value.subcity) if value.subcity else None,
                region=list(value.region) if value.region else None,
                country=value.country, continent=value.continent,
            )

        def literal_option_value(option: _Option) -> LocationValue:
            """Represent exactly the named entity without modern completion."""
            field = LOCATION_FIELDS[option.level]
            raw_value: object = (
                (option.label,) if field in {"subcity", "region"} else option.label
            )
            return LocationValue(level=option.level, **{field: raw_value})

        def option_value(option: _Option) -> LocationValue | None:
            """Represent one candidate branch without inventing finer fields."""
            option_metadata = metadata(option.qid)
            fields: dict[str, object] = {}
            required = LOCATION_FIELDS[option.level]
            if required in {"region", "subcity"}:
                aliases = [option.label]
                aliases.extend(option_metadata.get(required) or [])
                fields[required] = tuple(dict.fromkeys(aliases))
            else:
                fields[required] = option.label
            for field in LOCATION_CONTEXT_FIELDS[option.level]:
                if field == required or option_metadata.get(field) is None:
                    continue
                value = option_metadata[field]
                fields[field] = tuple(value) if field in {"region", "subcity"} else value
            # A transnational physical region is not a contradictory country.
            # Explicit country context can select one supported portion, but
            # cannot prove city containment or invent an unsupported country.
            if option.level is LocationLevel.REGION and fields.get("country") is None:
                stated_countries = {candidate.label for values in options.values()
                                    for candidate in values
                                    if candidate.level is LocationLevel.COUNTRY
                                    and not candidate.historical and candidate.lexical_score == 2}
                country_candidates = set(option_metadata.get("country_candidates") or [])
                supported = (stated_countries & country_candidates
                             if country_candidates else stated_countries)
                if len(supported) == 1:
                    fields["country"] = next(iter(supported))
            try:
                return LocationValue(level=option.level, **fields)
            except ValueError:
                return None

        def enrich_traces(traces: list[ComponentTrace]) -> list[ComponentTrace]:
            """Expose candidate identity and every plausible hierarchy branch.

            Search results remain visible even when they are only fuzzy matches.
            Completed hierarchies are limited to the best lexical candidates plus
            candidates actually used by the primary or consensus decisions.
            Historical entities are shown, but deliberately have no present-day
            hierarchy value.
            """
            enriched = []
            for index, trace in enumerate(traces):
                candidates = [
                    CandidateEntity(
                        qid=option.qid, label=option.label, level=option.level,
                        historical=option.historical,
                        administrative_region_role=option.administrative_region_role,
                        lexical_score=option.lexical_score,
                        search_rank=option.search_rank,
                    )
                    for option in options[index]
                ]
                # Every retained candidate receives a hierarchy. Selection is a
                # later joint decision; hiding lower-ranked candidates here
                # would make search-rank fallback impossible to audit.
                plausible = list(options[index])

                hierarchies = []
                for option in plausible:
                    candidate_value = None if option.historical else option_value(option)
                    hierarchies.append(CandidateHierarchy(
                        qid=option.qid,
                        label=option.label,
                        level=option.level,
                        historical=option.historical,
                        value=(as_normalized(candidate_value)
                               if candidate_value is not None else None),
                        used_for_primary=(
                            trace.selected is not None and trace.selected.qid == option.qid
                        ),
                        used_for_consensus=option.qid in trace.consensus_candidate_qids,
                        note=(
                            "selected as historical-only fallback; no present-day hierarchy completion"
                            if option.historical and trace.selected is not None
                            and trace.selected.qid == option.qid else
                            "historical context; excluded from present-day hierarchy"
                            if option.historical else
                            "candidate could not be completed into a valid hierarchy"
                            if candidate_value is None else
                            "country constrained by explicit answer context; Wikidata country is not unique"
                            if candidate_value.country and not metadata(option.qid).get("country")
                            else None
                        ),
                    ))
                hint = source_component_hint(raw[index], answer)
                if index in alias_targets:
                    hint = {**hint, "alias_of": raw[alias_targets[index]],
                            "evidence": "explicit alias relation in source answer"}
                enriched.append(trace.model_copy(update={
                    "candidates": candidates,
                    "candidate_hierarchies": hierarchies,
                    "source_hint": hint,
                }))
            return enriched

        if not all_options:
            if not place_indices:
                historical_options = [
                    option for values in options.values() for option in values
                    if option.historical
                ]
                if historical_options:
                    chosen = max(
                        historical_options,
                        key=lambda option: (
                            LOCATION_PRECISION[option.level], option.lexical_score,
                            -option.search_rank, -option.component_index,
                        ),
                    )
                    same_rank = [
                        option for option in historical_options
                        if option.level is chosen.level
                        and option.lexical_score == chosen.lexical_score
                    ]
                    alternative_qids = list(dict.fromkeys(
                        option.qid for option in same_rank if option.qid != chosen.qid
                    ))
                    value = as_normalized(literal_option_value(chosen))
                    traces = []
                    for index, component in enumerate(raw):
                        historical = [option for option in options[index] if option.historical]
                        modern = [option for option in options[index] if not option.historical]
                        selected_here = next(
                            (option for option in historical if option.qid == chosen.qid), None
                        )
                        if selected_here is not None:
                            traces.append(ComponentTrace(
                                component=component,
                                status="selected",
                                selected=ResolvedEntity(
                                    component=component, qid=chosen.qid,
                                    label=chosen.label, level=chosen.level,
                                    role="head", historical=True,
                                ),
                                candidate_qids=list(dict.fromkeys(
                                    option.qid for option in historical
                                )),
                                consensus_candidate_qids=[chosen.qid],
                                note=("historical-only fallback selected because no "
                                      "present-day location candidate exists"),
                                consensus_note="historical-only fallback retained for consensus",
                            ))
                        elif historical and not modern:
                            traces.append(ComponentTrace(
                                component=component,
                                status="omitted",
                                candidate_qids=list(dict.fromkeys(
                                    option.qid for option in historical
                                )),
                                note="other historical context; excluded from fallback value",
                            ))
                        else:
                            traces.append(ComponentTrace(
                                component=component,
                                status="unresolved",
                                note=("Wikidata search returned no supported present-day "
                                      "or historical location entity"),
                            ))
                    disambiguation = DisambiguationTrace(
                        status=("search_rank_fallback" if alternative_qids else "unique"),
                        ambiguous=bool(alternative_qids),
                        selected_qid=chosen.qid,
                        alternative_qids=alternative_qids,
                        evidence_score=[LOCATION_PRECISION[chosen.level], chosen.lexical_score],
                        note=("historical-only fallback; no present-day hierarchy completion"
                              if not alternative_qids else
                              "historical-only fallback selected by Wikidata search rank"),
                    )
                    return ComponentResolution(
                        status="partial", components=raw, value=value,
                        consensus_value=value, head_qid=chosen.qid,
                        disambiguation=disambiguation,
                        traces=enrich_traces(traces),
                    )
                traces = []
                for index, component in enumerate(raw):
                    historical = [option for option in options[index] if option.historical]
                    modern = [option for option in options[index] if not option.historical]
                    if historical and not modern:
                        traces.append(ComponentTrace(
                            component=component,
                            status="omitted",
                            candidate_qids=list(dict.fromkeys(
                                option.qid for option in historical
                            )),
                            note="historical context; excluded from present-day hierarchy",
                        ))
                    else:
                        traces.append(ComponentTrace(
                            component=component,
                            status="unresolved",
                            note="Wikidata search returned no supported present-day location entity",
                        ))
                return ComponentResolution(status="unresolved", components=raw, traces=enrich_traces(traces))

            ordered_places = [raw[index] for index in sorted(place_indices)]
            named = [index for index in place_indices if not is_generic_place(raw[index])]
            chosen_place_index = (named or sorted(place_indices))[0]
            fields: dict[str, object] = {"place": ordered_places}
            chosen_case, chosen_option, _ = place_decision(chosen_place_index, None, fields)
            if chosen_option is not None and chosen_case != "wikidata_conflict":
                option_metadata = metadata(chosen_option.qid)
                for field in ("subcity", "city", "region", "country", "continent"):
                    if option_metadata.get(field) is not None:
                        fields[field] = option_metadata[field]
            value = NormalizedLocation(
                level=LocationLevel.PLACE, explicit_levels=[LocationLevel.PLACE], **fields,
            )
            traces = [
                place_trace(index, None, fields) if index in place_indices
                else ComponentTrace(component=component, status="unresolved")
                for index, component in enumerate(raw)
            ]
            unsafe = any(trace.status == "unresolved"
                         or trace.place_case == "wikidata_conflict" for trace in traces)
            return ComponentResolution(
                status="partial" if unsafe else "normalized", components=raw,
                value=value, consensus_value=None if unsafe else value,
                traces=enrich_traces(traces),
            )

        def supporting(head: _Option) -> list[_Option]:
            head_ancestors = ancestors(head.qid)
            selected = []
            for index, values in options.items():
                if index not in hierarchy_indices:
                    continue
                compatible = [option for option in values
                              if option.level is not LocationLevel.PLACE
                              and not option.historical
                              and (
                                  option.qid == head.qid or option.qid in head_ancestors
                              )]
                if compatible:
                    compatible.sort(key=lambda option: (
                        -option.lexical_score,
                        -self._lexical_score(option.component, option.label),
                        0 if option.qid == head.qid else 1,
                        -LOCATION_PRECISION[option.level], option.search_rank,
                    ))
                    selected.append(compatible[0])
            return selected

        # Score complete directed hierarchy branches, not individual components.
        # Search rank is deliberately excluded from evidence_score: it is only
        # the deterministic prior used when equally supported branches remain.
        branches = {}
        for head in all_options:
            selected = supporting(head)
            selected_indices = {option.component_index for option in selected}
            venue_support = []
            named_venue_support = []
            for index in (hierarchy_indices - selected_indices) | place_indices:
                values = options[index]
                best_lexical = max((option.lexical_score for option in values), default=0)
                compatible_venues = [option for option in values
                                     if option.level is LocationLevel.PLACE
                                     and option.lexical_score == best_lexical
                                     and head.qid in ancestors(option.qid) | {option.qid}]
                if compatible_venues:
                    supported = max(compatible_venues, key=lambda option: (
                        option.lexical_score, -option.search_rank))
                    (named_venue_support if index in place_indices else venue_support).append(supported)
            alias_convergence = len(selected) - len({
                option.qid for option in selected
            })
            evidence_score = (
                len(selected) + len(venue_support),
                sum(self._lexical_score(option.component, option.label)
                    for option in selected + venue_support),
                alias_convergence,
                sum(option.lexical_score for option in selected + venue_support),
            )
            hides_other_city = any(
                index != head.component_index and index not in selected_indices
                and any(option.level is LocationLevel.CITY and option.lexical_score == 2
                        for option in options[index])
                for index in hierarchy_indices
            )
            prior_score = (
                # A named venue disambiguates equally supported geography;
                # it cannot outweigh stronger separately asserted geography.
                0 if hides_other_city else len(named_venue_support),
                # Equally supported incompatible claims preserve source order;
                # interpretations of the same name follow Wikidata search rank.
                -head.component_index,
                -head.search_rank,
                LOCATION_PRECISION[head.level],
            )
            branch_key = tuple(sorted(
                (option.component_index, option.qid) for option in selected
            ))
            row = (evidence_score, prior_score, head, selected)
            if branch_key not in branches or row[:2] > branches[branch_key][:2]:
                branches[branch_key] = row

        ranked = sorted(branches.values(), key=lambda row: row[:2], reverse=True)
        winning_evidence, _, head, selected = ranked[0]
        tied = [row for row in ranked[1:] if row[0] == winning_evidence
                and row[1][0] == ranked[0][1][0]]
        ambiguous = bool(tied)
        competing_qids = list(dict.fromkeys(row[2].qid for row in tied))
        retained_ambiguity = any(
            len({option.qid for option in values if not option.historical}) > 1
            for index, values in options.items() if index in hierarchy_indices
        )
        disambiguation = DisambiguationTrace(
            status=("search_rank_fallback" if ambiguous else
                    "context_resolved" if retained_ambiguity else "unique"),
            ambiguous=ambiguous,
            selected_qid=head.qid,
            alternative_qids=competing_qids,
            evidence_score=list(winning_evidence),
            note=(
                "equally supported hierarchy branches; selected earliest Wikidata search result"
                if ambiguous else
                "other candidates were rejected by directed hierarchy context"
                if retained_ambiguity else
                "only one retained hierarchy branch"
            ),
        )
        selected_by_index = {option.component_index: option for option in selected}

        # A finer selected component than the provisional head is the actual head.
        head = max(selected, key=lambda option: (
            option.qid == head.qid, option.lexical_score, -option.search_rank,
            LOCATION_PRECISION[option.level], -option.component_index,
        ))
        head_ancestors = ancestors(head.qid)
        selected = []
        for index, values in options.items():
            if index not in hierarchy_indices:
                continue
            compatible = [option for option in values
                          if option.level is not LocationLevel.PLACE
                          and not option.historical
                          and (option.qid == head.qid or option.qid in head_ancestors)]
            if compatible:
                compatible.sort(key=lambda option: (
                    -option.lexical_score,
                    -self._lexical_score(option.component, option.label),
                    0 if option.qid == head.qid else 1,
                    option.search_rank, -LOCATION_PRECISION[option.level],
                ))
                selected.append(compatible[0])
        selected_by_index = {option.component_index: option for option in selected}

        # Resolve mixed same-name PLACE/non-PLACE ambiguity using the selected
        # geographic head.  This admits the Manhattan St. Vincent's entity but
        # does not turn the unrelated South African "Century City" venue into
        # the Los Angeles neighborhood mentioned by the answer.
        for index, values in options.items():
            if index in place_indices or index not in hierarchy_indices or not values:
                continue
            if index in selected_by_index:
                continue
            best_lexical = max(option.lexical_score for option in values)
            compatible_places = [
                option for option in candidate_place_options[index]
                if option.lexical_score == best_lexical
                and head.qid in ancestors(option.qid) | {option.qid}
            ]
            if compatible_places:
                place_indices.add(index)
                place_options[index] = compatible_places
                selected = [option for option in selected if option.component_index != index]
                selected_by_index.pop(index, None)

        explicit_levels: list[LocationLevel] = []
        fields: dict[str, object] = {}
        entities: list[ResolvedEntity] = []
        for option in selected:
            role = "head" if option.component_index == head.component_index else (
                "alias" if option.qid == head.qid else "container"
            )
            effective_level = option.level
            if (option.level is LocationLevel.CITY and option.administrative_region_role
                    and any(child.level is LocationLevel.CITY
                            and child.qid != option.qid
                            and option.qid in ancestors(child.qid)
                            for child in selected)):
                effective_level = LocationLevel.REGION
            entities.append(ResolvedEntity(component=option.component, qid=option.qid,
                                            label=option.label, level=effective_level,
                                            role=role, historical=option.historical))
            if effective_level not in explicit_levels:
                explicit_levels.append(effective_level)
            field = LOCATION_FIELDS[effective_level]
            if field in {"region", "subcity"}:
                fields.setdefault(field, [])
                if option.label not in fields[field]:
                    fields[field].append(option.label)
            else:
                fields.setdefault(field, option.label)

        if place_indices:
            fields["place"] = [raw[index] for index in sorted(place_indices)]
            if LocationLevel.PLACE not in explicit_levels:
                explicit_levels.append(LocationLevel.PLACE)

        head_metadata = metadata(head.qid)
        # A separately searched country alias must not overwrite the country
        # attached to the selected geographic entity (e.g. Leiden's Q55).
        if head.level is not LocationLevel.COUNTRY and head_metadata.get("country"):
            fields["country"] = head_metadata["country"]
            if head_metadata.get("continent"):
                fields["continent"] = head_metadata["continent"]
        # Modern containers can repair stale historical completion on a finer
        # head. They must actually belong to its directed administrative chain.
        for option in sorted(selected, key=lambda item: LOCATION_PRECISION[item.level], reverse=True):
            if option.qid == head.qid:
                continue
            container_metadata = metadata(option.qid)
            if option.level is LocationLevel.COUNTRY:
                # An explicitly selected territory/constituent country can be
                # geographically remote from its completed sovereign country.
                # E.g. French Polynesia -> France, but physically Oceania, not
                # Europe. Ordinary sovereign countries must not overwrite the
                # finer head's physical continent with their own continent.
                sovereign = container_metadata.get("country")
                if not sovereign or canonical_component(str(sovereign)) == canonical_component(option.label):
                    continue
                # Keep the head's country naming/comparison convention intact
                # (Netherlands vs Kingdom of the Netherlands, for example).
                # This repair concerns physical continent, not sovereignty.
                if container_metadata.get("continent"):
                    fields["continent"] = container_metadata["continent"]
                    break
                continue
            if container_metadata.get("country"):
                fields["country"] = container_metadata["country"]
                if container_metadata.get("continent"):
                    fields["continent"] = container_metadata["continent"]
                break
        # Wikidata models constituent countries such as England as entities
        # contained by a sovereign country. Preserve the explicit COUNTRY level
        # in the trace, but use the completed sovereign country for comparison.
        if (head.level is LocationLevel.COUNTRY and not head.historical
                and fields.get("country") and head_metadata.get("country")
                and fields["country"] != head_metadata["country"]):
            fields["country"] = head_metadata["country"]
        for field in LOCATION_CONTEXT_FIELDS[head.level]:
            if field in {"region", "subcity"} and head_metadata.get(field) is not None:
                fields.setdefault(field, [])
                for item in head_metadata[field]:
                    if item not in fields[field]:
                        fields[field].append(item)
            elif fields.get(field) is None and head_metadata.get(field) is not None:
                fields[field] = head_metadata[field]

        # PLACE is outside the administrative precision chain, but a resolved
        # venue can still supply missing hierarchy fields.  Merge those fields
        # only after the venue has been proven to contain the selected head;
        # an unrelated same-name hospital or house must never change geography.
        for index in sorted(place_indices):
            place_case, place_option, _ = place_decision(index, head.qid, fields)
            if place_case != "wikidata_compatible" or place_option is None:
                continue
            place_metadata = metadata(place_option.qid)
            for field in ("subcity", "region"):
                if place_metadata.get(field) is None:
                    continue
                fields.setdefault(field, [])
                for item in place_metadata[field]:
                    if item not in fields[field]:
                        fields[field].append(item)
            for field in ("city", "country", "continent"):
                if fields.get(field) is None and place_metadata.get(field) is not None:
                    fields[field] = place_metadata[field]

        level = max(explicit_levels, key=LOCATION_PRECISION.get)
        # A selected PLACE entity supplies the required place field; otherwise an
        # explicit textual place expression above does.
        if level is LocationLevel.PLACE and not fields.get("place"):
            fields["place"] = next(entity.label for entity in entities
                                    if entity.level is LocationLevel.PLACE)
        ordered_levels = sorted(set(explicit_levels), key=LOCATION_PRECISION.get, reverse=True)
        value = NormalizedLocation(level=level, explicit_levels=ordered_levels, **fields)

        def conflicts_with_selected_hierarchy(candidate: LocationValue | None) -> bool:
            """Reject a homonym only when its broad geography clearly conflicts.

            The extracted components remain the primary evidence.  This check
            is deliberately limited to country and continent: missing metadata
            is not a conflict, and finer administrative differences may be the
            very ambiguity that the other extracted components resolve.
            """
            if candidate is None:
                return False
            selected = value.to_value()
            return any(
                getattr(selected, field) is not None
                and getattr(candidate, field) is not None
                and getattr(selected, field) != getattr(candidate, field)
                for field in ("country", "continent")
            )

        def safely_historical_omission(index: int) -> bool:
            candidates = options[index]
            if not candidates:
                return False
            best_lexical = max(option.lexical_score for option in candidates)
            best = [option for option in candidates if option.lexical_score == best_lexical]
            historical = [option for option in best if option.historical]
            modern = [option for option in best if not option.historical]
            # Selection cannot erase independent evidence against itself. A
            # directly named modern country remains a conflicting assertion,
            # even when search also returns a historical polity alias (Iran /
            # Qajar Iran). Likewise a stronger modern label match cannot be
            # reinterpreted as historical merely because its geography differs.
            historical_label_score = max(
                (self._lexical_score(option.component, option.label)
                 for option in historical), default=-1,
            )
            if any(
                (option.level is LocationLevel.COUNTRY
                 and self._lexical_score(option.component, option.label) == 2)
                or self._lexical_score(option.component, option.label) > historical_label_score
                for option in modern
            ):
                return False
            compatible_modern = [
                option for option in best
                if not option.historical
                and not conflicts_with_selected_hierarchy(option_value(option))
            ]
            # A modern homonym in an incompatible country must not prevent a
            # clearly historical reading from being omitted safely.
            return bool(historical) and not compatible_modern

        traces = []
        entity_by_index = {option.component_index: entity for option, entity in zip(selected, entities)}
        for index, component in enumerate(raw):
            candidate_qids = list(dict.fromkeys(option.qid for option in options[index]))
            context = contextual_component(component)
            if context:
                relation, _ = context
                context_options = options[index]
                if context_options:
                    hierarchy_qids = {entity.qid for entity in entities}
                    ranked_context = []
                    for option in context_options:
                        overlap = len(hierarchy_qids.intersection(ancestors(option.qid) | {option.qid}))
                        # The written context name is primary.  Hierarchy overlap
                        # is only a tie-breaker: otherwise "near Xi'an" can be
                        # resolved back to the already-selected Chang'an through
                        # a corroborated alias candidate.
                        ranked_context.append(((option.lexical_score, overlap,
                                                -option.search_rank), option))
                    option = max(ranked_context, key=lambda row: row[0])[1]
                    context_entity = ResolvedEntity(
                        component=component, qid=option.qid, label=option.label,
                        level=option.level, role="context", historical=option.historical,
                    )
                    traces.append(ComponentTrace(
                        component=component, status="context", selected=context_entity,
                        candidate_qids=candidate_qids, relation=relation,
                        note="resolved proximity/context entity; excluded from containment and granularity",
                    ))
                else:
                    traces.append(ComponentTrace(
                        component=component, status="context", candidate_qids=[],
                        relation=relation, note="context qualifier preserved; entity unresolved",
                    ))
            elif index in place_indices:
                traces.append(place_trace(index, head.qid, fields))
            elif index in entity_by_index:
                traces.append(ComponentTrace(component=component, status="selected",
                                             selected=entity_by_index[index], candidate_qids=candidate_qids,
                                             note=(
                                                 "dual CITY/REGION entity used as the administrative "
                                                 "REGION containing a selected city"
                                                 if entity_by_index[index].level is LocationLevel.REGION
                                                 and selected_by_index[index].level is LocationLevel.CITY
                                                 else None
                                             )))
            elif candidate_qids:
                historical_omission = safely_historical_omission(index)
                has_modern_homonym = any(not option.historical for option in options[index])
                traces.append(ComponentTrace(component=component, status="omitted",
                                             candidate_qids=candidate_qids,
                                             note=("historical context; incompatible modern homonyms ignored"
                                                   if historical_omission and has_modern_homonym
                                                   else "historical context; excluded from present-day hierarchy"
                                                   if historical_omission
                                                   else "no candidate belongs to the selected hierarchy")))
            else:
                traces.append(ComponentTrace(component=component, status="unresolved",
                                             note="Wikidata search returned no supported location entity"))
        unsafe = any(
            trace.status == "unresolved" or trace.place_case == "wikidata_conflict" or (
                trace.status == "omitted"
                and not safely_historical_omission(index)
            )
            for index, trace in enumerate(traces)
        )
        context_precisions = [LOCATION_PRECISION[trace.selected.level] for trace in traces
                              if trace.status == "context" and trace.selected is not None]
        # "Near Xi'an" alone cannot be normalized as Xi'an. A separate head at
        # least as specific as the context entity is required for a full value.
        if context_precisions and LOCATION_PRECISION[level] < max(context_precisions):
            unsafe = True

        consensus_value = value
        if unsafe:
            primary_branch: LocationValue | None = value.to_value()
            # A Wikidata PLACE hit in a contradictory hierarchy does not make
            # the answer's literal venue unsafe to display, but it does mean
            # that venue is not normalized evidence. Conservative consensus
            # therefore starts at the most specific verified administrative
            # level written/completed for the answer (Berlin, France, etc.).
            # Otherwise identical venue strings in incompatible countries can
            # incorrectly survive agreement at PLACE.
            if any(trace.place_case == "wikidata_conflict" for trace in traces):
                primary_branch = next((
                    value.to_value().to_level(candidate_level)
                    for candidate_level in (
                        LocationLevel.SUBCITY, LocationLevel.CITY,
                        LocationLevel.REGION, LocationLevel.COUNTRY,
                        LocationLevel.CONTINENT,
                    )
                    if getattr(value, LOCATION_FIELDS[candidate_level])
                ), None)
            branches: list[LocationValue | None] = [primary_branch]
            consensus_candidates: dict[int, list[str]] = {}
            consensus_notes: dict[int, str] = {}

            for index, trace in enumerate(traces):
                if trace.place_case == "wikidata_conflict":
                    consensus_notes[index] = (
                        "conflicting Wikidata PLACE excluded from consensus; "
                        "verified broader hierarchy retained"
                    )
                elif trace.status == "omitted":
                    candidates = options[index]
                    if candidates:
                        best_lexical = max(option.lexical_score for option in candidates)
                        candidate_pairs = [
                            (option, option_value(option)) for option in candidates
                            if option.lexical_score == best_lexical and not option.historical
                        ]
                        candidate_pairs = [
                            (option, candidate) for option, candidate in candidate_pairs
                            if candidate is not None
                            and not (
                                safely_historical_omission(index)
                                and conflicts_with_selected_hierarchy(candidate)
                            )
                        ]
                        # Resolve this alternative with ITS stated context,
                        # using the same hierarchy evidence/search-rank prior
                        # as the main value. Do not intersect every homonymous
                        # search hit (Paris, France vs Paris, Texas).
                        candidate_qids = {option.qid for option, _ in candidate_pairs}
                        branch = next((row for row in ranked if any(
                            option.component_index == index and option.qid in candidate_qids
                            for option in row[3]
                        )), None)
                        chosen = next((option for option in branch[3]
                                       if option.component_index == index
                                       and option.qid in candidate_qids), None) if branch else None
                        if chosen is None and candidate_pairs:
                            chosen = min((option for option, _ in candidate_pairs),
                                         key=lambda option: option.search_rank)
                        used_pairs = [(option, candidate) for option, candidate in candidate_pairs
                                      if chosen is not None and option.qid == chosen.qid]
                        branches.extend(
                            [candidate for _, candidate in used_pairs]
                        )
                        if used_pairs:
                            consensus_candidates[index] = [
                                option.qid for option, _ in used_pairs
                            ]
                            consensus_notes[index] = (
                                "candidate branch used to derive consensus_value"
                                if len(used_pairs) == 1
                                else "candidate branches used to derive consensus_value"
                            )
                        else:
                            consensus_notes[index] = (
                                "no usable modern entity branch; resolved hierarchy retained"
                            )
                    else:
                        consensus_notes[index] = (
                            "no entity candidate; resolved hierarchy retained"
                        )
                elif trace.status == "unresolved":
                    # Missing resolution is not evidence of a conflicting
                    # location. Keep partial status, without inventing a branch.
                    consensus_notes[index] = (
                        "unresolved component; not evidence of geographic conflict; "
                        "resolved hierarchy retained for consensus"
                    )

            # An unresolved string is missing evidence, not a competing
            # geographic branch. Preserve the complete resolved hierarchy when
            # there is no actual alternative; round-tripping a single REGION
            # value through location_consensus would otherwise discard all but
            # one of its valid nested regions.
            if len(branches) == 1:
                consensus_value = (
                    as_normalized(branches[0]) if branches[0] is not None else None
                )
            else:
                safe = location_consensus(branches, min_share=1.0)
                consensus_value = as_normalized(safe) if safe is not None else None
            traces = [
                trace.model_copy(update={
                    "consensus_candidate_qids": consensus_candidates.get(index, []),
                    "consensus_note": consensus_notes.get(index),
                })
                for index, trace in enumerate(traces)
            ]
        elif ambiguous:
            # Ambiguity is retained for auditing, but the earliest sensible
            # Wikidata result is the operational resolution.  It therefore
            # remains eligible for downstream consensus like any other value.
            consensus_value = value
        traces = enrich_traces(traces)
        return ComponentResolution(status="partial" if unsafe else "normalized",
                                   components=raw, value=value,
                                   consensus_value=consensus_value,
                                   head_qid=head.qid,
                                   disambiguation=disambiguation, traces=traces)

def _cache_identity(components: Sequence[str], *, answer: str | None = None,
                    normalizer_id: str | None = None) -> dict:
    return {"normalizer": normalizer_id or normalizer_identity(),
            "components": [canonical_component(component) for component in components],
            "source_answer_sha256": hashlib.sha256(answer.encode()).hexdigest() if answer else None,
            "wikidata_language": "en",
            "wikidata_search_limit": COMPONENT_SEARCH_LIMIT}

def prepare_resolution_components(components):
    """Drop blank separator slots, not malformed substantive values.

    Shared by the filesystem and SQLite boundaries, including old extraction
    caches. Keep original extractor output in its audit/cache record.
    """
    if isinstance(components, (str, bytes)) or not isinstance(components, (list, tuple)):
        return [], "normalization components must be a list or tuple of strings"
    cleaned = [component.strip() for component in components
               if isinstance(component, str) and component.strip()]
    if any(not isinstance(component, str) for component in components):
        return cleaned, "normalization components contain a non-string value"
    if not cleaned:
        return [], "normalization requires nonblank extraction components"
    return cleaned, None

def split_location_parts(phrase: str, *, separator: str = ",") -> list[str] | None:
    """Split explicit components; never split commas inside semicolon components.

    The comma default keeps archived extraction experiments reproducible.
    New extraction-only datasets explicitly use separator=';'.
    """
    if separator not in {",", ";"}:
        raise ValueError("Location separator must be comma or semicolon")
    parts, start, depth = [], 0, 0
    for index, character in enumerate(phrase):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return None
        elif character == separator and depth == 0:
            parts.append(phrase[start:index])
            start = index + 1
    if depth:
        return None
    return parts + [phrase[start:]]

def normalizer_identity():
    from .versions import resolution_cache_identity
    return resolution_cache_identity()

