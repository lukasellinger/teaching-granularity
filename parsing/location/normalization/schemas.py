"""Strict normalization outputs and component-level evidence traces."""
from __future__ import annotations
import json
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator
from .values import LocationLevel, LOCATION_PRECISION, LocationValue
NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class SchemaModel(BaseModel):
    """Shared strictness and stable serialization for persisted artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def canonical_json(self) -> str:
        """Serialize deterministically for hashing and regression tests."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

class NormalizedLocation(SchemaModel):
    level: LocationLevel
    explicit_levels: list[LocationLevel] = Field(min_length=1)
    place: list[NonBlankString] | None = None
    subcity: list[NonBlankString] | None = None
    city: NonBlankString | None = None
    region: list[NonBlankString] | None = None
    country: NonBlankString | None = None
    continent: NonBlankString | None = None

    @field_validator("place", mode="before")
    @classmethod
    def accept_legacy_place(cls, value):
        return [value] if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_location(self) -> NormalizedLocation:
        if len(self.explicit_levels) != len(set(self.explicit_levels)):
            raise ValueError("explicit_levels must be unique")
        if self.level not in self.explicit_levels:
            raise ValueError("explicit_levels must contain level")
        if self.level != max(self.explicit_levels, key=LOCATION_PRECISION.get):
            raise ValueError("level must be the most specific explicit level")
        fields = ("place", "subcity", "city", "region", "country", "continent")
        allowed = {
            LocationLevel.PLACE: set(fields),
            LocationLevel.SUBCITY: set(fields[1:]),
            LocationLevel.CITY: set(fields[2:]),
            LocationLevel.REGION: set(fields[3:]),
            LocationLevel.COUNTRY: set(fields[4:]),
            LocationLevel.CONTINENT: set(fields[5:]),
        }[self.level]
        forbidden = [
            field
            for field in fields
            if field not in allowed and getattr(self, field) is not None
        ]
        if forbidden:
            raise ValueError(
                f"fields finer than {self.level.value} are forbidden: {forbidden}"
            )
        self.to_value()
        return self

    def to_value(self) -> LocationValue:
        return LocationValue.from_dict(
            {
                "level": self.level.value,
                "place": self.place,
                "subcity": self.subcity,
                "city": self.city,
                "region": self.region,
                "country": self.country,
                "continent": self.continent,
            }
        )

class ResolvedEntity(SchemaModel):
    component: NonBlankString
    qid: NonBlankString
    label: NonBlankString
    level: LocationLevel
    role: Literal["head", "container", "alias", "context", "place"]
    historical: bool = False

class CandidateEntity(SchemaModel):
    """One retained Wikidata search result, before hierarchy selection."""

    qid: NonBlankString
    label: NonBlankString
    level: LocationLevel
    historical: bool = False
    administrative_region_role: bool = False
    lexical_score: int
    search_rank: int

class CandidateHierarchy(SchemaModel):
    """A plausible interpretation and its completed present-day hierarchy."""

    qid: NonBlankString
    label: NonBlankString
    level: LocationLevel
    historical: bool = False
    value: NormalizedLocation | None = None
    used_for_primary: bool = False
    used_for_consensus: bool = False
    note: str | None = None

class DisambiguationTrace(SchemaModel):
    status: Literal[
        "not_applicable", "unique", "context_resolved", "search_rank_fallback",
    ] = "not_applicable"
    ambiguous: bool = False
    selected_qid: NonBlankString | None = None
    alternative_qids: list[NonBlankString] = Field(default_factory=list)
    evidence_score: list[int] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode="after")
    def valid_ambiguity(self):
        if self.ambiguous != (self.status == "search_rank_fallback"):
            raise ValueError("ambiguous is true exactly for search-rank fallback")
        if self.status == "search_rank_fallback" and not self.alternative_qids:
            raise ValueError("search-rank fallback requires alternatives")
        return self

class ComponentTrace(SchemaModel):
    component: NonBlankString
    status: Literal["selected", "context", "place_expression", "omitted", "unresolved"]
    selected: ResolvedEntity | None = None
    candidate_qids: list[NonBlankString] = Field(default_factory=list)
    candidates: list[CandidateEntity] = Field(default_factory=list)
    candidate_hierarchies: list[CandidateHierarchy] = Field(default_factory=list)
    consensus_candidate_qids: list[NonBlankString] = Field(default_factory=list)
    relation: Literal["near", "outside", "around"] | None = None
    place_case: Literal[
        "generic", "explicit", "wikidata_compatible",
        "wikidata_recognized", "wikidata_conflict",
    ] | None = None
    note: str | None = None
    consensus_note: str | None = None
    source_hint: dict = Field(default_factory=dict)

class ComponentResolution(SchemaModel):
    status: Literal["normalized", "partial", "unresolved", "idk", "failed"]
    components: list[NonBlankString]
    value: NormalizedLocation | None = None
    consensus_value: NormalizedLocation | None = None
    head_qid: str | None = None
    disambiguation: DisambiguationTrace = Field(default_factory=DisambiguationTrace)
    traces: list[ComponentTrace] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def restore_legacy_place_components(cls, payload):
        if not isinstance(payload, dict):
            return payload
        places = list(dict.fromkeys(t["component"] for t in payload.get("traces", [])
                                   if isinstance(t, dict) and t.get("status") == "place_expression"
                                   and t.get("component")))
        if not places:
            return payload
        payload = dict(payload)
        for field in ("value", "consensus_value"):
            value = payload.get(field)
            if isinstance(value, dict) and isinstance(value.get("place"), str):
                payload[field] = {**value, "place": places}
        return payload

    @model_validator(mode="after")
    def valid_status(self):
        if (self.status in {"normalized", "partial"}) != (self.value is not None):
            raise ValueError("only normalized/partial resolutions contain a candidate value")
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("only failed resolutions contain an error")
        if self.status == "idk" and self.components:
            raise ValueError("IDK has no components")
        if self.status not in {"normalized", "partial"} and self.consensus_value is not None:
            raise ValueError("only normalized/partial resolutions contain a consensus value")
        return self
