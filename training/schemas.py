"""Strict prepared examples: one primary target per answer, including Granola."""
from __future__ import annotations
from enum import StrEnum
from typing import Annotated, Literal
import json
import re
import unicodedata
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from parsing.temporal.values import TemporalLevel
from parsing.location.normalization.values import LocationLevel
from shared.text import _SENTENCE_END, _is_nonterminal_period

IDK_ANSWER = "I don't know."
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]
NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
GranularityLevel = TemporalLevel | LocationLevel

def canonical_answer_key(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()

def has_complete_sentence_ending(text: str) -> bool:
    """Return whether the final non-whitespace character closes a sentence."""
    stripped = text.rstrip()
    if not stripped:
        return False
    for match in _SENTENCE_END.finditer(stripped):
        if match.end() == len(stripped):
            return not _is_nonterminal_period(stripped, match)
    return False

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

class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

class Relation(StrEnum):
    BIRTH = "birth"
    DEATH = "death"
    BIRTHPLACE = "birthplace"
    DEATHPLACE = "deathplace"

class Modality(StrEnum):
    TEMPORAL = "temporal"
    LOCATION = "location"

class AbstentionLevel(StrEnum):
    IDK = "IDK"

CandidateLevel = GranularityLevel | AbstentionLevel

def primary_modality(relation: Relation) -> Modality:
    if relation in {Relation.BIRTH, Relation.DEATH}:
        return Modality.TEMPORAL
    return Modality.LOCATION

def level_type(modality: Modality) -> type[TemporalLevel] | type[LocationLevel]:
    return TemporalLevel if modality is Modality.TEMPORAL else LocationLevel

class Candidate(SchemaModel):
    answer: str = Field(min_length=1)
    score_start: NonNegativeInt
    score_end: PositiveInt
    level: CandidateLevel

    @model_validator(mode="after")
    def validate_score_span(self) -> Candidate:
        if self.score_end <= self.score_start:
            raise ValueError("score_end must be greater than score_start")
        if self.score_end > len(self.answer):
            raise ValueError("score span extends beyond the candidate answer")
        if not self.answer[self.score_start:self.score_end].strip():
            raise ValueError("score span must contain non-whitespace text")
        return self

class ContrastiveGroup(SchemaModel):
    modality: Modality
    target_level: GranularityLevel
    target_index: NonNegativeInt
    candidates: list[Candidate] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_group(self) -> ContrastiveGroup:
        if self.target_index >= len(self.candidates):
            raise ValueError("target_index is outside the candidate list")

        answers = [candidate.answer for candidate in self.candidates]
        if len(answers) != len(set(answers)):
            raise ValueError("candidate answers must be unique within a group")

        expected_type = (
            TemporalLevel
            if self.modality is Modality.TEMPORAL
            else LocationLevel
        )
        if not all(
            isinstance(candidate.level, expected_type)
            or candidate.level is AbstentionLevel.IDK
            for candidate in self.candidates
        ):
            raise ValueError(
                f"{self.modality.value} groups contain an incompatible level"
            )
        if sum(
            candidate.level is AbstentionLevel.IDK
            for candidate in self.candidates
        ) > 1:
            raise ValueError("a contrastive group may contain at most one IDK")
        if not isinstance(self.target_level, expected_type):
            raise ValueError(
                f"target_level is incompatible with {self.modality.value}"
            )
        if self.candidates[self.target_index].level != self.target_level:
            raise ValueError("target_level does not match the target candidate")
        return self

class TrainingExample(SchemaModel):
    case_id: NonBlankString
    qid: NonBlankString
    split: Split
    source_model: NonBlankString
    relation: Relation
    prompt_template_id: NonNegativeInt
    prompt: NonBlankString
    source_answer: str
    positive_answer: str = Field(min_length=1)
    is_idk: bool
    primary_target_level: GranularityLevel | None = None
    secondary_target_level: None = None
    contrastive_groups: list[ContrastiveGroup] = Field(default_factory=list, max_length=1)

    @model_validator(mode="after")
    def validate_example(self) -> TrainingExample:
        if any(group.modality is not primary_modality(self.relation) for group in self.contrastive_groups):
            raise ValueError("Only the primary target may be supervised")
        modalities = [group.modality for group in self.contrastive_groups]
        if len(modalities) != len(set(modalities)):
            raise ValueError("an example may contain at most one group per modality")

        if self.is_idk:
            if self.positive_answer != IDK_ANSWER:
                raise ValueError(f"IDK positive_answer must be {IDK_ANSWER!r}")
            if self.primary_target_level is not None:
                raise ValueError("IDK examples cannot have a primary target level")
            if self.contrastive_groups:
                raise ValueError(
                    "IDK examples cannot contain primary contrastive groups"
                )
        elif self.positive_answer == IDK_ANSWER:
            raise ValueError("a non-IDK example cannot use the IDK answer")
        else:
            primary = primary_modality(self.relation)
            if not isinstance(self.primary_target_level, level_type(primary)):
                raise ValueError(
                    "primary_target_level is missing or incompatible with "
                    f"relation {self.relation.value}"
                )

        for group in self.contrastive_groups:
            target = group.candidates[group.target_index]
            if target.answer != self.positive_answer:
                raise ValueError(
                    "every target candidate must equal positive_answer"
                )
            if group.target_level != self.primary_target_level:
                raise ValueError(
                    "contrastive group target does not match the stored "
                    "positive target level"
                )
        return self

RELATION_FRAMES = {
    "P112": "{entity} was founded by ",
    "P170": "{entity} was created by ",
    "P175": "{entity} was performed by ",
    "P176": "{entity} was produced by ",
    "P40": "{entity} is a parent of ",
    "P69": "{entity} was educated ",
    "P50": "{entity} was written by ",
    "P26": "{entity} was married to ",
    "P127": "{entity} is owned by ",
    "P264": "The record-label affiliation of {entity} is ",
}

_INSTITUTION = re.compile(
    r"\b(?:university|universiteit|college|school|academy|institute|institution|polytechnic)\b",
    re.I,
)

_DEMONYM = r"(?:American|Australian|Austrian|British|Canadian|Chinese|English|Estonian|Filipino|Finnish|French|German|Greek|Indian|Indonesian|Irish|Italian|Japanese|Norwegian|Russian|Scottish|South African|South Korean|Swedish)"

_DESCRIPTOR_HEAD = r"(?:actor|archduke|artist|author|automaker|businessman|businesswoman|company|conglomerate|critic|developer|designer|director|ecologist|editor|entertainer|illustrator|lawyer|manufacturer|monarch|musician|novelist|painter|pastor|physicist|poet|politician|producer|programmer|publisher|rapper|ruler|scholar|screenwriter|singer|writer)"

def _nominal_phrase(text: str) -> str:
    """Lexical determiner rules; never look up factual entity values."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:a|an|the)\s+(?=(?:a|an|the)\s)", "", text, flags=re.I)
    if re.match(r"^(?:a|an|the)\s", text, re.I):
        return text if text.startswith("The Colony ") else text[0].lower() + text[1:]
    if re.match(
        r"^(?:University of\b|State University of\b|United States\b|United Kingdom\b|Government of\b|Archduchess of\b|Archduke of\b|Ottoman Sultan\b|Maharaja of\b)",
        text,
    ):
        return "the " + text
    if re.search(
        r"\b(?:Department of\b|state government$|Ports Authority$|State Highway Administration$|Institute of\b|group$)",
        text,
    ):
        return "the " + text
    descriptor = (
        re.fullmatch(_DEMONYM, text, re.I)
        or (
            re.match(_DEMONYM + r"\s", text, re.I)
            and re.search(r"\b" + _DESCRIPTOR_HEAD + r"\b", text, re.I)
        )
        or re.match(
            r"^(?:university\b|Ivy League (?:school|university)\b|ASX-listed\b|World War II prisoner\b)",
            text,
        )
    )
    if descriptor:
        if re.search(
            r"\b(?:companies|manufacturers|shipbuilders|writers|actors)\b", text, re.I
        ):
            return text
        article = "an" if re.match(r"^[aeiou]", text, re.I) else "a"
        if re.match(r"^(?:university|European)\b", text, re.I):
            article = "a"
        return article + " " + text
    return text

class GranolaRelation(StrEnum):
    FOUNDER = "P112"
    CREATOR = "P170"
    PERFORMER = "P175"
    MANUFACTURER = "P176"
    CHILD = "P40"
    EDUCATED_AT = "P69"
    AUTHOR = "P50"
    SPOUSE = "P26"
    OWNER = "P127"
    RECORD_LABEL = "P264"

class GranolaModality(StrEnum):
    HIERARCHY = "granola"

class GranolaLevel(StrEnum):
    L1 = "GRANOLA_1"
    L2 = "GRANOLA_2"
    L2_5 = "GRANOLA_2_5"
    L3 = "GRANOLA_3"
    L4 = "GRANOLA_4"

    @classmethod
    def from_number(cls, value) -> GranolaLevel:
        if isinstance(value, bool):
            raise ValueError("boolean is not a Granola level")
        levels = {1.0: cls.L1, 2.0: cls.L2, 2.5: cls.L2_5, 3.0: cls.L3, 4.0: cls.L4}
        try:
            return levels[float(value)]
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"invalid Granola level: {value!r}") from error

GRANOLA_PRECISION = {
    GranolaLevel.L1: 5,
    GranolaLevel.L2: 4,
    GranolaLevel.L2_5: 3,
    GranolaLevel.L3: 2,
    GranolaLevel.L4: 1,
}

def granola_level_key(relation: str, level: str) -> str:
    return f"{relation}_{level}"

class GranolaCandidate(Candidate):
    level: GranolaLevel | AbstentionLevel

class GranolaGroup(SchemaModel):
    modality: GranolaModality = GranolaModality.HIERARCHY
    target_level: GranolaLevel
    target_index: NonNegativeInt
    candidates: list[GranolaCandidate] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_group(self) -> GranolaGroup:
        if self.target_index >= len(self.candidates):
            raise ValueError("target_index is outside the candidate list")
        target = self.candidates[self.target_index]
        if target.level != self.target_level:
            raise ValueError("target_level does not match target candidate")
        answers = [
            canonical_answer_key(candidate.answer) for candidate in self.candidates
        ]
        if len(set(answers)) != len(answers):
            raise ValueError("candidate answers must be canonically unique")
        if len({candidate.level for candidate in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("Granola requires one realization per candidate level")
        if any(
            not isinstance(candidate.level, GranolaLevel)
            and candidate.level is not AbstentionLevel.IDK
            for candidate in self.candidates
        ):
            raise ValueError("Granola group contains an incompatible level")
        for candidate in self.candidates:
            if candidate.level is AbstentionLevel.IDK:
                if (
                    candidate.answer != IDK_ANSWER
                    or candidate.score_start != 0
                    or candidate.score_end != len(IDK_ANSWER)
                ):
                    raise ValueError("Granola IDK candidate must score the full answer")
                continue
            if (
                candidate.answer[: candidate.score_start]
                != target.answer[: target.score_start]
                or candidate.answer[candidate.score_end :]
                != target.answer[target.score_end :]
            ):
                raise ValueError("candidate changes text outside the score span")
            if not has_complete_sentence_ending(candidate.answer):
                raise ValueError("candidate is not a complete sentence")
        return self

def realize_candidate(
    entity: str, relation: str, level: GranolaLevel, text: str
) -> GranolaCandidate:
    prefix = RELATION_FRAMES[relation].format(entity=entity)
    phrase = _nominal_phrase(text).rstrip(".!?").strip()
    if not phrase:
        raise ValueError("empty_candidate_phrase")
    if relation == "P69":
        institution = bool(_INSTITUTION.search(text)) or level is GranolaLevel.L1
        phrase = ("at " if institution else "in ") + phrase
    return GranolaCandidate(
        answer=prefix + phrase + ".",
        score_start=len(prefix),
        score_end=len(prefix) + len(phrase),
        level=level,
    )

class GranolaTrainingExample(SchemaModel):
    data_source: Literal["granola"] = "granola"
    case_id: NonBlankString
    qid: str = Field(pattern=r"^Q[0-9]+$")
    entity: NonBlankString
    original_row_id: NonBlankString
    split: Split
    source_model: NonBlankString
    relation: GranolaRelation
    prompt_template_id: NonNegativeInt = 0
    prompt: NonBlankString
    # Unmodified model-selected phrase, not an old generated sentence.
    source_answer: NonBlankString
    positive_answer: str = Field(min_length=1)
    is_idk: Literal[False] = False
    primary_target_level: GranolaLevel
    secondary_target_level: None = None
    contrastive_groups: list[GranolaGroup] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def validate_example(self) -> GranolaTrainingExample:
        group = self.contrastive_groups[0]
        target = group.candidates[group.target_index]
        if (
            target.answer != self.positive_answer
            or group.target_level != self.primary_target_level
        ):
            raise ValueError(
                "Granola target candidate must equal the positive answer and level"
            )
        if (
            self.case_id
            != f"granola:{self.source_model}:{self.qid}:{self.relation.value}:{self.original_row_id}"
        ):
            raise ValueError("Granola case_id disagrees with its provenance")
        expected = realize_candidate(
            self.entity, self.relation.value, target.level, self.source_answer
        )
        if expected != target:
            raise ValueError(
                "target disagrees with its selected phrase or deterministic frame"
            )
        prefix = RELATION_FRAMES[self.relation.value].format(entity=self.entity)
        if any(
            candidate.level is not AbstentionLevel.IDK
            and (
                candidate.answer[: candidate.score_start] != prefix
                or candidate.answer[candidate.score_end :] != "."
            )
            for candidate in group.candidates
        ):
            raise ValueError("candidate does not use the deterministic relation frame")
        return self
