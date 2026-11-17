"""Primary-only eligibility, temporal edits and aligned location candidates."""

import hashlib
import json
import os
from pathlib import Path
import re

from parsing.location.rewriter.validation import replacement_expression_spans, plain_text
from parsing.temporal import parse_temporal_text, strip_brackets_with_mapping
from parsing.temporal.relations import relation_match_is_relevant
from parsing.temporal.values import FIVE_LEVEL_TEMPORAL_LEVELS
from .temporal_edit import plan_temporal_edit, first_complete_sentence_end

IDK_ANSWER = "I don't know."
PROMPTS = json.loads((Path(__file__).parent / "prompts.json").read_text())


def select_prompt(relation, name, seed, qid, sample_index):
    identity = f"{seed}:{qid}:{relation}:{sample_index}".encode()
    index = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % len(PROMPTS[relation])
    return index, PROMPTS[relation][index].format(person=name)


def balanced_markdown(text):
    return text.count("**") % 2 == 0 and text.count("__") % 2 == 0


def temporal_spans(answer, components):
    searchable = answer.translate(str.maketrans({char: " " for char in "()[]{}*"}))
    for component in components:
        address = re.match(r"^(\d+)\s+[^,;\n]+\b(?:Street|Road|Avenue|Lane|Drive|Way|Boulevard)\b", component, re.IGNORECASE)
        if address:
            for occurrence in re.finditer(re.escape(component), answer, re.IGNORECASE):
                start = occurrence.start()
                searchable = searchable[:start] + " " * len(address[1]) + searchable[start + len(address[1]):]
    parsed = parse_temporal_text(searchable)
    return bool(parsed.matches or parsed.issues)


def example(row, split, name, level, positive, groups, seed, *, idk=False):
    index, prompt = select_prompt(row["source_relation"], name, seed, row["qid"], -1 if idk else row["sample_index"])
    return {"case_id": row["case_id"], "qid": row["qid"], "split": split,
            "source_model": row["source_model"], "relation": row["source_relation"],
            "prompt_template_id": index, "prompt": prompt, "source_answer": "" if idk else row["answer"],
            "positive_answer": positive, "is_idk": idk, "primary_target_level": level,
            "secondary_target_level": None, "contrastive_groups": groups}


def temporal_example(row, common, split, name, seed, include_idk=False):
    extraction = row.get("extraction", {})
    if extraction.get("status") != "ok":
        raise ValueError("secondary location extraction unavailable")
    if extraction.get("components"):
        raise ValueError("secondary location present")
    full = parse_temporal_text(row["answer"], include_brackets=True)
    if full.issues or len(full.matches) != 1:
        raise ValueError("source must contain exactly one valid temporal expression")
    text, _ = strip_brackets_with_mapping(row["answer"])
    parsed = parse_temporal_text(text)
    if len(parsed.matches) != 1:
        raise ValueError("exactly one date must remain after bracket removal")
    match = parsed.matches[0]
    if not relation_match_is_relevant(text, match.start, match.end, row["source_relation"]):
        raise ValueError("date not anchored to requested event")
    end = first_complete_sentence_end(text, match.end)
    if end is None:
        raise ValueError("no complete temporal answer prefix")
    text = text[:end].rstrip()
    if not balanced_markdown(text):
        raise ValueError("unbalanced temporal answer styling")
    values = [common]
    for level in FIVE_LEVEL_TEMPORAL_LEVELS:
        try:
            candidate = match.value.to_level(level)
        except ValueError:
            continue
        if candidate.consistent_with(common) and candidate not in values:
            values.append(candidate)
    candidates, seen = [], set()
    for value in values:
        edit = plan_temporal_edit(text, match, value)
        rendered = text[:edit.start] + edit.replacement + text[edit.end:]
        if rendered not in seen:
            seen.add(rendered)
            candidates.append({"answer": rendered, "score_start": edit.start + edit.score_start,
                               "score_end": edit.start + edit.score_end, "level": value.level.value})
    if include_idk:
        candidates.append({"answer": IDK_ANSWER, "score_start": 0, "score_end": len(IDK_ANSWER), "level": "IDK"})
    groups = [{"modality": "temporal", "target_level": common.level.value, "target_index": 0,
               "candidates": candidates}] if len(candidates) > 1 else []
    return example(row, split, name, common.level.value, candidates[0]["answer"], groups, seed)


def location_eligible(row):
    safety = row.get("filter", {})
    if safety.get("filter_status") != "ok" or safety.get("filter_prediction") != "Single":
        return "missing or unsafe Single filter"
    if (row.get("extraction", {}).get("status") != "ok"
            or not row.get("extraction", {}).get("components")
            or row.get("normalization", {}).get("status") not in {"normalized", "partial"}
            or not row.get("normalization", {}).get("consensus_value")):
        return "unsafe source normalization"
    if temporal_spans(row["answer"], row["extraction"]["components"]):
        return "temporal content in primary-only location answer"
    return None


def location_candidates(rows, predictions):
    if not rows or len(rows) != len(predictions):
        raise ValueError("Candidate rows and predictions must be aligned")
    spans = []
    for row, answer in zip(rows, predictions, strict=True):
        matches = replacement_expression_spans(row, answer)
        if len(matches) != 1:
            raise ValueError("location expression has no unique score span")
        spans.append(matches[0])
    prefix = min(len(os.path.commonprefix(predictions)), *(start for start, _ in spans))
    suffix = min(len(os.path.commonprefix([text[::-1] for text in predictions])),
                 *(len(text)-end for text, (_, end) in zip(predictions, spans, strict=True)))
    if not re.search(r"\b(?:born|died|passed away|assassinated)\b", predictions[0][:prefix]):
        raise ValueError("rewriter changed the shared subject/event prefix")
    candidates = []
    for row, text, (start, end) in zip(rows, predictions, spans, strict=True):
        before = plain_text(text[prefix:start]).strip()
        if before not in {"", "the", "a", "an", "in", "in the", "in a", "at", "at the", "at a", "on", "on the", "near", "outside", "within"}:
            raise ValueError("rewriter changed context before the location")
        if re.search(r"[^\s*_,.;:!?]", text[end:len(text)-suffix]):
            raise ValueError("rewriter changed context after the location")
        candidates.append({"answer": text, "score_start": prefix, "score_end": len(text)-suffix,
                           "level": row["target_level"]})
    return candidates


def validate_example(row):
    if row["split"] not in {"train", "dev"}:
        raise ValueError("Test people must never appear in training examples")
    if row["is_idk"]:
        if row["positive_answer"] != IDK_ANSWER or row["primary_target_level"] is not None or row["contrastive_groups"]:
            raise ValueError("Invalid IDK training example")
        return
    levels = {level.value for level in FIVE_LEVEL_TEMPORAL_LEVELS} if row["relation"] in {"birth", "death"} else {"CITY", "REGION", "COUNTRY", "CONTINENT"}
    if row["primary_target_level"] not in levels or not row["positive_answer"].strip():
        raise ValueError("Invalid primary training target")
    if len(row["contrastive_groups"]) > 1:
        raise ValueError("Primary-only examples have at most one contrastive group")
    for group in row["contrastive_groups"]:
        modality = "temporal" if row["relation"] in {"birth", "death"} else "location"
        if group["modality"] != modality or group["target_level"] != row["primary_target_level"]:
            raise ValueError("Contrastive group does not match the primary modality/level")
        candidates = group["candidates"]
        if len(candidates) < 2 or len({c["answer"] for c in candidates}) != len(candidates):
            raise ValueError("Contrastive group requires distinct candidates")
        target = candidates[group["target_index"]]
        if target["answer"] != row["positive_answer"] or target["level"] != row["primary_target_level"]:
            raise ValueError("Contrastive positive does not match supervised target")
        for candidate in candidates:
            start, end = candidate["score_start"], candidate["score_end"]
            if candidate["level"] not in levels | {"IDK"} or not 0 <= start < end <= len(candidate["answer"]) or not candidate["answer"][start:end].strip():
                raise ValueError("Invalid contrastive score span or level")
