"""Consensus before filtering, then person-grouped target-aware train/dev splits."""

from shared.geography import FOUR_LEVELS as LOCATION_LEVELS, project_four_level_location as project_location
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import random

from parsing.location.normalization.values import (
    LocationLevel, LocationValue, location_consensus, consensus_count, LOCATION_FIELDS,
)
from parsing.temporal import TemporalValue, TemporalLevel, TemporalModifier, Season, parse_temporal_text
from parsing.temporal.relations import relation_match_is_relevant
from parsing.temporal.values import FIVE_LEVEL_TEMPORAL_LEVELS

RELATIONS = ("birth", "death", "birthplace", "deathplace")


def temporal_value(payload):
    fields = dict(payload)
    fields["level"] = TemporalLevel(fields["level"])
    for key in ("decade_modifier", "year_modifier", "century_modifier"):
        if fields.get(key):
            fields[key] = TemporalModifier(fields[key])
    if fields.get("season"):
        fields["season"] = Season(fields["season"])
    return TemporalValue(**fields)


def evidence(row, relation):
    if relation in {"birth", "death"}:
        matches = parse_temporal_text(row["answer"], include_brackets=True).matches
        relevant = [match.value for match in matches if relation_match_is_relevant(
            row["answer"], match.start, match.end, relation)]
        return max(relevant, key=lambda value: value.precision, default=None)
    extraction, resolution = row.get("extraction", {}), row.get("normalization", {})
    if extraction.get("status") != "ok" or not extraction.get("components"):
        return None
    if resolution.get("status") not in {"normalized", "partial"} or not resolution.get("consensus_value"):
        return None
    return project_location(LocationValue.from_dict(resolution["consensus_value"]))


def temporal_consensus(values, threshold):
    required = consensus_count(len(values), threshold)
    for level in FIVE_LEVEL_TEMPORAL_LEVELS:
        projected = []
        for value in values:
            if value is not None:
                try:
                    projected.append(value.to_level(level))
                except ValueError:
                    pass
        eligible = [(count, value) for value, count in Counter(projected).items() if count >= required]
        if eligible:
            return sorted(eligible, key=lambda item: (-item[0], item[1].to_text()))[0][1]
    return None


def summarize(rows, *, model, qid, relation, threshold=.8, expected_samples=20):
    if any(type(row["sample_index"]) is not int for row in rows) or len(rows) != expected_samples or sorted(row["sample_index"] for row in rows) != list(range(expected_samples)):
        raise ValueError(f"Expected every original sample exactly once: {model}/{qid}/{relation}")
    rows = sorted(rows, key=lambda row: row["sample_index"])
    if any(row["qid"] != qid or row["source_model"] != model or row["source_relation"] != relation for row in rows):
        raise ValueError("Mixed consensus identities")
    values = [evidence(row, relation) for row in rows]
    target = (temporal_consensus(values, threshold) if relation in {"birth", "death"}
              else location_consensus(values, threshold))
    available = sum(value is not None for value in values)
    return {"case_id": f"{model}:{qid}:{relation}", "qid": qid, "source_model": model, "relation": relation,
            "sample_count": expected_samples, "required_primary_values": consensus_count(expected_samples, threshold),
            "available_primary_values": available,
            "target_status": "known" if target else "idk" if available >= consensus_count(expected_samples, threshold) else "unavailable",
            "target_level": target.level.value if target else None, "target": asdict(target) if target else None,
            "evidence": [{"case_id": row["case_id"], "sample_index": row["sample_index"],
                          "value": asdict(value) if value else None} for row, value in zip(rows, values, strict=True)]}


def target_aware_split(development, test, targets, *, fraction=1/9, seed=42):
    development, test = set(development), set(test)
    if development & test or not 0 < fraction < 1:
        raise ValueError("Invalid development/test pools or fraction")
    labels = {qid: [] for qid in development}
    for target in targets:
        if target["qid"] in labels:
            labels[target["qid"]].append(f"{target['relation']}:{target['target_level'] or target['target_status']}")
    if any(len(values) != len(RELATIONS) for values in labels.values()):
        raise ValueError("Each development person needs all four targets before splitting")
    total = Counter(label for values in labels.values() for label in values)
    desired = {label: count * fraction for label, count in total.items()}
    selected_counts = Counter()
    remaining = sorted(development, key=lambda qid: hashlib.sha256(f"{seed}:{qid}".encode()).digest())
    chosen = set()
    for _ in range(round(len(development) * fraction)):
        qid = max(remaining, key=lambda q: sum(
            (2 * (desired[label] - selected_counts[label]) - 1) / max(desired[label], 1)
            for label in labels[q]))
        chosen.add(qid)
        selected_counts.update(labels[qid])
        remaining.remove(qid)
    assignments = {qid: "dev" if qid in chosen else "train" for qid in development}
    assignments.update({qid: "test" for qid in test})
    return assignments, {"seed": seed, "dev_fraction": fraction, "people": dict(Counter(assignments.values())),
                         "development_strata": dict(total), "dev_strata": dict(selected_counts)}


def supports(value, common):
    try:
        projected = value.to_level(common.level)
    except ValueError:
        return False
    actual, wanted = getattr(projected, LOCATION_FIELDS[common.level]), getattr(common, LOCATION_FIELDS[common.level])
    return bool(set(actual or ()) & set(wanted)) if isinstance(wanted, tuple) else actual == wanted


def replacements(target, seed):
    common = LocationValue.from_dict(target["target"])
    def render(value):
        output = []
        for level in LOCATION_LEVELS:
            try:
                output.append({"target_level": level.value, "replacement_location": value.to_level(level).to_text()})
            except ValueError:
                pass
        return output
    base = [{**row, "replacement_origin": "model_consensus"} for row in render(common)]
    compatible = []
    for item in target["evidence"]:
        if not item["value"]:
            continue
        value = LocationValue.from_dict(item["value"])
        if not supports(value, common) or any(getattr(common, field) and getattr(value, field)
                and getattr(common, field) != getattr(value, field) for field in ("country", "continent")):
            continue
        finer = [row for row in render(value) if LOCATION_LEVELS.index(LocationLevel(row["target_level"])) < LOCATION_LEVELS.index(common.level)]
        if finer:
            compatible.append((item, finer))
    if compatible:
        compatible.sort(key=lambda pair: (pair[0]["sample_index"], pair[0]["case_id"]))
        identity = [seed, target["source_model"], target["qid"], target["relation"]]
        group_seed = int(hashlib.sha256(json.dumps(identity).encode()).hexdigest(), 16)
        item, finer = random.Random(group_seed).choice(compatible)
        base.extend({**row, "replacement_origin": "single_sample", "replacement_source_case_id": item["case_id"],
                     "replacement_source_value": item["value"]} for row in finer)
    return sorted(base, key=lambda row: LOCATION_LEVELS.index(LocationLevel(row["target_level"])))
