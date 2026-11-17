"""Summarize paired correctness labels with question-cluster intervals."""
from collections import Counter, defaultdict
import random
import numpy as np

LABELS = ("Correct", "Incorrect", "Abstain", "Cannot determine")


def paired_interval(rows, bootstrap=1000, seed=42):
    if bootstrap < 1:
        raise ValueError("Positive bootstrap count required")
    if not rows:
        return None
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        group = groups[str(row["qid"])]
        group[0] += int(row["judge_finetuned_correct"] == "Correct") - int(row["judge_base_correct"] == "Correct")
        group[1] += 1
    values = np.array([groups[qid] for qid in sorted(groups)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap):
        sample = rng.choice(len(values), size=len(values), replace=True)
        totals = values[sample].sum(axis=0)
        estimates.append(float(totals[0] / totals[1]))
    alpha = 1 - .95
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(values[:, 0].sum() / values[:, 1].sum()), "confidence_level": .95,
            "num_resamples": bootstrap, "num_qid_clusters": len(values), "seed": seed,
            "ci_lower": float(lower), "ci_upper": float(upper)}


def summary(rows, bootstrap=1000, seed=42):
    if not rows or any(row["dataset"] != "granola" for row in rows):
        raise ValueError("Nonempty Granola pairs required")
    variants = {}
    for variant in ("base", "finetuned"):
        counts = Counter(row[f"judge_{variant}_correct"] for row in rows)
        if set(counts) - set(LABELS):
            raise ValueError("Invalid correctness label")
        determinate = len(rows) - counts["Cannot determine"]
        answered = counts["Correct"] + counts["Incorrect"]
        variants[variant] = {
            "label_counts": {label: counts[label] for label in LABELS},
            "correct_share_all_items": counts["Correct"] / len(rows),
            "correct_rate_determinate_including_abstentions": counts["Correct"] / determinate if determinate else None,
            "accuracy_among_judged_answers": counts["Correct"] / answered if answered else None,
            "judged_answer_coverage": answered / len(rows),
            "abstention_rate": counts["Abstain"] / len(rows),
            "cannot_determine_rate": counts["Cannot determine"] / len(rows)}
    transitions = {a: {b: 0 for b in LABELS} for a in LABELS}
    for row in rows:
        transitions[row["judge_base_correct"]][row["judge_finetuned_correct"]] += 1
    paired = [r for r in rows if all(r[f"judge_{v}_correct"] != "Cannot determine" for v in ("base", "finetuned"))]
    return {"items": len(rows), "variants": variants, "transitions_base_to_finetuned": transitions,
        "paired_correctness_difference": {"determinate_pairs": len(paired),
            "excluded_pairs_with_cannot_determine": len(rows)-len(paired), "interval": paired_interval(paired, bootstrap, seed),
            "definition": "FT minus base Correct rate on pairs with neither label Cannot determine; abstentions remain in denominator"},
        "pairs_with_ambiguous_references": sum(bool(r["reference_answers"].get("ambiguous_reference_answers"))
            if isinstance(r["reference_answers"], dict) else False for r in rows),
        "interpretation": "Reference-supported correctness at any compatible informative granularity; not an independent external fact check."}


def agreement(gold, predicted):
    if not gold or len(gold) != len(predicted) or any(x not in LABELS for x in [*gold,*predicted]):
        raise ValueError("Equal nonempty arrays of four-class labels required")
    confusion = {a:{b:0 for b in LABELS} for a in LABELS}
    for a,b in zip(gold,predicted): confusion[a][b] += 1
    n=len(gold); actual=Counter(gold); estimated=Counter(predicted)
    exact=sum(a==b for a,b in zip(gold,predicted))/n
    chance=sum(actual[x]*estimated[x] for x in LABELS)/(n*n)
    kappa=(exact-chance)/(1-chance) if chance < 1 else None
    per_label={}
    for label in LABELS:
        tp=confusion[label][label]; support=actual[label]; emitted=estimated[label]
        per_label[label]={"support":support,"predicted":emitted,"precision":tp/emitted if emitted else None,
                          "recall":tp/support if support else None,
                          "f1":2*tp/(support+emitted) if support+emitted else None}
    present=[v["f1"] or 0. for v in per_label.values() if v["support"]]
    return {"answers":n,"cohen_kappa":kappa,"exact_agreement":exact,"confusion_human_rows_judge_columns":confusion,
            "per_label":per_label,"macro_f1_on_human_present_labels":sum(present)/len(present)}


def score_pairs(rows, bootstrap=2000, seed=42, gold_field="reference_correctness"):
    prefix = {"reference_correctness":"human_reference", "correctness":"human_final"}[gold_field]
    def flatten(group, variant=None):
        vs=[variant] if variant else ["base","finetuned"]
        return ([r[f"{prefix}_{v}"] for r in group for v in vs], [r[f"judge_{v}_correct"] for r in group for v in vs])
    result=agreement(*flatten(rows))
    result["by_variant"]={v:agreement(*flatten(rows,v)) for v in ("base","finetuned")}
    result["by_model"]={m:agreement(*flatten([r for r in rows if r["model"]==m])) for m in sorted({r["model"] for r in rows})}
    if bootstrap < 1: raise ValueError("Positive bootstrap count required")
    rng=np.random.default_rng(seed); values=[]
    # Questions remain paired, and model allocation stays fixed in every draw.
    groups=[[r for r in rows if r["model"]==m] for m in sorted({r["model"] for r in rows})]
    for _ in range(bootstrap):
        sampled=[group[i] for group in groups for i in rng.integers(0,len(group),len(group))]
        metric=agreement(*flatten(sampled))["cohen_kappa"]
        if metric is not None: values.append(metric)
    result["kappa_interval"]={"method":"95% percentile bootstrap over question pairs, stratified by model",
                              "resamples":bootstrap,"defined_resamples":len(values),"seed":seed,
                              "lower":float(np.quantile(values,.025)) if values else None,
                              "upper":float(np.quantile(values,.975)) if values else None}
    return result


def split_pairs(pairs, dev_per_model=5, seed=42):
    if dev_per_model < 1:
        raise ValueError("Positive development quota required")
    pairs = sorted(pairs, key=lambda r: r["pair_id"])
    if len({r["pair_id"] for r in pairs}) != len(pairs):
        raise ValueError("Duplicate pair IDs")
    parent = list(range(len(pairs)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    normalize = lambda text: " ".join(str(text).casefold().split())
    seen = {}
    for i, row in enumerate(pairs):
        for key in (("qid", normalize(row["qid"])), ("question", normalize(row["question"]))):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    components = defaultdict(list)
    for i, row in enumerate(pairs):
        components[find(i)].append(row)
    sizes = Counter(r["model"] for r in pairs)
    if not sizes or any(n <= dev_per_model for n in sizes.values()):
        raise ValueError("Every model needs development and evaluation pairs")
    groups = sorted(components.values(), key=lambda g: min(r["pair_id"] for r in g))
    random.Random(seed).shuffle(groups)
    dev, evaluation, counts = [], [], Counter()
    for group in groups:
        addition = Counter(r["model"] for r in group)
        if all(counts[m] + n <= dev_per_model for m, n in addition.items()):
            dev.extend(group)
            counts.update(addition)
        else:
            evaluation.extend(group)
    if any(counts[m] != dev_per_model for m in sizes):
        raise ValueError("Cannot meet model quotas using whole question groups")
    for field in ("qid", "question"):
        if {normalize(r[field]) for r in dev} & {normalize(r[field]) for r in evaluation}:
            raise ValueError("Question/entity leakage between partitions")
    return sorted(dev, key=lambda r: r["pair_id"]), sorted(evaluation, key=lambda r: r["pair_id"])


