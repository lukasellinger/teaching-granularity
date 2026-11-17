"""Unchanged chat-token alignment, score masks and epoch-wise negative selection."""
from __future__ import annotations
import hashlib
import multiprocessing
import random
from typing import Mapping, Protocol, Sequence
import torch
from torch.utils.data import Dataset
from shared.chat import render_chat
from .data import WeightedRealization, group_level_key, _level_id
from .schemas import AbstentionLevel, ContrastiveGroup, Modality, GranolaGroup, GranolaModality, GRANOLA_PRECISION
from parsing.location.normalization.values import LOCATION_PRECISION
from parsing.temporal.values import TEMPORAL_PRECISION


class Tokenizer(Protocol):
    """Tokenizer operations required by the realization encoder."""

    padding_side: str

    def apply_chat_template(self, messages, **kwargs): ...

    def __call__(self, text: str, **kwargs): ...

    def pad(self, features, **kwargs): ...

def _stable_rng(*, seed: int, case_id: str, modality: Modality) -> random.Random:
    identity = f"{seed}:{case_id}:{modality.value}".encode()
    digest = hashlib.sha256(identity).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))

def rotating_negative_indices(
    negative_indices: Sequence[int],
    *,
    limit: int,
    seed: int,
    epoch: int,
    case_id: str,
    modality: Modality | GranolaModality,
) -> list[int]:
    """Select a deterministic window from one seeded negative permutation.

    Reading consecutive windows through a circular permutation ensures that no
    candidate repeats until every candidate has occurred. Sorting only restores
    stored candidate order after the window has been selected.
    """
    if limit < 1:
        raise ValueError("negative limit must be positive")
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    ordered = list(negative_indices)
    if len(ordered) <= limit:
        return ordered
    _stable_rng(seed=seed, case_id=case_id, modality=modality).shuffle(ordered)
    start = (epoch * limit) % len(ordered)
    return sorted(ordered[(start + offset) % len(ordered)] for offset in range(limit))

def selected_negative_indices(
    group: ContrastiveGroup | GranolaGroup,
    *,
    limit: int,
    seed: int,
    epoch: int,
    case_id: str,
) -> list[int]:
    """Select ordinal neighbors and IDK, with rotating fallback candidates."""
    negative_indices = [
        index
        for index in range(len(group.candidates))
        if index != group.target_index
    ]
    idk_indices = [
        index
        for index in negative_indices
        if group.candidates[index].level is AbstentionLevel.IDK
    ]
    if not idk_indices:
        if len(negative_indices) <= limit:
            return negative_indices
        return rotating_negative_indices(
            negative_indices,
            limit=limit,
            seed=seed,
            epoch=epoch,
            case_id=case_id,
            modality=group.modality,
        )
    # Every ordinal-IDK group has the same fixed three-negative budget.
    ordinal_limit = 3
    if len(negative_indices) <= ordinal_limit:
        return negative_indices
    limit = ordinal_limit

    if group.modality is Modality.TEMPORAL:
        precision = TEMPORAL_PRECISION
    elif group.modality is Modality.LOCATION:
        precision = LOCATION_PRECISION
    else:
        precision = GRANOLA_PRECISION
    target_precision = precision[group.target_level]
    concrete = [
        index
        for index in negative_indices
        if group.candidates[index].level in precision
    ]
    selected: list[int] = []

    def add_boundary(*, finer: bool) -> None:
        candidates = [
            index
            for index in concrete
            if (
                precision[group.candidates[index].level] > target_precision
                if finer
                else precision[group.candidates[index].level] < target_precision
            )
        ]
        if not candidates:
            return
        boundary_precision = (
            min if finer else max
        )(precision[group.candidates[index].level] for index in candidates)
        tied = [
            index
            for index in candidates
            if precision[group.candidates[index].level] == boundary_precision
        ]
        selected.extend(
            rotating_negative_indices(
                tied,
                limit=1,
                seed=seed,
                epoch=epoch,
                case_id=f"{case_id}:{'finer' if finer else 'coarser'}",
                modality=group.modality,
            )
        )

    add_boundary(finer=True)
    add_boundary(finer=False)
    selected.append(idk_indices[0])

    remaining_slots = limit - len(selected)
    if remaining_slots:
        remaining = [
            index for index in negative_indices if index not in selected
        ]
        selected.extend(
            rotating_negative_indices(
                remaining,
                limit=min(remaining_slots, len(remaining)),
                seed=seed,
                epoch=epoch,
                case_id=f"{case_id}:fallback",
                modality=group.modality,
            )
        )
    return sorted(selected)

def _encode_text(tokenizer: Tokenizer, text: str, *, offsets: bool):
    return tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=offsets,
        truncation=False,
    )

def encode_realization(
    row: WeightedRealization,
    tokenizer: Tokenizer,
    *,
    seed: int,
    epoch: int,
    max_negatives_per_group: int | None,
    max_length: int | None,
    system_prompt: str | None,
    user_prompt_template: str = "{question}",
    rank_balance_factors: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Tokenize one positive and its locally factorized alternatives."""
    example = row.example
    user_prompt = user_prompt_template.format(question=example.prompt)
    prompt_text = render_chat(tokenizer, user_prompt, None, system_prompt)
    prompt_encoding = _encode_text(tokenizer, prompt_text, offsets=False)
    prompt_ids = list(prompt_encoding["input_ids"])

    sequences: list[dict[str, list[int]]] = []
    sequence_indices: dict[tuple[int, ...], int] = {}
    assistant_encodings = {}

    def encode_assistant(
        assistant: str,
    ) -> tuple[int, list[tuple[int, int]], int]:
        if assistant in assistant_encodings:
            return assistant_encodings[assistant]
        full_text = render_chat(
            tokenizer,
            user_prompt,
            assistant,
            system_prompt,
        )
        assistant_start = full_text.find(assistant, len(prompt_text))
        if assistant_start < 0:
            assistant_start = full_text.rfind(assistant)
        if assistant_start < 0:
            raise ValueError(
                f"{example.case_id}: assistant answer is absent from chat template"
            )
        encoded = _encode_text(tokenizer, full_text, offsets=True)
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        if max_length is not None and len(input_ids) > max_length:
            raise ValueError(
                f"{example.case_id}: supervised sequence has {len(input_ids)} "
                f"tokens, exceeding max_length={max_length}"
            )
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"{example.case_id}: generation prompt is not a token prefix "
                "of the full chat template"
            )
        sft_mask = [
            int(index >= len(prompt_ids) and attention)
            for index, attention in enumerate(attention_mask)
        ]
        if not any(sft_mask):
            raise ValueError(
                f"{example.case_id}: chat template removed the assistant answer"
            )
        identity = tuple(input_ids)
        if identity not in sequence_indices:
            sequence_indices[identity] = len(sequences)
            sequences.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "sft_mask": sft_mask,
                }
            )
        result = sequence_indices[identity], offsets, assistant_start
        assistant_encodings[assistant] = result
        return result

    positive_index, _, _ = encode_assistant(example.positive_answer)
    candidate_sequence_indices: list[int] = []
    candidate_score_masks: list[list[int]] = []
    candidate_group_ids: list[int] = []
    candidate_is_target: list[bool] = []
    group_modalities: list[str] = []
    group_target_levels: list[str] = []
    group_balance_weights: list[float] = []

    if max_negatives_per_group != 0:
        for group_id, group in enumerate(example.contrastive_groups):
            negative_indices = [
                index
                for index in range(len(group.candidates))
                if index != group.target_index
            ]
            if max_negatives_per_group is not None:
                negative_indices = selected_negative_indices(
                    group,
                    limit=max_negatives_per_group,
                    seed=seed,
                    epoch=epoch,
                    case_id=example.case_id,
                )
            selected_indices = sorted([group.target_index, *negative_indices])
            if len(selected_indices) < 2:
                raise ValueError(
                    f"{example.case_id}: contrastive group has no negative"
                )
            for candidate_index in selected_indices:
                candidate = group.candidates[candidate_index]
                sequence_index, offsets, assistant_start = encode_assistant(
                    candidate.answer
                )
                absolute_start = assistant_start + candidate.score_start
                absolute_end = assistant_start + candidate.score_end
                score_mask = [
                    int(
                        attention
                        and end > absolute_start
                        and start < absolute_end
                        and end > start
                    )
                    for (start, end), attention in zip(
                        offsets,
                        sequences[sequence_index]["attention_mask"],
                    )
                ]
                covered = [
                    offsets[index]
                    for index, selected in enumerate(score_mask)
                    if selected
                ]
                if (
                    not covered
                    or min(start for start, _ in covered) > absolute_start
                    or max(end for _, end in covered) < absolute_end
                ):
                    raise ValueError(
                        f"{example.case_id}: tokenization does not cover score "
                        f"span {candidate.score_start}:{candidate.score_end}"
                    )
                candidate_sequence_indices.append(sequence_index)
                candidate_score_masks.append(score_mask)
                candidate_group_ids.append(group_id)
                candidate_is_target.append(candidate_index == group.target_index)
            group_modalities.append(group.modality.value)
            level_key = group_level_key(example, group.target_level.value)
            group_target_levels.append(level_key)
            group_balance_weights.append(
                (rank_balance_factors or {}).get(level_key, 1.0)
            )

    return {
        "case_id": example.case_id,
        "sequences": sequences,
        "positive_sequence_index": positive_index,
        "candidate_sequence_indices": candidate_sequence_indices,
        "candidate_score_masks": candidate_score_masks,
        "candidate_group_ids": candidate_group_ids,
        "candidate_is_target": candidate_is_target,
        "group_modalities": group_modalities,
        "group_target_levels": group_target_levels,
        "group_balance_weights": group_balance_weights,
        "num_groups": len(group_modalities),
        "sft_case_weight": row.sft_case_weight,
        "contrastive_case_weight": row.contrastive_case_weight,
    }

class RealizationDataset(Dataset):
    """All realizations; only local negative selection varies by epoch."""

    def __init__(
        self,
        rows: Sequence[WeightedRealization],
        tokenizer: Tokenizer,
        *,
        seed: int,
        max_negatives_per_group: int | None,
        max_length: int | None = None,
        system_prompt: str | None = None,
        user_prompt_template: str = "{question}",
        rank_balance_factors: Mapping[str, float] | None = None,
    ) -> None:
        if not rows:
            raise ValueError("cannot construct an empty realization dataset")
        if max_negatives_per_group is not None and max_negatives_per_group < 0:
            raise ValueError("max_negatives_per_group cannot be negative")
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_negatives_per_group = max_negatives_per_group
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.rank_balance_factors = dict(rank_balance_factors or {})
        self._epoch = multiprocessing.Value("q", 0)
        self._encoded: dict[int, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        full = self._encode_once(index)
        limit = self.max_negatives_per_group
        if limit is None:
            return full
        example = self.rows[index].example
        selected = []
        offset = 0
        if limit:
            for group in example.contrastive_groups:
                negatives = [
                    i for i in range(len(group.candidates)) if i != group.target_index
                ]
                negatives = selected_negative_indices(
                    group,
                    limit=limit,
                    seed=self.seed,
                    epoch=self._epoch.value,
                    case_id=example.case_id,
                )
                selected.extend(
                    offset + i for i in sorted([group.target_index, *negatives])
                )
                offset += len(group.candidates)
        # Keep only selected sequences in the GPU batch, not all cached negatives.
        used = [full["positive_sequence_index"]]
        for i in selected:
            sequence = full["candidate_sequence_indices"][i]
            if sequence not in used:
                used.append(sequence)
        mapping = {old: new for new, old in enumerate(used)}
        result = dict(full)
        result["sequences"] = [full["sequences"][i] for i in used]
        result["positive_sequence_index"] = 0
        result["candidate_sequence_indices"] = [
            mapping[full["candidate_sequence_indices"][i]] for i in selected
        ]
        for name in (
            "candidate_score_masks",
            "candidate_group_ids",
            "candidate_is_target",
        ):
            result[name] = [full[name][i] for i in selected]
        if not limit:
            for name in (
                "group_modalities",
                "group_target_levels",
                "group_balance_weights",
            ):
                result[name] = []
            result["num_groups"] = 0
        return result

    def _encode_once(self, index: int) -> dict[str, object]:
        if index not in self._encoded:
            self._encoded[index] = encode_realization(
                self.rows[index],
                self.tokenizer,
                seed=self.seed,
                epoch=0,
                max_negatives_per_group=None,
                max_length=self.max_length,
                system_prompt=self.system_prompt,
                user_prompt_template=self.user_prompt_template,
                rank_balance_factors=self.rank_balance_factors,
            )
        return self._encoded[index]

    def preflight(self) -> dict[str, int]:
        """Validate and cache every candidate before allocating the model."""
        sequences = groups = maximum_tokens = 0
        for index in range(len(self)):
            encoded = self._encode_once(index)
            sequences += len(encoded["sequences"])
            groups += encoded["num_groups"]
            maximum_tokens = max(
                maximum_tokens, *(len(s["input_ids"]) for s in encoded["sequences"])
            )
            if (index + 1) % 1_000 == 0:
                print(f"tokenization preflight: {index + 1:,}/{len(self):,} rows")
        return {
            "realizations": len(self),
            "candidate_groups": groups,
            "unique_sequences_within_realizations": sequences,
            "maximum_sequence_tokens": maximum_tokens,
        }

def _pad_mask(mask: list[int], width: int, padding_side: str) -> list[int]:
    padding = [0] * (width - len(mask))
    return padding + mask if padding_side == "left" else mask + padding

class RealizationCollator:
    """Flatten and globally deduplicate candidate sequences in a row batch."""

    def __init__(self, tokenizer: Tokenizer, *, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        sequences: list[dict[str, list[int]]] = []
        sequence_indices: dict[tuple[int, ...], int] = {}
        local_to_global: list[list[int]] = []
        for feature in features:
            mapping = []
            for sequence in feature["sequences"]:
                identity = tuple(sequence["input_ids"])
                global_index = sequence_indices.get(identity)
                if global_index is None:
                    global_index = len(sequences)
                    sequence_indices[identity] = global_index
                    sequences.append(sequence)
                else:
                    existing = sequences[global_index]
                    if (
                        existing["attention_mask"] != sequence["attention_mask"]
                        or existing["sft_mask"] != sequence["sft_mask"]
                    ):
                        raise ValueError(
                            "identical token sequences have incompatible masks"
                        )
                mapping.append(global_index)
            local_to_global.append(mapping)

        padded = self.tokenizer.pad(
            {
                "input_ids": [sequence["input_ids"] for sequence in sequences],
                "attention_mask": [
                    sequence["attention_mask"] for sequence in sequences
                ],
            },
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        width = int(padded["input_ids"].shape[1])
        sft_mask = torch.tensor(
            [
                _pad_mask(sequence["sft_mask"], width, self.tokenizer.padding_side)
                for sequence in sequences
            ],
            dtype=torch.bool,
        )

        positive_indices = []
        candidate_indices = []
        candidate_masks = []
        candidate_group_ids = []
        group_realization_indices = []
        group_modalities = []
        group_target_levels = []
        group_balance_weights = []
        group_offset = 0
        for realization_index, (feature, mapping) in enumerate(
            zip(features, local_to_global)
        ):
            positive_indices.append(mapping[feature["positive_sequence_index"]])
            candidate_indices.extend(
                mapping[index] for index in feature["candidate_sequence_indices"]
            )
            candidate_masks.extend(
                _pad_mask(mask, width, self.tokenizer.padding_side)
                for mask in feature["candidate_score_masks"]
            )
            candidate_group_ids.extend(
                group_offset + group_id for group_id in feature["candidate_group_ids"]
            )
            group_realization_indices.extend(
                [realization_index] * feature["num_groups"]
            )
            group_modalities.extend(feature["group_modalities"])
            group_target_levels.extend(feature["group_target_levels"])
            group_balance_weights.extend(feature["group_balance_weights"])
            group_offset += feature["num_groups"]

        score_masks = (
            torch.tensor(candidate_masks, dtype=torch.bool)
            if candidate_masks
            else torch.zeros((0, width), dtype=torch.bool)
        )
        batch = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "sft_mask": sft_mask,
            "positive_sequence_indices": torch.tensor(
                positive_indices, dtype=torch.long
            ),
            "candidate_sequence_indices": torch.tensor(
                candidate_indices, dtype=torch.long
            ),
            "candidate_score_masks": score_masks,
            "candidate_group_ids": torch.tensor(candidate_group_ids, dtype=torch.long),
            "candidate_is_target": torch.tensor(
                [
                    target
                    for feature in features
                    for target in feature["candidate_is_target"]
                ],
                dtype=torch.bool,
            ),
            "group_realization_indices": torch.tensor(
                group_realization_indices, dtype=torch.long
            ),
            "group_modality_ids": torch.tensor(
                [
                    {"temporal": 0, "location": 1, "granola": 2}[modality]
                    for modality in group_modalities
                ],
                dtype=torch.long,
            ),
            "group_target_level_ids": torch.tensor(
                [_level_id(level) for level in group_target_levels],
                dtype=torch.long,
            ),
            "group_balance_weights": torch.tensor(
                group_balance_weights, dtype=torch.float
            ),
            "sft_case_weights": torch.tensor(
                [feature["sft_case_weight"] for feature in features],
                dtype=torch.float,
            ),
            "contrastive_case_weights": torch.tensor(
                [feature["contrastive_case_weight"] for feature in features],
                dtype=torch.float,
            ),
        }
        from .loss import validate_loss_inputs

        # Validate on CPU so the loss has no device-to-host validation barriers.
        validate_loss_inputs(batch)
        batch["loss_inputs_validated"] = True
        needed = torch.zeros_like(sft_mask[:, 1:])
        needed[positive_indices] = sft_mask[positive_indices, 1:]
        for sequence, mask in zip(candidate_indices, score_masks):
            needed[sequence] |= mask[1:]
        batch["logits_to_keep"] = needed.any(dim=0).nonzero(as_tuple=True)[0]
        return batch
