# Teaching Granularity
Code and data for the paper [Teaching Language Models to Adapt Answer Granularity to Their Confidence](https://arxiv.org/abs/2611.21211).

When precise details are uncertain, a coarser answer can retain useful information without guessing or abstaining. 
We investigated whether language models can learn this behavior through model-conditioned supervision derived from consistency across sampled responses and recognition of answers at different granularity levels.

## Data

| Contents | Location |
| --- | --- |
| Vital Persons cohort, reviewed gold and shared development/test membership | [metadata](data_preparation/vital_persons/metadata/) |
| Four model-specific training/validation splits and case targets | [datasets](data_preparation/vital_persons/datasets/) |
| Twenty responses per person--relation pair: 524,800 responses across four models | [samples](data_preparation/vital_persons/samples/) |
| Human granularity labels, adjudications and correctness judgments | [annotations](human_annotations/generalization/) |
| Temporal-parser audit | [temporal annotations](parsing/temporal/human_validation/) |
| Geographic pipeline audit | [geographic annotations](parsing/location/normalization/human_validation/audits/base-ft-20pairs-v1/) |
| Answer-presence and abstention judgments | [answer-status annotations](evaluation/vital_persons/answer_status/human_validation/) |

The model-specific datasets include `train.jsonl`, `dev.jsonl`,
`case_targets.jsonl` and split assignments. The held-out test people are shared
across models; `test.jsonl` contains no training response realizations.

Extractor, filter and rewriter training/reference annotations are included under
`parsing/location/`. These references are teacher-assisted or selectively reviewed,
not uniformly independent human-gold labels. Provenance fields are retained.

Full evaluation generations, automated judgments, caches, processing traces,
and generated tables and figures remain local. Small annotated response sets
are included with their labels. Evaluation uses the fitted parameters in
[metric.json](generalization/human-study/results/metric.json).

## GRANOLA dataset construction

We use a processed version of the [GRANOLA-EQ dataset](https://huggingface.co/datasets/google/granola-entity-questions)
introduced by [Yona et al. (2024)](https://aclanthology.org/2024.acl-long.365/).
Due to licensing restrictions of upstream resources, the processed dataset used
in this work is not redistributed in this repository or on external hosting
platforms. You can reconstruct it using: [build_granola_dataset.ipynb](https://github.com/lukasellinger/granuscore/blob/main/notebooks/build_granola_dataset.ipynb)

Use the notebook's dataset construction procedure. This paper's six training
relations and four held-out relations are specified in
`shared/constants.py` and `generalization/datasets.json`; they differ from the
notebook's optional split experiment.

Each model's `data_preparation/granola/datasets/<model>/split_assignments.json`
records the accepted source row IDs, entity IDs, split membership and selected
levels, without question or answer text.

Reconstruct the exact local training files without rerunning recognition:

```bash
uv run --locked --extra normalization python -m data_preparation.granola.restore \
  --model lfm --granola-jsonl /path/to/locally-reconstructed-granola.jsonl
```

Repeat for `ministral`, `qwen` and `gemma`. Omitting `--granola-jsonl` downloads
the pinned upstream revision instead. The command verifies each reconstructed
example and split against the recorded hashes before writing ignored local
`train.jsonl`, `dev.jsonl` and `test.jsonl` files. The latter contains accepted
recognition examples from held-out relations; full generalization evaluation
uses all eligible questions from those relations. Raw recognition outputs remain
local; `data_preparation/granola/recognition.py` supports rerunning the procedure.

## Setup

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Run commands from
the repository root:

```bash
uv sync --locked
```

Install the relevant extras for GPU generation/training, geographic normalization
or API judging: `generation`, `training`, `normalization`, `scoring`, `analysis`
and `judge`. Credentials can be supplied through an ignored `.env`, using
`.env.template` as the template. Paid API calls require explicit opt-in.

## Training and evaluation

Model identifiers are `lfm`, `ministral`, `qwen` and `gemma`. Pinned adapter
references are in [models](models/); weights are available on
[Hugging Face](https://huggingface.co/lukasellinger/teaching-granularity-adapters).

### Generate with released adapters

These commands generate Vital Persons baseline and fine-tuned responses, and
Base/FT responses on the generalization datasets, using the released adapters.
Generation requires a CUDA GPU.

```bash
uv sync --locked --extra generation --extra normalization

uv run --no-sync python -m evaluation.vital_persons \
  --model lfm --split test --stage generate --output-dir runs/evaluation/lfm

uv run --no-sync python -m generalization.generate \
  --models lfm --output-dir runs/generalization
```

### Optional: train your own adapter

Training requires a BF16-capable CUDA GPU and the locally reconstructed Granola
training files described above. Configurations are in [training/configs](training/configs/).

```bash
uv sync --locked --extra training --extra generation --extra normalization
uv run --no-sync python -m training --model lfm --output-dir runs/lfm
```

To generate with this newly trained adapter, add `--adapter runs/lfm/final` to
either generation command and choose a new `--output-dir`.

### Scoring

The commands above generate responses; scoring is a separate step. For Vital
Persons, use `--stage score` with the same output directory. Geographic scoring
also requires `--normalization-cache data/cache/location-normalization.sqlite3`;
the snapshot is available on request and can be verified against
`data/cache/SHA256SUMS`. Options are listed by
`python -m evaluation.vital_persons --help` and
`python -m generalization.generate --help`.

Parsing code is under `parsing/`; generalization scoring and judge scripts are
under `generalization/`. Ablations use the same trainer with `--config` and
configurations under `ablations/lfm/`.

## Human annotations

The generalization export contains 400 independent granularity judgments on
200 response pairs, 64 adjudications and 136 individual correctness judgments.
It retains sampled model responses, source IDs, numerical features and blind
mappings, while omitting benchmark question strings and reference hierarchies.
Recompute the human-study metrics offline:

```bash
uv run --locked python -m human_annotations.report_generalization
```

To restore question/reference tasks locally after reconstructing the benchmarks:

```bash
uv run --locked python -m human_annotations.restore_generalization \
  --output-dir /tmp/restored-human-study \
  --granola-jsonl /path/to/locally-reconstructed-granola.jsonl
```

## Tests

```bash
uv run --locked --extra test --extra normalization --extra analysis python -m pytest -q -rs
```

Granola training checks require the reconstructed splits described above.
Matched-control replay checks require locally prepared control datasets and
rewrite traces; use `python -m data_preparation.vital_persons.greedy_control --help`.
Historical evaluation checks require saved local generations and results.
These checks skip with an explicit reason when their inputs are absent.
Available inputs still undergo the original checksum and consistency checks;
changed data fails rather than being skipped.

## License

Original code is released under the [MIT license](LICENSE). Vital Persons factual
metadata is derived from [Wikidata structured data](https://www.wikidata.org/wiki/Wikidata:Licensing)
(CC0), with the person cohort drawn from Wikipedia's Vital Articles.
Processed GRANOLA-EQ data is not redistributed; see the reconstruction instructions
above and its [upstream license](https://huggingface.co/datasets/google/granola-entity-questions).

## Citation

If you use any of the work, please cite the following paper:

```tex
@misc{ellinger2026teachinglanguagemodelsadapt,
      title={Teaching Language Models to Adapt Answer Granularity to Their Confidence},
      author={Lukas Ellinger and Georg Groh},
      year={2026},
      eprint={2611.21211},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2611.21211},
}
```
