# Character-Name Concentration in LLM Fiction: Measuring Displacement and Mitigating Repetition

Alejandro Mozo Quesada · Sergio Gálvez Rojas · Francisco López Valverde — Universidad de Málaga

![arXiv](https://img.shields.io/badge/arXiv-PENDING-b31b1b.svg)
[![Hugging Face dataset](https://img.shields.io/badge/dataset-amozo%2Fcharacter--name--concentration-yellow.svg)](https://huggingface.co/datasets/amozo/character-name-concentration)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Character names in LLM fiction collapse onto a handful of attractors (*Elara*,
*Elias*), and the usual fixes either require logit access or merely move the
concentration somewhere else. This repository releases the generation pipelines
that produced our 44,782-story corpus — twenty models, four languages, text and
visual stimuli, eight name-selection strategies — together with the evaluation
that produced every number in the paper from the released dataset.

## The method

1. **Draft.** Verbalized sampling asks the model for K = 5 candidate name-sets
   with their probabilities, instead of one deterministic choice.
2. **Verify.** A SlopTracker scores each candidate against the model's own
   baseline fingerprint plus a runtime counter of names already used in the
   session.
3. **Reject.** A λ-penalized acceptance rule (or its zero-tolerance variant,
   HARD) selects the least-concentrated candidate set before any prose exists.

The paper's measurement claim: list-based strategies displace concentration
(the balloon effect), speculative naming removes it — mean collision
probability falls from 7.3% to 0.4% — and readers prefer the resulting names.

## Repository layout

```
generator/    six pipelines -> SQLite per run
  generate.py                 baseline
  generate_speculative.py     speculative naming (λ-penalized + HARD, τ = 0.02)
  generate_multiagent.py      multiagent reviewer
  generate_multiphase*.py     two-pass revision, plain and verbalized sampling
  generate_without_stimulus.py
  prompts/ icons/ reelsData.json        prompt templates and stimulus pictograms
  model*.json reviewer_model.json       OpenRouter and Ollama endpoints
  slop_fingerprints.json                per-model baseline name fingerprints (K = 2%)
evaluation/   released CSVs -> the paper's per-cell metrics
  diversity_metrics.py        Slop Ratio with known/emergent decomposition,
                              C_k, Shannon, Gini, ENP, stimulus ratio
```

## Quickstart

```bash
# base install; keys from environment (OPENROUTER_APIKEY) or generator/.env
pip install -r generator/requirements.txt

# 1. baseline run for one language (local models via model-local.json + Ollama)
python generator/generate.py -m generator/model.json -d stories.db -l es -t 4

# 2. speculative naming: λ = 10 and the HARD variant
python generator/generate_speculative.py -m generator/model.json \
    --slop-fingerprints generator/slop_fingerprints.json \
    --lambdas 10 --hard --hard-threshold 0.02 -d stories.db

# 3. per-cell metrics straight from the released dataset
python evaluation/diversity_metrics.py --data-dir <dataset-dir> --out metrics.csv
```

Full pipeline documentation: `generator/README.md`; metric definitions:
`evaluation/README.md`. Every script accepts `--dry-run` and `--limit` for
smoke tests.

## Data

The 44,782 stories and 88,919 character extractions behind the paper are on
[Hugging Face](https://huggingface.co/datasets/amozo/character-name-concentration)
(CC BY 4.0, public). Generation is not deterministically reproducible for the
online models, so the dataset is the definitive artifact.

## Citation

```bibtex
@article{mozoqueesada2026charactername,
  title     = {Character-Name Concentration in LLM Fiction: Measuring
               Displacement and Mitigating Repetition},
  author    = {Mozo Quesada, Alejandro and G{\'a}lvez Rojas, Sergio and
               L{\'o}pez Valverde, Francisco},
  year      = {2026},
  note      = {arXiv preprint coming soon}
}
```

## License

Code is released under the [MIT License](LICENSE). The dataset on Hugging Face
is CC BY 4.0 as a whole.
