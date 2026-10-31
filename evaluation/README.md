# Evaluation code

Computes every per-cell name-diversity metric reported in the paper from
the two released CSV files (no SQLite required).

## Metrics

Per cell (model x strategy), over `PROPER_NAME` characters by default:

- Concentration: C1, C3, C5, C10 (share of the top-k names)
- Shannon entropy H (bits), Pielou evenness, Gini coefficient
- **Slop Ratio (SR)**: collision probability SR = sum(c_i (c_i - 1)) / (N (N - 1))
- SR decomposition: `sr_known` (concentration in the known-slop set) and
  `sr_emergent` (concentration in everything else)
- Stimulus ratio (names matching one of the 80 stimulus concepts)
- ENP = 2^H and coverage thresholds ENP_50/80/95
- Unique names, top-1 and top-10 names

## Usage

```
python diversity_metrics.py --data-dir ../../dataset --out metrics.csv

# decompose against the universal SLOP_NAMES list (variant used for the
# published figures):
python diversity_metrics.py --data-dir ../../dataset --known-set static --out metrics_static.csv

# include NO_NAME and STIMULUS_AS_NAME characters:
python diversity_metrics.py --data-dir ../../dataset --include-all-names
```

## Known-slop sets

The SR decomposition splits concentration into a known component and an
emergent component. Two known-set variants are supported:

- `fingerprint` (default): each model's baseline fingerprint, `pct_2` names
  (>= 2% frequency in that model's baseline output), from
  `dataset/slop_fingerprints.json`. This is the definition stated in the
  paper.
- `static`: the universal `SLOP_NAMES` list defined in this module. This is
  the variant used to produce the published figure values.

## Requirements

Python 3.10+, pandas, numpy.
