"""
Name diversity metrics for the release dataset (two CSV files).

Standalone version of the paper's evaluation code: reads stories.csv and
characters.csv instead of the SQLite databases. Computes, per experimental
cell (model x strategy): C_k concentration, Shannon entropy, evenness,
Gini, Slop Ratio (collision probability) with its known/emergent
decomposition, stimulus ratio, ENP, and top-10 names.

Usage:
    python diversity_metrics.py --data-dir ../dataset --out metrics.csv
    python diversity_metrics.py --data-dir ../dataset --known-set static --out metrics_static.csv

Known set for the SR decomposition:
    fingerprint  per-model baseline fingerprint (pct_2: names with >=2%
                 frequency in the model's baseline output), loaded from
                 slop_fingerprints.json
    static       the universal SLOP_NAMES list (same for every model)

Requires: pandas, numpy.
"""

import argparse
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical slop name set (universal, cross-model).
# Sources: name-slop analysis of this corpus and Paech et al. 2025.
SLOP_NAMES = {
    "elara", "elias", "leo", "maya", "luna",
    "kai", "aria", "aurora", "orion", "lyra",
    "fenrir", "sage", "rowan", "finn", "zara",
    "iris", "silas", "juno", "cassian", "elena", "milo",
    "tiber", "tiberius", "lila", "clara", "lena", "lina",
    "maren", "callista", "tober", "aedan",
    "flora", "fitzwilliam", "hemlock", "wren", "bram",
    "cormac", "archibald", "willow",
}

# The 80 stimulus image concepts. When these appear as character names they
# are prompt-leakage artifacts, not creative name choices.
STIMULUS_WORDS = {
    "alien", "angry face", "apple", "astronaut", "backpack", "beach",
    "binoculars", "boat", "broken heart", "campfire", "candle", "castle",
    "cave", "cloud", "compass", "confused", "cowboy hat", "crown",
    "crying face", "crystal ball", "desert island", "detective", "dragon",
    "evil face", "explorer", "explosion", "eye", "fairy", "feather",
    "fist", "flag", "flower", "footprint", "forest", "ghost", "heart",
    "key", "laser gun", "laughing face", "leaf", "lightbulb", "luck",
    "magic wand", "map", "moon", "motorcycle", "mountain", "ninja",
    "open book", "phoenix", "pirate", "prison", "rainbow",
    "raining cloud", "river", "road", "robot", "rocket", "running",
    "sad face", "scientist", "shield", "skull", "smiley face",
    "snowflake", "snowman", "spaceship", "sun", "surprised face",
    "sword", "tent", "train", "tree", "typewriter", "unicorn",
    "water drop", "wave", "winking face", "witch", "wizard hat",
}


def normalize_name(name: str) -> str:
    """Normalize for comparison: strip accents, lowercase."""
    if not name or not isinstance(name, str):
        return ""
    name = name.strip().lower()
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )
    return name


def normalize_name_counter(raw_names):
    """Aggregate raw names into a Counter keyed by normalized name.

    Also returns a mapping from normalized name to the most frequent
    original variant (title-case preferred on ties).
    """
    variant_counts: dict[str, Counter] = {}
    name_counts = Counter()
    for raw_name in raw_names:
        norm = normalize_name(raw_name)
        if not norm:
            continue
        name_counts[norm] += 1
        if norm not in variant_counts:
            variant_counts[norm] = Counter()
        variant_counts[norm][raw_name] += 1

    canonical_forms = {}
    for norm, variants in variant_counts.items():
        max_count = max(variants.values())
        candidates = [v for v, c in variants.items() if c == max_count]
        title_cased = [v for v in candidates if v == v.title()]
        canonical_forms[norm] = title_cased[0] if title_cased else candidates[0]

    return name_counts, canonical_forms


def concentration_k(counts, k: int) -> float:
    """Proportion of the total accounted for by the top-k items."""
    n = sum(counts)
    if n == 0:
        return 0.0
    return sum(counts[:k]) / n


def shannon_entropy(counts) -> float:
    """Shannon entropy in bits."""
    n = sum(counts)
    if n == 0 or len(counts) == 0:
        return 0.0
    probs = [c / n for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def evenness(entropy: float, v: int) -> float:
    """Pielou's evenness: H / log2(V). Range [0, 1]."""
    if v <= 1:
        return 0.0
    return entropy / math.log2(v)


def gini(counts) -> float:
    """Gini coefficient. 0 = uniform, 1 = one name dominates."""
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    if n == 0 or np.sum(sorted_counts) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(
        (2 * np.sum(index * sorted_counts) - (n + 1) * np.sum(sorted_counts))
        / (n * np.sum(sorted_counts))
    )


def slop_ratio_list(name_counts, slop_set=SLOP_NAMES) -> float:
    """Legacy list-based slop ratio: fraction of characters with slop names."""
    n = sum(name_counts.values())
    if n == 0:
        return 0.0
    return sum(c for name, c in name_counts.items() if name in slop_set) / n


def slop_ratio(name_counts) -> float:
    """Slop Ratio (SR): probability two random characters share a name.

    List-free collision probability, SR = sum(c_i*(c_i-1)) / (N*(N-1)).
    0 = all names unique, 1 = all characters share one name.
    """
    n = sum(name_counts.values())
    if n <= 1:
        return 0.0
    return sum(c * (c - 1) for c in name_counts.values()) / (n * (n - 1))


def slop_decomposition(name_counts, known_set) -> dict:
    """Split SR into the known component (names in known_set) and emergent."""
    n = sum(name_counts.values())
    denom = n * (n - 1) if n > 1 else 1.0
    sr_known = 0.0
    sr_emergent = 0.0
    for name, count in name_counts.items():
        contrib = count * (count - 1) / denom
        if name in known_set:
            sr_known += contrib
        else:
            sr_emergent += contrib
    return {"sr_known": sr_known, "sr_emergent": sr_emergent,
            "sr_total": sr_known + sr_emergent}


def effective_number_of_names(entropy: float) -> float:
    """ENP = 2^H."""
    return 2 ** entropy


def enp_alpha(counts, alpha: float) -> int:
    """Minimum number of names covering alpha of the characters."""
    n = sum(counts)
    if n == 0:
        return 0
    cumulative = 0
    for k, count in enumerate(counts, 1):
        cumulative += count
        if cumulative / n >= alpha:
            return k
    return len(counts)


def stimulus_name_ratio(name_counts) -> float:
    """Fraction of characters whose name matches a stimulus concept."""
    n = sum(name_counts.values())
    if n == 0:
        return 0.0
    return sum(c for name, c in name_counts.items() if name in STIMULUS_WORDS) / n


def load_data(data_dir):
    """Load stories.csv and characters.csv from a directory."""
    data_dir = Path(data_dir)
    stories = pd.read_csv(data_dir / "stories.csv",
                          usecols=["story_id", "model", "strategy"])
    characters = pd.read_csv(data_dir / "characters.csv",
                             usecols=["story_id", "character_name", "name_type"])
    characters = characters.merge(stories, on="story_id", how="inner")
    characters = characters[characters["character_name"].notna()
                            & (characters["character_name"] != "")]
    return stories, characters


def load_fingerprints(path):
    """Load per-model baseline fingerprints (slop_fingerprints.json)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_cell_metrics(characters, model, strategy, known_set,
                         proper_only=True):
    """Compute all diversity metrics for one cell (model x strategy).

    known_set: dict model -> set of fingerprint names (fingerprint mode),
               or None to use the static SLOP_NAMES list.
    """
    cell = characters[(characters["model"] == model)
                      & (characters["strategy"] == strategy)]

    cat_counts = Counter(cell["name_type"].fillna("NO_NAME"))
    if proper_only:
        cell = cell[cell["name_type"] == "PROPER_NAME"]
    raw_names = cell["character_name"].tolist()

    name_counts, canonical_forms = normalize_name_counter(raw_names)
    sorted_counts = sorted(name_counts.values(), reverse=True)
    total = sum(sorted_counts)
    unique = len(sorted_counts)

    h = shannon_entropy(sorted_counts)
    if known_set is not None:
        base = known_set.get(model, set())
    else:
        base = SLOP_NAMES
    decomp = slop_decomposition(name_counts, base)

    top10 = name_counts.most_common(10)
    top10_display = [(canonical_forms.get(norm, norm), c) for norm, c in top10]
    top1_name, top1_count = top10_display[0] if top10_display else ("N/A", 0)

    total_all = sum(cat_counts.values())
    return {
        "model": model,
        "strategy": strategy,
        "total_characters": total,
        "total_all_chars": total_all,
        "pct_proper_name": round(cat_counts.get("PROPER_NAME", 0) / total_all * 100, 1) if total_all else 0,
        "pct_no_name": round(cat_counts.get("NO_NAME", 0) / total_all * 100, 1) if total_all else 0,
        "pct_stimulus": round(cat_counts.get("STIMULUS_AS_NAME", 0) / total_all * 100, 1) if total_all else 0,
        "unique_names": unique,
        "unique_ratio": round(unique / total, 3) if total > 0 else 0.0,
        "top1_name": top1_name,
        "top1_count": top1_count,
        "C1": concentration_k(sorted_counts, 1),
        "C3": concentration_k(sorted_counts, 3),
        "C5": concentration_k(sorted_counts, 5),
        "C10": concentration_k(sorted_counts, 10),
        "entropy_H": round(h, 3),
        "evenness_E": round(evenness(h, unique), 3),
        "gini_G": round(gini(sorted_counts), 3),
        "slop_ratio_list": round(slop_ratio_list(name_counts), 3),
        "slop_ratio_v2": round(slop_ratio(name_counts), 4),
        "sr_known": round(decomp["sr_known"], 4),
        "sr_emergent": round(decomp["sr_emergent"], 4),
        "stimulus_ratio": round(stimulus_name_ratio(name_counts), 3),
        "ENP": round(effective_number_of_names(h), 1),
        "ENP_50": enp_alpha(sorted_counts, 0.50),
        "ENP_80": enp_alpha(sorted_counts, 0.80),
        "ENP_95": enp_alpha(sorted_counts, 0.95),
        "top10_names": ", ".join(f"{n}({c})" for n, c in top10_display),
    }


def compute_all_cells(characters, known_set=None, proper_only=True):
    """Compute metrics for every (model x strategy) cell."""
    cells = (characters[["model", "strategy"]]
             .drop_duplicates()
             .sort_values(["model", "strategy"]))
    rows = []
    for model, strategy in cells.itertuples(index=False):
        rows.append(compute_cell_metrics(characters, model, strategy,
                                         known_set, proper_only))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Name diversity metrics from the release CSVs")
    parser.add_argument("--data-dir", default="../dataset",
                        help="Directory containing stories.csv and characters.csv")
    parser.add_argument("--fingerprints", default=None,
                        help="Path to slop_fingerprints.json (default: <data-dir>/slop_fingerprints.json)")
    parser.add_argument("--known-set", choices=["fingerprint", "static"],
                        default="fingerprint",
                        help="Known-slop set for the SR decomposition")
    parser.add_argument("--include-all-names", action="store_true",
                        help="Include NO_NAME and STIMULUS_AS_NAME characters (default: PROPER_NAME only)")
    parser.add_argument("--model", default=None, help="Filter by model (substring)")
    parser.add_argument("--out", default="diversity_metrics.csv")
    args = parser.parse_args()

    stories, characters = load_data(args.data_dir)
    print(f"Loaded {len(stories):,} stories, {len(characters):,} character rows")

    if args.known_set == "fingerprint":
        fp_path = args.fingerprints or str(Path(args.data_dir) / "slop_fingerprints.json")
        fingerprints = load_fingerprints(fp_path)
        known_set = {model: set(data.get("pct_2", []))
                     for model, data in fingerprints.items()}
        print(f"Known set: per-model fingerprint (pct_2), {len(known_set)} models")
    else:
        known_set = None
        print("Known set: static SLOP_NAMES list")

    df = compute_all_cells(characters, known_set,
                           proper_only=not args.include_all_names)
    if args.model:
        df = df[df["model"].str.contains(args.model, case=False)]

    df.drop(columns=["top10_names"], errors="ignore").to_csv(args.out, index=False)
    print(f"{len(df)} cells -> {args.out}")


if __name__ == "__main__":
    main()
