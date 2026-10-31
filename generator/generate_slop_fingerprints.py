"""
Generate model-specific slop fingerprints from baseline stories.

Reads character extractions from baseline stories (prompt_modification IS NULL)
and derives slop name sets for each model using 3 methods:
  - top-N most frequent names (default N=20)
  - names with frequency >= threshold (default 2%)
  - knee detection on sorted frequency curve (kneedle algorithm)

The fingerprints are consumed by generate_speculative.py as the static
verification set for speculative naming. They are frozen at experiment
start and are NOT updated during generation (avoiding contamination
from strategy artifacts).

Usage:
    uv run python generate_slop_fingerprints.py
    uv run python generate_slop_fingerprints.py --db path/to/custom_analysis.db
    uv run python generate_slop_fingerprints.py --top-n 20 --pct-threshold 2.0
    uv run python generate_slop_fingerprints.py --model "Gemma 3:27b"
"""

import argparse
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import sqlite3


def normalize_name(name: str) -> str:
    """Normalize name: strip accents, lowercase. Matches analyze_diversity.py."""
    if not name or not isinstance(name, str):
        return ""
    name = name.strip().lower()
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )
    return name


def extract_first_name(name: str) -> str:
    """Extract first name (given name) from a possibly full name."""
    if not name:
        return ""
    name = name.strip()
    # Remove titles
    for prefix in ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Sra.", "Dr ", "Mr ", "Mrs ", "Ms "]:
        if name.startswith(prefix):
            name = name[len(prefix):].strip()
    # Take first word (given name)
    parts = name.split()
    if parts:
        return parts[0]
    return name


def knee_detection(sorted_freqs):
    """
    Kneedle algorithm: find the knee (maximum curvature) in a sorted
    frequency distribution.

    Returns the index where the curve transitions from steep to flat.
    Names above this index are 'slop', below are 'tail'.
    """
    if len(sorted_freqs) <= 2:
        return len(sorted_freqs)

    freqs = np.array(sorted_freqs, dtype=float)

    # Normalize x and y to [0, 1]
    n = len(freqs)
    x = np.linspace(0, 1, n)
    y = (freqs - freqs.min()) / (freqs.max() - freqs.min() + 1e-12)

    # Compute distance from each point to the line connecting first and last
    # The line from (x[0], y[0]) to (x[-1], y[-1])
    # Distance = |(y[-1]-y[0])*x - (x[-1]-x[0])*y + x[-1]*y[0] - x[-1]*y[0]| / sqrt(...)
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    line_len = math.sqrt(dx ** 2 + dy ** 2) + 1e-12

    distances = np.abs(dy * x - dx * y + x[-1] * y[0] - y[-1] * x[0]) / line_len

    # Knee is the point of maximum distance
    knee_idx = int(np.argmax(distances))

    # Include at least 3 names
    return max(knee_idx + 1, 3)


def compute_name_frequencies(db_path):
    """
    Read character extractions from baseline stories and compute
    per-model name frequency distributions.

    Returns: {model_origin: {normalized_name: count}}
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    query = """
    SELECT sa.model_origin, ce.character_name
    FROM character_extractions ce
    JOIN story_analysis sa ON ce.story_id = sa.story_id
    WHERE sa.prompt_modification IS NULL
      AND ce.character_name IS NOT NULL
      AND TRIM(ce.character_name) != ''
    """
    cur.execute(query)

    model_names = {}
    for model_origin, raw_name in cur.fetchall():
        first = extract_first_name(raw_name)
        norm = normalize_name(first)
        if not norm or len(norm) < 2:
            continue
        model_names.setdefault(model_origin, Counter())[norm] += 1

    conn.close()
    return model_names


def derive_fingerprints(model_frequencies, top_n=20, pct_threshold=2.0):
    """
    For each model, derive slop sets using 3 methods.

    Returns a dict suitable for JSON serialization.
    """
    fingerprints = {}

    for model, counter in sorted(model_frequencies.items()):
        total = sum(counter.values())
        if total == 0:
            continue

        # Sorted (name, count) descending
        sorted_items = counter.most_common()
        sorted_freqs = [cnt / total for _, cnt in sorted_items]
        names_sorted = [name for name, _ in sorted_items]

        # Method 1: top-N
        top_set = names_sorted[:top_n]

        # Method 2: frequency >= threshold
        pct_set = [
            name for name, cnt in sorted_items
            if (cnt / total) * 100 >= pct_threshold
        ]

        # Method 3: knee detection
        knee_idx = knee_detection(sorted_freqs)
        knee_set = names_sorted[:knee_idx]

        # Frequency dict for slop_score lookup (all names, not just slop)
        freqs = {name: cnt / total for name, cnt in sorted_items}

        fingerprints[model] = {
            "total_chars": total,
            "unique_names": len(counter),
            "freqs": freqs,
            "top_20": top_set,
            "pct_2": pct_set,
            "knee": knee_set,
            "knee_idx": knee_idx,
        }

    return fingerprints


def main():
    parser = argparse.ArgumentParser(
        description="Generate model-specific slop fingerprints from baseline stories."
    )
    parser.add_argument(
        "--db", type=str,
        default="wiki/code/name-analyzer/custom_analysis.db",
        help="Path to custom_analysis.db with character extractions.",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="wiki/code/generator/slop_fingerprints.json",
        help="Output JSON path for fingerprints.",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="N for top-N method (default: 20).",
    )
    parser.add_argument(
        "--pct-threshold", type=float, default=2.0,
        help="Frequency %% threshold for pct method (default: 2.0).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Filter by model name (substring match). If omitted, process all.",
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"Error: {db_path} does not exist.")
        return

    print(f"Reading baseline character extractions from {db_path}...")
    model_frequencies = compute_name_frequencies(db_path)
    print(f"Found {len(model_frequencies)} models with baseline data.")

    if args.model:
        model_frequencies = {
            k: v for k, v in model_frequencies.items()
            if args.model.lower() in k.lower()
        }
        print(f"Filtered to {len(model_frequencies)} models matching '{args.model}'.")

    print(f"Deriving fingerprints (top-{args.top_n}, >= {args.pct_threshold}%, knee)...")
    fingerprints = derive_fingerprints(
        model_frequencies,
        top_n=args.top_n,
        pct_threshold=args.pct_threshold,
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fingerprints, f, ensure_ascii=False, indent=2)

    print(f"\nSaved fingerprints to {output_path}")
    print(f"\n{'Model':<40} {'chars':>6} {'uniq':>5} {'top20':>6} {'≥2%':>5} {'knee':>5}")
    print("-" * 75)
    for model, fp in fingerprints.items():
        print(
            f"{model:<40} {fp['total_chars']:>6} {fp['unique_names']:>5} "
            f"{len(fp['top_20']):>6} {len(fp['pct_2']):>5} {len(fp['knee']):>5}"
        )
        # Show top slop names
        slop_display = fp["knee"][:8]
        print(f"  slop (knee): {', '.join(slop_display)}")


if __name__ == "__main__":
    main()
