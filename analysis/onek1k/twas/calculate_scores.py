"""Compute summary TWAS Z = w.T @ z using LD-normalized, allele-aligned weights.

Weights already satisfy w.T R w = 1. Do not divide by an LD denominator again.
This command performs coordinate matching; use harmonize_gwas before scoring.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from ..common import (METHODS, canonical_coordinate, finite, positive_int,
                      require_columns, table, token, write_json, write_table)


def load_weights(path):
    frame = table(path)
    require_columns(frame, ["POS", "weights"])
    frame["POS"] = frame.POS.map(canonical_coordinate)
    frame["weights"] = finite(frame.weights, "TWAS weights")
    if frame.empty or frame.POS.duplicated().any():
        raise ValueError(f"Empty or duplicate weight coordinates: {path}")
    return frame


def load_zscores(path, positions, chunksize, dosage_allele, assume_harmonized=False, duplicates="error"):
    retained, seen, removed = [], set(), 0
    for frame in table(path, chunksize=chunksize):
        require_columns(frame, ["POS", "zscore"])
        if not assume_harmonized:
            require_columns(frame, ["target_allele", "alignment_status"])
            if not frame.target_allele.eq(dosage_allele).all():
                raise ValueError("GWAS target allele differs from weight dosage allele")
            allowed = ["aligned", "flipped", "strand_aligned", "strand_flipped"]
            frame = frame.loc[frame.alignment_status.isin(allowed)].copy()
        frame["POS"] = frame.POS.map(canonical_coordinate)
        frame = frame.loc[frame.POS.isin(positions), ["POS", "zscore"]].copy()
        frame["zscore"] = finite(frame.zscore, "GWAS Z-scores")
        duplicate = frame.POS.duplicated() | frame.POS.isin(seen)
        if duplicate.any() and duplicates == "error":
            raise ValueError("Duplicate GWAS coordinates among tested variants; resolve multiallelic/duplicate rows")
        removed += int(duplicate.sum())
        frame = frame.loc[~duplicate]
        seen.update(frame.POS)
        retained.append(frame)
    if not retained:
        return pd.Series(dtype=float), removed
    return pd.concat(retained, ignore_index=True).set_index("POS").zscore, removed


def score_weights(weights, gwas, missing_policy="zero", max_missing_rate=1.0):
    aligned = gwas.reindex(weights.POS)
    missing = aligned.isna()
    rate = float(missing.mean())
    if (missing.any() and missing_policy == "error") or rate > max_missing_rate:
        raise ValueError(f"GWAS missing rate {rate:.3%} exceeds the specified policy")
    score = float(np.dot(weights.weights, aligned.fillna(0)))
    return score, math.erfc(abs(score) / math.sqrt(2)), rate, int((~missing).sum())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cell-types", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--genes", nargs="+")
    p.add_argument("--traits", nargs="+", required=True)
    p.add_argument("--gwas-template", required=True, help="Aligned GWAS CSV/TSV path with {trait} placeholder.")
    p.add_argument("--dosage-allele", choices=["A1", "A2"], required=True)
    p.add_argument("--assume-harmonized", action="store_true", help="Explicitly accept external POS,zscore tables without alignment metadata.")
    p.add_argument("--missing-z", choices=["zero", "error"], default="zero", help="zero retains original behavior and reports the missing rate.")
    p.add_argument("--max-missing-rate", type=float, default=1.0)
    p.add_argument("--duplicate-positions", choices=["error", "first"], default="error")
    p.add_argument("--chunksize", type=positive_int, default=200000)
    args = p.parse_args(argv)
    if not 0 <= args.max_missing_rate <= 1:
        p.error("--max-missing-rate must be in [0,1]")
    files, positions = [], set()
    for ct in map(token, args.cell_types):
        for method in map(token, args.methods):
            directory = args.weights_dir / ct / f"{method}_weights"
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for path in sorted(directory.glob("*.csv")):
                if args.genes and path.stem not in args.genes:
                    continue
                positions.update(load_weights(path).POS)
                files.append((ct, method, path.stem, path))
    if not files:
        raise ValueError("No weight files selected")
    rows, duplicate_counts = [], {}
    for trait in map(token, args.traits):
        gwas, removed = load_zscores(Path(args.gwas_template.format(trait=trait)), positions,
            args.chunksize, args.dosage_allele, args.assume_harmonized, args.duplicate_positions)
        duplicate_counts[trait] = removed
        for ct, method, gene, path in files:
            weights = load_weights(path)
            z, pv, rate, matched = score_weights(weights, gwas, args.missing_z, args.max_missing_rate)
            rows.append(dict(trait=trait, cell_type=ct, method=method, gene=gene,
                zscore=z, pvalue=pv, missing_rate=rate, n_variants=len(weights), n_matched=matched,
                status="scored" if matched else "no_gwas_data"))
    write_table(pd.DataFrame(rows), args.output)
    write_json({"arguments": vars(args), "removed_duplicates": duplicate_counts, "tested_pairs": len(rows)}, args.output.with_suffix(".run.json"))


if __name__ == "__main__":
    main()
