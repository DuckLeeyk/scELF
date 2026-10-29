"""Export a locus evidence table (PIP, CS, GWAS Z and reference LD r-squared).

No plotting is performed. Coordinates and donor IDs are explicitly aligned.
The reference variant, gene, cell type and trait are all supplied by the caller.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .calculate_scores import load_zscores
from ..common import (METHODS, canonical_coordinate, load_cs, load_donors, load_genotype,
    read_pips, standardize_genotype, token, write_json, write_table)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, required=True, help="Prepared input root.")
    p.add_argument("--finemapping-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cell-type", required=True)
    p.add_argument("--gene", required=True)
    p.add_argument("--trait", required=True)
    p.add_argument("--reference-pos", required=True)
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--gwas", type=Path, help="Optional aligned POS,zscore table.")
    p.add_argument("--dosage-allele", choices=["A1", "A2"], default="A2")
    p.add_argument("--assume-harmonized", action="store_true")
    args = p.parse_args(argv)
    ct, gene, trait = map(token, [args.cell_type, args.gene, args.trait])
    data = args.analysis_dir / ct / "data"
    donors = load_donors(data / f"genotype_IDs_{ct}.txt")
    raw = load_genotype(data / "genotype" / f"genotype_{ct}_{gene}.csv", donors)
    keys = [canonical_coordinate(x) for x in raw.columns]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate genotype coordinates")
    reference = canonical_coordinate(args.reference_pos)
    if reference not in keys:
        raise ValueError("Reference variant is absent from the locus genotype matrix")
    genotype = standardize_genotype(raw)
    r = genotype.T @ genotype[:, keys.index(reference)] / len(donors)
    locus = pd.DataFrame({"POS": keys, "ld_r2": np.clip(r*r, 0, 1)})
    if args.gwas:
        gwas, _ = load_zscores(args.gwas, set(keys), 200000, args.dosage_allele, args.assume_harmonized)
        locus["gwas_zscore"] = locus.POS.map(gwas)
    rows = []
    for method in map(token, args.methods):
        base = args.finemapping_dir / ct
        pips = read_pips(base / f"{method}_pip" / f"{gene}.csv")
        if set(pips.POS) != set(keys):
            raise ValueError(f"{method} PIPs do not match the locus genotype variants")
        membership = {pos: [] for pos in keys}
        for name, indices in load_cs(base / f"{method}_cs" / f"{gene}.json", len(pips)).items():
            for pos in pips.iloc[indices].POS:
                membership[pos].append(name)
        frame = locus.merge(pips[["POS", "PIP"]], on="POS", validate="one_to_one")
        frame["credible_sets"] = frame.POS.map(lambda x: ";".join(membership[x]))
        frame["method"], frame["cell_type"], frame["gene"], frame["trait"] = method, ct, gene, trait
        rows.append(frame)
    write_table(pd.concat(rows, ignore_index=True), args.output_dir / "locus_variants.tsv")
    write_json({"arguments": vars(args), "reference_POS": reference, "n_donors": len(donors), "n_variants": len(keys)}, args.output_dir / "locus_metadata.json")


if __name__ == "__main__":
    main()
