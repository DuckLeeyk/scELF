"""Extract per-gene genotypes and explicitly align donor rows to count inputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ..common import (add_selection, align_rows, load_donors, selected_celltypes,
                      selected_genes, write_json, write_table)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_selection(p)
    p.add_argument("--plink-dir", type=Path, required=True, help="Directory of gene.bed/.bim/.fam files.")
    p.add_argument("--output-dir", type=Path, required=True, help="Analysis root receiving cell_type/data/genotype.")
    p.add_argument("--missing-genotypes", choices=["zero", "mean", "error"], default="zero", help="zero retains the source preparation rule.")
    args = p.parse_args(argv)
    from pandas_plink import read_plink
    audit = []
    for ct in selected_celltypes(args.celltype_table, args.cell_types):
        donors = load_donors(args.analysis_dir / ct.name / "data" / f"genotype_IDs_{ct.name}.txt")
        for gene in selected_genes(args, ct):
            bim, fam, bed = read_plink(str(args.plink_dir / gene), verbose=False)
            # Select only requested samples before materializing the lazy BED.
            index = pd.Index(fam.iid.astype(str))
            if index.has_duplicates:
                raise ValueError("PLINK donor IDs are not unique")
            positions = index.get_indexer(donors)
            if (positions < 0).any():
                raise ValueError(f"PLINK donor missing for {ct.name}/{gene}")
            keys = bim.chrom.astype(str) + ":" + bim.pos.astype(str)
            if keys.duplicated().any():
                raise ValueError("Multiple variants share one coordinate; resolve alleles before analysis")
            frame = pd.DataFrame(bed[:, positions].compute().T, index=donors, columns=keys)
            missing = int(frame.isna().sum().sum())
            if missing and args.missing_genotypes == "error":
                raise ValueError(f"Missing genotypes for {gene}")
            frame = frame.fillna(0 if args.missing_genotypes == "zero" else frame.mean())
            # Keep the original discrete-genotype preparation policy.
            if args.missing_genotypes == "zero":
                frame = np.rint(frame).astype(np.int64)
            keep = frame.nunique() > 1
            frame = frame.loc[:, keep]
            if frame.empty or not np.isfinite(frame.to_numpy()).all():
                raise ValueError(f"No valid variable variants for {gene}")
            target = args.output_dir / ct.name / "data" / "genotype" / f"genotype_{ct.name}_{gene}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.index.name = "genotype_ID"
            frame.to_csv(target)
            # Preserve BIM alleles for effect-direction audits in downstream TWAS.
            alleles = pd.DataFrame({"POS": keys, "rsid": bim.snp.astype(str),
                                    "A1": bim.a0.astype(str), "A2": bim.a1.astype(str)})
            write_table(alleles.loc[keep.to_numpy()], target.with_suffix(".variants.tsv"))
            audit.append({"cell_type": ct.name, "gene": gene, "donors": len(frame),
                          "variants": frame.shape[1], "missing_dosages": missing,
                          "dropped_constant_variants": int((~keep).sum())})
    write_json(audit, args.output_dir / "genotype_preparation.json")


if __name__ == "__main__":
    main()
