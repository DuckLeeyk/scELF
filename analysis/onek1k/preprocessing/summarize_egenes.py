"""Summarize per-gene BIM regions and released OneK1K cell-type eGene flags."""
import argparse
from pathlib import Path
import pandas as pd
from ..common import require_columns, table, token, write_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gene-list", type=Path, required=True, help="One gene per line without a header.")
    p.add_argument("--plink-dir", type=Path, required=True)
    p.add_argument("--esnp-table", type=Path, required=True, help="Table with GENE,GENE_ID,CELL_ID.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    esnp = table(args.esnp_table, dtype=str)
    require_columns(esnp, ["GENE", "GENE_ID", "CELL_ID"])
    esnp = esnp.dropna(subset=["GENE", "GENE_ID", "CELL_ID"])
    mapping = esnp[["GENE", "GENE_ID"]].drop_duplicates()
    if mapping.GENE.duplicated().any():
        raise ValueError("Gene symbols map to multiple gene IDs; resolve the annotation first")
    ids = mapping.set_index("GENE").GENE_ID
    flags = esnp.assign(flag=True).drop_duplicates(["GENE", "CELL_ID"]).pivot(index="GENE", columns="CELL_ID", values="flag").fillna(False)
    rows = []
    for gene in dict.fromkeys(args.gene_list.read_text(encoding="utf-8-sig").split()):
        bim = pd.read_csv(args.plink_dir / f"{token(gene)}.bim", sep=r"\s+", header=None)
        if bim.empty or bim.iloc[:, 0].nunique() != 1:
            raise ValueError(f"Expected one nonempty chromosome region for {gene}")
        rows.append({"gene": gene, "gene_id": ids.get(gene), "chr": bim.iloc[0, 0],
                     "pos_min": bim.iloc[:, 3].min(), "pos_max": bim.iloc[:, 3].max(), "cis_snp_count": len(bim)})
    result = pd.DataFrame(rows).merge(flags, left_on="gene", right_index=True, how="left")
    result[flags.columns] = result[flags.columns].fillna(False)
    write_table(result, args.output)


if __name__ == "__main__":
    main()
