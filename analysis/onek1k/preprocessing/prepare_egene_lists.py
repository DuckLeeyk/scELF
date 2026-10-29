"""Make per-original-cell-type eGene CSVs from a region summary and expression IDs."""
import argparse
from pathlib import Path
from ..common import require_columns, table, token, write_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--cell-types", nargs="+", required=True, help="Boolean cell-type flag columns in the eGene summary (original 14-cell labels).")
    p.add_argument("--min-snps", type=int, default=11, help="Inclusive minimum; 11 preserves the source cis_snp_count > 10 rule.")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    import anndata as ad
    data = ad.read_h5ad(args.h5ad, backed="r")
    genes = set(data.var_names.astype(str))
    data.file.close()
    summary = table(args.summary)
    columns = ["gene", "gene_id", "chr", "pos_min", "pos_max", "cis_snp_count"]
    require_columns(summary, columns + args.cell_types)
    for ct in args.cell_types:
        truth = summary[ct].astype(str).str.lower().isin(["true", "1"])
        subset = summary.loc[truth & summary.gene_id.isin(genes) & summary.cis_snp_count.ge(args.min_snps), columns]
        write_table(subset.dropna(subset=["gene", "gene_id"]).drop_duplicates(), args.output_dir / f"{token(ct)}_eGene.csv")


if __name__ == "__main__":
    main()
