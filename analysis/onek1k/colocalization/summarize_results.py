"""Summarize colocalization and export unique significant gene sets."""
import argparse
from pathlib import Path
import pandas as pd
from ._workflow import RESULT_COLUMNS, summarize
from ..common import require_columns, table, write_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="Explicit *_coloc.tsv paths.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--h4-threshold", type=float, default=0.5)
    p.add_argument("--gene-annotation", type=Path, help="Optional unique-gene table with gene plus annotation columns.")
    args = p.parse_args(argv)
    if not 0 <= args.h4_threshold <= 1:
        p.error("--h4-threshold must lie in [0,1]")
    frames = [table(path) for path in args.inputs]
    for frame in frames:
        require_columns(frame, RESULT_COLUMNS)
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["cell_type", "trait", "method", "gene", "eqtl_cs", "gwas_cs"]).any():
        raise ValueError("Duplicate CS pairs; supply each result once")
    if data.pip_policy.nunique() > 1:
        raise ValueError("Summarize each PIP policy separately")
    write_table(summarize(data, args.h4_threshold), args.output_dir / "cs_summary.tsv")
    positive = data.loc[data.coloc_pip_score > args.h4_threshold]
    genes = positive.groupby(["cell_type", "trait", "method", "gene"], as_index=False).agg(max_h4=("coloc_pip_score", "max"))
    if args.gene_annotation:
        annotation = table(args.gene_annotation)
        require_columns(annotation, ["gene"])
        genes = genes.merge(annotation, on="gene", how="left", validate="many_to_one")
    write_table(genes, args.output_dir / "colocalized_genes.tsv")
    write_table(genes[["method", "gene"]].drop_duplicates(), args.output_dir / "unique_method_genes.tsv")


if __name__ == "__main__":
    main()
