"""Summarize TWAS tests and hits by trait, cell type and method."""
import argparse
from pathlib import Path
import pandas as pd
from ..common import finite, require_columns, table, write_json, write_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--p-threshold", type=float, default=1e-6, help="Unadjusted manuscript display threshold; strict < comparison.")
    p.add_argument("--traits", nargs="+")
    p.add_argument("--max-missing-rate", type=float, default=1.0)
    args = p.parse_args(argv)
    if not 0 < args.p_threshold <= 1 or not 0 <= args.max_missing_rate <= 1:
        p.error("Invalid p-value or missing-rate threshold")
    data = pd.concat([table(x) for x in args.inputs], ignore_index=True)
    keys = ["trait", "cell_type", "method", "gene"]
    require_columns(data, keys + ["zscore", "pvalue", "missing_rate", "status"])
    if data.duplicated(keys).any():
        raise ValueError("Duplicate TWAS tests")
    finite(data[["zscore", "pvalue", "missing_rate"]], "TWAS statistics")
    if args.traits:
        data = data.loc[data.trait.isin(args.traits)]
    data = data.loc[(data.status == "scored") & (data.missing_rate <= args.max_missing_rate)].copy()
    data["significant"] = data.pvalue < args.p_threshold
    summary = data.groupby(keys[:3], as_index=False).agg(n_tests=("gene", "size"), n_significant=("significant", "sum"), min_pvalue=("pvalue", "min"))
    write_table(summary, args.output_dir / "twas_summary.tsv")
    write_table(data.loc[data.significant].sort_values("pvalue"), args.output_dir / "twas_hits.tsv")
    write_json({"arguments": vars(args), "included_tests": len(data)}, args.output_dir / "summary_run.json")


if __name__ == "__main__":
    main()
