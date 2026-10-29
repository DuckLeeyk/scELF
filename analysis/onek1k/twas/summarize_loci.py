"""Join TWAS tests to maximum per-gene colocalization H4, preserving missingness."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ..common import require_columns, table, write_json, write_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--twas", type=Path, nargs="+", required=True)
    p.add_argument("--coloc", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--traits", nargs="+")
    p.add_argument("--cell-types", nargs="+")
    p.add_argument("--genes", nargs="+")
    p.add_argument("--flip-methods", nargs="+", default=[], help="Optional explicit sign reversal for selected method labels; never applied automatically.")
    args = p.parse_args(argv)
    keys = ["trait", "cell_type", "method", "gene"]
    twas = pd.concat([table(x) for x in args.twas], ignore_index=True)
    coloc = pd.concat([table(x) for x in args.coloc], ignore_index=True)
    require_columns(twas, keys + ["zscore", "pvalue"])
    require_columns(coloc, keys + ["coloc_pip_score", "eqtl_cs", "gwas_cs", "status", "pip_policy"])
    if twas.duplicated(keys).any():
        raise ValueError("Duplicate TWAS tests")
    if coloc.pip_policy.nunique() > 1:
        raise ValueError("Do not combine different PIP policies")
    # Take the best scored CS pair, while retaining a status for unscored genes.
    best = coloc.sort_values("coloc_pip_score", ascending=False, na_position="last").drop_duplicates(keys)
    best = best[keys + ["coloc_pip_score", "eqtl_cs", "gwas_cs", "status", "pip_policy"]].rename(columns={"status": "coloc_status"})
    result = twas.merge(best, on=keys, how="outer", validate="one_to_one")
    result["coloc_status"] = result.coloc_status.fillna("no_coloc_record")
    if "status" in result:
        result["status"] = result.status.fillna("no_twas_record")
    result["zscore_original"] = result.zscore
    result["sign_flipped"] = result.method.isin(args.flip_methods)
    result.loc[result.sign_flipped, "zscore"] *= -1
    result["direction"] = np.select([result.zscore > 0, result.zscore < 0, result.zscore.eq(0)], ["positive", "negative", "zero"], default="missing")
    for column, values in [("trait", args.traits), ("cell_type", args.cell_types), ("gene", args.genes)]:
        if values:
            result = result.loc[result[column].isin(values)]
    write_table(result.sort_values(keys), args.output)
    write_json({"arguments": vars(args), "rows": len(result)}, args.output.with_suffix(".run.json"))


if __name__ == "__main__":
    main()
