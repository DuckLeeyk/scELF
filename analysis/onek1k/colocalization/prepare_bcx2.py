"""Prepare one BCX2 credible-set table without a fixed trait list."""
import argparse
from pathlib import Path
import pandas as pd
from ..common import canonical_coordinate, finite, require_columns, table, write_json, write_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--genome-build", required=True, help="Coordinate build of both input and downstream eQTLs, e.g. GRCh37. No conversion is performed.")
    p.add_argument("--variant-column", default="MarkerName")
    p.add_argument("--cs-column", default="Sentinel_MarkerName")
    p.add_argument("--pip-column", default="PP")
    args = p.parse_args(argv)
    raw = table(args.input, dtype=str)
    require_columns(raw, [args.variant_column, args.cs_column, args.pip_column])
    pos = raw[args.variant_column].map(canonical_coordinate)
    # Keep marker alleles for audits; the score itself matches coordinates.
    result = pd.DataFrame({"MarkerName": pos,
        "Sentinel_MarkerName": raw[args.cs_column].map(canonical_coordinate),
        "PP": finite(raw[args.pip_column], "GWAS PIPs"), "source_variant": raw[args.variant_column]})
    result["chr"] = pos.str.split(":").str[0]
    result["pos"] = pos.str.split(":").str[1].astype(int)
    if not result.PP.between(0, 1).all() or result.duplicated(["Sentinel_MarkerName", "MarkerName"]).any():
        raise ValueError("Invalid PIPs or duplicate coordinates within a GWAS credible set")
    write_table(result, args.output)
    write_json({"arguments": vars(args), "genome_build": args.genome_build, "rows": len(result)}, args.output.with_suffix(".metadata.json"))


if __name__ == "__main__":
    main()
