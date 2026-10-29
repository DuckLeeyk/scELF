"""Prepare FinnGen signal probabilities with explicit genome builds.

Cross-build analysis requires an auditable coordinate map supplied by the user;
network-based coordinate annotation is intentionally replaced by a local input.
"""
import argparse
from pathlib import Path
import pandas as pd
from ..common import canonical_coordinate, finite, require_columns, table, write_json, write_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input-build", required=True)
    p.add_argument("--output-build", required=True)
    p.add_argument("--coordinate-map", type=Path, help="CSV/TSV: source_POS,target_POS, 1-based chr:pos; one-to-one mappings only.")
    p.add_argument("--unmapped", choices=["error", "drop"], default="error")
    p.add_argument("--chromosome-column", default="chromosome")
    p.add_argument("--position-column", default="position")
    p.add_argument("--region-column", default="region")
    p.add_argument("--cs-column", default="cs")
    p.add_argument("--pip-column", default="cs_specific_prob")
    p.add_argument("--include-sex-chromosomes", action="store_true", help="Default retains autosomes, as in the source script.")
    args = p.parse_args(argv)
    if args.input_build != args.output_build and not args.coordinate_map:
        p.error("Different builds require --coordinate-map; no implicit conversion")
    raw = table(args.input, dtype=str)
    cols = [args.chromosome_column, args.position_column, args.region_column, args.cs_column, args.pip_column]
    require_columns(raw, cols)
    if raw[cols].isna().any().any():
        raise ValueError("Missing FinnGen coordinate, signal identifier, or probability")
    raw["source_POS"] = (raw[args.chromosome_column] + ":" + raw[args.position_column]).map(canonical_coordinate)
    if not args.include_sex_chromosomes:
        raw = raw.loc[raw.source_POS.str.split(":").str[0].isin([str(i) for i in range(1, 23)])].copy()
    n_input = len(raw)
    if args.coordinate_map:
        mapping = table(args.coordinate_map, dtype=str)
        require_columns(mapping, ["source_POS", "target_POS"])
        for col in ["source_POS", "target_POS"]:
            mapping[col] = mapping[col].map(canonical_coordinate)
            if mapping[col].duplicated().any():
                raise ValueError("Coordinate map must be one-to-one")
        raw["POS"] = raw.source_POS.map(mapping.set_index("source_POS").target_POS)
        if raw.POS.isna().any() and args.unmapped == "error":
            raise ValueError("Some FinnGen coordinates are absent from --coordinate-map")
        raw = raw.dropna(subset=["POS"])
    else:
        raw["POS"] = raw.source_POS
    result = pd.DataFrame({"cs_id": raw[args.region_column] + "_" + raw[args.cs_column],
        "chr:pos": raw.POS, "PP": finite(raw[args.pip_column], "GWAS PIPs"), "source_POS": raw.source_POS})
    result["chr"] = result["chr:pos"].str.split(":").str[0]
    result["pos"] = result["chr:pos"].str.split(":").str[1].astype(int)
    if not result.PP.between(0, 1).all() or result.duplicated(["cs_id", "chr:pos"]).any():
        raise ValueError("Invalid PIPs or duplicate coordinates within a GWAS credible set")
    write_table(result.sort_values(["cs_id", "PP"], ascending=[True, False]), args.output)
    write_json({"arguments": vars(args), "genome_build": args.output_build,
        "eligible_input_rows": n_input, "unmapped_rows": n_input-len(result), "rows": len(result)}, args.output.with_suffix(".metadata.json"))


if __name__ == "__main__":
    main()
