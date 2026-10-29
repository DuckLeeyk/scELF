"""Prepare donor covariates, genotype PCs, sex indicators and pool indicators."""
import argparse
from pathlib import Path
import pandas as pd
from ..common import align_rows, finite, require_columns, table, write_json, write_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--sample-mapping", type=Path, required=True, help="CSV/TSV with donor_id and genotype_ID.")
    p.add_argument("--genotype-pcs", type=Path, required=True, help="Headerless PLINK eigenvec: FID IID PC1 ...")
    p.add_argument("--n-pcs", type=int, default=6)
    p.add_argument("--donor-column", default="donor_id")
    p.add_argument("--pool-column", default="pool_number")
    p.add_argument("--age-column", default="age")
    p.add_argument("--sex-column", default="sex")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.n_pcs < 1:
        p.error("--n-pcs must be positive")
    import anndata as ad
    data = ad.read_h5ad(args.h5ad, backed="r")
    columns = [args.donor_column, args.pool_column, args.age_column, args.sex_column]
    require_columns(data.obs, columns)
    donors = data.obs[columns].drop_duplicates().copy()
    data.file.close()
    if donors[args.donor_column].duplicated().any():
        raise ValueError("Inconsistent age/sex/pool metadata within a donor")
    mapping = table(args.sample_mapping, dtype=str)
    require_columns(mapping, ["donor_id", "genotype_ID"])
    if mapping.donor_id.duplicated().any() or mapping.genotype_ID.duplicated().any():
        raise ValueError("Sample mapping must be one-to-one")
    donors[args.donor_column] = donors[args.donor_column].astype(str)
    donors = donors.merge(mapping, left_on=args.donor_column, right_on="donor_id", validate="one_to_one").dropna()
    pc_names = [f"PC{i+1}" for i in range(args.n_pcs)]
    pcs = pd.read_csv(args.genotype_pcs, sep=r"\s+", header=None, dtype={1: str})
    if pcs.shape[1] < args.n_pcs + 2:
        raise ValueError("Insufficient genotype PC columns")
    pcs = pcs.iloc[:, :args.n_pcs + 2]
    pcs.columns = ["FID", "genotype_ID", *pc_names]
    donors = donors.merge(pcs.drop(columns="FID"), on="genotype_ID", validate="one_to_one")
    if donors.empty:
        raise ValueError("No donors matched across metadata, mapping, and PCs")
    donors = donors.sort_values("genotype_ID", kind="stable").reset_index(drop=True)
    result = donors[["genotype_ID", args.age_column, *pc_names]].rename(columns={args.age_column: "age"})
    levels = {}
    for column, prefix in [(args.sex_column, "is"), (args.pool_column, "is_pool")]:
        levels[column] = sorted(donors[column].astype(str).unique())
        categorical = pd.Categorical(donors[column].astype(str), categories=levels[column])
        result = pd.concat([result, pd.get_dummies(categorical, prefix=prefix, drop_first=True, dtype=int)], axis=1)
    finite(result.drop(columns="genotype_ID"), "Donor covariates")
    write_table(result, args.output)
    write_json({"category_levels": levels, "reference_level": {k: v[0] for k,v in levels.items()},
                "n_donors": len(result)}, args.output.with_suffix(".encoding.json"))


if __name__ == "__main__":
    main()
