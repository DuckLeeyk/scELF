"""Export aligned gene counts, cell covariates, donor IDs and incidence matrices.

Inputs may be raw or corrected counts; this script never normalizes expression.
Use normalize_counts.R first for the manuscript corrected-UMI workflow.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ..common import (add_selection, finite, require_columns, selected_celltypes,
                      selected_genes, table, token, write_json, write_table)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_selection(p, analysis=False)
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--sample-mapping", type=Path, required=True)
    p.add_argument("--donor-covariates", type=Path, required=True)
    p.add_argument("--pcs-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="Prepared analysis root.")
    p.add_argument("--celltype-column", default="predicted.celltype.l2")
    p.add_argument("--donor-column", default="donor_id")
    p.add_argument("--umi-column", default="nCount_RNA", help="Metadata depth covariate; normalization preserves original metadata.")
    p.add_argument("--mitochondrial-column", default="percent.mt")
    p.add_argument("--missing-pcs", choices=["error", "zero"], default="error")
    args = p.parse_args(argv)
    import anndata as ad
    from scipy import sparse
    data = ad.read_h5ad(args.h5ad, backed="r")
    if data.obs_names.has_duplicates or data.var_names.has_duplicates:
        raise ValueError("Cell and gene identifiers must be unique")
    mapping = table(args.sample_mapping, dtype=str)
    require_columns(mapping, ["donor_id", "genotype_ID"])
    if mapping.donor_id.duplicated().any() or mapping.genotype_ID.duplicated().any():
        raise ValueError("Sample mapping must be one-to-one")
    lookup = mapping.set_index("donor_id").genotype_ID
    cov = table(args.donor_covariates, dtype={"genotype_ID": str})
    require_columns(cov, ["genotype_ID"])
    audit = []
    try:
        for ct in selected_celltypes(args.celltype_table, args.cell_types):
            require_columns(data.obs, [args.celltype_column, args.donor_column, args.umi_column, args.mitochondrial_column])
            cells = data.obs.loc[data.obs[args.celltype_column].astype(str).eq(ct.label)].copy()
            n_before = len(cells)
            cells["genotype_ID"] = cells[args.donor_column].astype(str).map(lookup)
            cells = cells.loc[cells.genotype_ID.isin(set(cov.genotype_ID))].copy()
            cells = cells.sort_values("genotype_ID", kind="stable")
            if cells.empty:
                raise ValueError(f"No matched donors/cells for {ct.name}")
            donors = cells.genotype_ID.drop_duplicates().tolist()
            pcs = pd.read_csv(args.pcs_dir / f"PCs_{ct.name}.csv", dtype={"barcode": str})
            require_columns(pcs, ["barcode"])
            if pcs.barcode.duplicated().any() or len(pcs.columns) < 2:
                raise ValueError("PC input needs unique barcodes and component columns")
            umi = finite(cells[args.umi_column], "UMI depth")
            if (umi <= 0).any():
                raise ValueError("UMI depth must be positive before taking log")
            xc = pd.DataFrame({"barcode": cells.index.astype(str), "log_UMI": np.log(umi),
                               "percent.mt": finite(cells[args.mitochondrial_column], "Mitochondrial fraction")})
            xc = xc.merge(pcs, on="barcode", how="left", sort=False, validate="one_to_one")
            pc_columns = [x for x in pcs.columns if x != "barcode"]
            missing = int(xc[pc_columns].isna().any(axis=1).sum())
            if missing and args.missing_pcs == "error":
                raise ValueError(f"{missing} cells have missing expression PCs")
            xc[pc_columns] = xc[pc_columns].fillna(0)
            finite(xc.drop(columns="barcode"), "Cell covariates")
            if xc.barcode.tolist() != cells.index.astype(str).tolist():
                raise ValueError("PC merge changed cell order")
            base = args.output_dir / ct.name / "data"
            base.mkdir(parents=True, exist_ok=True)
            (base / f"genotype_IDs_{ct.name}.txt").write_text("\n".join(donors) + "\n", encoding="utf-8")
            codes = pd.Categorical(cells.genotype_ID, categories=donors).codes
            z = np.zeros((len(cells), len(donors)), dtype=np.uint8)
            z[np.arange(len(cells)), codes] = 1
            np.savetxt(base / f"design_matrix_{ct.label}.txt", z, fmt="%d")
            write_table(xc, base / f"cell_covariates_{ct.label}.csv")
            genes = table(args.egene_dir / f"{ct.original}_eGene.csv", dtype=str)
            require_columns(genes, ["gene", "gene_id"])
            genes = genes.dropna(subset=["gene", "gene_id"]).drop_duplicates(["gene", "gene_id"])
            genes = genes.loc[genes.gene.isin(selected_genes(args, ct))]
            if genes.gene.duplicated().any():
                raise ValueError("Each gene symbol must map to one expression identifier")
            row_index = data.obs_names.get_indexer(cells.index)
            # Dense backed h5ad indexing requires ascending integer indices.
            # Restore the selected donor/cell order after reading each gene.
            read_order = np.argsort(row_index)
            restore_order = np.argsort(read_order)
            skipped = []
            counts_dir = base / "counts"
            counts_dir.mkdir(exist_ok=True)
            for gene, gene_id in genes[["gene", "gene_id"]].itertuples(index=False, name=None):
                if gene_id not in data.var_names:
                    skipped.append(gene)
                    continue
                column = data.var_names.get_loc(gene_id)
                # h5ad backed slicing follows the exact ordered barcode list.
                values = data[row_index[read_order], [column]].X
                values = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
                values = values[restore_order]
                values = finite(values, f"Counts for {gene}")
                if (values < 0).any() or not np.allclose(values, np.rint(values)):
                    raise ValueError("Expression must contain nonnegative integer counts")
                np.savetxt(counts_dir / f"GReX_{ct.name}_{token(gene)}.txt", values, fmt="%d")
            audit.append({"cell_type": ct.name, "input_cells": n_before, "retained_cells": len(cells),
                          "donors": len(donors), "missing_pc_cells": missing, "absent_genes": skipped})
    finally:
        data.file.close()
    write_json(audit, args.output_dir / "counts_preparation.json")


if __name__ == "__main__":
    main()
