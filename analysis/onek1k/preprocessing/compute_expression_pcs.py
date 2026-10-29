"""Compute per-cell-type expression components using the source TruncatedSVD algorithm.

TruncatedSVD does not center expression columns. It is retained explicitly for
compatibility with the original analysis, rather than relabelled centered PCA.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ..common import selected_celltypes, positive_int, write_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", required=True, type=Path)
    p.add_argument("--celltype-table", required=True, type=Path)
    p.add_argument("--cell-types", nargs="+")
    p.add_argument("--celltype-column", default="predicted.celltype.l2")
    p.add_argument("--fit-components", type=positive_int, default=50)
    p.add_argument("--save-components", type=positive_int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args(argv)
    if args.save_components > args.fit_components:
        p.error("--save-components must not exceed --fit-components")
    import anndata as ad
    from sklearn.decomposition import TruncatedSVD
    data = ad.read_h5ad(args.h5ad)
    for ct in selected_celltypes(args.celltype_table, args.cell_types):
        mask = data.obs[args.celltype_column].astype(str).eq(ct.label).to_numpy()
        x = data.X[mask].astype(np.float32)
        n = min(args.fit_components, min(x.shape) - 1)
        if n < args.save_components:
            raise ValueError(f"Not enough cells/genes for {ct.name} components")
        scores = TruncatedSVD(n_components=n, random_state=args.seed).fit_transform(x)[:, :args.save_components]
        sd = scores.std(axis=0)
        scores = (scores - scores.mean(axis=0)) / np.where(sd < 1e-12, 1, sd)
        scores[:, sd < 1e-12] = 0
        frame = pd.DataFrame(scores, columns=[f"gene_PC{i+1}" for i in range(args.save_components)])
        frame.insert(0, "barcode", data.obs_names[mask].astype(str))
        write_table(frame, args.output_dir / f"PCs_{ct.name}.csv")


if __name__ == "__main__":
    main()
