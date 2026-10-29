"""Select the union of genes with nonempty credible sets across methods."""
import argparse
from pathlib import Path
import pandas as pd
from ..common import METHODS, load_cs, token, write_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--cell-types", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--genes", nargs="+")
    p.add_argument("--output", type=Path, required=True, help="Candidate TSV: cell_type,gene,cs_methods.")
    args = p.parse_args(argv)
    rows = []
    for ct in map(token, args.cell_types):
        support = {}
        for method in map(token, args.methods):
            directory = args.analysis_dir / ct / f"{method}_cs"
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for path in sorted(directory.glob("*.json")):
                if (not args.genes or path.stem in args.genes) and load_cs(path):
                    support.setdefault(path.stem, []).append(method)
        rows.extend({"cell_type": ct, "gene": gene, "cs_methods": ";".join(methods)} for gene, methods in sorted(support.items()))
    write_table(pd.DataFrame(rows, columns=["cell_type", "gene", "cs_methods"]), args.output)


if __name__ == "__main__":
    main()
