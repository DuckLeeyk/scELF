"""Subset a released OneK1K eQTL table to an explicit gene list, retaining SNP IDs."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import pandas as pd
from ..common import positive_int, table, write_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Released tab-separated eQTL table, optionally gzip compressed.")
    p.add_argument("--gene-list", type=Path, required=True, help="One gene symbol per line, without a header.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunksize", type=positive_int, default=1000000)
    p.add_argument("--columns", nargs="+", default=["RSID", "GENE", "CHR", "POS"])
    args = p.parse_args(argv)
    if "GENE" not in args.columns:
        p.error("--columns must include GENE")
    genes = set(args.gene_list.read_text(encoding="utf-8-sig").split())
    if not genes:
        raise ValueError("Gene list is empty")
    # Disk-backed global deduplication avoids reloading the whole filtered table.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scelf_eqtl_") as temporary:
        with closing(sqlite3.connect(str(Path(temporary) / "rows.sqlite"))) as conn, conn:
            initialized = False
            for chunk in pd.read_csv(args.input, sep="\t", usecols=args.columns, chunksize=args.chunksize, dtype=str):
                kept = chunk.loc[chunk.GENE.isin(genes), args.columns].drop_duplicates()
                if kept.empty:
                    continue
                kept.to_sql("records", conn, if_exists="append", index=False)
                initialized = True
            if not initialized:
                write_table(pd.DataFrame(columns=args.columns), args.output)
                return
            # Column names are validated before interpolation into this local query.
            if any(not col.replace("_", "").isalnum() for col in args.columns):
                raise ValueError("--columns must contain simple identifier names")
            query = 'SELECT DISTINCT ' + ','.join('"'+x+'"' for x in args.columns) + ' FROM records'
            first = True
            for chunk in pd.read_sql_query(query, conn, chunksize=args.chunksize):
                chunk.to_csv(args.output, sep="," if args.output.name.removesuffix(".gz").endswith(".csv") else "\t",
                             index=False, mode="w" if first else "a", header=first, compression="infer")
                first = False


if __name__ == "__main__":
    main()
