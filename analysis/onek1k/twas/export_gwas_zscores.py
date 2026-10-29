"""Export only successfully aligned GWAS coordinates and Z-scores, with QC."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from ..common import canonical_coordinate, positive_int, require_columns, table, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Uncompressed CSV/TSV with POS,zscore,alignment_status,target_allele.")
    p.add_argument("--chunksize", type=positive_int, default=200000)
    args = p.parse_args(argv)
    if args.input.resolve() == args.output.resolve() or args.output.suffix == ".gz":
        p.error("Use a distinct, uncompressed output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    first, counts = True, Counter()
    for frame in table(args.input, chunksize=args.chunksize):
        require_columns(frame, ["POS", "zscore", "alignment_sign", "alignment_status", "target_allele"])
        good = frame.alignment_sign.isin([-1, 1]) & np.isfinite(frame.zscore)
        counts.update({"input": len(frame), "aligned": int(good.sum()), "excluded": int((~good).sum())})
        result = frame.loc[good, ["POS", "zscore", "alignment_status", "target_allele"]].copy()
        result["POS"] = result.POS.map(canonical_coordinate)
        result.to_csv(args.output, sep="," if args.output.suffix == ".csv" else "\t", index=False, mode="w" if first else "a", header=first)
        first = False
    if first:
        raise ValueError("Empty harmonized input")
    write_json({"arguments": vars(args), "counts": dict(counts)}, args.output.with_suffix(".qc.json"))


if __name__ == "__main__":
    main()
