"""Stream GWAS statistics and align their effects to a specified PLINK allele.

Only biallelic A/C/G/T SNPs are aligned. Ambiguous BIM coordinates are rejected.
Unmatched rows retain an explicit status; the safe default drops these rows.
"""
import argparse
from collections import Counter
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import numpy as np
import pandas as pd
from ..common import canonical_coordinate, positive_int, require_columns, table, write_json

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def build_bim_database(path, database, chunksize):
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute("CREATE TABLE alleles (POS TEXT PRIMARY KEY, A1 TEXT, A2 TEXT)")
        for part in pd.read_csv(path, sep=r"\s+", header=None, names=["chr", "rsid", "cm", "bp", "A1", "A2"], dtype=str, chunksize=chunksize):
            part["POS"] = (part.chr + ":" + part.bp).map(canonical_coordinate)
            for col in ["A1", "A2"]:
                part[col] = part[col].str.strip().str.upper()
            # A position-only join cannot distinguish multiallelic records, even
            # when one of the records would subsequently be filtered as non-SNP.
            try:
                conn.executemany("INSERT INTO alleles VALUES (?, ?, ?)", part[["POS", "A1", "A2"]].itertuples(index=False, name=None))
            except sqlite3.IntegrityError as exc:
                raise ValueError("BIM has duplicate chr:pos; resolve ambiguous records before harmonization") from exc


def harmonize_chunk(chunk, bim, effect_col, other_col, z_col, beta_col, target, palindromic):
    require_columns(chunk, ["POS", effect_col, other_col, z_col] + ([beta_col] if beta_col else []))
    # Prefix reference fields so GWAS columns named A1/A2 remain usable as CLI
    # effect/non-effect inputs, without pandas adding ambiguous merge suffixes.
    chunk = chunk.merge(bim.rename(columns={"A1": "bim_A1", "A2": "bim_A2"}), on="POS", how="left", validate="many_to_one", sort=False)
    effect = chunk[effect_col].astype(str).str.strip().str.upper()
    other = chunk[other_col].astype(str).str.strip().str.upper()
    wanted, opposite = chunk[f"bim_{target}"], chunk["bim_A1" if target == "A2" else "bim_A2"]
    valid = wanted.str.fullmatch("[ACGT]", na=False) & opposite.str.fullmatch("[ACGT]", na=False) & wanted.ne(opposite)
    direct_plus = valid & effect.eq(wanted) & other.eq(opposite)
    direct_minus = valid & effect.eq(opposite) & other.eq(wanted)
    complement_wanted = wanted.str.translate(COMPLEMENT)
    complement_other = opposite.str.translate(COMPLEMENT)
    strand_plus = valid & effect.eq(complement_wanted) & other.eq(complement_other) & ~direct_plus & ~direct_minus
    strand_minus = valid & effect.eq(complement_other) & other.eq(complement_wanted) & ~direct_plus & ~direct_minus
    status = pd.Series("unmatched_alleles", index=chunk.index)
    status.loc[wanted.isna() | opposite.isna()] = "missing_in_bim"
    sign = pd.Series(np.nan, index=chunk.index)
    for mask, label, value in [(direct_plus, "aligned", 1), (direct_minus, "flipped", -1), (strand_plus, "strand_aligned", 1), (strand_minus, "strand_flipped", -1)]:
        sign.loc[mask], status.loc[mask] = value, label
    # Palindromic SNPs cannot be oriented by allele strings alone across strands.
    palindrome = valid & wanted.eq(complement_other)
    if palindromic == "drop":
        sign.loc[palindrome], status.loc[palindrome] = np.nan, "palindromic"
    original = pd.to_numeric(chunk[z_col], errors="coerce")
    invalid = ~np.isfinite(original)
    sign.loc[invalid], status.loc[invalid] = np.nan, "invalid_z"
    chunk["z_original"], chunk["alignment_sign"], chunk["alignment_status"] = original, sign, status
    chunk["target_allele"] = target
    chunk["zscore"] = original * sign
    if beta_col:
        chunk["beta_original"] = pd.to_numeric(chunk[beta_col], errors="raise")
        chunk["beta_aligned"] = chunk.beta_original * sign
    return chunk


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--bim", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Uncompressed CSV or TSV; alignment QC is saved beside it.")
    p.add_argument("--genome-build", required=True, help="Declare a common GWAS/BIM build; this command does not lift coordinates.")
    p.add_argument("--position-column", default="rs_number", help="chr:pos or chr:pos_A_G field.")
    p.add_argument("--effect-allele-column", required=True, help="Must follow the GWAS documentation.")
    p.add_argument("--non-effect-allele-column", required=True)
    p.add_argument("--z-column", default="z")
    p.add_argument("--beta-column")
    p.add_argument("--target-allele", choices=["A1", "A2"], required=True, help="Must equal the dosage allele used to fit weights.")
    p.add_argument("--palindromic", choices=["drop", "keep"], default="drop", help="keep explicitly assumes matching strands.")
    p.add_argument("--unmatched", choices=["drop", "keep", "error"], default="drop", help="keep writes NA aligned statistics and retains originals for auditing.")
    p.add_argument("--chunksize", type=positive_int, default=200000)
    args = p.parse_args(argv)
    if args.output.suffix == ".gz" or args.input.resolve() == args.output.resolve():
        p.error("Use a distinct, uncompressed output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts, written = Counter(), 0
    # Stage output locally: a failed chunk never leaves a partly valid final file.
    with tempfile.TemporaryDirectory(prefix="scelf_harmonize_") as temporary:
        database = Path(temporary) / "bim.sqlite"
        staged = Path(temporary) / "aligned.tsv"
        build_bim_database(args.bim, database, args.chunksize)
        first = True
        with closing(sqlite3.connect(database)) as conn, conn:
            for chunk in table(args.input, chunksize=args.chunksize):
                require_columns(chunk, [args.position_column])
                chunk["POS"] = chunk[args.position_column].map(canonical_coordinate)
                pd.DataFrame({"POS": chunk.POS.unique()}).to_sql("query", conn, if_exists="replace", index=False)
                bim = pd.read_sql_query("SELECT q.POS,a.A1,a.A2 FROM query q LEFT JOIN alleles a ON q.POS=a.POS", conn)
                result = harmonize_chunk(chunk, bim, args.effect_allele_column, args.non_effect_allele_column,
                    args.z_column, args.beta_column, args.target_allele, args.palindromic)
                counts.update(result.alignment_status.value_counts().to_dict())
                unmatched = result.alignment_sign.isna()
                if unmatched.any() and args.unmatched == "error":
                    raise ValueError(f"Unharmonizable rows: {result.loc[unmatched, 'alignment_status'].value_counts().to_dict()}")
                if args.unmatched == "drop":
                    result = result.loc[~unmatched]
                result.to_csv(staged, sep="," if args.output.suffix == ".csv" else "\t", index=False, mode="w" if first else "a", header=first)
                first = False
                written += len(result)
        if first:
            raise ValueError("Empty GWAS input")
        import shutil
        shutil.copyfile(staged, args.output)
    write_json({"arguments": vars(args), "status_counts": dict(counts), "output_rows": written}, args.output.with_suffix(".qc.json"))


if __name__ == "__main__":
    main()
