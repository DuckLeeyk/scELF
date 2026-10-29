"""Shared input validation and CLI conventions for the OneK1K workflows.

This module has no analysis side effects on import. Executable scripts live in
preprocessing/, finemapping/, colocalization/, and twas/.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

METHODS = ("scelf", "susie", "logsusie", "saigeqtl_susie")


def table(path, **kwargs):
    """Read CSV or tab-separated tables, including compressed files."""
    name = str(path).lower().removesuffix(".gz")
    return pd.read_csv(path, sep="," if name.endswith(".csv") else "\t", **kwargs)


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def token(value):
    """Reject path components instead of silently changing gene/cell IDs."""
    value = str(value).strip()
    if not value or value in {".", ".."} or re.search(r'[\\/:*?"<>|\x00-\x1f]', value):
        raise ValueError(f"Unsafe file-name component: {value!r}")
    return value


def write_table(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="," if str(path).removesuffix(".gz").endswith(".csv") else "\t", index=False)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_dict()
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=jsonable) + "\n", encoding="utf-8")


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return result


def add_selection(parser, *, analysis=True):
    if analysis:
        parser.add_argument("--analysis-dir", required=True, type=Path, help="Prepared per-cell-type data directory.")
    parser.add_argument("--celltype-table", required=True, type=Path, help="CSV/TSV with predicted.celltype.l2, l2_name, 14_celltype_name.")
    parser.add_argument("--egene-dir", required=True, type=Path, help="Directory of {14_celltype_name}_eGene.csv files.")
    parser.add_argument("--cell-types", nargs="+", help="l2_name values to analyze; default: all mapped types.")
    parser.add_argument("--genes", nargs="+", help="Gene symbols to analyze; default: all eligible genes.")
    parser.add_argument("--gene-pairs", type=Path, help="Optional TSV with cell_type,gene; filters existing eGene lists.")


def selected_celltypes(path, names=None):
    frame = table(path, dtype=str)
    columns = ["predicted.celltype.l2", "l2_name", "14_celltype_name"]
    require_columns(frame, columns)
    frame = frame.dropna(subset=columns).copy()
    for column in columns:
        frame[column] = frame[column].str.strip()
    frame = frame.loc[frame[columns].ne("").all(axis=1)]
    if frame.l2_name.duplicated().any() or frame[columns[0]].duplicated().any():
        raise ValueError("Cell-type labels and l2_name values must be unique")
    if names:
        unknown = set(names) - set(frame.l2_name)
        if unknown:
            raise ValueError(f"Unknown --cell-types: {sorted(unknown)}")
        frame = frame.loc[frame.l2_name.isin(names)]
    if frame.empty:
        raise ValueError("No mapped cell types selected")
    for raw, name, original in frame[columns].itertuples(index=False, name=None):
        yield CellType(token(raw), token(name), token(original))


@dataclass(frozen=True)
class CellType:
    label: str
    name: str
    original: str


def gene_pairs(path):
    pairs = table(path, dtype=str)
    require_columns(pairs, ["cell_type", "gene"])
    pairs = pairs.dropna(subset=["cell_type", "gene"])
    return {ct: set(part.gene.astype(str).str.strip()) for ct, part in pairs.groupby("cell_type")}


def selected_genes(args, ct):
    frame = table(args.egene_dir / f"{ct.original}_eGene.csv", dtype=str)
    require_columns(frame, ["gene"])
    genes = list(dict.fromkeys(token(x) for x in frame.gene.dropna() if str(x).strip()))
    if args.genes:
        genes = [g for g in genes if g in args.genes]
    if args.gene_pairs:
        allowed = gene_pairs(args.gene_pairs).get(ct.name, set())
        genes = [g for g in genes if g in allowed]
    return genes


def load_donors(path):
    donors = [x.strip() for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if not donors or len(set(donors)) != len(donors):
        raise ValueError(f"Empty or duplicate donor IDs: {path}")
    return donors


def align_rows(frame, ids, label):
    """Reindex by identifiers; equal dimensions alone do not establish alignment."""
    frame = frame.copy()
    frame.index = frame.index.astype(str)
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate {label} IDs")
    missing = set(ids) - set(frame.index)
    if missing:
        raise ValueError(f"Missing {label} IDs: {sorted(missing)[:5]}")
    return frame.loc[ids]


def finite(values, label):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN/Inf")
    return values


def standardize_genotype(values):
    values = finite(values, "Genotype")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Genotype must contain >=2 donors and >=1 variant")
    sd = values.std(axis=0, ddof=0)
    if (sd == 0).any():
        raise ValueError("Genotype contains constant variants")
    return np.asfortranarray((values - values.mean(axis=0)) / sd)


def load_genotype(path, donors):
    # Pandas can infer numeric index values even with dtype={0: str} when
    # index_col is used. Read identifiers as a normal string column first.
    frame = pd.read_csv(path, dtype=str)
    frame = frame.set_index(frame.columns[0]).apply(pd.to_numeric, errors="raise")
    keys = [canonical_coordinate(x) for x in frame.columns]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate genotype coordinates: {path}")
    return align_rows(frame, donors, "genotype donor")


def load_shared(args, ct):
    data = args.analysis_dir / ct.name / "data"
    donors = load_donors(data / f"genotype_IDs_{ct.name}.txt")
    cov = table(args.donor_covariates, dtype={args.donor_id_column: str})
    require_columns(cov, [args.donor_id_column])
    xd = finite(align_rows(cov.set_index(args.donor_id_column), donors, "covariate donor"), "Donor covariates")
    cells = pd.read_csv(data / f"cell_covariates_{ct.label}.csv", dtype={"barcode": str})
    require_columns(cells, ["barcode"])
    if cells.barcode.duplicated().any():
        raise ValueError("Duplicate cell barcodes")
    xc = finite(cells.drop(columns="barcode"), "Cell covariates")
    z = np.loadtxt(data / f"design_matrix_{ct.label}.txt", ndmin=2)
    if z.shape != (len(cells), len(donors)) or not np.isin(z, [0, 1]).all() or not np.all(z.sum(axis=1) == 1):
        raise ValueError("Invalid cell-to-donor design matrix")
    return data, donors, xc, xd, z


def load_gene(data, ct, gene, donors, n_cells):
    y = finite(np.loadtxt(data / "counts" / f"GReX_{ct.name}_{gene}.txt", ndmin=1), "Counts")
    if y.shape != (n_cells,) or (y < 0).any() or not np.allclose(y, np.rint(y)):
        raise ValueError("Counts must be aligned, nonnegative integers")
    frame = load_genotype(data / "genotype" / f"genotype_{ct.name}_{gene}.csv", donors)
    return y, frame, standardize_genotype(frame)


def load_cs(path, size=None):
    sets = json.loads(Path(path).read_text(encoding="utf-8")).get("cs") or {}
    if not isinstance(sets, dict):
        raise ValueError(f"Expected a CS dictionary: {path}")
    result = {}
    for key, indices in sets.items():
        if indices is None or len(indices) == 0:
            continue
        raw = np.asarray(indices)
        idx = raw.astype(int)
        if not np.array_equal(raw, idx) or (idx < 0).any() or (size is not None and (idx >= size).any()):
            raise ValueError(f"Out-of-range or noninteger CS indices: {path}/{key}")
        result[str(key)] = np.unique(idx)
    return result


def canonical_coordinate(value):
    match = re.fullmatch(r"(?:chr)?([0-9]+|X|Y|M|MT):([0-9]+)(?:[_:].*)?", str(value).strip(), re.I)
    if not match or int(match[2]) < 1:
        raise ValueError(f"Invalid variant coordinate: {value!r}")
    chrom = match[1].upper()
    if chrom.isdigit():
        chrom = str(int(chrom))
    chrom = {"23": "X", "24": "Y", "25": "MT", "M": "MT"}.get(chrom, chrom)
    if chrom == "0":
        raise ValueError("Chromosome 0 is not a genomic coordinate")
    return f"{chrom}:{int(match[2])}"


def read_pips(path):
    frame = pd.read_csv(path)
    if {"variant_id", "pip"}.issubset(frame):
        frame = frame.rename(columns={"variant_id": "POS", "pip": "PIP"})
    require_columns(frame, ["POS", "PIP"])
    frame["POS"] = frame.POS.map(canonical_coordinate)
    frame["PIP"] = finite(frame.PIP, "PIPs")
    if frame.POS.duplicated().any() or not frame.PIP.between(0, 1).all():
        raise ValueError(f"Duplicate variants or PIPs outside [0,1]: {path}")
    return frame


def normalize_weights(weights, genotype):
    weights = finite(weights, "Weights").reshape(-1)
    if genotype.shape[1] != weights.size:
        raise ValueError("Genotype and weight dimensions disagree")
    variance = float(np.mean(np.square(genotype @ weights)))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("TWAS genetic variance is non-positive")
    return weights / np.sqrt(variance)
