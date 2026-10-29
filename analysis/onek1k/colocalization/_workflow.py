"""CS-pair scoring shared by BCX2 and FinnGen, with explicit PIP policies."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scelf import coloc_pip
from ..common import (METHODS, canonical_coordinate, finite, load_cs, read_pips,
                      require_columns, table, token, write_json, write_table)

RESULT_COLUMNS = ["cell_type", "trait", "method", "gene", "eqtl_cs", "gwas_cs",
                  "coloc_pip_score", "h3", "status", "n_eqtl_variants", "n_gwas_variants", "n_shared_variants", "pip_policy"]


def aligned_pips(eqtl, gwas, policy, unique_cs):
    """Align indexed PIPs; legacy also fills residual mass and normalizes."""
    union = sorted(set(eqtl.index) | set(gwas.index))
    vectors = []
    for values, fill_residual in [(eqtl, unique_cs), (gwas, True)]:
        aligned = values.reindex(union)
        missing = aligned.isna()
        fill = 0.0
        if policy == "legacy" and fill_residual and missing.any() and values.sum() <= 1:
            fill = (1 - float(values.sum())) / int(missing.sum())
        # Normalization needs an owned, writable array under pandas copy-on-write.
        vector = aligned.fillna(fill).to_numpy(dtype=float, copy=True)
        if policy == "legacy" and vector.sum() > 0:
            vector /= vector.sum()
        vectors.append(vector)
    return vectors


def load_gwas(path, source):
    raw = table(path, dtype=str)
    variant, cs = ("MarkerName", "Sentinel_MarkerName") if source == "bcx2" else ("chr:pos", "cs_id")
    require_columns(raw, [variant, cs, "PP"])
    if raw[[variant, cs, "PP"]].isna().any().any():
        raise ValueError("Missing GWAS coordinate, CS identifier or PIP")
    result = pd.DataFrame({"POS": raw[variant].map(canonical_coordinate), "cs": raw[cs], "PIP": finite(raw.PP, "GWAS PIPs")})
    if not result.PIP.between(0, 1).all() or result.duplicated(["cs", "POS"]).any():
        raise ValueError("Invalid PIPs or duplicate GWAS CS coordinates; resolve multiallelic sites first")
    return result


def index_gwas(gwas):
    """Index once per trait, rather than scanning every GWAS row for each gene."""
    by_cs = {str(k): v.set_index("POS").PIP for k, v in gwas.groupby("cs", sort=False)}
    signals = {}
    for cs, values in by_cs.items():
        for pos in values.index:
            signals.setdefault(pos, set()).add(cs)
    return by_cs, signals


def analyze_gene(pips, sets, gwas, policy, priors):
    """Yield each pair and explicit missing statuses. Marginal eQTL PIPs are used."""
    by_cs, signals = index_gwas(gwas) if isinstance(gwas, pd.DataFrame) else gwas
    for key, indices in sets.items():
        eqtl = pips.iloc[indices].set_index("POS").PIP
        candidates = sorted(set().union(*(signals.get(x, set()) for x in eqtl.index)))
        base = {"eqtl_cs": key, "n_eqtl_variants": len(eqtl), "pip_policy": policy}
        if not candidates:
            yield {**base, "status": "no_gwas_overlap", "n_shared_variants": 0}
        for cs in candidates:
            complete = by_cs[cs]
            shared = eqtl.index.intersection(complete.index)
            # Historical code discarded GWAS CS members outside the eQTL CS.
            g = complete.loc[shared] if policy == "legacy" else complete
            v1, v2 = aligned_pips(eqtl, g, policy, len(sets) == 1)
            row = {**base, "gwas_cs": cs, "n_gwas_variants": len(g), "n_shared_variants": len(shared)}
            if v1.sum() <= 0 or v2.sum() <= 0:
                yield {**row, "status": "zero_pip_support"}
            else:
                score = coloc_pip(v1, v2, **priors)
                yield {**row, "status": "scored", "coloc_pip_score": score.h4, "h3": score.h3}


def summarize(frame, threshold):
    """Count unique eQTL CSs separately within each cell type/trait/method."""
    cols = ["cell_type", "trait", "method"]
    rows = []
    for key, group in frame.groupby(cols, sort=False):
        cs = group.groupby(["gene", "eqtl_cs"], dropna=False)
        maxima = cs.coloc_pip_score.max()
        overlap = cs.n_shared_variants.max()
        rows.append(dict(zip(cols, key), total_cs=len(maxima),
            cs_with_intersection=int((overlap > 0).sum()), cs_scored=int(maxima.notna().sum()),
            cs_colocalized=int((maxima > threshold).sum()),
            median_pair_h4=group.coloc_pip_score.median(), h4_threshold=threshold))
    return pd.DataFrame(rows, columns=cols + ["total_cs", "cs_with_intersection", "cs_scored", "cs_colocalized", "median_pair_h4", "h4_threshold"])


def main_for(source, argv=None):
    p = argparse.ArgumentParser(description=f"Score {source} GWAS/eQTL CS pairs (conditional H3/H4).")
    p.add_argument("--analysis-dir", type=Path, required=True, help="Fine-mapping output root.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cell-types", nargs="+", required=True)
    p.add_argument("--traits", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=list(METHODS), help="Method directory prefixes, such as scelf or susie.")
    p.add_argument("--genes", nargs="+")
    p.add_argument("--gwas-template", required=True, help="Prepared CSV/TSV path with {trait} placeholder.")
    p.add_argument("--genome-build", required=True, help="Required declaration that both inputs use this build.")
    p.add_argument("--pip-policy", choices=["legacy", "union-zero"], required=True,
                   help="legacy: intersection first, residual filling/normalization; union-zero: complete CS union, zero padding.")
    p.add_argument("--h4-threshold", type=float, default=0.5, help="Strict > comparison.")
    p.add_argument("--p1", type=float, default=1e-4)
    p.add_argument("--p2", type=float, default=1e-4)
    p.add_argument("--p12", type=float, default=1e-7)
    args = p.parse_args(argv)
    if not 0 <= args.h4_threshold <= 1:
        p.error("--h4-threshold must lie in [0,1]")
    priors = {k: getattr(args, k) for k in ("p1", "p2", "p12")}
    if not all(np.isfinite(v) and 0 < v <= 1 for v in priors.values()):
        p.error("Prior weights must be finite and in (0,1]")
    summaries = []
    for trait in map(token, args.traits):
        path = Path(args.gwas_template.format(trait=trait))
        metadata = path.with_suffix(".metadata.json")
        if metadata.exists():
            import json
            if json.loads(metadata.read_text())["genome_build"] != args.genome_build:
                raise ValueError(f"GWAS build mismatch: {path}")
        gwas = index_gwas(load_gwas(path, source))
        for ct in map(token, args.cell_types):
            for method in map(token, args.methods):
                base = args.analysis_dir / ct
                directory = base / f"{method}_cs"
                if not directory.is_dir():
                    raise FileNotFoundError(directory)
                rows = []
                for cs_path in sorted(directory.glob("*.json")):
                    gene = cs_path.stem
                    if args.genes and gene not in args.genes:
                        continue
                    sets = load_cs(cs_path)
                    if not sets:
                        continue
                    pips = read_pips(base / f"{method}_pip" / f"{gene}.csv")
                    sets = load_cs(cs_path, len(pips))
                    for row in analyze_gene(pips, sets, gwas, args.pip_policy, priors):
                        rows.append({"cell_type": ct, "trait": trait, "method": method, "gene": gene, **row})
                frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
                write_table(frame, args.output_dir / ct / f"{trait}_{method}_coloc.tsv")
                result = summarize(frame, args.h4_threshold)
                if result.empty:
                    result = pd.DataFrame([dict(cell_type=ct, trait=trait, method=method, total_cs=0,
                        cs_with_intersection=0, cs_scored=0, cs_colocalized=0, median_pair_h4=np.nan, h4_threshold=args.h4_threshold)])
                summaries.append(result)
    write_table(pd.concat(summaries, ignore_index=True), args.output_dir / f"{source}_summary.tsv")
    write_json({"source": source, "arguments": vars(args)}, args.output_dir / f"{source}_run.json")
