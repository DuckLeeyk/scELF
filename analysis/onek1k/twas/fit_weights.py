"""Generate LD-normalized TWAS weights from saved effects or an explicit refit.

Select candidates using --gene-pairs from select_cs_genes. Default effects mode
reuses fine-mapping posterior means; refit uses the same shared fitting code.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ..common import (METHODS, canonical_coordinate, finite, load_donors, load_gene,
    load_genotype, load_shared, normalize_weights, require_columns, selected_celltypes,
    selected_genes, standardize_genotype, write_json, write_table)
from ..finemapping._workflow import fit_model, fit_rss, parser_for


def main(argv=None):
    # Parse the method first to expose the appropriate donor-covariate requirement.
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--method", choices=METHODS, default="scelf")
    method = preliminary.parse_known_args(argv)[0].method
    p = parser_for(method, require_covariates=False)
    p.description = __doc__
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--mode", choices=["effects", "refit"], default="effects")
    p.add_argument("--effects-dir", type=Path, help="Fine-mapping results root; defaults to analysis-dir.")
    args = p.parse_args(argv)
    if args.mode == "refit" and method != "saigeqtl_susie" and not args.donor_covariates:
        p.error("--donor-covariates is required for individual-level refits")
    if not args.gene_pairs:
        p.error("--gene-pairs is required: use select_cs_genes to build the union candidate set")
    audit, failures = [], []
    for ct in selected_celltypes(args.celltype_table, args.cell_types):
        data = args.analysis_dir / ct.name / "data"
        donors = load_donors(data / f"genotype_IDs_{ct.name}.txt")
        if args.mode == "refit" and method != "saigeqtl_susie":
            data, donors, xc, xd, z = load_shared(args, ct)
        for gene in selected_genes(args, ct):
            try:
                frame = load_genotype(data / "genotype" / f"genotype_{ct.name}_{gene}.csv", donors)
                genotype = standardize_genotype(frame)
                keys = [canonical_coordinate(x) for x in frame.columns]
                if len(set(keys)) != len(keys):
                    raise ValueError("Duplicate canonical genotype coordinates")
                if args.mode == "effects":
                    effects = pd.read_csv((args.effects_dir or args.analysis_dir) / ct.name / f"{method}_effects" / f"{gene}.csv")
                    require_columns(effects, ["POS", "posterior_mean"])
                    effects["POS"] = effects.POS.map(canonical_coordinate)
                    if effects.POS.duplicated().any() or set(effects.POS) != set(keys):
                        raise ValueError("Posterior effects must match the genotype variant set exactly")
                    mean = finite(effects.set_index("POS").loc[keys, "posterior_mean"], "Posterior means")
                elif method == "saigeqtl_susie":
                    result = (args.saige_results_dir or args.analysis_dir) / ct.name / "saigeqtl_results" / f"result_{gene}.csv"
                    _, mean = fit_rss(genotype, keys, result, args.components, args.saige_z_sign)
                else:
                    y, _, genotype = load_gene(data, ct, gene, donors, len(z))
                    _, mean = fit_model(method, y, xc, xd, z, genotype, args)
                weights = normalize_weights(mean, genotype)
                write_table(pd.DataFrame({"POS": keys, "weights": weights}), args.output_dir / ct.name / f"{method}_weights" / f"{gene}.csv")
                audit.append({"cell_type": ct.name, "gene": gene, "variants": len(weights), "normalized_genetic_variance": float(np.mean((genotype @ weights)**2))})
            except Exception as exc:
                failures.append({"cell_type": ct.name, "gene": gene, "error": str(exc)})
                if not args.continue_on_error:
                    write_json(failures, args.output_dir / f"{method}_weight_errors.json")
                    raise
    if not audit and not failures:
        raise ValueError("No candidate genes selected")
    write_json({"arguments": vars(args), "completed": audit, "failures": failures}, args.output_dir / f"{method}_weights_run.json")
    if failures:
        raise SystemExit(f"{len(failures)} weight fits failed; see run log")


if __name__ == "__main__":
    main()
