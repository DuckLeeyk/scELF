"""Shared fitting implementation; public run_*.py files provide argparse CLIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

from ..common import (add_selection, finite, load_gene, load_shared, positive_int,
                      selected_celltypes, selected_genes, write_json, write_table)


def parser_for(method, *, require_covariates=True):
    p = argparse.ArgumentParser(description=f"Run OneK1K {method} fine-mapping on prepared inputs.")
    add_selection(p)
    p.add_argument("--output-dir", type=Path, required=True, help="Results root; each cell type gets method_pip/method_cs subdirectories.")
    p.add_argument("--donor-covariates", type=Path, required=require_covariates and method != "saigeqtl_susie", help="Numeric covariates with donor identifier column.")
    p.add_argument("--donor-id-column", default="genotype_ID")
    p.add_argument("--components", type=positive_int, default=10, help="Maximum number of single effects L.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-outer-iter", type=positive_int, default=20)
    p.add_argument("--tolerance", type=float, default=1e-5)
    p.add_argument("--backend", choices=["auto", "numba", "cpp"], default="auto")
    p.add_argument("--fit-options", type=Path, help="Optional JSON with ScELF.fit keyword overrides, including stage1_options/stage2_options.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--continue-on-error", action="store_true", help="Write error log, continue, then exit nonzero if any gene failed.")
    p.add_argument("--saige-results-dir", type=Path, help="Root containing cell_type/saigeqtl_results/result_gene.csv (default: analysis-dir).")
    p.add_argument("--saige-z-sign", type=int, choices=[-1, 1], default=1,
                   help="RSS only: explicit direction correction for imported legacy SAIGE z-scores. New exported results already target A2.")
    if method == "saigeqtl":
        add_saige_arguments(p)
    return p


def add_saige_arguments(p):
    p.add_argument("--saige-dir", type=Path, required=True, help="SAIGE-QTL checkout with extdata/step1_fitNULLGLMM_qtl.R and step2_tests_qtl.R.")
    p.add_argument("--rscript", default="Rscript", help="Rscript executable, resolved in PATH unless an explicit path is supplied.")
    p.add_argument("--pixi-manifest", type=Path, help="Optional pixi manifest; runs pixi run --manifest-path PATH Rscript ...")
    p.add_argument("--pixi", default="pixi")
    p.add_argument("--saige-options", type=Path, help="Optional JSON with step1 and step2 argument dictionaries (SAIGE option names without --).")


def fit_model(method, y, xc, xd, z, genotype, args):
    """Retain the original four statistical workflows on one aligned dataset."""
    if method == "scelf":
        from scelf import ScELF
        options = dict(use_covariates=True, max_outer_iter=args.max_outer_iter,
                       tol_F=args.tolerance, verbose=not args.quiet,
                       stage1_options={"estimate_phi": True, "phi_min": 1e-4, "phi_max": 1e4, "phi_init": 100},
                       stage2_options={"tau": {"method": "ai"},
                                       "phi": {"update": True, "phi_min": 1e-4, "phi_max": 1e4},
                                       "e_step": {"ibss_sweeps": 1, "update_prior_var": True}})
        if args.fit_options:
            options.update(json.loads(args.fit_options.read_text(encoding="utf-8")))
        model = ScELF(y=y, Xc=xc, Xd=xd, Z=z, G=genotype,
                      L=args.components, seed=args.seed, backend=args.backend)
        model.fit(**options)
        mean = np.asarray(model.susie_q["m_comp"]).sum(axis=0)
        return model, mean
    if method not in {"susie", "logsusie"}:
        raise ValueError(f"Not an individual-level fit: {method}")
    from pysusie import SuSiE
    import statsmodels.api as sm
    # Z.T aggregates cells in exactly the donor order of the genotype matrix.
    sizes = z.sum(axis=0)
    if (sizes == 0).any():
        raise ValueError("Pseudobulk donors must each have at least one cell")
    response = (z.T @ y) / sizes if method == "susie" else np.log2(1 + z.T @ y)
    cell_means = (z.T @ xc) / sizes[:, None]
    design = sm.add_constant(np.column_stack([xd, cell_means]), has_constant="add")
    residual = sm.OLS(response, design).fit().resid
    model = SuSiE(L=args.components).fit(genotype, residual)
    return model, np.sum(np.asarray(model.alpha) * np.asarray(model.mu), axis=0)


def fit_rss(genotype, positions, result_file, components, z_sign=1):
    from pysusie import SuSiE_SS
    frame = pd.read_csv(result_file, dtype={"POS": str})
    from ..common import canonical_coordinate, require_columns
    require_columns(frame, ["POS", "zscore"])
    frame["POS"] = frame.POS.map(canonical_coordinate)
    keys = [canonical_coordinate(x) for x in positions]
    if frame.POS.duplicated().any() or set(keys) != set(frame.POS):
        raise ValueError("SAIGE z-score variants must match genotype variants exactly")
    zscore = z_sign * finite(frame.set_index("POS").loc[keys, "zscore"], "SAIGE z-scores")
    ld = genotype.T @ genotype / genotype.shape[0]
    np.fill_diagonal(ld, 1)
    model = SuSiE_SS(L=components).fit(zscore, ld, genotype.shape[0])
    return model, np.sum(np.asarray(model.alpha) * np.asarray(model.mu), axis=0)


def align_saige_output(result, positions):
    """Map synthetic markers and convert SAIGE effect-allele statistics to A2.

    Temporary BIM A1='2', A2='1'. SAIGE's default alt-first mode reports
    Allele2='2' (BIM A1), so its beta must be negated for input A2 dosage.
    Checking returned alleles also supports ref-first mode without assumptions.
    """
    from ..common import require_columns
    require_columns(result, ["BETA", "SE", "Allele1", "Allele2"])
    synthetic = [f"rs{i+1}" for i in range(len(positions))]
    id_column = next((x for x in ["MarkerID", "SNPID", "SNP", "ID"] if x in result), None)
    if id_column is not None:
        keys = result[id_column].astype(str)
    elif "POS" in result:
        keys = "rs" + pd.to_numeric(result.POS, errors="raise").astype(int).astype(str)
    else:
        raise ValueError("SAIGE results need MarkerID/SNPID/SNP/ID or synthetic POS")
    result = result.assign(_id=keys).set_index("_id")
    if result.index.has_duplicates or set(result.index) != set(synthetic):
        raise ValueError("SAIGE returned a different marker set; inspect its QC filters")
    result = result.loc[synthetic]
    a1, a2 = result.Allele1.astype(str), result.Allele2.astype(str)
    plus, minus = a1.eq("2") & a2.eq("1"), a1.eq("1") & a2.eq("2")
    if not (plus | minus).all():
        raise ValueError("SAIGE returned alleles inconsistent with the temporary PLINK coding")
    sign = np.where(plus, 1, -1)
    beta, se = finite(result.BETA, "SAIGE beta"), finite(result.SE, "SAIGE SE")
    if (se <= 0).any():
        raise ValueError("SAIGE standard errors must be positive")
    output = pd.DataFrame({"POS": positions, "BETA": sign*beta, "SE": se, "zscore": sign*beta/se,
        "BETA_original": beta, "alignment_sign": sign, "target_allele": "A2"})
    if "p.value" in result:
        output["pvalue"] = result["p.value"].to_numpy()
    return output


def run_saige(y, xc, xd, z, raw_genotype, donors, positions, args):
    """Generate temporary per-gene PLINK/phenotype files and map SAIGE IDs back.

    Synthetic markers rs1...rsP refer to input columns, never to real dbSNP IDs.
    Fixed synthetic alleles retain the historical SAIGE coding; exported effects
    are then explicitly aligned to the original input matrix's A2 dosage.
    """
    import xarray as xr
    from pandas_plink import write_plink1_bin
    with tempfile.TemporaryDirectory(prefix="scelf_saige_") as temporary:
        work = Path(temporary)
        n, p = raw_genotype.shape
        synthetic = [f"rs{i + 1}" for i in range(p)]
        if not np.isin(raw_genotype, [0, 1, 2]).all():
            raise ValueError("SAIGE PLINK input requires integer dosages 0/1/2")
        array = xr.DataArray(raw_genotype.astype("float32"), dims=["sample", "variant"],
            coords={"sample": donors, "variant": synthetic, "fid": ("sample", donors),
                    "father": ("sample", np.zeros(n, int)), "mother": ("sample", np.zeros(n, int)),
                    "sex": ("sample", np.zeros(n, int)), "trait": ("sample", np.full(n, -9)),
                    "snp": ("variant", synthetic), "chrom": ("variant", ["1"] * p),
                    "cm": ("variant", np.zeros(p)), "pos": ("variant", np.arange(1, p + 1)),
                    "a0": ("variant", ["2"] * p), "a1": ("variant", ["1"] * p)})
        write_plink1_bin(array, str(work / "geno.bed"))
        # Expand donor covariates to cells, preserving the design matrix row order.
        donor_index = z.argmax(axis=1)
        names_d = [f"donor_cov_{i}" for i in range(xd.shape[1])]
        names_c = [f"cell_cov_{i}" for i in range(xc.shape[1])]
        phenotype = pd.DataFrame(np.column_stack([xd[donor_index], xc]), columns=names_d + names_c)
        phenotype.insert(0, "IND_ID", np.asarray(donors)[donor_index])
        phenotype["expression"] = y
        phenotype.to_csv(work / "phenotype.tsv", sep="\t", index=False)
        # Each temporary PLINK file already contains only the selected gene's
        # markers; no external range file or artificial coordinate filter is needed.
        step1 = {"useSparseGRMtoFitNULL": "FALSE", "useGRMtoFitNULL": "FALSE",
                 "phenoFile": work / "phenotype.tsv", "phenoCol": "expression",
                 "covarColList": ",".join(names_d + names_c), "sampleCovarColList": ",".join(names_d),
                 "sampleIDColinphenoFile": "IND_ID", "traitType": "count", "outputPrefix": work / "null",
                 "skipVarianceRatioEstimation": "FALSE", "isRemoveZerosinPheno": "FALSE",
                 "isCovariateOffset": "FALSE", "isCovariateTransform": "TRUE", "skipModelFitting": "FALSE",
                 "tol": args.tolerance, "plinkFile": work / "geno", "IsOverwriteVarianceRatioFile": "TRUE"}
        step2 = {"bedFile": work / "geno.bed", "bimFile": work / "geno.bim", "famFile": work / "geno.fam",
                 "SAIGEOutputFile": work / "result.txt", "chrom": "1", "minMAF": "0", "minMAC": "0.5",
                 "LOCO": "FALSE", "GMMATmodelFile": work / "null.rda", "SPAcutoff": "2",
                 "varianceRatioFile": work / "null.varianceRatio.txt",
                 "markers_per_chunk": "10000"}
        overrides = json.loads(args.saige_options.read_text(encoding="utf-8")) if args.saige_options else {}
        prefix = [args.rscript]
        if args.pixi_manifest:
            prefix = [args.pixi, "run", "--manifest-path", str(args.pixi_manifest), args.rscript]
        for number, (script, options) in enumerate([
            ("step1_fitNULLGLMM_qtl.R", step1), ("step2_tests_qtl.R", step2)], 1):
            # Structural input/output options are protected: temporary paths must
            # remain coupled to the files that this function actually reads.
            extra = overrides.get(f"step{number}", {})
            protected = {k for k in options if "File" in k or k == "outputPrefix"}
            if set(extra) & protected:
                raise ValueError(f"Cannot override structural SAIGE options: {sorted(set(extra) & protected)}")
            options.update(extra)
            subprocess.run(prefix + [str(args.saige_dir / "extdata" / script)] +
                           [f"--{key}={value}" for key, value in options.items()], check=True)
        result = pd.read_csv(work / "result.txt", sep="\t", dtype={"Allele1": str, "Allele2": str})
        return align_saige_output(result, positions)


def main_for(method, argv=None):
    args = parser_for(method).parse_args(argv)
    failures, runs = [], []
    for ct in selected_celltypes(args.celltype_table, args.cell_types):
        genes = selected_genes(args, ct)
        if not genes:
            continue
        # RSS only needs genotype and donor order; all other methods share Xc,Xd,Z.
        if method != "saigeqtl_susie":
            data, donors, xc, xd, z = load_shared(args, ct)
        else:
            from ..common import load_donors
            data = args.analysis_dir / ct.name / "data"
            donors = load_donors(data / f"genotype_IDs_{ct.name}.txt")
        for gene in genes:
            start = time.perf_counter()
            try:
                if method == "saigeqtl_susie":
                    from ..common import load_genotype, standardize_genotype
                    frame = load_genotype(data / "genotype" / f"genotype_{ct.name}_{gene}.csv", donors)
                    genotype = standardize_genotype(frame)
                    result_file = (args.saige_results_dir or args.analysis_dir) / ct.name / "saigeqtl_results" / f"result_{gene}.csv"
                    model, mean = fit_rss(genotype, frame.columns, result_file, args.components, args.saige_z_sign)
                else:
                    y, frame, genotype = load_gene(data, ct, gene, donors, len(z))
                    if method == "saigeqtl":
                        result = run_saige(y, xc, xd, z, frame.to_numpy(float), donors, frame.columns, args)
                        write_table(result, args.output_dir / ct.name / "saigeqtl_results" / f"result_{gene}.csv")
                    else:
                        model, mean = fit_model(method, y, xc, xd, z, genotype, args)
                if method != "saigeqtl":
                    base = args.output_dir / ct.name
                    write_table(pd.DataFrame({"POS": frame.columns, "PIP": finite(model.pip, "PIP")}),
                                base / f"{method}_pip" / f"{gene}.csv")
                    write_json(model.sets, base / f"{method}_cs" / f"{gene}.json")
                    # Saving posterior means avoids expensive refits for TWAS.
                    write_table(pd.DataFrame({"POS": frame.columns, "posterior_mean": finite(mean, "Posterior mean")}),
                                base / f"{method}_effects" / f"{gene}.csv")
                runs.append({"cell_type": ct.name, "gene": gene, "method": method, "seconds": time.perf_counter() - start})
            except Exception as exc:
                failures.append({"cell_type": ct.name, "gene": gene, "error": str(exc)})
                write_json({"error": str(exc), "traceback": traceback.format_exc()},
                           args.output_dir / ct.name / f"{method}_errors" / f"{gene}.json")
                if not args.continue_on_error:
                    raise
    if not runs and not failures:
        raise ValueError("No genes selected; check cell-type and eGene filters")
    write_json({"method": method, "arguments": vars(args), "completed": runs, "failures": failures},
               args.output_dir / f"{method}_run.json")
    if failures:
        raise SystemExit(f"{len(failures)} genes failed; see error logs")
