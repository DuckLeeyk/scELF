# OneK1K analysis workflows

The `analysis/onek1k/` directory contains command-line scripts for the manuscript's
OneK1K analyses: data preparation, four-method fine-mapping, GWAS colocalization,
and TWAS with locus evidence export. The scripts produce analysis tables and
model results. Plotting and external validation scripts are excluded.

## Getting started

Run the Python entry points from the repository root using `python -m`.
Install the analysis dependencies and pySuSiE comparators in your environment:

```bash
python -m pip install -e ".[analysis,comparators]"
python -m analysis.onek1k.finemapping.run_scelf --help
```

Each executable uses argparse. Input/output paths and cell-type, gene and trait
selections are supplied through command-line arguments. Shared helper modules
such as `common.py` and `_workflow.py` are used by these entry points.

PLINK and an R environment for SAIGE-QTL are separate requirements. Count
normalization requires the R packages `argparse`, `Matrix`, `sctransform`,
`reticulate` and `future`, plus a reticulate Python environment with AnnData,
NumPy and SciPy. See [Installation](installation.md) for the scELF package.

## 1. Prepare OneK1K inputs

Start with an AnnData count matrix, donor-to-genotype mapping, genotype PCs,
released eQTL/eSNP tables and per-gene PLINK files or SNP lists. Preparation
aligns expression, covariates and genotypes by cell barcode and donor ID.

The following entry points live in `analysis/onek1k/preprocessing/`:

| Scripts | Purpose |
| --- | --- |
| `normalize_counts.R` | Generate SCT v1 corrected UMI counts with optional exact-depth integer postprocessing. |
| `prepare_donor_covariates.py` | Combine age, sex, genotype PCs and sequencing-pool indicators. |
| `compute_expression_pcs.py` | Compute per-cell-type expression components using the original uncentered TruncatedSVD procedure. |
| `filter_eqtl_table.py`, `extract_gene_genotypes.py` | Filter released eQTL variants to selected genes and extract genotypes using supplied per-gene SNP lists. |
| `summarize_egenes.py`, `prepare_egene_lists.py` | Summarize gene regions and construct eligible eGene lists for the mapped cell groups. |
| `prepare_counts.py`, `prepare_genotypes.py` | Export aligned count vectors, covariates, cell-to-donor matrices and cis-genotype matrices. |

Prepared files are organized as `prepared/{cell_type}/data/`. The cell-type
mapping table contains `predicted.celltype.l2`, `l2_name` and `14_celltype_name`;
most `--cell-types` arguments select the `l2_name` directory label. The R
normalizer selects raw cell-type labels, while `prepare_egene_lists` selects
original OneK1K group names.

Genotype imputation and construction of SNP lists encoding the intended cis
window, MAF filter and released-variant intersection are upstream prerequisites.
See [Input data](input-data.md) for the model's array and alignment requirements.

## 2. Run four-method fine-mapping

The `analysis/onek1k/finemapping/` entry points use the same prepared samples and
variants for the four comparisons:

| Method | Entry point | Expression input |
| --- | --- | --- |
| scELF | `run_scelf.py` | Cell-level counts with donor random effects and covariate adjustment. |
| SuSiE | `run_susie.py` | Within-donor mean counts, residualized against covariates. |
| logSuSiE | `run_logsusie.py` | Log-transformed within-donor total counts, residualized against covariates. |
| SAIGE-QTL + SuSiE-RSS | `run_saigeqtl.py`, then `run_susie_rss.py` | Cell-level association statistics followed by fine-mapping with matching in-sample LD. |

For example, replace the paths, cell-type label and gene symbol with your inputs:

```bash
python -m analysis.onek1k.finemapping.run_scelf \
  --analysis-dir work/prepared \
  --output-dir work/fits \
  --celltype-table input/celltypes.csv \
  --egene-dir work/egenes \
  --donor-covariates work/donor_covariates.csv \
  --cell-types CELL_TYPE \
  --genes GENE_SYMBOL
```

Omit `--cell-types` and `--genes` to run all eligible entries in the mapping and
eGene lists. Results include variant PIPs, credible sets and posterior mean
effects under `{cell_type}/{method}_pip`, `{method}_cs` and `{method}_effects`.
The four method prefixes are `scelf`, `susie`, `logsusie` and `saigeqtl_susie`.
Saved effects can be reused for TWAS without fitting the models again.

## 3. Compare eQTL and GWAS credible sets

The `analysis/onek1k/colocalization/` scripts prepare released GWAS credible sets
and score eQTL–GWAS CS pairs sharing at least one variant.

| Scripts | Purpose |
| --- | --- |
| `prepare_bcx2.py`, `prepare_finngen.py` | Standardize one trait's released BCX2 or FinnGen credible-set table. |
| `run_bcx2.py`, `run_finngen.py` | Calculate conditional H3/H4 scores for selected traits, cell types and methods. |
| `summarize_results.py` | Summarize credible-set support and export colocalized gene lists. |

Supply traits with `--traits` and input files with `--gwas-template`, for example
`"work/gwas_cs/{trait}.tsv"`. Both datasets must use the same genome build;
cross-build FinnGen preparation requires a supplied local coordinate map.

Scoring requires an explicit `--pip-policy`: `union-zero` uses the complete CS
union and zero padding described in the manuscript; `legacy` retains the
original scripts' overlap restriction, residual-mass filling and normalization.
Keep this choice consistent when comparing results. Output tables retain
missing-score statuses and report both CS-pair scores and per-CS summaries.

The package also exposes the underlying `from scelf import coloc_pip` function.
See [PIP colocalization](colocalization.md) for its API and conditional H3/H4
interpretation.

## 4. Calculate TWAS and export locus evidence

The `analysis/onek1k/twas/` scripts connect fine-mapping effects to GWAS summary
statistics and produce tables for gene- and locus-level comparisons.

| Scripts | Purpose |
| --- | --- |
| `select_cs_genes.py` | Select the union of genes with a nonempty CS in any method, within each cell type. |
| `fit_weights.py` | Generate LD-normalized TWAS weights from saved posterior means or an explicit refit. |
| `harmonize_gwas.py`, `export_gwas_zscores.py` | Align GWAS effects to the genotype dosage allele and export successfully aligned Z-scores. |
| `calculate_scores.py` | Calculate TWAS Z-scores and two-sided P-values, recording GWAS variant coverage. |
| `summarize_scores.py`, `summarize_loci.py` | Summarize TWAS associations and join them to gene-level colocalization support. |
| `export_locus.py` | Export per-method PIPs, CS membership, reference-variant LD r-squared and optional GWAS Z-scores. |

Generate weights for each method on the same candidate gene set. Weights already
include the in-sample LD normalization, so scoring uses `weights.T @ GWAS_Z`.
Match the GWAS target allele to the weight dosage allele; the provided genotype
preparation counts BIM A2. The harmonization and scoring commands expose allele
and missing-variant policies explicitly. Locus export requires a cell type, gene,
trait and reference coordinate, and produces a table for further interpretation.

## Detailed commands and validation

The checkout's `analysis/onek1k/README.md` contains full command sequences,
input/output schemas and parameter conventions. `source_manifest.json` records
source labels and original content checksums, marking publication aliases
explicitly. The repository's CI runs synthetic Python checks with the appropriate
optional dependencies. Complete SCT and SAIGE-QTL R analyses
require validation in the target R environment; the bundled and synthetic checks
do not establish reproduction of all manuscript results.
