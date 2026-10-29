# OneK1K manuscript analysis

Command-line workflows for OneK1K input preparation, four-method eQTL fine-mapping,
GWAS colocalization, TWAS, and locus evidence export. External validation and
plotting programs are excluded. These scripts prepare analysis tables; they do
not download private data or include study-wide results.

Run Python commands **from the repository root**, using `python -m`, after
installing scELF. For example:

```sh
python -m pip install -e ".[analysis,comparators]"
python -m analysis.onek1k.finemapping.run_scelf --help
```

Use a separate Python 3.10-3.12 environment for these workflows. The `analysis`
extra adds AnnData, scikit-learn, pandas-plink and xarray, with NumPy 1.x and
pandas 2.x as required by the current pandas-plink dependency. These optional
constraints do not apply to a core-only installation. `comparators`
installs the authors' [pySuSiE implementation](https://github.com/DuckLeeyk/pySuSiE)
at the revision recorded in `pyproject.toml`. A compatible SAIGE-QTL R environment
and PLINK executable are separate requirements. `normalize_counts.R` requires R
packages `argparse`, `Matrix`, `sctransform`, `reticulate`, and `future`, plus a
reticulate Python environment containing `anndata`, `numpy`, and `scipy`.
Dependencies are never installed by an analysis script.

## Organization and naming

| Directory | Entry points | Purpose |
| --- | --- | --- |
| `preprocessing/` | `normalize_counts.R` | SCT v1 corrected UMI counts and exact-depth integer postprocessing |
| | `prepare_donor_covariates`, `compute_expression_pcs` | Donor covariates and per-cell-type expression components |
| | `filter_eqtl_table`, `extract_gene_genotypes` | Released eQTL variant intersection and per-gene PLINK extraction |
| | `summarize_egenes`, `prepare_egene_lists` | Gene regions, original cell-group flags, eligible eGene lists |
| | `prepare_counts`, `prepare_genotypes` | Aligned per-cell-type analysis inputs |
| `finemapping/` | `run_scelf`, `run_susie`, `run_logsusie` | Cell-level scELF and two pseudobulk comparators |
| | `run_saigeqtl`, `run_susie_rss` | SAIGE-QTL single-variant statistics, followed by SuSiE-RSS |
| `colocalization/` | `prepare_bcx2`, `prepare_finngen` | Released GWAS credible sets |
| | `run_bcx2`, `run_finngen`, `summarize_results` | CS-pair H3/H4 scoring and summary tables |
| `twas/` | `select_cs_genes`, `fit_weights` | Union candidate genes and LD-normalized prediction weights |
| | `harmonize_gwas`, `export_gwas_zscores`, `calculate_scores` | Allele alignment and TWAS Z-scores |
| | `summarize_scores`, `summarize_loci`, `export_locus` | Association summaries and locus evidence tables |

Python names in this table omit `.py`. `common.py` and the two `_workflow.py`
modules are shared libraries, without standalone execution or analysis on import.
All executable entry points use argparse. The package feature is
[`scelf.coloc_pip`](../../docs/colocalization.md), independent of these workflows.

Method prefixes are `scelf`, `susie`, `logsusie`, and `saigeqtl_susie`.
`run_saigeqtl` generates the statistics needed by the fourth method; it is not a
fifth fine-mapping method. Use the `scelf` prefix for scELF results in downstream
`--methods` selections. There are no built-in cell/gene/trait lists.

## Input contracts

Paths in the examples are illustrative relative paths. Supply your own files.
CSV filenames imply comma separation; other table filenames imply tabs. `.gz`
inputs are supported; streaming GWAS exporters require uncompressed outputs.
Gene lists contain one symbol per line, without a header.

* **Cell-type mapping**: CSV/TSV with `predicted.celltype.l2` (raw AnnData label),
  `l2_name` (safe directory name), `14_celltype_name` (original OneK1K group).
  Labels and directory names must be unique. Incomplete mapping rows are removed.
  Most `--cell-types` options select `l2_name`; the R normalizer selects raw labels,
  and `prepare_egene_lists` selects original group/flag names.
* **Sample mapping**: unique `donor_id,genotype_ID` pairs. IDs are strings,
  including any leading zeros. Donor rows are reindexed by ID, never assumed to
  match genotype, covariate, or PLINK row order.
* **AnnData**: counts in `X`, unique barcodes in `obs_names`, gene identifiers in
  `var_names`. Default metadata fields are `predicted.celltype.l2`, `donor_id`,
  `pool_number`, `age`, `sex`, `nCount_RNA`, and `percent.mt`. CLI options expose
  these field names where used.
* **Genotype PCs**: headerless PLINK eigenvec: FID, IID, PC1, PC2, ... .
* **eGene lists**: `egene-dir/{14_celltype_name}_eGene.csv`, with `gene,gene_id`.
* **Coordinates**: 1-based `chrom:pos`. A `chr` prefix and allele suffixes such as
  `1:123_A_G` are normalized. Position-only matching requires one biallelic variant
  per coordinate; duplicate coordinates are rejected rather than arbitrarily
  assigning an allele. Match genome builds before any join.

Prepared inputs follow this layout, retaining source file conventions:

```text
prepared/{l2_name}/data/
  genotype_IDs_{l2_name}.txt
  cell_covariates_{predicted.celltype.l2}.csv  # barcode, numeric covariates
  design_matrix_{predicted.celltype.l2}.txt   # cells x donors, one-hot rows
  counts/GReX_{l2_name}_{gene}.txt            # counts in barcode order
  genotype/genotype_{l2_name}_{gene}.csv      # donor index, chr:pos columns
  genotype/genotype_{l2_name}_{gene}.variants.tsv  # POS,rsid,A1,A2
```

Counts, cell covariates and design rows must have identical cell order. Genotypes
are standardized across donors with population SD (`ddof=0`). pandas-plink dosage
0/1/2 counts the second BIM allele (A2); the allele sidecar preserves this link.
Do not replace the genotype preparation tool without checking dosage coding.

## 1. Prepare inputs

Genotype imputation, reference-panel choice, and construction of per-gene SNP-ID
lists are **prerequisites**. Those upstream jobs were not fully specified in the
source directory. For the manuscript, lists should encode gene body +/-1 Mb,
MAF >=0.05, and intersection with the released OneK1K eQTL variants. This workflow
does not claim to reproduce those unavailable steps.

```sh
Rscript analysis/onek1k/preprocessing/normalize_counts.R --input input/raw.h5ad --output work/corrected.h5ad --summary work/sct_groups.tsv --python python
python -m analysis.onek1k.preprocessing.prepare_donor_covariates --h5ad input/raw.h5ad --sample-mapping input/sample_mapping.tsv --genotype-pcs input/genotypes.eigenvec --output work/donor_covariates.csv
python -m analysis.onek1k.preprocessing.compute_expression_pcs --h5ad work/corrected.h5ad --celltype-table input/celltypes.csv --output-dir work/expression_pcs
python -m analysis.onek1k.preprocessing.filter_eqtl_table --input input/released_eqtl.tsv.gz --gene-list input/genes.txt --output work/eqtl_variants.tsv.gz
python -m analysis.onek1k.preprocessing.extract_gene_genotypes --bfile input/imputed --snp-list-dir input/gene_snp_lists --gene-list input/genes.txt --maf 0.05 --output-dir work/gene_plink
python -m analysis.onek1k.preprocessing.summarize_egenes --gene-list input/genes.txt --plink-dir work/gene_plink --esnp-table input/released_esnp.tsv --output work/egene_summary.csv
python -m analysis.onek1k.preprocessing.prepare_egene_lists --summary work/egene_summary.csv --h5ad work/corrected.h5ad --cell-types ORIGINAL_GROUP --output-dir work/egenes
python -m analysis.onek1k.preprocessing.prepare_counts --h5ad work/corrected.h5ad --celltype-table input/celltypes.csv --egene-dir work/egenes --sample-mapping input/sample_mapping.tsv --donor-covariates work/donor_covariates.csv --pcs-dir work/expression_pcs --output-dir work/prepared
python -m analysis.onek1k.preprocessing.prepare_genotypes --analysis-dir work/prepared --celltype-table input/celltypes.csv --egene-dir work/egenes --plink-dir work/gene_plink --output-dir work/prepared
```

`ORIGINAL_GROUP` is a placeholder for a `CELL_ID` flag in the released eSNP table;
pass all intended group names. `summarize_egenes` requires columns
`GENE,GENE_ID,CELL_ID`; `filter_eqtl_table` defaults to `RSID,GENE,CHR,POS`.
Use `--columns` if the released file has additional fields to retain. Filtering
exports an intersection table, not the per-gene SNP lists themselves.

The normalizer retains the original SCT v1 algorithm: split cell types over
10,000 cells by pool, merge small pool groups, minimum 200 cells, target 1,261
UMIs, and largest-remainder integer allocation. These thresholds and seed are
CLI parameters. By default a normalization failure stops execution; allowing raw
fallback after fitting errors requires `--allow-error-fallback`. Undersized groups
retain raw counts before exact-depth scaling, as in the original algorithm.
Input metadata, including the original
`nCount_RNA` used as a depth covariate, is preserved. Corrected `X` contains counts,
not Pearson residuals. Existing R outputs are protected from overwrite.

Expression components retain the source's **uncentered TruncatedSVD**, despite
the historical name "gene PCs". They are standardized and saved with barcodes.
Donor covariates use age, six genotype PCs, and drop-first sex/pool indicators;
category reference levels are saved in an encoding JSON. The number of pool
indicators depends on the supplied metadata. Missing genotypes default to zero
as in the source; `--missing-genotypes mean` or `error` makes another policy
explicit. Monomorphic variants are removed. `prepare_counts` logs excluded cells
and missing expression genes; selected absent genes cannot silently be fitted.

## 2. Run the four fine-mapping workflows

Replace `CELL_NAME` and `GENE_SYMBOL` below, or omit these optional filters to run
all mapped eGenes. `--gene-pairs` optionally selects a TSV of `cell_type,gene`.

```sh
python -m analysis.onek1k.finemapping.run_scelf --analysis-dir work/prepared --output-dir work/fits --celltype-table input/celltypes.csv --egene-dir work/egenes --donor-covariates work/donor_covariates.csv --cell-types CELL_NAME --genes GENE_SYMBOL
python -m analysis.onek1k.finemapping.run_susie --analysis-dir work/prepared --output-dir work/fits --celltype-table input/celltypes.csv --egene-dir work/egenes --donor-covariates work/donor_covariates.csv --cell-types CELL_NAME --genes GENE_SYMBOL
python -m analysis.onek1k.finemapping.run_logsusie --analysis-dir work/prepared --output-dir work/fits --celltype-table input/celltypes.csv --egene-dir work/egenes --donor-covariates work/donor_covariates.csv --cell-types CELL_NAME --genes GENE_SYMBOL
python -m analysis.onek1k.finemapping.run_saigeqtl --analysis-dir work/prepared --output-dir work/fits --celltype-table input/celltypes.csv --egene-dir work/egenes --donor-covariates work/donor_covariates.csv --saige-dir software/SAIGE-QTL --cell-types CELL_NAME --genes GENE_SYMBOL
python -m analysis.onek1k.finemapping.run_susie_rss --analysis-dir work/prepared --output-dir work/fits --celltype-table input/celltypes.csv --egene-dir work/egenes --saige-results-dir work/fits --cell-types CELL_NAME --genes GENE_SYMBOL
```

SuSiE uses donor mean counts; logSuSiE uses `log2(1 + donor total counts)`.
Both residualize against donor covariates and donor means of cell covariates,
including an OLS intercept. SuSiE-RSS aligns SAIGE Z-scores by variant, and uses
the matching genotype-derived LD and donor count. Default maximum components is
10. scELF exposes seed, outer iterations, tolerance and backend; `--fit-options`
accepts a JSON of `ScELF.fit` overrides. This workflow retains the original `ai`
tau update configuration, which differs from other example configurations.

Model results are written as:

```text
fits/{cell_type}/{method}_pip/{gene}.csv       # POS,PIP
fits/{cell_type}/{method}_cs/{gene}.json       # cs dict, zero-based PIP-row indices
fits/{cell_type}/{method}_effects/{gene}.csv   # POS,posterior_mean
fits/{cell_type}/saigeqtl_results/result_{gene}.csv  # POS,BETA,SE,zscore; original beta, allele QC
```

SAIGE uses temporary per-gene PLINK/phenotype inputs, synthetic marker IDs and
coordinates, then maps results back to real variant coordinates by marker ID.
No external gene-range file is needed because the temporary file contains only
that gene's selected variants. Return variants must match input variants exactly;
QC filtering does not silently shift rows. Returned `Allele1,Allele2` determine
whether BETA must be reversed to match the original A2 dosage. Aligned BETA/Z,
original BETA and the alignment sign are all retained. This corrects the source
workflow's opposite SAIGE direction before RSS/TWAS. Imported old SAIGE files
must already use A2; `--saige-z-sign -1` makes a known opposite direction explicit
when running RSS or refitting RSS weights. Pass `--rscript`, or optionally
`--pixi-manifest` and `--pixi`, for your environment. `--saige-options` accepts
`{"step1": {...}, "step2": {...}}` argument overrides, protecting structural
input/output paths. Verify compatibility with your SAIGE-QTL release.

The default stops on errors. `--continue-on-error` records failures, continues,
and exits nonzero at the end. Outputs include run arguments and per-gene error
logs. Use a fresh output directory for a new selection; old result files are
not removed automatically.

## 3. Prepare and score GWAS credible sets

Process one released trait file per preparation call:

```sh
python -m analysis.onek1k.colocalization.prepare_bcx2 --input input/TRAIT_EA_Credible_Sets_VEP.txt --output work/gwas_cs/TRAIT.tsv --genome-build GRCh37
python -m analysis.onek1k.colocalization.prepare_finngen --input input/finngen_trait.tsv --output work/finngen_cs/TRAIT.tsv --input-build GRCh38 --output-build GRCh37 --coordinate-map input/grch38_to_grch37.tsv
python -m analysis.onek1k.colocalization.run_bcx2 --analysis-dir work/fits --output-dir work/coloc_bcx2 --cell-types CELL_NAME --traits TRAIT --gwas-template "work/gwas_cs/{trait}.tsv" --genome-build GRCh37 --pip-policy union-zero
python -m analysis.onek1k.colocalization.run_finngen --analysis-dir work/fits --output-dir work/coloc_finngen --cell-types CELL_NAME --traits TRAIT --gwas-template "work/finngen_cs/{trait}.tsv" --genome-build GRCh37 --pip-policy union-zero
python -m analysis.onek1k.colocalization.summarize_results --inputs work/coloc_bcx2/CELL_NAME/TRAIT_scelf_coloc.tsv --output-dir work/coloc_summary
```

BCX2 defaults to `MarkerName,Sentinel_MarkerName,PP`. FinnGen defaults to
`chromosome,position,region,cs,cs_specific_prob`; chromosome filtering retains
autosomes unless requested otherwise. Prepared fields are respectively
`MarkerName,Sentinel_MarkerName,PP` and `chr:pos,cs_id,PP`, with chromosome/position
and source-coordinate audit columns. Column-name overrides are available.

Use identical build names for corresponding CLI declarations and metadata.
Cross-build FinnGen preparation requires a local one-to-one CSV/TSV mapping
`source_POS,target_POS`; unmatched coordinates error unless `--unmapped drop` is
specified. The mapping is an upstream, independently verified input. This
replaces the original optional MyVariant network lookup. If both inputs already
share a build, specify the same input/output build and omit the map. A build
declaration alone cannot detect a mislabeled input; no implicit liftOver occurs.

**Choose a PIP policy explicitly.** The old scripts and manuscript describe
different operations, so neither is an invisible default:

* `legacy`: restrict GWAS rows to eQTL-CS overlaps *before* grouping by GWAS CS;
  fill remaining mass on absent coordinates where the original code did so,
  and normalize each aligned vector. eQTL residual filling applies only to
  single-CS genes. This retains the old algorithm for valid, unambiguous inputs.
* `union-zero`: identify GWAS CSs by at least one overlap, then use each complete
  GWAS CS and complete eQTL CS on their union, padding absent PIPs with zero and
  adding no residual mass. This follows the current manuscript definition.

Both use the source's marginal variant PIPs within each eQTL CS, rather than
newly estimating component-specific probabilities. The package applies a
`1e-16` numerical floor; its H4 is conditional on H3 versus H4, not a five-model
coloc posterior. Prior weights default to `p1=p2=1e-4`, `p12=1e-7` and are CLI
parameters. No-overlap/zero-support records have missing H4 and a status.

Each result is `output-dir/{cell_type}/{trait}_{method}_coloc.tsv`.
Counts use one maximum H4 per eQTL CS per trait; `median_pair_h4` uses all scored
CS pairs. `cs_colocalized` uses strict H4 >0.5 by default. Summary denominators
are per cell type/trait/method: do not sum them across traits and call that a
unique-CS count. `unique_method_genes.tsv` explicitly deduplicates genes across
cell types and traits. Mixed PIP policies cannot be summarized together.

## 4. TWAS and locus analysis

Use the union of genes with a nonempty CS from any of the four methods, within
each cell type. Generate weights for **each** method on that same candidate set:

```sh
python -m analysis.onek1k.twas.select_cs_genes --analysis-dir work/fits --cell-types CELL_NAME --output work/cs_gene_pairs.tsv
python -m analysis.onek1k.twas.fit_weights --method scelf --analysis-dir work/prepared --effects-dir work/fits --output-dir work/weights --celltype-table input/celltypes.csv --egene-dir work/egenes --gene-pairs work/cs_gene_pairs.tsv
python -m analysis.onek1k.twas.harmonize_gwas --input input/TRAIT_GWAMA.out.gz --bim input/imputed.bim --output work/gwas/TRAIT_aligned.tsv --genome-build GRCh37 --effect-allele-column other_allele --non-effect-allele-column reference_allele --target-allele A2
python -m analysis.onek1k.twas.export_gwas_zscores --input work/gwas/TRAIT_aligned.tsv --output work/gwas/TRAIT_zscores.tsv
python -m analysis.onek1k.twas.calculate_scores --weights-dir work/weights --cell-types CELL_NAME --traits TRAIT --gwas-template "work/gwas/{trait}_zscores.tsv" --dosage-allele A2 --output work/twas.tsv
python -m analysis.onek1k.twas.summarize_scores --inputs work/twas.tsv --p-threshold 1e-6 --output-dir work/twas_summary
python -m analysis.onek1k.twas.summarize_loci --twas work/twas.tsv --coloc work/coloc_bcx2/CELL_NAME/TRAIT_scelf_coloc.tsv --output work/locus_summary.tsv
python -m analysis.onek1k.twas.export_locus --analysis-dir work/prepared --finemapping-dir work/fits --output-dir work/locus --cell-type CELL_NAME --gene GENE_SYMBOL --trait TRAIT --reference-pos CHROM:POSITION --gwas work/gwas/TRAIT_zscores.tsv
```

Repeat `fit_weights` with `--method susie`, `logsusie`, and `saigeqtl_susie` before
running the default four-method scoring command. Use `--methods` to analyze a
subset. The GWAS effect/non-effect column names in this example reflect the
historical assumption; **check the source GWAS documentation** and supply the
actual convention. The script never infers an effect allele from a column name.

`fit_weights` defaults to reusing saved posterior means. `--mode refit` uses the
shared fine-mapping code; individual-level refits require donor covariates, and
RSS refits can use `--saige-results-dir`. All four methods use
`w = posterior_mean / sqrt(mean((G_standardized @ posterior_mean)**2))`.
Zero genetic variance is an error. Consequently `calculate_scores` computes
`w.T @ GWAS_Z` directly: the LD denominator is already included.

Harmonization streams inputs against a disk-backed BIM index. It handles direct
and complemented SNP alleles, records original and aligned statistics, rejects
ambiguous duplicate BIM coordinates, and drops palindromic/unmatched rows by
default. `--palindromic keep` explicitly assumes matching strands. `--unmatched
keep` retains rows with missing aligned Z and original Z for audit; the slim
export only includes successful alignments. `--unmatched error` stops on these
rows. The selected `--target-allele` must match the weight dosage allele.

TWAS scoring requires alignment status and target-allele metadata. External
pre-aligned `POS,zscore` tables can be accepted explicitly with
`--assume-harmonized`. Duplicate GWAS positions error by default;
`--duplicate-positions first` reproduces the source's first-record policy.
Missing GWAS Z defaults to zero as in the original calculation, with missing
rate and matched-variant count exported. Use `--missing-z error` or
`--max-missing-rate` to impose stricter requirements. Entirely missing loci have
`status=no_gwas_data` and are excluded from significance summaries.

`summarize_loci` joins maximum gene-level H4 with TWAS while keeping missing H4
as missing and preserving absence statuses. It applies no hidden sign reversal;
`--flip-methods` records any deliberate reversal. `export_locus` emits per-method
PIP, CS membership, reference-variant LD r-squared, and optional aligned GWAS Z.
The reference coordinate is required. It produces no image.

## Provenance, changes and verification

[`source_manifest.json`](source_manifest.json) records source labels and original
SHA256 checksums before migration. Entries marked `source_name_is_alias` use
scELF publication names; their hashes still identify the original source content.
Other source paths are relative to the original project. Related weight scripts
were merged into one entry point. Analysis calculations originally embedded in plot programs were
separated into table exporters; plotting code and external-validation programs
were not copied into this workflow. Public model naming consistently uses scELF.

Beyond CLI conversion, input ID validation, explicit PIP/missingness/allele
policies, saved posterior means, and temporary-file handling are intentional
changes. Do not expect unverified bit-for-bit agreement with old server outputs.
The current manuscript's RA disease list also differs from some historical
plot configurations; choose the intended diseases through `--traits` and input
paths. This directory does not impose either historical trait list.

Run checks from the repository root after installation:

```sh
python -m unittest discover -s tests -p "test_coloc.py" -v
python -m unittest discover -s tests -p "test_onek1k_workflows.py" -v
Rscript tests/test_onek1k_normalization.R
```

Tests check the H3/H4 analytical formula, donor/variant reordering, PIP policies,
allele flips and palindromes, LD normalization, and the small synthetic downstream
workflow. AnnData/PLINK preparation tests run when analysis dependencies are
installed. Full SCT and SAIGE-QTL study analyses require their R environments
and real input data; synthetic checks do not establish manuscript reproduction.
The GitHub Actions workflow runs the synthetic Python checks in dedicated core,
native-backend and optional-analysis jobs. Historical baseline comparisons skip
without a user-supplied reference snapshot. Full SCT and SAIGE-QTL R execution
remains a separate environment-dependent check.
