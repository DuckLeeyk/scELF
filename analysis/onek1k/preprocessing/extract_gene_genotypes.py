"""Extract per-gene PLINK files from explicit candidate-SNP lists.

Candidate lists must already encode the chosen cis window, MAF and OneK1K
variant intersection. This step does not perform genotype imputation.
"""
import argparse
from pathlib import Path
import subprocess
from ..common import token


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfile", type=Path, required=True, help="Source PLINK bed/bim/fam prefix.")
    p.add_argument("--snp-list-dir", type=Path, required=True, help="Contains gene.txt lists of PLINK variant IDs.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--genes", nargs="+")
    group.add_argument("--gene-list", type=Path, help="One gene per line, without header.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--plink", default="plink")
    p.add_argument("--maf", type=float, help="Optional additional PLINK MAF threshold; input lists define the primary filters.")
    args = p.parse_args(argv)
    if args.maf is not None and not 0 <= args.maf <= .5:
        p.error("--maf must be in [0,0.5]")
    genes = args.genes or args.gene_list.read_text(encoding="utf-8-sig").split()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for gene in dict.fromkeys(genes):
        gene = token(gene)
        snps = args.snp_list_dir / f"{gene}.txt"
        if not snps.is_file() or not snps.read_text().strip():
            raise ValueError(f"Missing or empty SNP list: {snps}")
        command = [args.plink, "--bfile", str(args.bfile), "--extract", str(snps),
                   "--make-bed", "--out", str(args.output_dir / gene)]
        if args.maf is not None:
            command += ["--maf", str(args.maf)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
