"""Scientific alignment and small end-to-end checks for the public workflows.

All fixtures are synthetic. Optional AnnData/PLINK checks skip when those
analysis dependencies are absent; no private study data are required.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from analysis.onek1k.common import (canonical_coordinate, load_genotype, normalize_weights,
    standardize_genotype, write_json, write_table)
from analysis.onek1k.colocalization._workflow import analyze_gene, main_for as coloc_main
from analysis.onek1k.twas.harmonize_gwas import harmonize_chunk, main as harmonize_main
from analysis.onek1k.twas.export_gwas_zscores import main as export_main
from analysis.onek1k.twas.select_cs_genes import main as candidates_main
from analysis.onek1k.twas.fit_weights import main as weights_main
from analysis.onek1k.twas.calculate_scores import score_weights, main as scores_main
from analysis.onek1k.twas.summarize_loci import main as loci_main
from analysis.onek1k.twas.export_locus import main as locus_main
from analysis.onek1k.preprocessing.filter_eqtl_table import main as filter_main
from analysis.onek1k.colocalization.prepare_finngen import main as finngen_main
from analysis.onek1k.colocalization.prepare_bcx2 import main as bcx_main


class WorkflowTests(unittest.TestCase):
    def test_saige_effect_alleles_and_marker_order(self):
        from analysis.onek1k.finemapping._workflow import align_saige_output
        raw = pd.DataFrame({"SNPID": ["rs2","rs1"], "Allele1": ["2","1"], "Allele2": ["1","2"],
                            "BETA": [1.,2.], "SE": [.5,.5]})
        result = align_saige_output(raw,["1:1","1:2"])
        assert_allclose(result.BETA,[-2,1])
        assert_allclose(result.zscore,[-4,2])
        assert_allclose(result.BETA_original,[2,1])
        self.assertTrue(result.target_allele.eq("A2").all())
        with self.assertRaises(ValueError):
            align_saige_output(raw.iloc[:1],["1:1","1:2"])

    def test_streamed_eqtl_deduplication_and_gwas_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw, genes, output = root/"eqtl.tsv.gz", root/"genes.txt", root/"filtered.csv.gz"
            genes.write_text("GeneA\n")
            write_table(pd.DataFrame({"RSID": ["r1","r2","r1","r3"], "GENE": ["GeneA","GeneB","GeneA","GeneA"], "CHR": [1]*4,"POS": [1,2,1,3]}),raw)
            filter_main(["--input",str(raw),"--gene-list",str(genes),"--output",str(output),"--chunksize","2"])
            self.assertEqual(pd.read_csv(output).RSID.tolist(),["r1","r3"])
            bcx = root/"bcx.tsv"
            write_table(pd.DataFrame({"MarkerName": ["1:3_A_C"],"Sentinel_MarkerName": ["1:3_A_C"],"PP": [.9]}),bcx)
            bcx_main(["--input",str(bcx),"--output",str(root/"bcx_prepared.tsv"),"--genome-build","GRCh37"])
            self.assertEqual(pd.read_csv(root/"bcx_prepared.tsv",sep="\t").MarkerName.iloc[0],"1:3")
            finn = root/"finn.tsv"
            write_table(pd.DataFrame({"chromosome": [1],"position": [3],"region": ["r"],"cs": [1],"cs_specific_prob": [.9]}),finn)
            mapping = root/"map.tsv"
            write_table(pd.DataFrame({"source_POS": ["1:3"],"target_POS": ["1:2"]}),mapping)
            finngen_main(["--input",str(finn),"--output",str(root/"finn_prepared.tsv"),"--input-build","GRCh38","--output-build","GRCh37","--coordinate-map",str(mapping)])
            self.assertEqual(pd.read_csv(root/"finn_prepared.tsv",sep="\t")["chr:pos"].iloc[0],"1:2")

    def test_donor_ids_are_reordered_and_leading_zeros_retained(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "g.csv"
            pd.DataFrame({"1:1": [0, 2], "1:2": [1, 0]}, index=["002", "001"]).to_csv(path)
            frame = load_genotype(path, ["001", "002"])
            assert_allclose(frame.to_numpy(), [[2, 0], [0, 1]])
            with self.assertRaises(ValueError):
                load_genotype(path, ["003"])
        self.assertEqual(canonical_coordinate("chr01:00012_A_G"), "1:12")
        self.assertEqual(canonical_coordinate("chrMT:12"), "MT:12")

    def test_complete_cs_union_is_distinct_from_legacy(self):
        pips = pd.DataFrame({"POS": ["1:1", "1:2"], "PIP": [.8, .1]})
        gwas = pd.DataFrame({"POS": ["1:1", "1:3"], "PIP": [.6, .3], "cs": ["s", "s"]})
        sets = {"0": np.array([0, 1])}
        legacy = list(analyze_gene(pips, sets, gwas, "legacy", {}))[0]
        union = list(analyze_gene(pips, sets, gwas, "union-zero", {}))[0]
        self.assertEqual(legacy["n_gwas_variants"], 1)
        self.assertEqual(union["n_gwas_variants"], 2)
        shared = .8 * .6
        expected = 1e-7 * shared / (1e-7 * shared + 1e-8 * (.9*.9-shared))
        self.assertAlmostEqual(union["coloc_pip_score"], expected)
        self.assertNotAlmostEqual(union["coloc_pip_score"], legacy["coloc_pip_score"])
        absent = list(analyze_gene(pips, sets, gwas.assign(POS=["2:1", "2:3"]), "union-zero", {}))[0]
        self.assertEqual(absent["status"], "no_gwas_overlap")
        self.assertNotIn("coloc_pip_score", absent)

    def test_allele_orientation_and_palindromes(self):
        chunk = pd.DataFrame({"POS": [f"1:{i}" for i in range(1, 7)],
            "ea": ["C", "A", "G", "T", "C", "T"], "oa": ["A", "C", "T", "G", "A", "A"], "z": [2]*6})
        bim = pd.DataFrame({"POS": ["1:1", "1:2", "1:3", "1:4", "1:6"], "A1": ["A"]*5, "A2": ["C"]*4+["T"]})
        result = harmonize_chunk(chunk, bim, "ea", "oa", "z", None, "A2", "drop")
        assert_allclose(result.zscore.iloc[:4], [2, -2, 2, -2])
        self.assertEqual(result.alignment_status.iloc[4], "missing_in_bim")
        self.assertEqual(result.alignment_status.iloc[5], "palindromic")
        self.assertTrue(result.zscore.iloc[4:].isna().all())
        named_alleles = chunk.rename(columns={"ea": "A2", "oa": "A1"})
        result = harmonize_chunk(named_alleles, bim, "A2", "A1", "z", None, "A2", "drop")
        assert_allclose(result.zscore.iloc[:4], [2, -2, 2, -2])

    def test_twas_denominator_is_applied_once(self):
        genotype = standardize_genotype(np.array([[0,0,1],[0,1,0],[1,1,2],[2,2,1],[2,1,0]]))
        mean = np.array([.3, -.2, .1])
        normalized = normalize_weights(mean, genotype)
        ld = genotype.T @ genotype / len(genotype)
        self.assertAlmostEqual(normalized @ ld @ normalized, 1)
        z = pd.Series([2., -1., 3.], index=["1:1", "1:2", "1:3"])
        frame = pd.DataFrame({"POS": z.index, "weights": normalized})
        observed, _, rate, n = score_weights(frame, z)
        self.assertAlmostEqual(observed, mean @ z / np.sqrt(mean @ ld @ mean))
        self.assertEqual((rate, n), (0, 3))
        with self.assertRaises(ValueError):
            score_weights(frame, z.iloc[:1], "error")

    def test_synthetic_cli_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ct, gene, trait, method = "CellA", "GeneA", "TraitA", "scelf"
            analysis, output = root/"inputs", root/"fits"
            data = analysis/ct/"data"
            (data/"genotype").mkdir(parents=True)
            donors = ["001", "002", "003", "004", "005"]
            keys = ["1:1", "1:2", "1:3"]
            (data/f"genotype_IDs_{ct}.txt").write_text("\n".join(donors))
            pd.DataFrame([[0,0,1],[0,1,0],[1,1,2],[2,2,1],[2,1,0]], index=donors, columns=keys).iloc[::-1].to_csv(data/"genotype"/f"genotype_{ct}_{gene}.csv")
            mapping = root/"celltypes.tsv"
            write_table(pd.DataFrame({"predicted.celltype.l2": ["RawA"], "l2_name": [ct], "14_celltype_name": ["GroupA"]}), mapping)
            egenes = root/"egenes"
            write_table(pd.DataFrame({"gene": [gene], "gene_id": ["id1"]}), egenes/"GroupA_eGene.csv")
            write_table(pd.DataFrame({"POS": keys, "PIP": [.8,.1,.01]}), output/ct/f"{method}_pip"/f"{gene}.csv")
            write_json({"cs": {"0": [0,1]}}, output/ct/f"{method}_cs"/f"{gene}.json")
            # Deliberately permuted effect rows test variant-ID alignment.
            write_table(pd.DataFrame({"POS": keys, "posterior_mean": [.3,-.2,.1]}).iloc[::-1], output/ct/f"{method}_effects"/f"{gene}.csv")
            pairs = root/"pairs.tsv"
            candidates_main(["--analysis-dir",str(output),"--cell-types",ct,"--methods",method,"--output",str(pairs)])
            weights = root/"weights"
            weights_main(["--method",method,"--analysis-dir",str(analysis),"--effects-dir",str(output),"--output-dir",str(weights),"--celltype-table",str(mapping),"--egene-dir",str(egenes),"--gene-pairs",str(pairs)])
            bim = root/"input.bim"
            bim.write_text("1 rs1 0 1 A C\n1 rs2 0 2 A C\n1 rs3 0 3 A C\n")
            raw = root/"gwas.tsv"
            write_table(pd.DataFrame({"rs_number": ["1:1_A_C","1:2_A_C","1:3_A_C"],"ea": ["C","A","C"],"oa": ["A","C","A"],"z": [2,1,3]}), raw)
            aligned, slim = root/"aligned.tsv", root/"slim.tsv"
            harmonize_main(["--input",str(raw),"--bim",str(bim),"--output",str(aligned),"--genome-build","GRCh37","--effect-allele-column","ea","--non-effect-allele-column","oa","--target-allele","A2","--chunksize","2"])
            export_main(["--input",str(aligned),"--output",str(slim),"--chunksize","2"])
            scores = root/"scores.tsv"
            scores_main(["--weights-dir",str(weights),"--output",str(scores),"--cell-types",ct,"--methods",method,"--traits",trait,"--gwas-template",str(slim),"--dosage-allele","A2"])
            self.assertEqual(pd.read_csv(scores,sep="\t").status.iloc[0],"scored")
            gwas_cs = root/"gwas_cs.tsv"
            write_table(pd.DataFrame({"MarkerName": keys,"Sentinel_MarkerName": ["signal1"]*3,"PP": [.6,.2,.1]}), gwas_cs)
            coloc = root/"coloc"
            coloc_main("bcx2",["--analysis-dir",str(output),"--output-dir",str(coloc),"--cell-types",ct,"--methods",method,"--traits",trait,"--gwas-template",str(gwas_cs),"--genome-build","GRCh37","--pip-policy","union-zero"])
            joined = root/"loci.tsv"
            loci_main(["--twas",str(scores),"--coloc",str(coloc/ct/f"{trait}_{method}_coloc.tsv"),"--output",str(joined)])
            self.assertTrue(np.isfinite(pd.read_csv(joined,sep="\t").coloc_pip_score.iloc[0]))
            locus = root/"locus"
            locus_main(["--analysis-dir",str(analysis),"--finemapping-dir",str(output),"--output-dir",str(locus),"--cell-type",ct,"--gene",gene,"--trait",trait,"--reference-pos","1:1","--methods",method,"--gwas",str(slim)])
            evidence = pd.read_csv(locus/"locus_variants.tsv",sep="\t")
            self.assertAlmostEqual(evidence.loc[evidence.POS.eq("1:1"),"ld_r2"].iloc[0],1)
            assert_allclose(evidence.gwas_zscore, [2,-1,3])


@unittest.skipUnless(importlib.util.find_spec("anndata"), "Install scelf[analysis] for h5ad checks")
class PreparationTests(unittest.TestCase):
    def test_metadata_pcs_and_counts_preserve_barcode_order(self):
        import anndata as ad
        from scipy import sparse
        from analysis.onek1k.preprocessing.prepare_donor_covariates import main as donor_main
        from analysis.onek1k.preprocessing.compute_expression_pcs import main as pc_main
        from analysis.onek1k.preprocessing.prepare_counts import main as counts_main
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Rows are interleaved by donor, and donor mapping reverses their order.
            obs = pd.DataFrame({"donor_id": ["d2","d1"]*4, "pool_number": ["p2","p1"]*4,
                "age": [30,40]*4, "sex": ["F","M"]*4, "predicted.celltype.l2": ["RawA"]*8,
                "nCount_RNA": [20]*8, "percent.mt": np.arange(8)/10}, index=[f"c{i}" for i in range(8)])
            matrix = np.arange(32).reshape(8,4) % 7
            mapping = root/"mapping.tsv"
            write_table(pd.DataFrame({"donor_id": ["d1","d2"],"genotype_ID": ["002","001"]}), mapping)
            eigenvec = root/"pcs.eigenvec"
            eigenvec.write_text("f 002 1 2\nf 001 3 4\n")
            ct_table = root/"celltypes.tsv"
            write_table(pd.DataFrame({"predicted.celltype.l2": ["RawA"],"l2_name": ["CellA"],"14_celltype_name": ["GroupA"]}), ct_table)
            egenes = root/"egenes"
            write_table(pd.DataFrame({"gene": ["GeneA"],"gene_id": ["id1"]}), egenes/"GroupA_eGene.csv")
            for kind, values in [("dense", matrix), ("sparse", sparse.csr_matrix(matrix))]:
                h5ad = root/f"{kind}.h5ad"
                ad.AnnData(values, obs=obs, var=pd.DataFrame(index=["id1","id2","id3","id4"])).write_h5ad(h5ad)
                cov = root/f"{kind}_cov.csv"
                donor_main(["--h5ad",str(h5ad),"--sample-mapping",str(mapping),"--genotype-pcs",str(eigenvec),"--n-pcs","2","--output",str(cov)])
                pcs = root/f"{kind}_pcs"
                pc_main(["--h5ad",str(h5ad),"--celltype-table",str(ct_table),"--fit-components","3","--save-components","2","--output-dir",str(pcs)])
                output = root/kind
                counts_main(["--h5ad",str(h5ad),"--celltype-table",str(ct_table),"--egene-dir",str(egenes),"--sample-mapping",str(mapping),"--donor-covariates",str(cov),"--pcs-dir",str(pcs),"--output-dir",str(output)])
                data = output/"CellA"/"data"
                cell_cov = pd.read_csv(data/"cell_covariates_RawA.csv")
                order = [int(x[1:]) for x in cell_cov.barcode]
                assert_allclose(np.loadtxt(data/"counts"/"GReX_CellA_GeneA.txt"), matrix[order,0])
                self.assertEqual(order, [0,2,4,6,1,3,5,7])
                self.assertEqual((data/"genotype_IDs_CellA.txt").read_text().split(), ["001","002"])
                assert_allclose(np.loadtxt(data/"design_matrix_RawA.txt"), np.eye(2)[[0]*4+[1]*4])

    @unittest.skipUnless(importlib.util.find_spec("pandas_plink"), "Install pandas-plink for genotype preparation")
    def test_plink_alleles_donor_reordering_and_constant_filter(self):
        import xarray as xr
        from pandas_plink import write_plink1_bin
        from analysis.onek1k.preprocessing.prepare_genotypes import main as genotype_main
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = np.array([[0,1,2],[2,1,0],[1,1,2]], dtype=np.float32)
            arr = xr.DataArray(original, dims=["sample","variant"], coords={
                "sample": ["002","001","003"], "variant": ["v1","v2","v3"],
                "snp": ("variant",["rs1","rs2","rs3"]),"chrom": ("variant",["1"]*3),
                "pos": ("variant",[1,2,3]),"a0": ("variant",["A"]*3),"a1": ("variant",["C"]*3)})
            plink = root/"plink"
            plink.mkdir()
            write_plink1_bin(arr,str(plink/"GeneA.bed"),verbose=False)
            inputs = root/"inputs"
            data = inputs/"CellA"/"data"
            data.mkdir(parents=True)
            (data/"genotype_IDs_CellA.txt").write_text("001\n002\n003\n")
            mapping = root/"celltypes.tsv"
            write_table(pd.DataFrame({"predicted.celltype.l2": ["RawA"],"l2_name": ["CellA"],"14_celltype_name": ["GroupA"]}), mapping)
            egenes = root/"egenes"
            write_table(pd.DataFrame({"gene": ["GeneA"]}), egenes/"GroupA_eGene.csv")
            genotype_main(["--analysis-dir",str(inputs),"--celltype-table",str(mapping),"--egene-dir",str(egenes),"--plink-dir",str(plink),"--output-dir",str(inputs)])
            result = load_genotype(data/"genotype"/"genotype_CellA_GeneA.csv",["001","002","003"])
            assert_allclose(result,original[[1,0,2]][:,[0,2]])
            alleles = pd.read_csv(data/"genotype"/"genotype_CellA_GeneA.variants.tsv",sep="\t")
            self.assertTrue(alleles.A2.eq("C").all())


@unittest.skipUnless(importlib.util.find_spec("pysusie"), "Install scelf[comparators] for RSS checks")
class ComparatorTests(unittest.TestCase):
    def test_rss_aligns_summary_statistics_by_variant_id(self):
        from analysis.onek1k.finemapping._workflow import fit_rss
        rng = np.random.default_rng(91)
        genotype = standardize_genotype(rng.binomial(2, .35, size=(60, 5)))
        response = genotype[:,0] + rng.normal(size=60)
        response = (response-response.mean())/response.std()
        correlation = genotype.T@response/len(genotype)
        z = correlation * np.sqrt((len(genotype)-2)/(1-correlation**2))
        keys = [f"1:{i+1}" for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"summary.csv"
            data = pd.DataFrame({"POS": keys,"zscore": z})
            write_table(data,path)
            baseline, mean = fit_rss(genotype,keys,path,2)
            write_table(data.iloc[::-1],path)
            permuted, mean_permuted = fit_rss(genotype,keys,path,2)
            assert_allclose(baseline.pip,permuted.pip,atol=1e-12)
            assert_allclose(mean,mean_permuted,atol=1e-12)


if __name__ == "__main__":
    unittest.main()
