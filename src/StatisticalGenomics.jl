# ============================================================================
# StatisticalGenomics.jl - Main Module
# ============================================================================
# A comprehensive Julia package for statistical genomics analysis
# Based on the Handbook of Statistical Genomics (4th Edition)
# Internationally top-tier software for genetic and genomic data analysis
# ============================================================================

"""
    StatisticalGenomics

A comprehensive, internationally top-tier Julia package for statistical genomics
analysis covering all aspects of the Handbook of Statistical Genomics (4th Edition):

## Core Analysis Domains
- **Population Genetics**: Hardy-Weinberg equilibrium, linkage disequilibrium, haplotypes
- **Evolutionary Genetics**: Wright-Fisher model, mutation, selection, drift
- **Coalescent Theory**: Kingman's coalescent, ARG, demographic inference
- **Phylogenetics**: Tree estimation, molecular evolution, adaptive evolution
- **GWAS**: Single-variant, mixed models, rare variants, gene-environment interaction
- **Fine-Mapping**: SuSiE, FINEMAP, credible sets, colocalization
- **PRS**: LDpred, PRS-CS, C+T methods, validation
- **Heritability**: LDSC, GREML, partitioning, genetic correlation
- **Meta-Analysis**: Fixed/random effects, trans-ancestry, heterogeneity
- **Expression**: eQTL, differential expression, co-expression networks
- **Epigenetics**: Methylation, EWAS, DMR detection
- **Single-Cell**: QC, normalization, clustering, trajectory analysis
- **Pharmacogenomics**: Star alleles, drug response, dosing
- **Causal Inference**: Mendelian randomization, effect estimation
- **Machine Learning**: Penalized regression, random forests, feature selection

## Quick Start
```julia
using StatisticalGenomics

# Load genotype data
gm = read_plink("data/study")

# Run GWAS
result = gwas_single_variant(gm, phenotype)

# Multiple testing correction
fdr = fdr_correction(result.pvalues)

# Fine-mapping
fm_result = susie_rss(z_scores, ld_matrix, n)

# Polygenic risk scores
prs_weights = ldpred2_auto(betas, se, R, n)

# Visualization
manhattan_plot(result)
qq_plot(result)
```

## Package Features
- High-performance implementations optimized for large-scale genomic data
- Comprehensive documentation with mathematical details
- Extensive test coverage (100% coverage target)
- Memory-efficient algorithms for biobank-scale data
- Parallel computing support via Julia's multi-threading
- Docker and HPC deployment ready

## References
- Handbook of Statistical Genomics, 4th Edition (Balding et al., 2019)
- See individual function documentation for method-specific references
"""
module StatisticalGenomics

# ============================================================================
# Version Information
# ============================================================================
const VERSION = v"2.0.0"
const PACKAGE_NAME = "StatisticalGenomics.jl"

# ============================================================================
# Standard Library Imports
# ============================================================================
using LinearAlgebra
using Statistics
using Random
using SparseArrays
using Dates

# ============================================================================
# External Package Imports
# ============================================================================
using Distributions
using StatsBase
using DataFrames
using CSV
using SpecialFunctions
using HypothesisTests
using Optim
using ProgressMeter
using JLD2

# ============================================================================
# Type Aliases for Clarity
# ============================================================================
const Chromosome = Union{Int, String}
const Position = Int
const Allele = Union{Char, String}
const GenotypeValue = Union{Int8, Missing}
const DosageValue = Union{Float64, Missing}
const AlleleFrequency = Float64

# ============================================================================
# Core Types Module
# ============================================================================
include("types/Types.jl")
include("types/Genotypes.jl")
include("types/Phenotypes.jl")
include("types/Populations.jl")

# ============================================================================
# Utility Modules
# ============================================================================
include("utils/Statistics.jl")
include("utils/IO.jl")
include("utils/Visualization.jl")
include("utils/Parallel.jl")

# ============================================================================
# Population Genetics Module
# ============================================================================
include("population/HardyWeinberg.jl")
include("population/LinkageDisequilibrium.jl")
include("population/Haplotypes.jl")
include("population/Imputation.jl")

# ============================================================================
# Evolution Module
# ============================================================================
include("evolution/WrightFisher.jl")
include("evolution/Mutation.jl")
include("evolution/Selection.jl")
include("evolution/Drift.jl")

# ============================================================================
# Coalescent Theory Module
# ============================================================================
include("coalescent/BasicCoalescent.jl")
include("coalescent/RecombinationCoalescent.jl")
include("coalescent/MultiSpeciesCoalescent.jl")
include("coalescent/CoalescentInference.jl")

# ============================================================================
# Phylogenetics Module
# ============================================================================
include("phylogenetics/TreeEstimation.jl")
include("phylogenetics/MolecularEvolution.jl")
include("phylogenetics/NaturalSelection.jl")
include("phylogenetics/AdaptiveEvolution.jl")

# ============================================================================
# Population Structure Module
# ============================================================================
include("structure/PCA.jl")
include("structure/Clustering.jl")
include("structure/Admixture.jl")
include("structure/AncientDNA.jl")

# ============================================================================
# GWAS Module
# ============================================================================
include("gwas/SingleVariant.jl")
include("gwas/MultipleTestingCorrection.jl")
include("gwas/MixedModels.jl")
include("gwas/GxE.jl")
include("gwas/RareVariant.jl")

# ============================================================================
# Bayesian Methods Module (NEW)
# ============================================================================
include("bayesian/MCMC.jl")
include("bayesian/VariationalBayes.jl")

# ============================================================================
# Heritability Module (NEW)
# ============================================================================
include("heritability/LDSC.jl")

# ============================================================================
# Fine-Mapping Module (NEW)
# ============================================================================
include("finemapping/SuSiE.jl")

# ============================================================================
# Polygenic Risk Scores Module (NEW)
# ============================================================================
include("prs/LDpred.jl")

# ============================================================================
# Meta-Analysis Module (NEW)
# ============================================================================
include("metaanalysis/MetaAnalysis.jl")

# ============================================================================
# Epistasis Module (NEW)
# ============================================================================
include("epistasis/GeneGeneInteraction.jl")

# ============================================================================
# Single-Cell Genomics Module (NEW)
# ============================================================================
include("singlecell/SingleCellAnalysis.jl")

# ============================================================================
# Pharmacogenomics Module (NEW)
# ============================================================================
include("pharmacogenomics/Pharmacogenomics.jl")

# ============================================================================
# Power Calculations Module (NEW)
# ============================================================================
include("power/PowerCalculations.jl")

# ============================================================================
# Machine Learning Module (NEW)
# ============================================================================
include("ml/MachineLearning.jl")

# ============================================================================
# Gene Expression Module
# ============================================================================
include("expression/EQTL.jl")
include("expression/DifferentialExpression.jl")
include("expression/CoExpressionNetworks.jl")
include("expression/PathwayEnrichment.jl")

# ============================================================================
# Epigenetics Module
# ============================================================================
include("epigenetics/MethylationProcessing.jl")
include("epigenetics/DifferentialMethylation.jl")
include("epigenetics/EWAS.jl")

# ============================================================================
# Microbiome Module
# ============================================================================
include("microbiome/CommunityAnalysis.jl")
include("microbiome/DiversityIndices.jl")
include("microbiome/MetagenomeAssociation.jl")

# ============================================================================
# Forensics and Conservation Module
# ============================================================================
include("forensics/ForensicDNA.jl")
include("forensics/Kinship.jl")
include("forensics/Conservation.jl")

# ============================================================================
# Causal Inference Module
# ============================================================================
include("causal/MendelianRandomization.jl")
include("causal/VariantAnnotation.jl")
include("causal/EffectSize.jl")

# ============================================================================
# Exports - Core Types
# ============================================================================
export AbstractGenotype, AbstractPhenotype, AbstractPopulation, AbstractVariant
export AbstractPhylogeneticTree, AbstractCoalescentTree, AbstractEvolutionaryModel
export AbstractAssociationResult, AbstractGeneticDistance
export StatisticalTestResult, ConfidenceInterval, EffectEstimate

# Genotype types
export SNPGenotype, GenotypeMatrix, DosageMatrix, VariantInfo
export n_samples, n_variants, minor_allele_frequency, missing_rate

# Phenotype types
export ContinuousPhenotype, BinaryPhenotype, CategoricalPhenotype
export SurvivalPhenotype, CovariateMatrix
export standardize, inverse_normal_transform

# Population types
export Population, PopulationSample, SubPopulation

# ============================================================================
# Exports - Statistics Utilities
# ============================================================================
export welch_t_test, chi_squared_test, fisher_exact_2x2
export linear_regression, logistic_regression
export permutation_test, correlation_test

# ============================================================================
# Exports - I/O Functions
# ============================================================================
export read_vcf, write_vcf, read_plink
export read_expression_matrix
export save_results, load_results

# ============================================================================
# Exports - Visualization
# ============================================================================
export manhattan_plot, qq_plot, pca_plot
export heatmap_plot, ld_heatmap, forest_plot

# ============================================================================
# Exports - Population Genetics
# ============================================================================
export allele_frequencies, genotype_frequencies
export hwe_test, inbreeding_coefficient, filter_hwe
export calculate_ld, ld_matrix, ld_prune, find_haplotype_blocks
export estimate_haplotypes, phase_genotypes, haplotype_frequencies
export impute_genotypes, imputation_quality

# ============================================================================
# Exports - Evolution
# ============================================================================
export wright_fisher_simulate, wright_fisher_trajectory
export fixation_probability, fixation_probability_theory
export expected_fixation_time, heterozygosity_decay

# Mutation models
export SubstitutionModel, JC69, K80, HKY85, GTR
export rate_matrix, transition_probability_matrix
export simulate_sequence, jukes_cantor_distance, kimura_distance

# Selection
export FitnessModel, AdditiveSelection, DominantSelection
export RecessiveSelection, OverdominantSelection
export fitness, mean_fitness, equilibrium_frequency
export selective_sweep_detect, ihs_score

# Drift
export genetic_drift_simulate, effective_population_size
export bottleneck_detect, founder_effect
export variance_effective_size

# ============================================================================
# Exports - Coalescent
# ============================================================================
export CoalescentTree, coalescent_simulate
export time_to_mrca, expected_branch_lengths
export simulate_mutations_on_tree, site_frequency_spectrum

# Recombination
export AncestralRecombinationGraph, coalescent_with_recombination
export arg_simulate, recombination_rate_estimate

# Multi-species
export SpeciesTree, multispecies_coalescent
export incomplete_lineage_sorting_probability

# Inference
export demographic_inference, skyline_plot_data
export estimate_theta_watterson, estimate_theta_pi
export tajima_D

# ============================================================================
# Exports - Phylogenetics
# ============================================================================
export PhyloTree, distance_matrix
export neighbor_joining, upgma, maximum_likelihood_tree
export phylogenetic_likelihood, gamma_rate_heterogeneity
export dn_ds_ratio, site_selection_test
export mcdonald_kreitman_test, positive_selection_sites

# ============================================================================
# Exports - Population Structure
# ============================================================================
export PCAResult, genetic_pca, pca_projection, tracy_widom_test
export ClusteringResult, structure_clustering, optimal_k
export admixture_proportions, f3_statistic, f4_statistic, d_statistic
export ancient_dna_damage, contamination_estimate

# ============================================================================
# Exports - GWAS
# ============================================================================
export GWASResult, gwas_single_variant
export gwas_linear, gwas_logistic, score_test
export gwas_to_dataframe, filter_gwas_results

# Multiple testing
export bonferroni_correction, fdr_correction
export genomic_control, permutation_threshold
export effective_number_of_tests

# Mixed models
export grm_matrix, mixed_model_gwas, emma_reml
export kinship_adjustment

# Gene-environment
export gxe_interaction, stratified_gwas
export heterogeneity_test, meta_analysis_gxe

# Rare variants (NEW)
export RareVariantResult
export burden_test, skat, skat_o
export cmc_test, vt_test, acatv_test
export gene_based_test

# ============================================================================
# Exports - Bayesian Methods (NEW)
# ============================================================================
export MCMCChain, MCMCDiagnostics
export metropolis_hastings, gibbs_sampler, hamiltonian_monte_carlo
export slice_sampler, parallel_tempering
export compute_diagnostics, summarize_chain

export VariationalResult, MeanFieldVariational
export advi, coordinate_ascent_vi
export variational_linear_regression, variational_spike_slab
export stochastic_vi, sample_from_variational

# ============================================================================
# Exports - Heritability (NEW)
# ============================================================================
export LDScoreResult, GeneticCorrelationResult, PartitionedHeritability
export compute_ld_scores, ldsc_regression
export genetic_correlation, partitioned_ldsc
export observed_to_liability
export compute_cell_type_enrichment, stratified_ldsc

# ============================================================================
# Exports - Fine-Mapping (NEW)
# ============================================================================
export SuSiEResult
export susie, susie_rss
export compute_pip, susie_get_cs_summary

# ============================================================================
# Exports - Polygenic Risk Scores (NEW)
# ============================================================================
export PRSResult, PRSWeights
export clump_threshold_prs
export ldpred2_grid, ldpred2_auto
export prs_cs
export compute_prs, validate_prs
export select_best_prs, stratify_prs, expected_prs_r2

# ============================================================================
# Exports - Meta-Analysis (NEW)
# ============================================================================
export MetaAnalysisResult
export fixed_effects_meta, random_effects_meta
export sample_size_weighted_meta
export trans_ancestry_meta, mr_mega
export gwas_meta_analysis
export forest_plot_data, leave_one_out_analysis
export publication_bias_test

# ============================================================================
# Exports - Epistasis (NEW)
# ============================================================================
export EpistasisResult
export pairwise_epistasis, boost_epistasis
export pathway_epistasis, mdr
export random_forest_epistasis

# ============================================================================
# Exports - Single-Cell (NEW)
# ============================================================================
export SingleCellData, SingleCellQC
export sc_qc, sc_normalize
export sc_highly_variable_genes, sc_pca
export sc_neighbors, sc_cluster
export sc_differential_expression, sc_umap

# ============================================================================
# Exports - Pharmacogenomics (NEW)
# ============================================================================
export StarAllele, PGxResult, DrugResponseResult
export call_star_alleles, predict_phenotype
export pgx_gwas, pgx_report
export warfarin_dose_prediction

# ============================================================================
# Exports - Power Calculations (NEW)
# ============================================================================
export PowerResult
export gwas_power, gwas_sample_size
export case_control_power
export rare_variant_power
export heritability_power
export prs_power
export genetic_correlation_power
export finemapping_power
export power_summary_plot_data, sample_size_table

# ============================================================================
# Exports - Machine Learning (NEW)
# ============================================================================
export PenalizedRegressionResult, RandomForestResult
export lasso, ridge, elastic_net
export random_forest, gradient_boosting
export feature_selection_stability

# ============================================================================
# Exports - Expression
# ============================================================================
export eqtl_mapping, cis_eqtl, trans_eqtl
export normalize_expression, differential_expression
export tmm_normalize, deseq_normalize
export coexpression_network, module_detection, module_eigengene
export pathway_enrichment, gsea

# ============================================================================
# Exports - Epigenetics
# ============================================================================
export beta_to_m_value, m_to_beta_value
export normalize_methylation
export differential_methylation, dmr_detection
export ewas_association, cell_type_adjust

# ============================================================================
# Exports - Microbiome
# ============================================================================
export community_profile, taxonomic_abundance
export shannon_diversity, simpson_diversity, chao1
export beta_diversity
export metagenome_association, compositional_analysis

# ============================================================================
# Exports - Forensics and Conservation
# ============================================================================
export match_probability, likelihood_ratio, str_analysis
export kinship_coefficient, ibd_estimation, relatedness_matrix
export inbreeding_coefficient_individual, population_viability

# ============================================================================
# Exports - Causal Inference
# ============================================================================
export mendelian_randomization, ivw_method, mr_egger, weighted_median
export variant_impact, conservation_score, regulatory_annotation
export heritability_estimate, genetic_variance
export polygenic_score, prs_from_gwas

# ============================================================================
# Utility Functions
# ============================================================================
"""
    version() -> VersionNumber

Return the package version.
"""
version() = VERSION

"""
    help_text() -> String

Return package help text.
"""
function help_text()
    return """
    StatisticalGenomics.jl v$(VERSION)
    ================================

    A comprehensive Julia package for statistical genomics analysis.

    Quick Start:
    - read_plink("prefix")     : Load PLINK files
    - gwas_single_variant()    : Run GWAS
    - susie_rss()              : Fine-mapping
    - ldpred2_auto()           : PRS computation
    - ldsc_regression()        : Heritability estimation

    Documentation: https://github.com/statgen/StatisticalGenomics.jl

    Type ?function_name for detailed help on any function.
    """
end

# ============================================================================
# Package Initialization
# ============================================================================
function __init__()
    # Set random seed for reproducibility if environment variable set
    if haskey(ENV, "STATGEN_SEED")
        Random.seed!(parse(Int, ENV["STATGEN_SEED"]))
    end

    @info "StatisticalGenomics.jl v$(VERSION) loaded successfully"
    @info "  - $(Threads.nthreads()) threads available"
end

end # module StatisticalGenomics
