# ============================================================================
# StatisticalGenomics.jl - Main Module
# ============================================================================
# A comprehensive Julia package for statistical genomics analysis
# Based on the Handbook of Statistical Genomics (4th Edition)
# ============================================================================

"""
    StatisticalGenomics

A comprehensive Julia package for statistical genomics analysis covering:
- Population genetics and Hardy-Weinberg equilibrium
- Linkage disequilibrium and haplotype analysis
- Coalescent theory and demographic inference
- Phylogenetics and molecular evolution
- Genome-wide association studies (GWAS)
- Population structure and admixture
- Gene expression analysis (eQTL, differential expression)
- Epigenetics and methylation analysis
- Microbiome and metagenomics
- Forensic and conservation genetics
- Causal inference and Mendelian randomization

# Quick Start
```julia
using StatisticalGenomics

# Load genotype data
gm = read_plink("data/study")

# Run GWAS
result = gwas_single_variant(gm, phenotype)

# Multiple testing correction
fdr = fdr_correction(result.pvalues)

# Visualization
manhattan_plot(result)
```
"""
module StatisticalGenomics

# ============================================================================
# Standard Library Imports
# ============================================================================
using LinearAlgebra
using Statistics
using Random
using SparseArrays

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
# Package Initialization
# ============================================================================
function __init__()
    # Package initialization code
    @info "StatisticalGenomics.jl loaded successfully"
end

end # module StatisticalGenomics
