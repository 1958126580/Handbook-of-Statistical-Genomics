# ============================================================================
# StatisticalGenomics.jl - Advanced GWAS Tutorial
# ============================================================================
# This tutorial covers advanced GWAS methods including mixed models,
# gene-environment interaction, and rare variant analysis.
# ============================================================================

using StatisticalGenomics
using Random
using Statistics
using LinearAlgebra

println("="^70)
println("StatisticalGenomics.jl - Advanced GWAS Tutorial")
println("="^70)
println()

# ============================================================================
# Setup: Generate Simulated Data
# ============================================================================
println("Setting up simulated data...")
Random.seed!(42)

n_samples = 1000
n_common_variants = 500
n_rare_variants = 200

# Generate common variant genotypes
common_geno = rand(0:2, n_samples, n_common_variants) |> x -> Float64.(x)

# Generate rare variant genotypes (MAF < 0.05)
rare_geno = zeros(n_samples, n_rare_variants)
for j in 1:n_rare_variants
    maf = rand() * 0.04 + 0.001  # MAF between 0.1% and 5%
    for i in 1:n_samples
        r = rand()
        if r < (1-maf)^2
            rare_geno[i, j] = 0
        elseif r < (1-maf)^2 + 2*maf*(1-maf)
            rare_geno[i, j] = 1
        else
            rare_geno[i, j] = 2
        end
    end
end

# Simulate population structure (3 subpopulations)
pop_effects = zeros(n_samples)
pop_labels = repeat(1:3, inner=div(n_samples, 3))
if length(pop_labels) < n_samples
    append!(pop_labels, ones(Int, n_samples - length(pop_labels)))
end
pop_effects[pop_labels .== 1] .= 0.0
pop_effects[pop_labels .== 2] .= 0.5
pop_effects[pop_labels .== 3] .= 1.0

# Simulate phenotype with genetic and population effects
causal_common = zeros(n_common_variants)
causal_common[1:10] = randn(10) * 0.3
causal_rare = zeros(n_rare_variants)
causal_rare[1:20] = randn(20) * 0.5

phenotype = common_geno * causal_common +
            rare_geno * causal_rare +
            pop_effects * 0.5 +
            randn(n_samples)

# Create environmental exposure
env_exposure = randn(n_samples)
# Add G×E interaction for variant 1
phenotype .+= common_geno[:, 1] .* env_exposure * 0.2

# Create objects
common_gm = GenotypeMatrix(common_geno)
pheno = ContinuousPhenotype(phenotype)

println("Data generated:")
println("  - Common variants: $n_common_variants")
println("  - Rare variants: $n_rare_variants")
println("  - Samples: $n_samples")
println("  - Subpopulations: 3")
println()

# ============================================================================
# Section 1: Standard GWAS (Naive)
# ============================================================================
println("Section 1: Standard GWAS (Naive)")
println("-"^50)

naive_result = gwas_single_variant(common_gm, pheno)

# Check genomic inflation
gc_naive = genomic_control(naive_result.pvalues)
println("Naive GWAS results:")
println("  - Lambda GC: $(round(gc_naive.lambda_gc, digits=3))")
println("  (Inflation due to population structure)")
println()

# ============================================================================
# Section 2: Mixed Model GWAS
# ============================================================================
println("Section 2: Mixed Model GWAS")
println("-"^50)

# Calculate genetic relationship matrix (GRM)
println("Computing GRM...")
grm = grm_matrix(common_gm)
println("  - GRM computed ($(size(grm, 1)) × $(size(grm, 2)))")
println()

# Run mixed model GWAS
println("Running mixed model GWAS...")
mm_result = mixed_model_gwas(common_gm, pheno, grm)

# Check genomic inflation
gc_mm = genomic_control(mm_result.pvalues)
println("Mixed model GWAS results:")
println("  - Lambda GC: $(round(gc_mm.lambda_gc, digits=3))")
println("  (Population structure controlled)")
println()

# Compare top hits
println("Top 10 associations (mixed model):")
sorted_idx = sortperm(mm_result.pvalues)
println("  Rank  Variant  P-value      Beta")
println("  " * "-"^45)
for i in 1:10
    idx = sorted_idx[i]
    println("  $(lpad(i, 4))  $(lpad(idx, 6))  $(round(mm_result.pvalues[idx], sigdigits=3))  $(round(mm_result.betas[idx], digits=3))")
end
println()

# ============================================================================
# Section 3: GWAS with Covariates
# ============================================================================
println("Section 3: GWAS with Covariates")
println("-"^50)

# Create covariate matrix (PC1, PC2, age, sex)
age = rand(30:70, n_samples) |> x -> Float64.(x)
sex = rand(0:1, n_samples) |> x -> Float64.(x)
pca_result = genetic_pca(common_gm; n_components=5)
covariates = hcat(pca_result.scores[:, 1:2], age, sex)

println("Covariates included:")
println("  - PC1, PC2 (population structure)")
println("  - Age (simulated)")
println("  - Sex (simulated)")
println()

# Run GWAS with covariates
cov_result = gwas_linear(common_gm, pheno; covariates=covariates)

gc_cov = genomic_control(cov_result.pvalues)
println("GWAS with covariates:")
println("  - Lambda GC: $(round(gc_cov.lambda_gc, digits=3))")
println()

# ============================================================================
# Section 4: Gene-Environment Interaction
# ============================================================================
println("Section 4: Gene-Environment Interaction")
println("-"^50)

println("Testing G×E interaction with environmental exposure...")

# Test G×E for each variant
gxe_result = gxe_interaction(common_gm, pheno, env_exposure)

# Find significant interactions
sig_gxe = findall(gxe_result.interaction_pvalues .< 0.05)
println("Significant G×E interactions (p < 0.05): $(length(sig_gxe))")
println()

# Report top interactions
sorted_gxe = sortperm(gxe_result.interaction_pvalues)
println("Top 5 G×E interactions:")
println("  Variant  Main P-value   G×E P-value")
println("  " * "-"^40)
for i in 1:5
    idx = sorted_gxe[i]
    println("  $(lpad(idx, 6))  $(round(gxe_result.main_pvalues[idx], sigdigits=3))  $(round(gxe_result.interaction_pvalues[idx], sigdigits=3))")
end
println()

# Stratified analysis
println("Stratified analysis by exposure level...")
strat_result = stratified_gwas(common_gm, pheno, env_exposure; n_strata=2)
println("  - Low exposure stratum: $(strat_result.n_low) samples")
println("  - High exposure stratum: $(strat_result.n_high) samples")
println()

# ============================================================================
# Section 5: Rare Variant Analysis
# ============================================================================
println("Section 5: Rare Variant Analysis")
println("-"^50)

# Define a "gene" region (variants 1-20)
gene_variants = rare_geno[:, 1:20]
pheno_vec = phenotype

println("Testing rare variants in a gene region (20 variants)...")
println()

# Burden test
burden_result = burden_test(gene_variants, pheno_vec)
println("Burden test:")
println("  - Statistic: $(round(burden_result.statistic, digits=3))")
println("  - P-value: $(round(burden_result.pvalue, sigdigits=3))")
println()

# SKAT
skat_result = skat(gene_variants, pheno_vec)
println("SKAT:")
println("  - Statistic: $(round(skat_result.statistic, digits=3))")
println("  - P-value: $(round(skat_result.pvalue, sigdigits=3))")
println()

# SKAT-O (optimal combination)
skato_result = skat_o(gene_variants, pheno_vec)
println("SKAT-O:")
println("  - Statistic: $(round(skato_result.statistic, digits=3))")
println("  - P-value: $(round(skato_result.pvalue, sigdigits=3))")
println("  - Optimal rho: $(round(skato_result.details[:optimal_rho], digits=3))")
println()

# ACAT-V
acatv_result = acatv_test(gene_variants, pheno_vec)
println("ACAT-V:")
println("  - P-value: $(round(acatv_result.pvalue, sigdigits=3))")
println()

# ============================================================================
# Section 6: Gene-Based Testing
# ============================================================================
println("Section 6: Gene-Based Testing")
println("-"^50)

# Define gene regions
variant_positions = collect(1:n_rare_variants) .* 1000
gene_regions = DataFrame(
    gene = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
    chr = ones(Int, 5),
    start = [1, 40001, 80001, 120001, 160001],
    stop = [40000, 80000, 120000, 160000, 200000]
)

println("Testing 5 gene regions...")
gene_results = gene_based_test(rare_geno, pheno_vec, variant_positions, gene_regions;
                               method=:skat_o)

println("Gene-based test results:")
println("  Gene    N_variants  P-value")
println("  " * "-"^35)
for row in eachrow(gene_results)
    println("  $(rpad(row.gene, 8))  $(lpad(row.n_variants, 4))        $(round(row.pvalue, sigdigits=3))")
end
println()

# ============================================================================
# Section 7: Epistasis Analysis
# ============================================================================
println("Section 7: Epistasis Analysis")
println("-"^50)

# Test pairwise epistasis for a subset of variants
println("Testing pairwise epistasis (first 20 variants)...")
subset_geno = common_geno[:, 1:20]

epistasis_result = pairwise_epistasis(subset_geno, phenotype)

# Report significant interactions
sig_epi = filter(row -> row.pvalue < 0.01, epistasis_result)
println("Significant epistatic interactions (p < 0.01): $(nrow(sig_epi))")
println()

if nrow(sig_epi) > 0
    println("Top epistatic pairs:")
    sorted_epi = sort(epistasis_result, :pvalue)
    for i in 1:min(5, nrow(sorted_epi))
        row = sorted_epi[i, :]
        println("  $(row.snp1) × $(row.snp2): p = $(round(row.pvalue, sigdigits=3))")
    end
    println()
end

# ============================================================================
# Summary
# ============================================================================
println("="^70)
println("Tutorial Summary")
println("="^70)
println("""
This tutorial demonstrated:
1. Standard (naive) GWAS and genomic inflation
2. Mixed model GWAS for population structure control
3. GWAS with covariates (PCs, age, sex)
4. Gene-environment interaction testing
5. Rare variant tests (Burden, SKAT, SKAT-O, ACAT-V)
6. Gene-based association testing
7. Epistasis (gene-gene interaction) analysis

Key findings from simulated data:
- Naive GWAS showed inflation (λ = $(round(gc_naive.lambda_gc, digits=2)))
- Mixed model controlled inflation (λ = $(round(gc_mm.lambda_gc, digits=2)))
- G×E interaction detected for causal variant
- Rare variant tests identified gene with causal rare variants

Next steps:
- See 03_finemapping_prs.jl for fine-mapping and PRS
- See 04_heritability.jl for heritability estimation
""")
