# ============================================================================
# StatisticalGenomics.jl - Getting Started Tutorial
# ============================================================================
# This tutorial introduces the basic functionality of StatisticalGenomics.jl
# covering data loading, quality control, and basic analysis workflows.
# ============================================================================

using StatisticalGenomics
using Random
using Statistics

println("="^70)
println("StatisticalGenomics.jl - Getting Started Tutorial")
println("="^70)
println()

# ============================================================================
# Section 1: Loading and Exploring Genotype Data
# ============================================================================
println("Section 1: Loading and Exploring Genotype Data")
println("-"^50)

# For this tutorial, we'll simulate genotype data
# In practice, you would use: gm = read_plink("path/to/data")

# Simulate genotype matrix (n_samples × n_variants)
Random.seed!(12345)
n_samples = 500
n_variants = 1000

# Generate random genotypes (0, 1, 2 = number of minor alleles)
genotype_data = rand(0:2, n_samples, n_variants) |> x -> Float64.(x)

# Create sample and variant identifiers
sample_ids = ["IND$(lpad(i, 4, '0'))" for i in 1:n_samples]
variant_ids = ["rs$(1000000 + i)" for i in 1:n_variants]

# Create GenotypeMatrix object
gm = GenotypeMatrix(genotype_data, sample_ids, variant_ids)

println("Genotype matrix loaded:")
println("  - Number of samples: $(n_samples(gm))")
println("  - Number of variants: $(n_variants(gm))")
println()

# ============================================================================
# Section 2: Quality Control
# ============================================================================
println("Section 2: Quality Control")
println("-"^50)

# Calculate minor allele frequencies
mafs = minor_allele_frequency(gm)
println("Minor Allele Frequency statistics:")
println("  - Min MAF: $(round(minimum(mafs), digits=4))")
println("  - Max MAF: $(round(maximum(mafs), digits=4))")
println("  - Mean MAF: $(round(mean(mafs), digits=4))")
println()

# Calculate missing rate (in real data)
miss_rates = missing_rate(gm)
println("Missing rate statistics:")
println("  - Mean missing rate: $(round(mean(miss_rates), digits=4))")
println()

# Hardy-Weinberg Equilibrium test for first variant
hwe_result = hwe_test(Int.(genotype_data[:, 1]))
println("HWE test for first variant:")
println("  - Chi-squared: $(round(hwe_result.chi_squared, digits=4))")
println("  - P-value: $(round(hwe_result.pvalue, digits=4))")
println()

# ============================================================================
# Section 3: Creating Phenotype Data
# ============================================================================
println("Section 3: Creating Phenotype Data")
println("-"^50)

# Simulate a quantitative phenotype influenced by some variants
# True causal variants: first 5 variants
causal_effects = zeros(n_variants)
causal_effects[1:5] = [0.3, -0.2, 0.25, 0.15, -0.1]
phenotype_values = genotype_data * causal_effects + randn(n_samples)

# Create ContinuousPhenotype object
phenotype = ContinuousPhenotype(phenotype_values, "SimulatedTrait")

println("Phenotype created:")
println("  - Name: $(phenotype.name)")
println("  - Mean: $(round(mean(phenotype.values), digits=4))")
println("  - Std: $(round(std(phenotype.values), digits=4))")
println()

# Standardize phenotype (mean=0, sd=1)
std_phenotype = standardize(phenotype)
println("After standardization:")
println("  - Mean: $(round(mean(std_phenotype.values), digits=6))")
println("  - Std: $(round(std(std_phenotype.values), digits=4))")
println()

# ============================================================================
# Section 4: Basic GWAS
# ============================================================================
println("Section 4: Basic GWAS")
println("-"^50)

# Run single-variant GWAS
println("Running GWAS...")
gwas_result = gwas_single_variant(gm, std_phenotype)

println("GWAS completed:")
println("  - Number of variants tested: $(length(gwas_result.pvalues))")
println("  - Minimum p-value: $(round(minimum(gwas_result.pvalues), sigdigits=3))")
println()

# Top 10 hits
sorted_idx = sortperm(gwas_result.pvalues)
println("Top 10 associations:")
println("  Rank  Variant       P-value")
println("  " * "-"^40)
for i in 1:10
    idx = sorted_idx[i]
    pval = gwas_result.pvalues[idx]
    println("  $(lpad(i, 4))  $(variant_ids[idx])  $(round(pval, sigdigits=3))")
end
println()

# Check if causal variants are detected
println("Causal variant detection:")
for i in 1:5
    pval = gwas_result.pvalues[i]
    rank = findfirst(==(i), sorted_idx)
    println("  Variant $i (effect=$(causal_effects[i])): rank=$rank, p=$(round(pval, sigdigits=3))")
end
println()

# ============================================================================
# Section 5: Multiple Testing Correction
# ============================================================================
println("Section 5: Multiple Testing Correction")
println("-"^50)

# Bonferroni correction
bonf = bonferroni_correction(gwas_result.pvalues)
n_bonf_sig = sum(gwas_result.pvalues .< bonf.threshold)
println("Bonferroni correction:")
println("  - Threshold: $(round(bonf.threshold, sigdigits=3))")
println("  - Significant variants: $n_bonf_sig")
println()

# FDR correction (Benjamini-Hochberg)
fdr = fdr_correction(gwas_result.pvalues)
n_fdr_sig = sum(fdr.qvalues .< 0.05)
println("FDR correction (q < 0.05):")
println("  - Significant variants: $n_fdr_sig")
println()

# Genomic control
gc = genomic_control(gwas_result.pvalues)
println("Genomic control:")
println("  - Lambda GC: $(round(gc.lambda_gc, digits=4))")
println("  (Lambda ≈ 1 indicates no inflation)")
println()

# ============================================================================
# Section 6: Linkage Disequilibrium
# ============================================================================
println("Section 6: Linkage Disequilibrium")
println("-"^50)

# Calculate LD between first two variants
ld_result = calculate_ld(Int.(genotype_data[:, 1]), Int.(genotype_data[:, 2]))
println("LD between rs1000001 and rs1000002:")
println("  - r²: $(round(ld_result.r_squared, digits=4))")
println("  - D': $(round(ld_result.d_prime, digits=4))")
println()

# Calculate LD matrix for first 10 variants
subset_gm = GenotypeMatrix(genotype_data[:, 1:10])
ld_mat = ld_matrix(subset_gm)
println("LD matrix (10×10) calculated")
println("  - Mean off-diagonal r²: $(round(mean(ld_mat[ld_mat .!= 1.0]), digits=4))")
println()

# ============================================================================
# Section 7: Population Structure with PCA
# ============================================================================
println("Section 7: Population Structure with PCA")
println("-"^50)

# Run genetic PCA
pca_result = genetic_pca(gm; n_components=10)

println("PCA completed:")
println("  - PC1 variance explained: $(round(pca_result.variance_explained[1] * 100, digits=2))%")
println("  - PC2 variance explained: $(round(pca_result.variance_explained[2] * 100, digits=2))%")
println("  - Total variance (10 PCs): $(round(sum(pca_result.variance_explained) * 100, digits=2))%")
println()

# ============================================================================
# Section 8: Summary
# ============================================================================
println("="^70)
println("Tutorial Summary")
println("="^70)
println("""
This tutorial demonstrated:
1. Loading and creating genotype matrices
2. Quality control: MAF calculation, missing rates, HWE testing
3. Creating and standardizing phenotype data
4. Running single-variant GWAS
5. Multiple testing correction (Bonferroni, FDR)
6. Calculating linkage disequilibrium
7. Population structure analysis with PCA

Next steps:
- See 02_advanced_gwas.jl for mixed models and rare variant tests
- See 03_finemapping_prs.jl for fine-mapping and polygenic scores
- See 04_heritability.jl for heritability estimation
""")
