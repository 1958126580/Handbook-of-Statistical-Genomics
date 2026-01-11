# ============================================================================
# StatisticalGenomics.jl - Heritability Analysis Tutorial
# ============================================================================
# This tutorial covers heritability estimation using LDSC, genetic correlation,
# and partitioned heritability analysis.
# ============================================================================

using StatisticalGenomics
using Random
using Statistics
using LinearAlgebra
using Distributions

println("="^70)
println("StatisticalGenomics.jl - Heritability Analysis Tutorial")
println("="^70)
println()

# ============================================================================
# Setup: Simulate GWAS Summary Statistics
# ============================================================================
println("Setting up simulated data...")
Random.seed!(2024)

n_snps = 10000
n_samples = 50000

# Simulate LD scores
# Real LD scores depend on genetic architecture and LD patterns
ld_scores = rand(Gamma(2, 15), n_snps) .+ 10  # Mean ~40

# True heritability
h2_true = 0.4

# Simulate chi-squared statistics under polygenicity
# E[chi2] = 1 + n * h2 * ld_score / M
inflation = 1 .+ (n_samples * h2_true / n_snps) .* ld_scores
chi2_stats = rand.(Chisq.(1)) .* inflation

# Convert to Z-scores
z_scores_trait1 = sign.(randn(n_snps)) .* sqrt.(chi2_stats)

println("Simulated GWAS for Trait 1:")
println("  - N SNPs: $n_snps")
println("  - N samples: $n_samples")
println("  - True h²: $h2_true")
println("  - Mean LD score: $(round(mean(ld_scores), digits=1))")
println("  - Mean χ²: $(round(mean(chi2_stats), digits=3))")
println()

# Simulate second trait (for genetic correlation)
h2_trait2 = 0.3
rg_true = 0.6  # Genetic correlation

# Correlated genetic effects
inflation2 = 1 .+ (n_samples * h2_trait2 / n_snps) .* ld_scores
chi2_trait2 = rand.(Chisq.(1)) .* inflation2
z_scores_trait2 = rg_true .* z_scores_trait1 .+ sqrt(1 - rg_true^2) .* sign.(randn(n_snps)) .* sqrt.(chi2_trait2)

println("Simulated GWAS for Trait 2:")
println("  - True h²: $h2_trait2")
println("  - True rg with Trait 1: $rg_true")
println()

# ============================================================================
# Part 1: Basic LD Score Regression
# ============================================================================
println("="^70)
println("Part 1: LD Score Regression")
println("="^70)
println()

# Run LDSC for heritability
ldsc_result = ldsc_regression(chi2_stats, ld_scores, n_samples)

println("LDSC Results for Trait 1:")
println("  - Estimated h²: $(round(ldsc_result.h2, digits=4))")
println("  - h² SE: $(round(ldsc_result.h2_se, digits=4))")
println("  - Intercept: $(round(ldsc_result.intercept, digits=4))")
println("  - Intercept SE: $(round(ldsc_result.intercept_se, digits=4))")
println("  - N SNPs used: $(ldsc_result.n_snps)")
println()

# Compare to true value
bias = ldsc_result.h2 - h2_true
println("Comparison to truth:")
println("  - True h²: $h2_true")
println("  - Bias: $(round(bias, digits=4))")
println("  - Within 2 SE: $(abs(bias) < 2 * ldsc_result.h2_se ? "Yes" : "No")")
println()

# ============================================================================
# Part 2: Liability Scale Conversion
# ============================================================================
println("="^70)
println("Part 2: Liability Scale Conversion")
println("="^70)
println()

# For binary traits, convert observed to liability scale
println("Example: Binary trait heritability conversion")
println()

observed_h2 = 0.15
population_prevalence = 0.01  # 1% disease
sample_prevalence = 0.5  # 50% cases in study

liability_h2 = observed_to_liability(observed_h2, population_prevalence, sample_prevalence)

println("Conversion parameters:")
println("  - Population prevalence: $(population_prevalence * 100)%")
println("  - Sample prevalence: $(sample_prevalence * 100)%")
println("  - Observed h²: $observed_h2")
println("  - Liability h²: $(round(liability_h2, digits=4))")
println()

# Different prevalences
println("Liability h² across prevalences (observed h² = 0.15):")
println("  Prevalence  Liability h²")
println("  " * "-"^30)
for prev in [0.001, 0.01, 0.05, 0.1, 0.2]
    liab = observed_to_liability(0.15, prev, 0.5)
    println("  $(rpad(prev, 10))  $(round(liab, digits=4))")
end
println()

# ============================================================================
# Part 3: Genetic Correlation
# ============================================================================
println("="^70)
println("Part 3: Genetic Correlation")
println("="^70)
println()

rg_result = genetic_correlation(z_scores_trait1, z_scores_trait2, ld_scores,
                                n_samples, n_samples)

println("Genetic Correlation Results:")
println("  - Estimated rg: $(round(rg_result.rg, digits=4))")
println("  - rg SE: $(round(rg_result.rg_se, digits=4))")
println("  - Z-score: $(round(rg_result.z, digits=3))")
println("  - P-value: $(round(rg_result.pvalue, sigdigits=3))")
println()

# Compare to truth
println("Comparison to truth:")
println("  - True rg: $rg_true")
println("  - Bias: $(round(rg_result.rg - rg_true, digits=4))")
println()

# Multiple trait correlations
println("Simulating correlations with multiple traits...")
n_traits = 5
rg_matrix = zeros(n_traits, n_traits)

# Simulate Z-scores for additional traits
z_all = zeros(n_snps, n_traits)
z_all[:, 1] = z_scores_trait1

for i in 2:n_traits
    h2_i = 0.3 + rand() * 0.2
    rg_i = 0.3 * randn()
    inflation_i = 1 .+ (n_samples * h2_i / n_snps) .* ld_scores
    chi2_i = rand.(Chisq.(1)) .* inflation_i
    z_all[:, i] = rg_i .* z_scores_trait1 .+ sqrt(1 - rg_i^2) .* sign.(randn(n_snps)) .* sqrt.(chi2_i)
end

# Compute pairwise correlations
println("Genetic correlation matrix:")
for i in 1:n_traits
    for j in i:n_traits
        if i == j
            rg_matrix[i, j] = 1.0
        else
            result = genetic_correlation(z_all[:, i], z_all[:, j], ld_scores,
                                        n_samples, n_samples)
            rg_matrix[i, j] = result.rg
            rg_matrix[j, i] = result.rg
        end
    end
end

println("      T1     T2     T3     T4     T5")
for i in 1:n_traits
    print("T$i  ")
    for j in 1:n_traits
        print("$(lpad(round(rg_matrix[i, j], digits=2), 6)) ")
    end
    println()
end
println()

# ============================================================================
# Part 4: Partitioned Heritability
# ============================================================================
println("="^70)
println("Part 4: Partitioned Heritability")
println("="^70)
println()

# Define functional annotations
n_categories = 3
annotations = zeros(n_snps, n_categories)

# Category 1: Coding regions (10% of SNPs, enriched for h²)
annotations[1:1000, 1] .= 1.0

# Category 2: Regulatory regions (20% of SNPs, enriched)
annotations[1001:3000, 2] .= 1.0

# Category 3: Other (70% of SNPs, baseline)
annotations[3001:end, 3] .= 1.0

# Generate category-specific LD scores
ld_scores_cat = zeros(n_snps, n_categories)
for j in 1:n_categories
    ld_scores_cat[:, j] = ld_scores .* annotations[:, j]
end

# Simulate with enrichment
# Coding: 5x enriched, Regulatory: 2x enriched
true_enrichment = [5.0, 2.0, 0.5]
chi2_part = ones(n_snps)
for i in 1:n_snps
    for j in 1:n_categories
        if annotations[i, j] == 1
            factor = true_enrichment[j]
            chi2_part[i] = rand(Chisq(1)) * (1 + n_samples * h2_true * factor * ld_scores[i] / n_snps)
            break
        end
    end
end

# Run partitioned LDSC
part_result = partitioned_ldsc(chi2_part, annotations, ld_scores_cat, n_samples)

println("Partitioned Heritability Results:")
println("  Category        h²       SE     Enrichment  SE")
println("  " * "-"^55)

categories = ["Coding", "Regulatory", "Other"]
for i in 1:n_categories
    println("  $(rpad(categories[i], 14))  $(lpad(round(part_result.h2_categories[i], digits=4), 6))  $(lpad(round(part_result.h2_se[i], digits=4), 6))  $(lpad(round(part_result.enrichment[i], digits=2), 8))    $(round(part_result.enrichment_se[i], digits=2))")
end
println()
println("  Total h²: $(round(part_result.h2_total, digits=4))")
println()

# ============================================================================
# Part 5: Cell Type Enrichment
# ============================================================================
println("="^70)
println("Part 5: Cell Type Enrichment")
println("="^70)
println()

# Simulate cell-type specific annotations
cell_types = Dict(
    "Neurons" => rand(Bool, n_snps),
    "Astrocytes" => rand(Bool, n_snps),
    "Microglia" => rand(Bool, n_snps),
    "Oligodendrocytes" => rand(Bool, n_snps)
)

# Compute cell-type enrichment
ct_result = compute_cell_type_enrichment(chi2_stats, cell_types, ld_scores, n_samples)

println("Cell Type Enrichment:")
println("  Cell Type           Enrichment  P-value")
println("  " * "-"^45)
for (ct, stats) in ct_result
    println("  $(rpad(ct, 20))  $(lpad(round(stats.enrichment, digits=2), 8))  $(round(stats.pvalue, sigdigits=3))")
end
println()

# ============================================================================
# Part 6: Power Calculations
# ============================================================================
println("="^70)
println("Part 6: Power Calculations")
println("="^70)
println()

println("Power to detect different h² values:")
println("  Sample Size   h² = 0.1   h² = 0.3   h² = 0.5")
println("  " * "-"^50)

sample_sizes = [10000, 25000, 50000, 100000, 200000]
for n in sample_sizes
    p1 = heritability_power(n, 0.1; n_snps=100000)
    p2 = heritability_power(n, 0.3; n_snps=100000)
    p3 = heritability_power(n, 0.5; n_snps=100000)
    println("  $(lpad(n, 10))   $(lpad(round(p1, digits=2), 8))   $(lpad(round(p2, digits=2), 8))   $(lpad(round(p3, digits=2), 8))")
end
println()

println("Power to detect genetic correlation:")
println("  N per trait   rg = 0.2   rg = 0.4   rg = 0.6")
println("  " * "-"^50)

for n in sample_sizes
    p1 = genetic_correlation_power(n, n, 0.2, 0.3, 0.3)
    p2 = genetic_correlation_power(n, n, 0.4, 0.3, 0.3)
    p3 = genetic_correlation_power(n, n, 0.6, 0.3, 0.3)
    println("  $(lpad(n, 11))   $(lpad(round(p1, digits=2), 8))   $(lpad(round(p2, digits=2), 8))   $(lpad(round(p3, digits=2), 8))")
end
println()

# ============================================================================
# Summary
# ============================================================================
println("="^70)
println("Tutorial Summary")
println("="^70)
println("""
This tutorial demonstrated:

Heritability Estimation:
1. LD Score Regression (LDSC) for h² estimation
2. Liability scale conversion for binary traits
3. Standard errors and significance testing

Genetic Correlation:
1. Cross-trait LDSC for genetic correlation
2. Multiple trait correlation matrices
3. Interpretation and visualization

Partitioned Heritability:
1. Functional annotation enrichment
2. Cell-type specific analysis
3. Interpreting enrichment statistics

Power Analysis:
1. Sample size requirements for h² detection
2. Power for genetic correlation
3. Study design considerations

Key findings from simulated data:
- Estimated h² = $(round(ldsc_result.h2, digits=3)) (true = $h2_true)
- Estimated rg = $(round(rg_result.rg, digits=3)) (true = $rg_true)
- Coding regions: $(round(part_result.enrichment[1], digits=1))x enriched

Recommendations:
- Use LDSC for summary-level heritability estimation
- Always report standard errors and confidence intervals
- Consider liability scale for binary traits
- Partitioned analysis reveals biological insights
""")
