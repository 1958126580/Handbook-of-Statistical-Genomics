# ============================================================================
# StatisticalGenomics.jl - Fine-Mapping and PRS Tutorial
# ============================================================================
# This tutorial covers fine-mapping with SuSiE and polygenic risk score
# calculation using LDpred2, PRS-CS, and clumping+thresholding methods.
# ============================================================================

using StatisticalGenomics
using Random
using Statistics
using LinearAlgebra

println("="^70)
println("StatisticalGenomics.jl - Fine-Mapping and PRS Tutorial")
println("="^70)
println()

# ============================================================================
# Setup: Generate Simulated Data
# ============================================================================
println("Setting up simulated data...")
Random.seed!(2024)

# Discovery cohort (large GWAS)
n_discovery = 50000
n_variants = 200

# Target cohort (for PRS validation)
n_target = 5000

# Generate genotypes
X_discovery = randn(n_discovery, n_variants)
X_discovery = (X_discovery .- mean(X_discovery, dims=1)) ./ std(X_discovery, dims=1)

X_target = randn(n_target, n_variants)
X_target = (X_target .- mean(X_target, dims=1)) ./ std(X_target, dims=1)

# True causal effects (sparse: only 10 causal variants)
β_true = zeros(n_variants)
causal_idx = [5, 23, 47, 89, 112, 134, 156, 167, 189, 195]
β_true[causal_idx] = randn(10) .* 0.1

# Heritability
h2 = 0.4

# Generate phenotypes
y_discovery = X_discovery * β_true
var_g = var(y_discovery)
var_e = var_g * (1 - h2) / h2
y_discovery = y_discovery + randn(n_discovery) * sqrt(var_e)

y_target = X_target * β_true + randn(n_target) * sqrt(var_e)

println("Discovery cohort: $n_discovery samples, $n_variants variants")
println("Target cohort: $n_target samples")
println("True heritability: $h2")
println("Causal variants: $(length(causal_idx))")
println()

# Compute summary statistics
println("Computing GWAS summary statistics...")
betas = zeros(n_variants)
ses = zeros(n_variants)
pvalues = zeros(n_variants)
z_scores = zeros(n_variants)

for j in 1:n_variants
    β_hat = cov(X_discovery[:, j], y_discovery) / var(X_discovery[:, j])
    se_hat = sqrt(var(y_discovery) / (n_discovery * var(X_discovery[:, j])))
    betas[j] = β_hat
    ses[j] = se_hat
    z_scores[j] = β_hat / se_hat
    pvalues[j] = 2 * ccdf(Normal(), abs(z_scores[j]))
end

# LD matrix
R = cor(X_discovery)

println("Summary statistics computed")
println("  - Min p-value: $(round(minimum(pvalues), sigdigits=3))")
println("  - Genome-wide significant (p < 5e-8): $(sum(pvalues .< 5e-8))")
println()

# ============================================================================
# Part 1: Fine-Mapping with SuSiE
# ============================================================================
println("="^70)
println("Part 1: Fine-Mapping with SuSiE")
println("="^70)
println()

# Run SuSiE with individual-level data
println("Running SuSiE with individual-level data...")
susie_result = susie(X_discovery, y_discovery; L=10, min_abs_corr=0.5)

println("SuSiE results:")
println("  - Converged: $(susie_result.converged)")
println("  - Iterations: $(susie_result.n_iterations)")
println("  - Number of credible sets: $(length(susie_result.cs))")
println()

# Display PIPs
println("Top 15 variants by PIP:")
pip_sorted = sortperm(susie_result.pip, rev=true)
println("  Rank  Variant  PIP      True Effect")
println("  " * "-"^45)
for i in 1:15
    idx = pip_sorted[i]
    is_causal = idx in causal_idx ? "✓" : " "
    println("  $(lpad(i, 4))  $(lpad(idx, 7))  $(round(susie_result.pip[idx], digits=3))    $(round(β_true[idx], digits=3)) $is_causal")
end
println()

# Display credible sets
println("Credible sets (95% coverage):")
for (i, cs) in enumerate(susie_result.cs)
    if !isempty(cs)
        coverage = susie_result.cs_coverage[i]
        contains_causal = any(v in causal_idx for v in cs)
        marker = contains_causal ? "✓" : " "
        println("  CS$i: variants $cs (coverage: $(round(coverage, digits=2))) $marker")
    end
end
println()

# SuSiE with summary statistics
println("Running SuSiE-RSS (summary statistics)...")
susie_rss_result = susie_rss(z_scores, R, n_discovery; L=10)

println("SuSiE-RSS results:")
println("  - Converged: $(susie_rss_result.converged)")
println("  - Number of credible sets: $(length(susie_rss_result.cs))")
println()

# Compare PIPs
pip_correlation = cor(susie_result.pip, susie_rss_result.pip)
println("Correlation of PIPs (individual vs summary): $(round(pip_correlation, digits=3))")
println()

# Get credible set summary
cs_summary = susie_get_cs_summary(susie_result, X_discovery)
println("Credible set summary:")
println(cs_summary)
println()

# ============================================================================
# Part 2: Polygenic Risk Scores
# ============================================================================
println("="^70)
println("Part 2: Polygenic Risk Scores")
println("="^70)
println()

# Method 1: Clumping and Thresholding (C+T)
println("Method 1: Clumping + Thresholding")
println("-"^40)

p_thresholds = [5e-8, 1e-5, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]
ct_results = Dict()

for p_thresh in p_thresholds
    weights = clump_threshold_prs(betas, pvalues, X_target;
                                  p_threshold=p_thresh,
                                  r2_threshold=0.1)
    scores = compute_prs(X_target, weights.weights)
    validation = validate_prs(scores, y_target)
    ct_results[p_thresh] = (weights=weights, r2=validation.r2, n_snps=sum(weights.weights .!= 0))
end

println("C+T results across p-value thresholds:")
println("  P-threshold  N_SNPs  R²")
println("  " * "-"^35)
for p_thresh in p_thresholds
    result = ct_results[p_thresh]
    println("  $(rpad(p_thresh, 12))  $(lpad(result.n_snps, 6))  $(round(result.r2, digits=4))")
end

# Best C+T
best_p = argmax(p -> ct_results[p].r2, p_thresholds)
best_ct = ct_results[best_p]
println()
println("Best C+T: p < $best_p, R² = $(round(best_ct.r2, digits=4))")
println()

# Method 2: LDpred2-auto
println("Method 2: LDpred2-auto")
println("-"^40)

ldpred_weights = ldpred2_auto(betas, ses, R, n_discovery;
                              n_iter=200, n_burn=50)
ldpred_scores = compute_prs(X_target, ldpred_weights)
ldpred_validation = validate_prs(ldpred_scores, y_target)

println("LDpred2-auto results:")
println("  - R²: $(round(ldpred_validation.r2, digits=4))")
println("  - Correlation: $(round(ldpred_validation.correlation, digits=4))")
println()

# Method 3: LDpred2-grid
println("Method 3: LDpred2-grid")
println("-"^40)

grid_results = ldpred2_grid(betas, ses, R, n_discovery;
                            h2_grid=[0.1, 0.3, 0.5],
                            p_grid=[0.01, 0.1, 1.0])

println("LDpred2-grid results:")
println("  (h2, p)        R²")
println("  " * "-"^25)
for ((h2, p), weights) in grid_results
    scores = compute_prs(X_target, weights)
    val = validate_prs(scores, y_target)
    println("  ($(h2), $(p))  $(round(val.r2, digits=4))")
end
println()

# Method 4: PRS-CS
println("Method 4: PRS-CS")
println("-"^40)

prscs_weights = prs_cs(betas, ses, R, n_discovery;
                       n_iter=200, n_burn=50)
prscs_scores = compute_prs(X_target, prscs_weights)
prscs_validation = validate_prs(prscs_scores, y_target)

println("PRS-CS results:")
println("  - R²: $(round(prscs_validation.r2, digits=4))")
println("  - Correlation: $(round(prscs_validation.correlation, digits=4))")
println()

# ============================================================================
# Part 3: PRS Model Comparison and Validation
# ============================================================================
println("="^70)
println("Part 3: PRS Model Comparison")
println("="^70)
println()

# Compare all methods
println("Summary of PRS methods:")
println("  Method        R²      Correlation")
println("  " * "-"^40)
println("  C+T (best)    $(round(best_ct.r2, digits=4))  $(round(sqrt(best_ct.r2), digits=4))")
println("  LDpred2-auto  $(round(ldpred_validation.r2, digits=4))  $(round(ldpred_validation.correlation, digits=4))")
println("  PRS-CS        $(round(prscs_validation.r2, digits=4))  $(round(prscs_validation.correlation, digits=4))")
println()

# Theoretical maximum
r2_max = expected_prs_r2(n_discovery, n_target, h2; n_causal=length(causal_idx))
println("Theoretical maximum R² (given sample sizes): $(round(r2_max, digits=4))")
println()

# Decile analysis for best PRS
println("PRS decile analysis (LDpred2-auto):")
stratification = stratify_prs(ldpred_scores, y_target; n_groups=10)
println("  Decile  Mean Phenotype  N")
println("  " * "-"^35)
for i in 1:10
    println("  $(lpad(i, 6))  $(round(stratification.group_means[i], digits=3))          $(stratification.group_sizes[i])")
end
println()

# Risk enrichment (top vs bottom decile)
enrichment = (stratification.group_means[10] - stratification.group_means[1]) /
             std(y_target)
println("Top vs bottom decile difference: $(round(enrichment, digits=2)) SD")
println()

# ============================================================================
# Part 4: Cross-Ancestry PRS Portability (Simulated)
# ============================================================================
println("="^70)
println("Part 4: PRS Portability Across Populations")
println("="^70)
println()

# Simulate target population with different LD
println("Simulating target population with different LD structure...")
X_target2 = randn(n_target, n_variants)
# Add correlation structure different from discovery
for j in 2:n_variants
    X_target2[:, j] = 0.3 * X_target2[:, j-1] + sqrt(1 - 0.3^2) * X_target2[:, j]
end
X_target2 = (X_target2 .- mean(X_target2, dims=1)) ./ std(X_target2, dims=1)
y_target2 = X_target2 * β_true + randn(n_target) * sqrt(var_e)

# Apply PRS
ldpred_scores2 = compute_prs(X_target2, ldpred_weights)
ldpred_val2 = validate_prs(ldpred_scores2, y_target2)

println("PRS portability:")
println("  - Same population R²: $(round(ldpred_validation.r2, digits=4))")
println("  - Different population R²: $(round(ldpred_val2.r2, digits=4))")
println("  - Portability ratio: $(round(ldpred_val2.r2 / ldpred_validation.r2, digits=2))")
println()

# ============================================================================
# Summary
# ============================================================================
println("="^70)
println("Tutorial Summary")
println("="^70)
println("""
This tutorial demonstrated:

Fine-Mapping:
1. SuSiE with individual-level data
2. SuSiE-RSS with summary statistics
3. Credible set identification
4. Posterior inclusion probabilities (PIPs)

Polygenic Risk Scores:
1. Clumping + Thresholding (C+T) method
2. LDpred2-auto (automatic parameter tuning)
3. LDpred2-grid (grid search for h² and p)
4. PRS-CS (continuous shrinkage prior)

Key findings:
- Fine-mapping identified $(length(susie_result.cs)) credible sets
- Best PRS R² = $(round(max(best_ct.r2, ldpred_validation.r2, prscs_validation.r2), digits=4))
- PRS portability across populations is limited

Recommendations:
- Use SuSiE for identifying causal variants
- Use LDpred2-auto or PRS-CS for optimal prediction
- Validate PRS in independent samples
- Consider ancestry-matched training data
""")
