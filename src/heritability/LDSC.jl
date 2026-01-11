# ============================================================================
# LDSC.jl - LD Score Regression for Heritability Estimation
# ============================================================================
# Implementation of LD Score Regression (Bulik-Sullivan et al. 2015)
# Estimates SNP heritability, genetic correlation, and partitioned h2
# ============================================================================

"""
    LDScoreResult

Structure containing results from LD score regression.

# Fields
- `h2::Float64`: SNP heritability estimate
- `h2_se::Float64`: Standard error of h2
- `intercept::Float64`: LD score regression intercept
- `intercept_se::Float64`: Standard error of intercept
- `lambda_gc::Float64`: Genomic control inflation factor
- `mean_chi2::Float64`: Mean chi-squared statistic
- `n_snps::Int`: Number of SNPs used
- `h2_liability::Union{Float64, Nothing}`: Liability scale h2 (for case-control)
"""
struct LDScoreResult
    h2::Float64
    h2_se::Float64
    intercept::Float64
    intercept_se::Float64
    lambda_gc::Float64
    mean_chi2::Float64
    n_snps::Int
    h2_liability::Union{Float64, Nothing}
end

"""
    GeneticCorrelationResult

Structure containing results from genetic correlation analysis.

# Fields
- `rg::Float64`: Genetic correlation estimate
- `rg_se::Float64`: Standard error of rg
- `z::Float64`: Z-score for rg
- `pvalue::Float64`: P-value for rg ≠ 0
- `h2_1::Float64`: Heritability of trait 1
- `h2_2::Float64`: Heritability of trait 2
- `gcov::Float64`: Genetic covariance
- `gcov_intercept::Float64`: Cross-trait intercept (sample overlap)
"""
struct GeneticCorrelationResult
    rg::Float64
    rg_se::Float64
    z::Float64
    pvalue::Float64
    h2_1::Float64
    h2_2::Float64
    gcov::Float64
    gcov_intercept::Float64
end

"""
    PartitionedHeritability

Structure containing partitioned heritability results.

# Fields
- `categories::Vector{String}`: Annotation category names
- `prop_snps::Vector{Float64}`: Proportion of SNPs in each category
- `prop_h2::Vector{Float64}`: Proportion of h2 in each category
- `enrichment::Vector{Float64}`: Enrichment (prop_h2 / prop_snps)
- `enrichment_se::Vector{Float64}`: SE of enrichment
- `enrichment_p::Vector{Float64}`: P-values for enrichment
- `tau::Vector{Float64}`: Per-SNP heritability coefficients
- `tau_se::Vector{Float64}`: SE of tau
"""
struct PartitionedHeritability
    categories::Vector{String}
    prop_snps::Vector{Float64}
    prop_h2::Vector{Float64}
    enrichment::Vector{Float64}
    enrichment_se::Vector{Float64}
    enrichment_p::Vector{Float64}
    tau::Vector{Float64}
    tau_se::Vector{Float64}
end

"""
    compute_ld_scores(genotypes::Matrix{Float64}; window_kb::Int=1000) -> Vector{Float64}

Compute LD scores for each variant.

# Arguments
- `genotypes`: Genotype matrix (n_samples × n_variants), coded 0/1/2
- `window_kb`: Window size in kilobases for LD computation

# Returns
- Vector of LD scores for each variant

# Mathematical Definition
LD score for variant j:
ℓ_j = Σ_k r²_{jk}

where the sum is over all variants k within the specified window.

For regression LD scores (Bulik-Sullivan et al. 2015):
ℓ_j = Σ_k (r²_{jk} - (1-r²_{jk})/(n-2))

This bias-correction accounts for sampling variance in r².

# Example
```julia
ld_scores = compute_ld_scores(genotypes; window_kb=1000)
```

# References
- Bulik-Sullivan et al. (2015) Nat. Genet.
"""
function compute_ld_scores(
    genotypes::Matrix{Float64},
    positions::Vector{Int};
    window_kb::Int=1000,
    min_maf::Float64=0.01,
    bias_correct::Bool=true
)
    n_samples, n_variants = size(genotypes)
    ld_scores = zeros(n_variants)

    # Calculate MAF and standardize genotypes
    mafs = vec(mean(genotypes, dims=1)) / 2
    valid = (mafs .>= min_maf) .& (mafs .<= 1 - min_maf)

    # Standardize: (G - 2p) / sqrt(2p(1-p))
    G_std = similar(genotypes)
    for j in 1:n_variants
        if valid[j]
            p = mafs[j]
            G_std[:, j] = (genotypes[:, j] .- 2p) ./ sqrt(2p * (1 - p))
        else
            G_std[:, j] .= 0
        end
    end

    # Window size in base pairs
    window_bp = window_kb * 1000

    # Compute LD scores
    @inbounds for j in 1:n_variants
        if !valid[j]
            continue
        end

        ld_sum = 0.0
        n_partners = 0

        # Find variants within window
        pos_j = positions[j]
        for k in 1:n_variants
            if !valid[k]
                continue
            end

            if abs(positions[k] - pos_j) <= window_bp
                # Compute r²
                r = dot(G_std[:, j], G_std[:, k]) / n_samples
                r2 = r^2

                if bias_correct && j != k
                    # Bias correction: E[r²] = r²_true + (1-r²_true)/(n-1)
                    # So: r²_true ≈ r² - (1-r²)/(n-2)
                    r2_corrected = r2 - (1 - r2) / (n_samples - 2)
                    ld_sum += max(0, r2_corrected)
                else
                    ld_sum += r2
                end
                n_partners += 1
            end
        end

        ld_scores[j] = ld_sum
    end

    return ld_scores
end

"""
    ldsc_regression(chi2::Vector{Float64}, ld_scores::Vector{Float64},
                   n_samples::Int; kwargs...) -> LDScoreResult

Run LD score regression to estimate SNP heritability.

# Arguments
- `chi2`: Vector of chi-squared statistics from GWAS
- `ld_scores`: LD scores for each variant
- `n_samples`: GWAS sample size

# Keyword Arguments
- `n_blocks::Int=200`: Number of blocks for jackknife SE estimation
- `intercept_constraint::Union{Float64, Nothing}=nothing`: Constrain intercept
- `prevalence::Union{Float64, Nothing}=nothing`: Disease prevalence (for liability h2)
- `sample_prevalence::Union{Float64, Nothing}=nothing`: Case proportion in sample
- `two_step::Bool=true`: Use two-step estimator (more robust)

# Mathematical Model
The LD score regression model:
E[χ²_j] = 1 + N * h²_g / M + Na

where:
- χ²_j is the chi-squared statistic for variant j
- ℓ_j is the LD score
- N is the sample size
- M is the number of variants
- h²_g is SNP heritability
- a is a constant capturing confounding bias

Rearranging:
χ²_j = (N * h²_g / M) * ℓ_j + (1 + Na)

This is a simple linear regression of χ² on LD scores.

# Two-Step Estimator
1. Initial regression with equal weights
2. Compute residual variance and reweight
3. Final weighted regression

# Example
```julia
# From GWAS summary statistics
chi2 = gwas_results.statistic
ld_scores = compute_ld_scores(ref_genotypes, positions)

result = ldsc_regression(chi2, ld_scores, n_samples)
println("h² = \$(result.h2) (SE = \$(result.h2_se))")
println("Intercept = \$(result.intercept)")
```

# References
- Bulik-Sullivan et al. (2015) Nat. Genet.
- Bulik-Sullivan et al. (2015) Nat. Genet. (partitioned)
"""
function ldsc_regression(
    chi2::Vector{Float64},
    ld_scores::Vector{Float64},
    n_samples::Int;
    n_blocks::Int=200,
    intercept_constraint::Union{Float64, Nothing}=nothing,
    prevalence::Union{Float64, Nothing}=nothing,
    sample_prevalence::Union{Float64, Nothing}=nothing,
    two_step::Bool=true,
    weights::Union{Vector{Float64}, Nothing}=nothing,
    min_maf::Float64=0.01
)
    # Filter to valid SNPs
    valid = (ld_scores .> 0) .& isfinite.(chi2) .& (chi2 .> 0)
    chi2_filt = chi2[valid]
    ld_filt = ld_scores[valid]
    n_snps = length(chi2_filt)
    M = n_snps  # Effective number of variants

    # Summary statistics
    mean_chi2 = mean(chi2_filt)
    lambda_gc = median(chi2_filt) / quantile(Chisq(1), 0.5)

    # Initial weights (inverse variance)
    if weights === nothing
        # Weight inversely by LD score (higher LD = higher variance)
        w = 1.0 ./ (1 .+ n_samples * ld_filt / M)
    else
        w = weights[valid]
    end
    w = w ./ sum(w)  # Normalize

    # Design matrix
    X = hcat(ones(n_snps), ld_filt)

    if intercept_constraint !== nothing
        # Constrain intercept
        y = chi2_filt .- intercept_constraint
        X_reg = reshape(ld_filt, :, 1)

        # Weighted least squares
        W = Diagonal(w)
        β = (X_reg' * W * X_reg) \ (X_reg' * W * y)

        slope = β[1]
        intercept = intercept_constraint
    else
        # Weighted least squares for full model
        W = Diagonal(w)
        β = (X' * W * X) \ (X' * W * chi2_filt)

        intercept = β[1]
        slope = β[2]
    end

    # Two-step estimator: re-estimate with refined weights
    if two_step && intercept_constraint === nothing
        residuals = chi2_filt .- (intercept .+ slope .* ld_filt)
        σ2 = mean(residuals.^2)

        # Heteroscedasticity-consistent weights
        predicted_var = (intercept .+ slope .* ld_filt).^2
        w_new = 1.0 ./ (predicted_var .+ 1.0)
        w_new = w_new ./ sum(w_new)

        W = Diagonal(w_new)
        β = (X' * W * X) \ (X' * W * chi2_filt)

        intercept = β[1]
        slope = β[2]
    end

    # Convert slope to heritability
    # slope = N * h² / M => h² = slope * M / N
    h2 = slope * M / n_samples
    h2 = clamp(h2, 0.0, 1.0)

    # Jackknife standard errors
    block_size = div(n_snps, n_blocks)
    h2_jk = zeros(n_blocks)
    intercept_jk = zeros(n_blocks)

    for b in 1:n_blocks
        # Leave-out block indices
        block_start = (b - 1) * block_size + 1
        block_end = min(b * block_size, n_snps)

        mask = trues(n_snps)
        mask[block_start:block_end] .= false

        X_jk = X[mask, :]
        y_jk = chi2_filt[mask]
        w_jk = w[mask]
        w_jk = w_jk ./ sum(w_jk)
        W_jk = Diagonal(w_jk)

        β_jk = (X_jk' * W_jk * X_jk) \ (X_jk' * W_jk * y_jk)

        intercept_jk[b] = β_jk[1]
        slope_jk = β_jk[2]
        h2_jk[b] = slope_jk * M / n_samples
    end

    # Jackknife SE: SE = sqrt((n-1)/n * Σ(θ_i - θ̄)²)
    h2_se = sqrt((n_blocks - 1) / n_blocks * sum((h2_jk .- h2).^2))
    intercept_se = sqrt((n_blocks - 1) / n_blocks * sum((intercept_jk .- intercept).^2))

    # Liability scale transformation for case-control
    h2_liability = nothing
    if prevalence !== nothing && sample_prevalence !== nothing
        h2_liability = observed_to_liability(h2, prevalence, sample_prevalence)
    end

    return LDScoreResult(
        h2,
        h2_se,
        intercept,
        intercept_se,
        lambda_gc,
        mean_chi2,
        n_snps,
        h2_liability
    )
end

"""
    genetic_correlation(z1::Vector{Float64}, z2::Vector{Float64},
                       ld_scores::Vector{Float64}, n1::Int, n2::Int;
                       kwargs...) -> GeneticCorrelationResult

Estimate genetic correlation between two traits using cross-trait LDSC.

# Arguments
- `z1`: Z-scores from GWAS of trait 1
- `z2`: Z-scores from GWAS of trait 2
- `ld_scores`: LD scores
- `n1`: Sample size for trait 1
- `n2`: Sample size for trait 2

# Keyword Arguments
- `n_overlap::Int=0`: Number of overlapping samples
- `n_blocks::Int=200`: Jackknife blocks

# Mathematical Model
For cross-trait LDSC:
E[z₁_j * z₂_j] = (√(n₁n₂) * ρ_g) / M * ℓ_j + (n_overlap * ρ) / √(n₁n₂)

where:
- ρ_g is the genetic covariance
- ρ is the phenotypic correlation due to sample overlap

Genetic correlation:
r_g = ρ_g / √(h²₁ * h²₂)

# Example
```julia
rg_result = genetic_correlation(z_bmi, z_t2d, ld_scores, n_bmi, n_t2d)
println("Genetic correlation: \$(rg_result.rg) (SE: \$(rg_result.rg_se))")
```

# References
- Bulik-Sullivan et al. (2015) Nat. Genet. (genetic correlation paper)
"""
function genetic_correlation(
    z1::Vector{Float64},
    z2::Vector{Float64},
    ld_scores::Vector{Float64},
    n1::Int,
    n2::Int;
    n_overlap::Int=0,
    n_blocks::Int=200
)
    # Filter valid SNPs
    valid = isfinite.(z1) .& isfinite.(z2) .& (ld_scores .> 0)
    z1_filt = z1[valid]
    z2_filt = z2[valid]
    ld_filt = ld_scores[valid]
    n_snps = length(z1_filt)
    M = n_snps

    # Cross-trait product
    z_cross = z1_filt .* z2_filt

    # Weights
    w = 1.0 ./ (1 .+ sqrt(n1 * n2) * ld_filt / M)
    w = w ./ sum(w)

    # Design matrix
    X = hcat(ones(n_snps), ld_filt)
    W = Diagonal(w)

    # Weighted regression for cross-trait
    β_cross = (X' * W * X) \ (X' * W * z_cross)
    intercept_cross = β_cross[1]
    slope_cross = β_cross[2]

    # Genetic covariance
    gcov = slope_cross * M / sqrt(n1 * n2)

    # Get individual h2 estimates
    chi2_1 = z1_filt.^2
    chi2_2 = z2_filt.^2

    β_1 = (X' * W * X) \ (X' * W * chi2_1)
    β_2 = (X' * W * X) \ (X' * W * chi2_2)

    h2_1 = β_1[2] * M / n1
    h2_2 = β_2[2] * M / n2

    h2_1 = max(h2_1, 0.01)
    h2_2 = max(h2_2, 0.01)

    # Genetic correlation
    rg = gcov / sqrt(h2_1 * h2_2)
    rg = clamp(rg, -1.0, 1.0)

    # Jackknife SE
    block_size = div(n_snps, n_blocks)
    rg_jk = zeros(n_blocks)

    for b in 1:n_blocks
        block_start = (b - 1) * block_size + 1
        block_end = min(b * block_size, n_snps)

        mask = trues(n_snps)
        mask[block_start:block_end] .= false

        X_jk = X[mask, :]
        w_jk = w[mask]
        w_jk = w_jk ./ sum(w_jk)
        W_jk = Diagonal(w_jk)

        # Cross-trait
        z_cross_jk = z_cross[mask]
        β_cross_jk = (X_jk' * W_jk * X_jk) \ (X_jk' * W_jk * z_cross_jk)
        gcov_jk = β_cross_jk[2] * M / sqrt(n1 * n2)

        # Individual traits
        β_1_jk = (X_jk' * W_jk * X_jk) \ (X_jk' * W_jk * chi2_1[mask])
        β_2_jk = (X_jk' * W_jk * X_jk) \ (X_jk' * W_jk * chi2_2[mask])

        h2_1_jk = max(β_1_jk[2] * M / n1, 0.01)
        h2_2_jk = max(β_2_jk[2] * M / n2, 0.01)

        rg_jk[b] = clamp(gcov_jk / sqrt(h2_1_jk * h2_2_jk), -1.0, 1.0)
    end

    rg_se = sqrt((n_blocks - 1) / n_blocks * sum((rg_jk .- rg).^2))
    rg_se = max(rg_se, 1e-6)

    z_stat = rg / rg_se
    pvalue = 2 * ccdf(Normal(), abs(z_stat))

    return GeneticCorrelationResult(
        rg,
        rg_se,
        z_stat,
        pvalue,
        h2_1,
        h2_2,
        gcov,
        intercept_cross
    )
end

"""
    partitioned_ldsc(chi2::Vector{Float64}, annotations::Matrix{Float64},
                    ld_scores::Matrix{Float64}, n_samples::Int;
                    kwargs...) -> PartitionedHeritability

Partitioned LD score regression for heritability enrichment.

# Arguments
- `chi2`: Chi-squared statistics from GWAS
- `annotations`: Binary annotation matrix (n_snps × n_categories)
- `ld_scores`: Category-specific LD scores (n_snps × n_categories)
- `n_samples`: GWAS sample size

# Keyword Arguments
- `category_names`: Names for each annotation category
- `baseline::Bool=true`: Include baseline LD scores
- `n_blocks::Int=200`: Jackknife blocks

# Mathematical Model
Extended LDSC with multiple annotations:
E[χ²_j] = Σ_c τ_c * ℓ_j,c + a

where:
- ℓ_j,c = Σ_k a_kc * r²_jk is the category-specific LD score
- τ_c is the per-SNP heritability for category c
- a_kc is 1 if SNP k is in category c, 0 otherwise

Enrichment = (prop_h2_c / prop_snps_c)

# Example
```julia
# Annotations: coding, enhancer, promoter, etc.
annotations = load_annotations("annotations.bed", snp_positions)
partition_results = partitioned_ldsc(chi2, annotations, cat_ld_scores, n)

# Plot enrichment
for (cat, enrich) in zip(partition_results.categories, partition_results.enrichment)
    println("\$cat: \$(round(enrich, digits=2))x enrichment")
end
```

# References
- Finucane et al. (2015) Nat. Genet.
"""
function partitioned_ldsc(
    chi2::Vector{Float64},
    annotations::Matrix{Float64},
    ld_scores_cat::Matrix{Float64},
    n_samples::Int;
    category_names::Union{Vector{String}, Nothing}=nothing,
    baseline_ld::Union{Vector{Float64}, Nothing}=nothing,
    n_blocks::Int=200
)
    n_snps, n_categories = size(annotations)
    M = n_snps

    # Filter valid SNPs
    valid = isfinite.(chi2) .& (chi2 .> 0) .& vec(all(isfinite.(ld_scores_cat), dims=2))
    chi2_filt = chi2[valid]
    annotations_filt = annotations[valid, :]
    ld_filt = ld_scores_cat[valid, :]
    n_valid = sum(valid)

    # Category names
    if category_names === nothing
        category_names = ["Category_$i" for i in 1:n_categories]
    end

    # Design matrix: [intercept, baseline_ld (optional), category ld_scores]
    if baseline_ld !== nothing
        X = hcat(ones(n_valid), baseline_ld[valid], ld_filt)
        offset = 2  # First category coefficient is at index 3
    else
        X = hcat(ones(n_valid), ld_filt)
        offset = 1
    end

    # Weights
    total_ld = vec(sum(ld_filt, dims=2))
    w = 1.0 ./ (1 .+ n_samples * total_ld / M)
    w = w ./ sum(w)
    W = Diagonal(w)

    # Weighted regression
    β = (X' * W * X) \ (X' * W * chi2_filt)

    # Extract category coefficients (τ values)
    tau = β[offset+1:end]

    # Proportion of SNPs in each category
    prop_snps = vec(sum(annotations_filt, dims=1)) / n_valid

    # Per-category heritability contribution
    # h2_c = τ_c * M_c where M_c is number of SNPs in category
    # Since we use category-specific LD scores, need different formula
    # τ represents per-SNP h2 contribution

    # Total heritability from partitioned model
    h2_total = sum(tau .* prop_snps) * M / n_samples

    # Category-specific heritability
    h2_cat = tau .* (M / n_samples) .* prop_snps
    h2_cat = max.(h2_cat, 0.0)

    # Proportion of h2
    total_h2 = sum(h2_cat)
    if total_h2 > 0
        prop_h2 = h2_cat ./ total_h2
    else
        prop_h2 = fill(1.0 / n_categories, n_categories)
    end

    # Enrichment
    enrichment = prop_h2 ./ (prop_snps .+ 1e-10)

    # Jackknife for standard errors
    block_size = max(1, div(n_valid, n_blocks))
    tau_jk = zeros(n_blocks, n_categories)

    for b in 1:n_blocks
        block_start = (b - 1) * block_size + 1
        block_end = min(b * block_size, n_valid)

        mask = trues(n_valid)
        mask[block_start:block_end] .= false

        X_jk = X[mask, :]
        y_jk = chi2_filt[mask]
        w_jk = w[mask]
        w_jk = w_jk ./ sum(w_jk)
        W_jk = Diagonal(w_jk)

        β_jk = (X_jk' * W_jk * X_jk) \ (X_jk' * W_jk * y_jk)
        tau_jk[b, :] = β_jk[offset+1:end]
    end

    tau_se = vec(sqrt.((n_blocks - 1) / n_blocks .* sum((tau_jk .- tau').^2, dims=1)))

    # Enrichment SE and p-value
    # Using delta method: SE(enrich) ≈ SE(tau) * |∂enrich/∂tau|
    enrichment_se = tau_se ./ (tau .+ 1e-10) .* enrichment
    enrichment_z = (enrichment .- 1.0) ./ (enrichment_se .+ 1e-10)
    enrichment_p = 2 .* ccdf.(Normal(), abs.(enrichment_z))

    return PartitionedHeritability(
        category_names,
        prop_snps,
        prop_h2,
        enrichment,
        enrichment_se,
        enrichment_p,
        tau,
        tau_se
    )
end

"""
    observed_to_liability(h2_obs::Float64, K::Float64, P::Float64) -> Float64

Convert observed scale heritability to liability scale.

# Arguments
- `h2_obs`: Observed scale heritability
- `K`: Population prevalence
- `P`: Sample prevalence (proportion of cases)

# Formula
h²_liability = h²_obs * (K * (1-K))² / (P * (1-P) * z²)

where z = φ(Φ⁻¹(K)) is the height of the normal density at the threshold.

# References
- Lee et al. (2011) Am. J. Hum. Genet.
"""
function observed_to_liability(h2_obs::Float64, K::Float64, P::Float64)
    # Threshold on liability scale
    t = quantile(Normal(), 1 - K)

    # Height of normal density at threshold
    z = pdf(Normal(), t)

    # Conversion factor
    conversion = K^2 * (1 - K)^2 / (P * (1 - P) * z^2)

    return h2_obs * conversion
end

"""
    compute_cell_type_enrichment(chi2::Vector{Float64},
                                cell_type_annotations::Matrix{Float64},
                                ld_scores::Vector{Float64},
                                n_samples::Int;
                                cell_type_names=nothing) -> DataFrame

Compute cell-type-specific heritability enrichment.

# Arguments
- `chi2`: GWAS chi-squared statistics
- `cell_type_annotations`: Binary matrix (SNPs × cell types)
- `ld_scores`: Baseline LD scores
- `n_samples`: Sample size

# Returns
DataFrame with enrichment results for each cell type

# Example
```julia
# Load cell type-specific annotations (e.g., from LDSC baseline model)
cell_annotations = load_cell_type_annotations("cell_types.annot")
enrichment = compute_cell_type_enrichment(chi2, cell_annotations, ld_scores, n)

# Find significant enrichments
significant = filter(r -> r.pvalue < 0.05/nrow(enrichment), enrichment)
```

# References
- Finucane et al. (2018) Nat. Genet. (cell-type specific analysis)
"""
function compute_cell_type_enrichment(
    chi2::Vector{Float64},
    cell_type_annotations::Matrix{Float64},
    ld_scores::Vector{Float64},
    n_samples::Int;
    cell_type_names::Union{Vector{String}, Nothing}=nothing,
    n_blocks::Int=200
)
    n_snps, n_cell_types = size(cell_type_annotations)

    if cell_type_names === nothing
        cell_type_names = ["CellType_$i" for i in 1:n_cell_types]
    end

    results = DataFrame(
        cell_type = String[],
        prop_snps = Float64[],
        coefficient = Float64[],
        coefficient_se = Float64[],
        enrichment = Float64[],
        enrichment_se = Float64[],
        pvalue = Float64[]
    )

    # Baseline-only regression for comparison
    valid = isfinite.(chi2) .& (chi2 .> 0) .& (ld_scores .> 0)

    for ct in 1:n_cell_types
        # Run partitioned LDSC with this cell type vs baseline
        ct_ld = compute_annotation_ld_scores(cell_type_annotations[valid, ct:ct],
                                             ld_scores[valid])

        # Simple single-annotation enrichment
        ct_annot = cell_type_annotations[valid, ct]
        prop_snps = mean(ct_annot)

        if prop_snps < 0.001 || prop_snps > 0.999
            continue  # Skip very rare or very common annotations
        end

        # Regression with baseline and cell type annotation
        X = hcat(ones(sum(valid)), ld_scores[valid], ct_ld[:, 1])
        y = chi2[valid]

        w = 1.0 ./ (1 .+ n_samples * ld_scores[valid] / n_snps)
        w = w ./ sum(w)
        W = Diagonal(w)

        β = (X' * W * X) \ (X' * W * y)
        ct_coef = β[3]

        # Jackknife SE
        n_valid = sum(valid)
        block_size = max(1, div(n_valid, n_blocks))
        coef_jk = zeros(n_blocks)

        for b in 1:n_blocks
            block_start = (b - 1) * block_size + 1
            block_end = min(b * block_size, n_valid)

            mask = trues(n_valid)
            mask[block_start:block_end] .= false

            X_jk = X[mask, :]
            y_jk = y[mask]
            w_jk = w[mask]
            w_jk = w_jk ./ sum(w_jk)
            W_jk = Diagonal(w_jk)

            β_jk = (X_jk' * W_jk * X_jk) \ (X_jk' * W_jk * y_jk)
            coef_jk[b] = β_jk[3]
        end

        coef_se = sqrt((n_blocks - 1) / n_blocks * sum((coef_jk .- ct_coef).^2))

        # Enrichment calculation
        # (Per-SNP h2 in annotation) / (Mean per-SNP h2)
        total_coef = β[2] + ct_coef * prop_snps
        enrichment = ct_coef / (total_coef / prop_snps + 1e-10)
        enrichment_se = coef_se / (total_coef / prop_snps + 1e-10)

        z_score = ct_coef / (coef_se + 1e-10)
        pvalue = 2 * ccdf(Normal(), abs(z_score))

        push!(results, (
            cell_type = cell_type_names[ct],
            prop_snps = prop_snps,
            coefficient = ct_coef,
            coefficient_se = coef_se,
            enrichment = enrichment,
            enrichment_se = enrichment_se,
            pvalue = pvalue
        ))
    end

    sort!(results, :pvalue)

    return results
end

"""
    compute_annotation_ld_scores(annotations::Matrix{Float64},
                                base_ld::Vector{Float64}) -> Matrix{Float64}

Compute annotation-weighted LD scores.
"""
function compute_annotation_ld_scores(annotations::Matrix{Float64},
                                     base_ld::Vector{Float64})
    n_snps, n_annot = size(annotations)

    # Simple approximation: annotation LD score ∝ annotation × base LD score
    # Full computation would require re-calculating r² for annotated SNPs only

    annot_ld = similar(annotations)
    for a in 1:n_annot
        annot_ld[:, a] = annotations[:, a] .* base_ld
    end

    return annot_ld
end

"""
    stratified_ldsc(chi2::Vector{Float64}, ld_scores::Vector{Float64},
                   maf::Vector{Float64}, n_samples::Int;
                   maf_bins::Int=10) -> DataFrame

Stratified LDSC by MAF bins to assess MAF-dependent architecture.

# Arguments
- `chi2`: Chi-squared statistics
- `ld_scores`: LD scores
- `maf`: Minor allele frequencies
- `n_samples`: Sample size
- `maf_bins`: Number of MAF bins

# Returns
DataFrame with h2 estimates for each MAF bin

# References
- Gazal et al. (2019) Nat. Genet.
"""
function stratified_ldsc(
    chi2::Vector{Float64},
    ld_scores::Vector{Float64},
    maf::Vector{Float64},
    n_samples::Int;
    maf_bins::Int=10,
    n_blocks::Int=200
)
    # Create MAF bins
    maf_breaks = quantile(maf[maf .> 0], range(0, 1, length=maf_bins+1))

    results = DataFrame(
        maf_lower = Float64[],
        maf_upper = Float64[],
        n_snps = Int[],
        h2 = Float64[],
        h2_se = Float64[],
        prop_h2 = Float64[]
    )

    total_h2 = 0.0
    bin_h2 = Float64[]

    for b in 1:maf_bins
        lower = maf_breaks[b]
        upper = maf_breaks[b+1]

        in_bin = (maf .>= lower) .& (maf .< upper) .& isfinite.(chi2)

        if sum(in_bin) < 100
            continue
        end

        result = ldsc_regression(
            chi2[in_bin],
            ld_scores[in_bin],
            n_samples;
            n_blocks=min(n_blocks, div(sum(in_bin), 10))
        )

        push!(results, (
            maf_lower = lower,
            maf_upper = upper,
            n_snps = sum(in_bin),
            h2 = result.h2,
            h2_se = result.h2_se,
            prop_h2 = 0.0  # Fill in after
        ))

        push!(bin_h2, max(0, result.h2))
    end

    # Compute proportion of h2
    total_h2 = sum(bin_h2)
    if total_h2 > 0
        results.prop_h2 = bin_h2 ./ total_h2
    end

    return results
end
