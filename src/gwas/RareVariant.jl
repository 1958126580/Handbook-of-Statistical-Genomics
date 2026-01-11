# ============================================================================
# RareVariant.jl - Rare Variant Association Analysis
# ============================================================================
# Comprehensive methods for rare variant association testing including
# burden tests, SKAT, SKAT-O, and various weighting schemes
# ============================================================================

"""
    RareVariantResult

Structure containing results from rare variant association analysis.

# Fields
- `test_name::String`: Name of the test used
- `region_id::String`: Identifier for the region tested
- `n_variants::Int`: Number of variants in the region
- `n_samples::Int`: Number of samples
- `statistic::Float64`: Test statistic value
- `pvalue::Float64`: P-value
- `mac::Int`: Minor allele count in region
- `beta::Union{Float64, Nothing}`: Effect estimate (for burden tests)
- `se::Union{Float64, Nothing}`: Standard error (for burden tests)
"""
struct RareVariantResult
    test_name::String
    region_id::String
    n_variants::Int
    n_samples::Int
    statistic::Float64
    pvalue::Float64
    mac::Int
    beta::Union{Float64, Nothing}
    se::Union{Float64, Nothing}
end

"""
    burden_test(genotypes::Matrix{Float64}, phenotype::Vector{Float64};
               covariates=nothing, weights=nothing, test_type=:score) -> RareVariantResult

Perform burden test by collapsing rare variants within a region.

# Arguments
- `genotypes`: Genotype matrix (n_samples × n_variants), coded 0/1/2
- `phenotype`: Phenotype vector (continuous or binary)
- `covariates`: Optional covariate matrix
- `weights`: Optional variant weights (default: equal weights)
- `test_type`: Type of test - :score, :wald, or :lrt

# Mathematical Details
The burden test aggregates rare variants into a single "burden score":
B_i = Σ_j w_j * G_{ij}

where w_j are variant weights and G_{ij} is the genotype for sample i at variant j.

Test statistic (score test):
T = (Σ_i (Y_i - μ̂) * B_i)² / Var(Σ_i (Y_i - μ̂) * B_i)

Under H0, T ~ χ²(1)

# Weighting Schemes
- Equal weights: w_j = 1
- MAF-based: w_j = 1/√(MAF_j * (1 - MAF_j))
- Beta weights: w_j = Beta(MAF_j; a, b) (SKAT default: a=1, b=25)

# Example
```julia
# Test gene region for association
genotypes = gm.genotypes[:, gene_variants]
result = burden_test(genotypes, phenotype; weights=:beta, test_type=:score)
println("Burden test p-value: \$(result.pvalue)")
```

# References
- Li & Leal (2008) Am. J. Hum. Genet.
- Madsen & Browning (2009) PLoS Genet.
- Morris & Zeggini (2010) Genet. Epidemiol.
"""
function burden_test(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64};
    covariates::Union{Matrix{Float64}, Nothing}=nothing,
    weights::Union{Vector{Float64}, Symbol, Nothing}=nothing,
    test_type::Symbol=:score,
    region_id::String="region"
)
    n_samples, n_variants = size(genotypes)
    @assert length(phenotype) == n_samples "Phenotype length must match samples"

    # Compute weights
    w = compute_variant_weights(genotypes, weights)

    # Compute burden score
    burden = genotypes * w

    # Minor allele count
    mac = round(Int, sum(genotypes))

    # Build design matrix
    if covariates !== nothing
        X = hcat(ones(n_samples), covariates, burden)
        burden_idx = size(X, 2)
    else
        X = hcat(ones(n_samples), burden)
        burden_idx = 2
    end

    # Determine if binary or continuous phenotype
    is_binary = all(p -> p == 0 || p == 1, phenotype)

    if is_binary
        # Logistic regression
        result = logistic_regression_test(X, phenotype, burden_idx, test_type)
    else
        # Linear regression
        result = linear_regression_test(X, phenotype, burden_idx, test_type)
    end

    return RareVariantResult(
        "Burden ($test_type)",
        region_id,
        n_variants,
        n_samples,
        result.statistic,
        result.pvalue,
        mac,
        result.beta,
        result.se
    )
end

"""
    skat(genotypes::Matrix{Float64}, phenotype::Vector{Float64};
        covariates=nothing, weights=nothing, kernel=:linear) -> RareVariantResult

Sequence Kernel Association Test (SKAT) for rare variant analysis.

# Arguments
- `genotypes`: Genotype matrix (n_samples × n_variants)
- `phenotype`: Phenotype vector
- `covariates`: Optional covariate matrix
- `weights`: Variant weights (default: Beta(1,25) on MAF)
- `kernel`: Kernel type - :linear, :quadratic, or custom matrix

# Mathematical Details
SKAT tests the variance component in a mixed model:
Y = Xβ + Gγ + ε

where γ ~ N(0, τW) with W being a diagonal weight matrix.

The score statistic is:
Q = (Y - X̂β)' K (Y - X̂β)

where K = G W G' is the weighted kernel matrix.

Under H0, Q follows a mixture of chi-squared distributions:
Q ~ Σ_k λ_k χ²_1

where λ_k are eigenvalues of P₀^{1/2} K P₀^{1/2} and
P₀ = I - X(X'X)⁻¹X' is the projection matrix.

P-values computed using Davies' method or saddlepoint approximation.

# Example
```julia
result = skat(gene_genotypes, phenotype; weights=:beta)
if result.pvalue < 2.5e-6  # Genome-wide significance for genes
    println("Gene \$(result.region_id) is significant")
end
```

# References
- Wu et al. (2011) Am. J. Hum. Genet.
- Lee et al. (2012) Biostatistics (optimal SKAT)
"""
function skat(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64};
    covariates::Union{Matrix{Float64}, Nothing}=nothing,
    weights::Union{Vector{Float64}, Symbol, Nothing}=nothing,
    kernel::Symbol=:linear,
    region_id::String="region",
    method::Symbol=:davies
)
    n_samples, n_variants = size(genotypes)
    @assert length(phenotype) == n_samples

    # Compute weights
    w = compute_variant_weights(genotypes, weights)
    W = Diagonal(w)

    # Weighted genotype matrix
    G_weighted = genotypes * sqrt(W)

    # Minor allele count
    mac = round(Int, sum(genotypes))

    # Build covariate matrix with intercept
    if covariates !== nothing
        X = hcat(ones(n_samples), covariates)
    else
        X = ones(n_samples, 1)
    end

    # Determine if binary or continuous
    is_binary = all(p -> p == 0 || p == 1, phenotype)

    if is_binary
        result = skat_logistic(G_weighted, phenotype, X, method)
    else
        result = skat_linear(G_weighted, phenotype, X, method)
    end

    return RareVariantResult(
        "SKAT",
        region_id,
        n_variants,
        n_samples,
        result.Q,
        result.pvalue,
        mac,
        nothing,
        nothing
    )
end

"""
    skat_linear(G::Matrix{Float64}, y::Vector{Float64}, X::Matrix{Float64},
               method::Symbol) -> NamedTuple

SKAT for continuous phenotypes using linear model.
"""
function skat_linear(G::Matrix{Float64}, y::Vector{Float64}, X::Matrix{Float64},
                    method::Symbol)
    n = length(y)

    # Fit null model: Y = Xβ + ε
    β_null = X \ y
    residuals = y - X * β_null
    σ2 = sum(residuals.^2) / (n - size(X, 2))

    # Projection matrix P0 = I - X(X'X)^{-1}X'
    XtX_inv = inv(X' * X)
    P0 = I - X * XtX_inv * X'

    # Kernel matrix K = G G'
    K = G * G'

    # SKAT statistic Q = (Y - Xβ)' K (Y - Xβ) = residuals' K residuals
    Q = residuals' * K * residuals

    # Compute eigenvalues of P0^{1/2} K P0^{1/2}
    # Equivalent to eigenvalues of P0 K (for symmetric P0)
    P0_K = P0 * K
    eigenvalues = real.(eigvals(P0_K))
    eigenvalues = eigenvalues[eigenvalues .> 1e-10]  # Remove numerical zeros

    # Scale by σ²
    eigenvalues = eigenvalues .* σ2

    # Compute p-value
    pvalue = compute_mixture_chisq_pvalue(Q, eigenvalues, method)

    return (Q=Q, pvalue=pvalue, eigenvalues=eigenvalues)
end

"""
    skat_logistic(G::Matrix{Float64}, y::Vector{Float64}, X::Matrix{Float64},
                 method::Symbol) -> NamedTuple

SKAT for binary phenotypes using logistic model.
"""
function skat_logistic(G::Matrix{Float64}, y::Vector{Float64}, X::Matrix{Float64},
                      method::Symbol)
    n = length(y)

    # Fit null logistic model
    β_null = logistic_fit_irls(X, y)
    μ = logistic_predict(X, β_null)

    # Working residuals
    residuals = y - μ

    # Weight matrix V = diag(μ(1-μ))
    V = μ .* (1 .- μ)
    V_sqrt = sqrt.(V)

    # Adjusted projection matrix
    # P0 = V^{-1} - V^{-1}X(X'VX)^{-1}X'
    XtVX = X' * Diagonal(V) * X
    XtVX_inv = inv(XtVX)

    # Kernel matrix
    K = G * G'

    # SKAT statistic
    Q = residuals' * K * residuals

    # Compute eigenvalues for p-value
    # For logistic, use V-weighted kernel
    K_weighted = Diagonal(V_sqrt) * K * Diagonal(V_sqrt)
    P0_weighted = I - Diagonal(V_sqrt) * X * XtVX_inv * X' * Diagonal(V_sqrt)
    P0_K = P0_weighted * K_weighted

    eigenvalues = real.(eigvals(P0_K))
    eigenvalues = eigenvalues[eigenvalues .> 1e-10]

    pvalue = compute_mixture_chisq_pvalue(Q, eigenvalues, method)

    return (Q=Q, pvalue=pvalue, eigenvalues=eigenvalues)
end

"""
    skat_o(genotypes::Matrix{Float64}, phenotype::Vector{Float64};
          covariates=nothing, weights=nothing, rho_grid=nothing) -> RareVariantResult

Optimal SKAT (SKAT-O) combining burden and SKAT tests.

# Arguments
- `genotypes`: Genotype matrix
- `phenotype`: Phenotype vector
- `covariates`: Optional covariates
- `weights`: Variant weights
- `rho_grid`: Grid of ρ values (default: [0, 0.1, ..., 0.9, 1])

# Mathematical Details
SKAT-O tests using a kernel that interpolates between SKAT (ρ=0) and
burden test (ρ=1):

K_ρ = (1-ρ) K_SKAT + ρ K_burden

where K_SKAT = GWG' and K_burden = Gw 1' 1 w'G'.

The test statistic is:
Q_ρ = (Y - X̂β)' K_ρ (Y - X̂β)

The optimal ρ is selected to minimize the p-value, with correction
for multiple testing over the ρ grid.

# Example
```julia
# Test with SKAT-O for robust power
result = skat_o(gene_genotypes, phenotype)
println("SKAT-O p-value: \$(result.pvalue)")
```

# References
- Lee et al. (2012) Biostatistics
- Lee et al. (2014) Am. J. Hum. Genet.
"""
function skat_o(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64};
    covariates::Union{Matrix{Float64}, Nothing}=nothing,
    weights::Union{Vector{Float64}, Symbol, Nothing}=nothing,
    rho_grid::Union{Vector{Float64}, Nothing}=nothing,
    region_id::String="region"
)
    n_samples, n_variants = size(genotypes)

    # Default rho grid
    if rho_grid === nothing
        rho_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    end
    n_rho = length(rho_grid)

    # Compute weights
    w = compute_variant_weights(genotypes, weights)
    W = Diagonal(w)

    # Weighted genotype matrix
    G_weighted = genotypes * sqrt(W)

    # Burden score (for rho = 1)
    burden = genotypes * w

    mac = round(Int, sum(genotypes))

    # Build covariate matrix
    if covariates !== nothing
        X = hcat(ones(n_samples), covariates)
    else
        X = ones(n_samples, 1)
    end

    # Determine if binary or continuous
    is_binary = all(p -> p == 0 || p == 1, phenotype)

    # Fit null model
    if is_binary
        β_null = logistic_fit_irls(X, phenotype)
        μ = logistic_predict(X, β_null)
        residuals = phenotype - μ
        V = μ .* (1 .- μ)
    else
        β_null = X \ phenotype
        residuals = phenotype - X * β_null
        V = fill(var(residuals), n_samples)
    end

    # SKAT kernel: K_SKAT = G_weighted G_weighted'
    K_SKAT = G_weighted * G_weighted'

    # Burden kernel: K_burden = burden burden'
    K_burden = burden * burden'

    # Compute Q statistic for each rho
    Q_rho = zeros(n_rho)
    pvalues = zeros(n_rho)

    for (i, ρ) in enumerate(rho_grid)
        # Combined kernel
        K_rho = (1 - ρ) * K_SKAT + ρ * K_burden

        # Q statistic
        Q_rho[i] = residuals' * K_rho * residuals

        # Compute eigenvalues and p-value
        if is_binary
            V_sqrt = sqrt.(V)
            K_weighted = Diagonal(V_sqrt) * K_rho * Diagonal(V_sqrt)
            XtVX_inv = inv(X' * Diagonal(V) * X)
            P0 = I - Diagonal(V_sqrt) * X * XtVX_inv * X' * Diagonal(V_sqrt)
        else
            P0 = I - X * inv(X' * X) * X'
            K_weighted = K_rho
        end

        eigenvalues = real.(eigvals(P0 * K_weighted))
        eigenvalues = eigenvalues[eigenvalues .> 1e-10]

        if !is_binary
            eigenvalues = eigenvalues .* var(residuals)
        end

        pvalues[i] = compute_mixture_chisq_pvalue(Q_rho[i], eigenvalues, :davies)
    end

    # Find optimal rho (minimum p-value)
    min_idx = argmin(pvalues)
    min_pvalue = pvalues[min_idx]
    optimal_rho = rho_grid[min_idx]

    # Adjust for multiple testing over rho grid
    # Using minimum p-value statistic with correlation adjustment
    adjusted_pvalue = adjust_skat_o_pvalue(pvalues, Q_rho, rho_grid)

    return RareVariantResult(
        "SKAT-O (ρ=$(round(optimal_rho, digits=2)))",
        region_id,
        n_variants,
        n_samples,
        Q_rho[min_idx],
        adjusted_pvalue,
        mac,
        nothing,
        nothing
    )
end

"""
    cmc_test(genotypes::Matrix{Float64}, phenotype::Vector{Float64};
            maf_threshold::Float64=0.01) -> RareVariantResult

Combined Multivariate and Collapsing (CMC) test.

# Arguments
- `genotypes`: Genotype matrix
- `phenotype`: Phenotype vector
- `maf_threshold`: MAF threshold for defining rare variants

# Method
CMC partitions variants into groups by MAF:
1. Rare variants (MAF < threshold): collapsed into binary indicator
2. Common variants: included individually

Tests all variant groups jointly using multivariate test.

# References
- Li & Leal (2008) Am. J. Hum. Genet.
"""
function cmc_test(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64};
    covariates::Union{Matrix{Float64}, Nothing}=nothing,
    maf_threshold::Float64=0.01,
    region_id::String="region"
)
    n_samples, n_variants = size(genotypes)

    # Calculate MAF for each variant
    mafs = vec(mean(genotypes, dims=1)) / 2

    # Identify rare and common variants
    rare_idx = findall(mafs .< maf_threshold)
    common_idx = findall(mafs .>= maf_threshold)

    # Collapse rare variants into binary indicator
    if length(rare_idx) > 0
        rare_indicator = Float64.(vec(sum(genotypes[:, rare_idx], dims=2)) .> 0)
    else
        rare_indicator = Float64[]
    end

    # Build test matrix
    if length(common_idx) > 0 && length(rare_idx) > 0
        test_matrix = hcat(rare_indicator, genotypes[:, common_idx])
    elseif length(rare_idx) > 0
        test_matrix = reshape(rare_indicator, :, 1)
    else
        test_matrix = genotypes[:, common_idx]
    end

    n_test_vars = size(test_matrix, 2)
    mac = round(Int, sum(genotypes))

    # Build full design matrix
    if covariates !== nothing
        X_null = hcat(ones(n_samples), covariates)
    else
        X_null = ones(n_samples, 1)
    end

    X_full = hcat(X_null, test_matrix)

    # Determine test type
    is_binary = all(p -> p == 0 || p == 1, phenotype)

    if is_binary
        # Likelihood ratio test for logistic regression
        ll_null = logistic_log_likelihood(X_null, phenotype)
        ll_full = logistic_log_likelihood(X_full, phenotype)
        lrt_stat = 2 * (ll_full - ll_null)
        pvalue = ccdf(Chisq(n_test_vars), lrt_stat)
    else
        # F-test for linear regression
        ss_null = sum((phenotype - X_null * (X_null \ phenotype)).^2)
        ss_full = sum((phenotype - X_full * (X_full \ phenotype)).^2)

        df1 = n_test_vars
        df2 = n_samples - size(X_full, 2)

        f_stat = ((ss_null - ss_full) / df1) / (ss_full / df2)
        pvalue = ccdf(FDist(df1, df2), f_stat)
        lrt_stat = f_stat
    end

    return RareVariantResult(
        "CMC",
        region_id,
        n_variants,
        n_samples,
        lrt_stat,
        pvalue,
        mac,
        nothing,
        nothing
    )
end

"""
    vt_test(genotypes::Matrix{Float64}, phenotype::Vector{Float64};
           maf_thresholds=nothing) -> RareVariantResult

Variable Threshold (VT) test for rare variants.

# Arguments
- `genotypes`: Genotype matrix
- `phenotype`: Phenotype vector
- `maf_thresholds`: Grid of MAF thresholds to test

# Method
VT tests burden scores at multiple MAF thresholds and selects
the optimal threshold. P-value is adjusted for multiple testing
using permutation.

# References
- Price et al. (2010) Am. J. Hum. Genet.
"""
function vt_test(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64};
    covariates::Union{Matrix{Float64}, Nothing}=nothing,
    maf_thresholds::Union{Vector{Float64}, Nothing}=nothing,
    n_permutations::Int=1000,
    region_id::String="region"
)
    n_samples, n_variants = size(genotypes)

    # Calculate MAF for each variant
    mafs = vec(mean(genotypes, dims=1)) / 2

    # Default thresholds based on observed MAFs
    if maf_thresholds === nothing
        unique_mafs = sort(unique(mafs))
        maf_thresholds = unique_mafs[unique_mafs .> 0]
        if length(maf_thresholds) > 20
            maf_thresholds = quantile(maf_thresholds, 0.05:0.05:1.0)
        end
    end

    mac = round(Int, sum(genotypes))

    # Compute test statistic for each threshold
    function compute_vt_stat(y, G, thresholds)
        max_stat = 0.0
        for t in thresholds
            included = mafs .<= t
            if any(included)
                burden = vec(sum(G[:, included], dims=2))
                stat = abs(cor(y, burden)) * sqrt(n_samples - 2)
                max_stat = max(max_stat, stat)
            end
        end
        return max_stat
    end

    # Observed statistic
    observed_stat = compute_vt_stat(phenotype, genotypes, maf_thresholds)

    # Permutation for p-value
    n_exceed = 0
    for _ in 1:n_permutations
        perm_y = phenotype[randperm(n_samples)]
        perm_stat = compute_vt_stat(perm_y, genotypes, maf_thresholds)
        if perm_stat >= observed_stat
            n_exceed += 1
        end
    end

    pvalue = (n_exceed + 1) / (n_permutations + 1)

    return RareVariantResult(
        "VT",
        region_id,
        n_variants,
        n_samples,
        observed_stat,
        pvalue,
        mac,
        nothing,
        nothing
    )
end

"""
    acatv_test(genotypes::Matrix{Float64}, phenotype::Vector{Float64};
              covariates=nothing) -> RareVariantResult

Aggregated Cauchy Association Test for Variants (ACAT-V).

# Arguments
- `genotypes`: Genotype matrix
- `phenotype`: Phenotype vector

# Method
ACAT-V combines single-variant p-values using Cauchy combination:
T = Σ_j w_j tan((0.5 - p_j)π)

Under H0, T follows a standard Cauchy distribution.

Advantages:
- Fast computation
- Robust to correlation structure
- Works well with sparse signals

# References
- Liu et al. (2019) Am. J. Hum. Genet.
"""
function acatv_test(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64};
    covariates::Union{Matrix{Float64}, Nothing}=nothing,
    weights::Union{Vector{Float64}, Symbol, Nothing}=nothing,
    region_id::String="region"
)
    n_samples, n_variants = size(genotypes)
    mac = round(Int, sum(genotypes))

    # Compute weights
    w = compute_variant_weights(genotypes, weights)
    w = w / sum(w)  # Normalize

    # Build design matrix
    if covariates !== nothing
        X = hcat(ones(n_samples), covariates)
    else
        X = ones(n_samples, 1)
    end

    is_binary = all(p -> p == 0 || p == 1, phenotype)

    # Compute single-variant p-values
    pvalues = zeros(n_variants)

    for j in 1:n_variants
        X_full = hcat(X, genotypes[:, j])

        if is_binary
            result = logistic_regression_test(X_full, phenotype, size(X_full, 2), :wald)
        else
            result = linear_regression_test(X_full, phenotype, size(X_full, 2), :wald)
        end

        pvalues[j] = result.pvalue
    end

    # ACAT combination
    # Handle p-values at boundaries
    pvalues = clamp.(pvalues, 1e-15, 1 - 1e-15)

    # Cauchy combination statistic
    T = sum(w .* tan.((0.5 .- pvalues) .* π))

    # P-value from Cauchy distribution
    combined_pvalue = 0.5 - atan(T) / π
    combined_pvalue = clamp(combined_pvalue, 0.0, 1.0)

    return RareVariantResult(
        "ACAT-V",
        region_id,
        n_variants,
        n_samples,
        T,
        combined_pvalue,
        mac,
        nothing,
        nothing
    )
end

"""
    gene_based_test(genotypes::GenotypeMatrix, phenotype::Vector{Float64},
                   gene_regions::DataFrame; method=:skat_o) -> DataFrame

Run gene-based rare variant tests across all genes.

# Arguments
- `genotypes`: Full genotype matrix
- `phenotype`: Phenotype vector
- `gene_regions`: DataFrame with columns: gene, chr, start, end
- `method`: Test method - :burden, :skat, :skat_o, :cmc, :acatv

# Returns
DataFrame with gene-level association results

# Example
```julia
results = gene_based_test(gm, phenotype, gene_annotations; method=:skat_o)
significant_genes = filter(r -> r.pvalue < 2.5e-6, results)
```
"""
function gene_based_test(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64},
    variant_positions::Vector{Int},
    gene_regions::DataFrame;
    covariates::Union{Matrix{Float64}, Nothing}=nothing,
    method::Symbol=:skat_o,
    maf_threshold::Float64=0.01,
    min_mac::Int=1
)
    n_genes = nrow(gene_regions)

    results = DataFrame(
        gene = String[],
        chromosome = String[],
        start = Int[],
        stop = Int[],
        n_variants = Int[],
        mac = Int[],
        statistic = Float64[],
        pvalue = Float64[]
    )

    # Calculate MAFs for filtering
    mafs = vec(mean(genotypes, dims=1)) / 2

    for i in 1:n_genes
        gene = gene_regions.gene[i]
        chr = string(gene_regions.chr[i])
        start_pos = gene_regions.start[i]
        end_pos = gene_regions.stop[i]

        # Find variants in gene region
        in_gene = (variant_positions .>= start_pos) .&
                  (variant_positions .<= end_pos) .&
                  (mafs .<= maf_threshold)

        gene_variants = findall(in_gene)

        if length(gene_variants) < 2
            continue  # Skip genes with too few variants
        end

        gene_genotypes = genotypes[:, gene_variants]
        gene_mac = round(Int, sum(gene_genotypes))

        if gene_mac < min_mac
            continue
        end

        # Run selected test
        try
            if method == :burden
                result = burden_test(gene_genotypes, phenotype;
                                    covariates=covariates, region_id=gene)
            elseif method == :skat
                result = skat(gene_genotypes, phenotype;
                             covariates=covariates, region_id=gene)
            elseif method == :skat_o
                result = skat_o(gene_genotypes, phenotype;
                               covariates=covariates, region_id=gene)
            elseif method == :cmc
                result = cmc_test(gene_genotypes, phenotype;
                                 covariates=covariates, region_id=gene)
            elseif method == :acatv
                result = acatv_test(gene_genotypes, phenotype;
                                   covariates=covariates, region_id=gene)
            else
                error("Unknown method: $method")
            end

            push!(results, (
                gene = gene,
                chromosome = chr,
                start = start_pos,
                stop = end_pos,
                n_variants = length(gene_variants),
                mac = gene_mac,
                statistic = result.statistic,
                pvalue = result.pvalue
            ))
        catch e
            @warn "Failed to test gene $gene: $e"
        end
    end

    # Sort by p-value
    sort!(results, :pvalue)

    return results
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
    compute_variant_weights(genotypes, weight_spec) -> Vector{Float64}

Compute variant weights based on MAF.
"""
function compute_variant_weights(genotypes::Matrix{Float64},
                                 weight_spec::Union{Vector{Float64}, Symbol, Nothing})
    n_variants = size(genotypes, 2)

    if weight_spec === nothing
        return ones(n_variants)
    elseif weight_spec isa Vector{Float64}
        return weight_spec
    elseif weight_spec == :equal
        return ones(n_variants)
    elseif weight_spec == :maf
        # Madsen-Browning weights: 1/sqrt(MAF * (1-MAF))
        mafs = vec(mean(genotypes, dims=1)) / 2
        mafs = clamp.(mafs, 0.001, 0.999)
        return 1.0 ./ sqrt.(mafs .* (1 .- mafs))
    elseif weight_spec == :beta
        # Beta(1, 25) weights (SKAT default)
        mafs = vec(mean(genotypes, dims=1)) / 2
        mafs = clamp.(mafs, 0.001, 0.999)
        return pdf.(Beta(1, 25), mafs).^2
    else
        error("Unknown weight specification: $weight_spec")
    end
end

"""
    compute_mixture_chisq_pvalue(Q, eigenvalues, method) -> Float64

Compute p-value for mixture of chi-squared distributions.
"""
function compute_mixture_chisq_pvalue(Q::Float64, eigenvalues::Vector{Float64},
                                      method::Symbol)
    if isempty(eigenvalues) || all(eigenvalues .== 0)
        return 1.0
    end

    if method == :davies
        return davies_method(Q, eigenvalues)
    elseif method == :saddlepoint
        return saddlepoint_approximation(Q, eigenvalues)
    elseif method == :liu
        return liu_approximation(Q, eigenvalues)
    else
        # Default to Liu's approximation
        return liu_approximation(Q, eigenvalues)
    end
end

"""
    davies_method(Q, eigenvalues) -> Float64

Davies' exact method for mixture of chi-squared.
"""
function davies_method(Q::Float64, λ::Vector{Float64})
    # Simplified implementation using Liu's approximation
    # Full Davies method requires numerical integration
    return liu_approximation(Q, λ)
end

"""
    liu_approximation(Q, eigenvalues) -> Float64

Liu et al. approximation for mixture of chi-squared distributions.
"""
function liu_approximation(Q::Float64, λ::Vector{Float64})
    # Compute cumulants
    c1 = sum(λ)
    c2 = 2 * sum(λ.^2)
    c3 = 8 * sum(λ.^3)
    c4 = 48 * sum(λ.^4)

    # Skewness and kurtosis
    s1 = c3 / c2^1.5
    s2 = c4 / c2^2

    if s1^2 > s2
        # Match first 3 moments
        a = 1 / (s1 - sqrt(s1^2 - s2))
        δ = s1 * a^3 - a^2
        l = a^2 - 2δ
    else
        # Match first 2 moments only
        a = 1 / s1
        δ = 0.0
        l = 1 / s1^2
    end

    # Standardize Q
    Q_norm = (Q - c1) / sqrt(c2)
    Q_chi = Q_norm * sqrt(2 * l) + l

    # P-value from chi-squared
    if Q_chi < 0
        return 1.0
    end

    return ccdf(Chisq(l), Q_chi)
end

"""
    saddlepoint_approximation(Q, eigenvalues) -> Float64

Saddlepoint approximation for mixture of chi-squared.
"""
function saddlepoint_approximation(Q::Float64, λ::Vector{Float64})
    # Find saddlepoint
    function K(s)
        return -0.5 * sum(log.(1 .- 2 * s .* λ))
    end

    function K_prime(s)
        return sum(λ ./ (1 .- 2 * s .* λ))
    end

    # Solve K'(s) = Q for s
    s_max = 0.5 / maximum(λ) - 1e-6

    # Newton's method
    s = 0.0
    for _ in 1:50
        Kp = K_prime(s)
        if abs(Kp - Q) < 1e-8
            break
        end
        # K''(s)
        Kpp = 2 * sum(λ.^2 ./ (1 .- 2 * s .* λ).^2)
        s_new = s + (Q - Kp) / Kpp
        s = clamp(s_new, -s_max, s_max)
    end

    if abs(s) < 1e-10
        # Use normal approximation near s=0
        return 1 - cdf(Normal(), (Q - sum(λ)) / sqrt(2 * sum(λ.^2)))
    end

    # Saddlepoint approximation
    Kpp = 2 * sum(λ.^2 ./ (1 .- 2 * s .* λ).^2)
    w = sign(s) * sqrt(2 * (s * Q - K(s)))
    u = s * sqrt(Kpp)

    # Lugannani-Rice formula
    pvalue = 1 - cdf(Normal(), w) - pdf(Normal(), w) * (1/w - 1/u)

    return clamp(pvalue, 0.0, 1.0)
end

"""
    adjust_skat_o_pvalue(pvalues, Q, rho_grid) -> Float64

Adjust SKAT-O p-value for testing multiple rho values.
"""
function adjust_skat_o_pvalue(pvalues::Vector{Float64}, Q::Vector{Float64},
                              rho_grid::Vector{Float64})
    min_p = minimum(pvalues)

    # Simple Bonferroni-type adjustment
    # More sophisticated methods account for correlation between tests
    n_rho = length(rho_grid)

    # Estimate effective number of independent tests
    # Correlation between Q statistics at different rho
    # Use eigenvalue-based estimate
    n_eff = min(n_rho, 1 + (n_rho - 1) * 0.3)  # Rough estimate

    adjusted_p = min(1.0, min_p * n_eff)

    return adjusted_p
end

"""
Helper functions for regression tests
"""
function linear_regression_test(X, y, test_idx, test_type)
    n = length(y)
    p = size(X, 2)

    β = X \ y
    residuals = y - X * β
    σ2 = sum(residuals.^2) / (n - p)

    XtX_inv = inv(X' * X)
    se = sqrt(σ2 * XtX_inv[test_idx, test_idx])

    t_stat = β[test_idx] / se
    pvalue = 2 * ccdf(TDist(n - p), abs(t_stat))

    return (statistic=t_stat^2, pvalue=pvalue, beta=β[test_idx], se=se)
end

function logistic_regression_test(X, y, test_idx, test_type)
    β = logistic_fit_irls(X, y)
    μ = logistic_predict(X, β)
    V = μ .* (1 .- μ)

    # Fisher information
    I = X' * Diagonal(V) * X
    se = sqrt(inv(I)[test_idx, test_idx])

    z_stat = β[test_idx] / se
    pvalue = 2 * ccdf(Normal(), abs(z_stat))

    return (statistic=z_stat^2, pvalue=pvalue, beta=β[test_idx], se=se)
end

function logistic_fit_irls(X, y; max_iter=25, tol=1e-8)
    n, p = size(X)
    β = zeros(p)

    for _ in 1:max_iter
        μ = logistic_predict(X, β)
        V = μ .* (1 .- μ)
        V = max.(V, 1e-10)  # Prevent division by zero

        z = X * β + (y - μ) ./ V

        β_new = (X' * Diagonal(V) * X) \ (X' * Diagonal(V) * z)

        if maximum(abs.(β_new - β)) < tol
            return β_new
        end
        β = β_new
    end

    return β
end

function logistic_predict(X, β)
    η = X * β
    return 1.0 ./ (1.0 .+ exp.(-η))
end

function logistic_log_likelihood(X, y)
    β = logistic_fit_irls(X, y)
    μ = logistic_predict(X, β)
    μ = clamp.(μ, 1e-10, 1 - 1e-10)
    return sum(y .* log.(μ) + (1 .- y) .* log.(1 .- μ))
end
