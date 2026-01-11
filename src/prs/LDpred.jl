# ============================================================================
# LDpred.jl - Polygenic Risk Score Methods
# ============================================================================
# Implementation of LDpred, LDpred2, PRS-CS, and classical C+T methods
# for computing polygenic risk scores from GWAS summary statistics
# ============================================================================

"""
    PRSResult

Structure containing results from PRS calculation.

# Fields
- `scores::Vector{Float64}`: PRS for each individual
- `weights::Vector{Float64}`: SNP weights used
- `n_snps::Int`: Number of SNPs included
- `method::String`: Method used to compute weights
- `parameters::Dict`: Method-specific parameters
"""
struct PRSResult
    scores::Vector{Float64}
    weights::Vector{Float64}
    n_snps::Int
    method::String
    parameters::Dict{Symbol, Any}
end

"""
    PRSWeights

Structure containing computed PRS weights.

# Fields
- `variant_ids::Vector{String}`: Variant identifiers
- `weights::Vector{Float64}`: Effect size weights
- `alleles::Vector{String}`: Effect alleles
- `method::String`: Method used
"""
struct PRSWeights
    variant_ids::Vector{String}
    weights::Vector{Float64}
    alleles::Vector{String}
    method::String
end

"""
    clump_threshold_prs(betas::Vector{Float64}, pvalues::Vector{Float64},
                       genotypes::Matrix{Float64}; kwargs...) -> PRSWeights

Classical Clumping + Thresholding (C+T) method for PRS.

# Arguments
- `betas`: Effect sizes from GWAS
- `pvalues`: P-values from GWAS
- `genotypes`: Reference genotype matrix for LD calculation

# Keyword Arguments
- `p_threshold::Float64=5e-8`: P-value threshold for inclusion
- `r2_threshold::Float64=0.1`: LD threshold for clumping
- `window_kb::Int=250`: Window size for clumping
- `variant_ids`: Variant identifiers

# Algorithm
1. Sort variants by p-value
2. Select lead variant (lowest p-value)
3. Remove variants in LD (r² > threshold) within window
4. Repeat until no variants remain below p-value threshold
5. Use GWAS betas as weights for selected variants

# Example
```julia
weights = clump_threshold_prs(betas, pvalues, ref_genotypes;
                              p_threshold=5e-8, r2_threshold=0.1)

# Apply to target sample
scores = compute_prs(target_genotypes, weights)
```

# References
- Purcell et al. (2009) Am. J. Hum. Genet. (original C+T)
- Choi et al. (2020) Nat. Protoc. (PRS tutorial)
"""
function clump_threshold_prs(
    betas::Vector{Float64},
    pvalues::Vector{Float64},
    genotypes::Matrix{Float64};
    p_threshold::Float64=5e-8,
    r2_threshold::Float64=0.1,
    window_kb::Int=250,
    positions::Union{Vector{Int}, Nothing}=nothing,
    variant_ids::Union{Vector{String}, Nothing}=nothing
)
    n_snps = length(betas)

    if variant_ids === nothing
        variant_ids = ["SNP_$i" for i in 1:n_snps]
    end

    if positions === nothing
        # Assume sequential positions
        positions = collect(1:n_snps) * 1000
    end

    # Standardize genotypes for LD calculation
    geno_std = similar(genotypes, Float64)
    for j in 1:n_snps
        p = mean(genotypes[:, j]) / 2
        if p > 0 && p < 1
            geno_std[:, j] = (genotypes[:, j] .- 2p) ./ sqrt(2p * (1 - p))
        else
            geno_std[:, j] .= 0
        end
    end

    # Sort by p-value
    sorted_idx = sortperm(pvalues)

    # Track selected and excluded variants
    selected = Int[]
    excluded = falses(n_snps)

    window_bp = window_kb * 1000

    for idx in sorted_idx
        if excluded[idx]
            continue
        end

        if pvalues[idx] > p_threshold
            break  # All remaining variants have p > threshold
        end

        # Select this variant
        push!(selected, idx)

        # Find variants within window
        pos_i = positions[idx]
        in_window = abs.(positions .- pos_i) .<= window_bp

        # Calculate LD with selected variant
        for j in 1:n_snps
            if !excluded[j] && in_window[j] && j != idx
                r = cor(geno_std[:, idx], geno_std[:, j])
                if r^2 > r2_threshold
                    excluded[j] = true
                end
            end
        end
    end

    # Get weights for selected variants
    weights = zeros(n_snps)
    weights[selected] = betas[selected]

    return PRSWeights(
        variant_ids,
        weights,
        ["A1" for _ in 1:n_snps],  # Placeholder
        "C+T (p<$p_threshold, r²<$r2_threshold)"
    )
end

"""
    ldpred2_grid(betas::Vector{Float64}, se::Vector{Float64},
                ld_matrix::Matrix{Float64}, n::Int; kwargs...) -> Dict

LDpred2 with grid of hyperparameters.

# Arguments
- `betas`: GWAS effect sizes
- `se`: Standard errors of betas
- `ld_matrix`: LD correlation matrix
- `n`: GWAS sample size

# Keyword Arguments
- `h2_grid::Vector{Float64}`: Grid of h² values
- `p_grid::Vector{Float64}`: Grid of polygenicity (p) values
- `sparse::Bool=false`: Use sparse prior
- `n_iter::Int=500`: Number of Gibbs iterations
- `burn_in::Int=100`: Burn-in iterations

# Model
LDpred models effect sizes with a spike-and-slab prior:
β_j | γ_j ~ γ_j * N(0, h²/(M*p)) + (1-γ_j) * δ_0
γ_j ~ Bernoulli(p)

where p is the proportion of causal variants.

# Algorithm (Gibbs sampling)
1. Initialize β = 0
2. For each variant j:
   a. Compute posterior inclusion probability
   b. Sample γ_j from Bernoulli
   c. If γ_j = 1, sample β_j from Gaussian posterior
3. Repeat for many iterations

# Example
```julia
results = ldpred2_grid(betas, se, R, n;
                       h2_grid=[0.1, 0.2, 0.3],
                       p_grid=[0.001, 0.01, 0.1, 1.0])

# Select best model using validation data
best_params = select_best_ldpred_model(results, val_genotypes, val_phenotype)
```

# References
- Privé et al. (2020) Bioinformatics (LDpred2)
- Vilhjálmsson et al. (2015) Am. J. Hum. Genet. (original LDpred)
"""
function ldpred2_grid(
    betas::Vector{Float64},
    se::Vector{Float64},
    ld_matrix::Matrix{Float64},
    n::Int;
    h2_grid::Vector{Float64}=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    p_grid::Vector{Float64}=[1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0],
    sparse::Bool=false,
    n_iter::Int=500,
    burn_in::Int=100,
    verbose::Bool=true
)
    p_snps = length(betas)

    # Convert SE to chi-squared
    chi2 = (betas ./ se).^2

    results = Dict{Tuple{Float64, Float64}, Vector{Float64}}()

    total_combos = length(h2_grid) * length(p_grid)
    combo_idx = 1

    for h2 in h2_grid
        for p in p_grid
            if verbose
                println("Running h²=$h2, p=$p ($combo_idx/$total_combos)")
            end

            if sparse
                weights = ldpred2_gibbs_sparse(betas, se, ld_matrix, n, h2, p;
                                               n_iter=n_iter, burn_in=burn_in)
            else
                weights = ldpred2_gibbs(betas, se, ld_matrix, n, h2, p;
                                        n_iter=n_iter, burn_in=burn_in)
            end

            results[(h2, p)] = weights
            combo_idx += 1
        end
    end

    return results
end

"""
    ldpred2_gibbs(betas, se, ld_matrix, n, h2, p; kwargs...) -> Vector{Float64}

Run LDpred2 Gibbs sampler for a single set of hyperparameters.
"""
function ldpred2_gibbs(
    betas::Vector{Float64},
    se::Vector{Float64},
    ld_matrix::Matrix{Float64},
    n::Int,
    h2::Float64,
    p::Float64;
    n_iter::Int=500,
    burn_in::Int=100
)
    m = length(betas)

    # Prior variance for non-zero effects
    σ2_β = h2 / (m * p)

    # Initialize
    β_curr = zeros(m)
    β_samples = zeros(n_iter - burn_in, m)

    # Marginal posterior for each SNP
    # Given LD matrix R and marginal estimates β̂:
    # β_j | β_{-j} ~ N(μ_j, σ²_j)

    for iter in 1:n_iter
        for j in randperm(m)  # Random order update
            # Residual effect (remove effect of SNP j)
            resid = betas[j] - dot(ld_matrix[j, :], β_curr) + ld_matrix[j, j] * β_curr[j]

            # Posterior parameters
            # Precision from likelihood: n * r_jj / se_j²
            # Precision from prior: 1 / σ2_β
            prec_lik = n / (se[j]^2 + 1e-10)
            prec_prior = 1 / σ2_β

            σ2_post = 1 / (prec_lik * ld_matrix[j, j] + prec_prior)
            μ_post = σ2_post * prec_lik * resid

            # Log Bayes factor for inclusion
            log_BF = 0.5 * log(σ2_post / σ2_β) + 0.5 * μ_post^2 / σ2_post

            # Posterior inclusion probability
            log_odds = log_BF + log(p / (1 - p + 1e-10))
            pip = 1 / (1 + exp(-log_odds))

            # Sample inclusion indicator
            if rand() < pip
                β_curr[j] = μ_post + sqrt(σ2_post) * randn()
            else
                β_curr[j] = 0.0
            end
        end

        # Store sample after burn-in
        if iter > burn_in
            β_samples[iter - burn_in, :] = β_curr
        end
    end

    # Return posterior mean
    return vec(mean(β_samples, dims=1))
end

"""
    ldpred2_gibbs_sparse(betas, se, ld_matrix, n, h2, p; kwargs...) -> Vector{Float64}

Sparse version of LDpred2 Gibbs sampler with shrinkage.
"""
function ldpred2_gibbs_sparse(
    betas::Vector{Float64},
    se::Vector{Float64},
    ld_matrix::Matrix{Float64},
    n::Int,
    h2::Float64,
    p::Float64;
    n_iter::Int=500,
    burn_in::Int=100
)
    m = length(betas)

    σ2_β = h2 / (m * p)

    β_curr = zeros(m)
    β_samples = zeros(n_iter - burn_in, m)

    # Additional shrinkage for sparse model
    shrink = 0.95  # Shrinkage factor

    for iter in 1:n_iter
        for j in randperm(m)
            resid = betas[j] - dot(ld_matrix[j, :], β_curr) + ld_matrix[j, j] * β_curr[j]

            prec_lik = n / (se[j]^2 + 1e-10)
            prec_prior = 1 / σ2_β

            σ2_post = 1 / (prec_lik * ld_matrix[j, j] + prec_prior)
            μ_post = σ2_post * prec_lik * resid

            log_BF = 0.5 * log(σ2_post / σ2_β) + 0.5 * μ_post^2 / σ2_post
            log_odds = log_BF + log(p / (1 - p + 1e-10))
            pip = 1 / (1 + exp(-log_odds))

            if rand() < pip
                # Apply additional shrinkage
                β_curr[j] = shrink * (μ_post + sqrt(σ2_post) * randn())
            else
                β_curr[j] = 0.0
            end
        end

        if iter > burn_in
            β_samples[iter - burn_in, :] = β_curr
        end
    end

    return vec(mean(β_samples, dims=1))
end

"""
    ldpred2_auto(betas::Vector{Float64}, se::Vector{Float64},
                ld_matrix::Matrix{Float64}, n::Int; kwargs...) -> Vector{Float64}

LDpred2-auto: automatically estimates hyperparameters.

# Arguments
- `betas`: GWAS effect sizes
- `se`: Standard errors
- `ld_matrix`: LD correlation matrix
- `n`: Sample size

# Keyword Arguments
- `n_iter::Int=1000`: Total iterations
- `burn_in::Int=500`: Burn-in iterations
- `initial_h2::Float64=0.3`: Initial h² estimate
- `initial_p::Float64=0.1`: Initial p estimate

# Algorithm
Extends LDpred2 to jointly sample effect sizes and hyperparameters:
- Sample β | h², p using Gibbs
- Sample h² | β using scaled inverse chi-squared
- Sample p | β using Beta posterior

# Example
```julia
weights = ldpred2_auto(betas, se, R, n)
```

# References
- Privé et al. (2021) Bioinformatics (LDpred2-auto)
"""
function ldpred2_auto(
    betas::Vector{Float64},
    se::Vector{Float64},
    ld_matrix::Matrix{Float64},
    n::Int;
    n_iter::Int=1000,
    burn_in::Int=500,
    initial_h2::Float64=0.3,
    initial_p::Float64=0.1,
    verbose::Bool=true
)
    m = length(betas)

    # Initialize hyperparameters
    h2 = initial_h2
    p = initial_p

    # Initialize effects
    β_curr = zeros(m)
    γ_curr = falses(m)  # Inclusion indicators

    # Storage
    β_samples = zeros(n_iter - burn_in, m)
    h2_samples = zeros(n_iter - burn_in)
    p_samples = zeros(n_iter - burn_in)

    if verbose
        prog = Progress(n_iter; desc="LDpred2-auto: ")
    end

    for iter in 1:n_iter
        σ2_β = h2 / (m * p + 1e-10)

        # Sample β and γ for each SNP
        for j in randperm(m)
            resid = betas[j] - dot(ld_matrix[j, :], β_curr) + ld_matrix[j, j] * β_curr[j]

            prec_lik = n / (se[j]^2 + 1e-10)
            prec_prior = 1 / (σ2_β + 1e-10)

            σ2_post = 1 / (prec_lik * ld_matrix[j, j] + prec_prior)
            μ_post = σ2_post * prec_lik * resid

            log_BF = 0.5 * log(σ2_post / (σ2_β + 1e-10)) + 0.5 * μ_post^2 / σ2_post
            log_odds = log_BF + log(p / (1 - p + 1e-10))
            pip = 1 / (1 + exp(-clamp(log_odds, -500, 500)))

            if rand() < pip
                β_curr[j] = μ_post + sqrt(σ2_post) * randn()
                γ_curr[j] = true
            else
                β_curr[j] = 0.0
                γ_curr[j] = false
            end
        end

        # Sample h² from posterior
        # h² | β ~ scaled-inv-χ² if β'β ~ χ² * h²/(Mp)
        n_causal = sum(γ_curr)
        if n_causal > 0
            ss_β = sum(β_curr[γ_curr].^2)
            # Posterior: h² ~ InverseGamma(a + n_causal/2, b + ss_β * m * p / 2)
            a_post = 1.0 + n_causal / 2
            b_post = 0.01 + ss_β * m * p / 2
            h2 = rand(InverseGamma(a_post, b_post))
            h2 = clamp(h2, 0.001, 0.99)
        end

        # Sample p from posterior
        # p | γ ~ Beta(α + Σγ, β + M - Σγ)
        a_p = 1.0 + n_causal
        b_p = 1.0 + m - n_causal
        p = rand(Beta(a_p, b_p))
        p = clamp(p, 1e-5, 1 - 1e-5)

        # Store samples
        if iter > burn_in
            β_samples[iter - burn_in, :] = β_curr
            h2_samples[iter - burn_in] = h2
            p_samples[iter - burn_in] = p
        end

        if verbose
            next!(prog)
        end
    end

    if verbose
        println("\nEstimated h² = $(round(mean(h2_samples), digits=3)) ± $(round(std(h2_samples), digits=3))")
        println("Estimated p = $(round(mean(p_samples), digits=4)) ± $(round(std(p_samples), digits=4))")
    end

    return vec(mean(β_samples, dims=1))
end

"""
    prs_cs(betas::Vector{Float64}, se::Vector{Float64},
          ld_matrix::Matrix{Float64}, n::Int; kwargs...) -> Vector{Float64}

PRS-CS: Continuous Shrinkage prior for PRS.

# Arguments
- `betas`: GWAS effect sizes
- `se`: Standard errors
- `ld_matrix`: LD correlation matrix
- `n`: Sample size

# Keyword Arguments
- `phi::Union{Float64, Nothing}=nothing`: Global shrinkage (auto if nothing)
- `n_iter::Int=1000`: MCMC iterations
- `burn_in::Int=500`: Burn-in

# Model
PRS-CS uses a continuous shrinkage prior (global-local horseshoe-like):
β_j ~ N(0, ψ_j τ² σ²)
ψ_j ~ Gamma(a, 1) (local shrinkage)
τ ~ Gamma(b, 1) (global shrinkage)

This allows adaptive shrinkage: large effects are shrunk less,
small effects are shrunk more toward zero.

# Example
```julia
weights = prs_cs(betas, se, R, n; phi=1e-2)
```

# References
- Ge et al. (2019) Nat. Commun.
"""
function prs_cs(
    betas::Vector{Float64},
    se::Vector{Float64},
    ld_matrix::Matrix{Float64},
    n::Int;
    phi::Union{Float64, Nothing}=nothing,
    a::Float64=1.0,
    b::Float64=0.5,
    n_iter::Int=1000,
    burn_in::Int=500,
    verbose::Bool=true
)
    m = length(betas)

    # Auto-estimate phi if not provided
    if phi === nothing
        # Use LD score regression estimate
        h2_est = min(0.9, max(0.01, sum(betas.^2 ./ se.^2) / n - m / n))
        phi = h2_est / m
    end

    # Initialize
    β_curr = zeros(m)
    ψ = ones(m)  # Local shrinkage
    τ = 1.0      # Global shrinkage

    β_samples = zeros(n_iter - burn_in, m)

    if verbose
        prog = Progress(n_iter; desc="PRS-CS: ")
    end

    for iter in 1:n_iter
        # Sample β | ψ, τ
        for j in 1:m
            resid = betas[j] - dot(ld_matrix[j, :], β_curr) + ld_matrix[j, j] * β_curr[j]

            prec_lik = n / (se[j]^2 + 1e-10)
            prec_prior = 1 / (ψ[j] * τ^2 * phi + 1e-10)

            σ2_post = 1 / (prec_lik * ld_matrix[j, j] + prec_prior)
            μ_post = σ2_post * prec_lik * resid

            β_curr[j] = μ_post + sqrt(σ2_post) * randn()
        end

        # Sample ψ | β, τ (local shrinkage)
        for j in 1:m
            # ψ_j | β_j ~ InverseGamma(a + 0.5, 1 + β_j²/(2τ²φ))
            rate = 1 + β_curr[j]^2 / (2 * τ^2 * phi + 1e-10)
            ψ[j] = rand(InverseGamma(a + 0.5, rate))
            ψ[j] = clamp(ψ[j], 1e-6, 1e6)
        end

        # Sample τ | β, ψ (global shrinkage)
        # τ² | β ~ InverseGamma(b + m/2, 1 + Σ β_j²/(2ψ_j φ))
        ss = sum(β_curr.^2 ./ (ψ .+ 1e-10)) / (2 * phi + 1e-10)
        τ_sq = rand(InverseGamma(b + m/2, 1 + ss))
        τ = sqrt(clamp(τ_sq, 1e-10, 1e6))

        if iter > burn_in
            β_samples[iter - burn_in, :] = β_curr
        end

        if verbose
            next!(prog)
        end
    end

    return vec(mean(β_samples, dims=1))
end

"""
    compute_prs(genotypes::Matrix{Float64}, weights::Vector{Float64}) -> Vector{Float64}

Compute PRS for individuals using SNP weights.

# Arguments
- `genotypes`: Genotype matrix (n_individuals × n_snps)
- `weights`: Effect size weights for each SNP

# Returns
Vector of PRS values for each individual

# Formula
PRS_i = Σ_j w_j * G_{ij}

where w_j is the weight for SNP j and G_{ij} is individual i's genotype.
"""
function compute_prs(genotypes::Matrix{Float64}, weights::Vector{Float64})
    @assert size(genotypes, 2) == length(weights) "Genotype columns must match weights"
    return genotypes * weights
end

"""
    compute_prs(genotypes::Matrix{Float64}, weights::PRSWeights) -> PRSResult

Compute PRS using a PRSWeights object.
"""
function compute_prs(genotypes::Matrix{Float64}, weights::PRSWeights)
    scores = genotypes * weights.weights
    n_snps = sum(weights.weights .!= 0)

    return PRSResult(
        scores,
        weights.weights,
        n_snps,
        weights.method,
        Dict{Symbol, Any}()
    )
end

"""
    validate_prs(scores::Vector{Float64}, phenotype::Vector{Float64};
                covariates=nothing) -> NamedTuple

Validate PRS performance in an independent sample.

# Arguments
- `scores`: PRS values
- `phenotype`: True phenotypes
- `covariates`: Optional covariate matrix

# Returns
Named tuple with:
- `r2`: Variance explained (R²) or Nagelkerke R²
- `auc`: Area under ROC curve (for binary)
- `beta`: Regression coefficient
- `se`: Standard error
- `pvalue`: P-value for association

# Example
```julia
val_results = validate_prs(scores, phenotype)
println("R² = \$(val_results.r2)")
println("P-value = \$(val_results.pvalue)")
```
"""
function validate_prs(
    scores::Vector{Float64},
    phenotype::Vector{Float64};
    covariates::Union{Matrix{Float64}, Nothing}=nothing
)
    n = length(scores)

    # Standardize PRS
    scores_std = (scores .- mean(scores)) ./ std(scores)

    # Build design matrix
    if covariates !== nothing
        X = hcat(ones(n), covariates, scores_std)
        X_null = hcat(ones(n), covariates)
    else
        X = hcat(ones(n), scores_std)
        X_null = ones(n, 1)
    end

    prs_idx = size(X, 2)

    # Determine if binary or continuous
    is_binary = all(p -> p == 0 || p == 1, phenotype)

    if is_binary
        # Logistic regression
        β = logistic_fit(X, phenotype)
        μ = logistic_predict(X, β)

        # Nagelkerke R²
        ll_full = sum(phenotype .* log.(μ .+ 1e-10) + (1 .- phenotype) .* log.(1 .- μ .+ 1e-10))

        β_null = logistic_fit(X_null, phenotype)
        μ_null = logistic_predict(X_null, β_null)
        ll_null = sum(phenotype .* log.(μ_null .+ 1e-10) + (1 .- phenotype) .* log.(1 .- μ_null .+ 1e-10))

        ll_0 = sum(phenotype .* log(mean(phenotype)) + (1 .- phenotype) .* log(1 - mean(phenotype)))

        r2_cox_snell = 1 - exp(-2/n * (ll_full - ll_null))
        r2_max = 1 - exp(2/n * ll_0)
        r2_nagelkerke = r2_cox_snell / r2_max

        # AUC
        auc = compute_auc(phenotype, scores)

        # SE and p-value
        V = μ .* (1 .- μ)
        I = X' * Diagonal(V) * X
        se = sqrt(inv(I)[prs_idx, prs_idx])
        z = β[prs_idx] / se
        pvalue = 2 * ccdf(Normal(), abs(z))

        return (
            r2 = r2_nagelkerke,
            auc = auc,
            beta = β[prs_idx],
            se = se,
            pvalue = pvalue,
            type = "binary"
        )
    else
        # Linear regression
        β = X \ phenotype
        residuals = phenotype - X * β
        σ2 = sum(residuals.^2) / (n - size(X, 2))

        # R² increment
        ss_null = sum((phenotype .- mean(phenotype)).^2)
        residuals_null = phenotype - X_null * (X_null \ phenotype)
        ss_null_model = sum(residuals_null.^2)
        ss_full = sum(residuals.^2)

        r2_full = 1 - ss_full / ss_null
        r2_null = 1 - ss_null_model / ss_null
        r2_increment = r2_full - r2_null

        # SE and p-value
        XtX_inv = inv(X' * X)
        se = sqrt(σ2 * XtX_inv[prs_idx, prs_idx])
        t_stat = β[prs_idx] / se
        pvalue = 2 * ccdf(TDist(n - size(X, 2)), abs(t_stat))

        return (
            r2 = r2_increment,
            auc = NaN,
            beta = β[prs_idx],
            se = se,
            pvalue = pvalue,
            type = "continuous"
        )
    end
end

"""
    select_best_prs(weights_dict::Dict, val_genotypes::Matrix{Float64},
                   val_phenotype::Vector{Float64}) -> Tuple

Select best PRS model from a grid of hyperparameters.

# Arguments
- `weights_dict`: Dictionary of (params => weights) from grid search
- `val_genotypes`: Validation genotype matrix
- `val_phenotype`: Validation phenotype

# Returns
(best_params, best_weights, best_r2)
"""
function select_best_prs(
    weights_dict::Dict,
    val_genotypes::Matrix{Float64},
    val_phenotype::Vector{Float64}
)
    best_r2 = -Inf
    best_params = nothing
    best_weights = nothing

    for (params, weights) in weights_dict
        scores = compute_prs(val_genotypes, weights)
        val_result = validate_prs(scores, val_phenotype)

        if val_result.r2 > best_r2
            best_r2 = val_result.r2
            best_params = params
            best_weights = weights
        end
    end

    return (best_params, best_weights, best_r2)
end

"""
    stratify_prs(scores::Vector{Float64}; n_quantiles::Int=10) -> Vector{Int}

Stratify individuals into risk groups based on PRS quantiles.
"""
function stratify_prs(scores::Vector{Float64}; n_quantiles::Int=10)
    quantiles = quantile(scores, range(0, 1, length=n_quantiles+1))
    groups = zeros(Int, length(scores))

    for i in eachindex(scores)
        for q in 1:n_quantiles
            if scores[i] <= quantiles[q+1]
                groups[i] = q
                break
            end
        end
        if groups[i] == 0
            groups[i] = n_quantiles
        end
    end

    return groups
end

# Helper functions
function logistic_fit(X, y; max_iter=25)
    n, p = size(X)
    β = zeros(p)

    for _ in 1:max_iter
        μ = 1.0 ./ (1.0 .+ exp.(-X * β))
        V = μ .* (1 .- μ)
        V = max.(V, 1e-10)
        z = X * β + (y - μ) ./ V
        β_new = (X' * Diagonal(V) * X) \ (X' * Diagonal(V) * z)

        if maximum(abs.(β_new - β)) < 1e-8
            return β_new
        end
        β = β_new
    end
    return β
end

function logistic_predict(X, β)
    return 1.0 ./ (1.0 .+ exp.(-X * β))
end

function compute_auc(y_true::Vector{Float64}, scores::Vector{Float64})
    # Mann-Whitney U statistic
    n1 = sum(y_true .== 1)
    n0 = sum(y_true .== 0)

    pos_scores = scores[y_true .== 1]
    neg_scores = scores[y_true .== 0]

    u = 0.0
    for p in pos_scores
        for n in neg_scores
            if p > n
                u += 1.0
            elseif p == n
                u += 0.5
            end
        end
    end

    return u / (n1 * n0)
end
