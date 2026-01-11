# ============================================================================
# SuSiE.jl - Sum of Single Effects Model for Fine-Mapping
# ============================================================================
# Implementation of SuSiE (Wang et al. 2020) for statistical fine-mapping
# Identifies causal variants and constructs credible sets
# ============================================================================

"""
    SuSiEResult

Structure containing results from SuSiE fine-mapping.

# Fields
- `alpha::Matrix{Float64}`: Posterior inclusion probabilities (L × p)
- `mu::Matrix{Float64}`: Posterior means for each effect (L × p)
- `mu2::Matrix{Float64}`: Posterior second moments (L × p)
- `pip::Vector{Float64}`: Per-SNP posterior inclusion probability
- `credible_sets::Vector{Vector{Int}}`: Credible sets for each causal signal
- `cs_coverage::Vector{Float64}`: Coverage of each credible set
- `cs_purity::Vector{Float64}`: Minimum pairwise correlation in CS
- `sigma2::Float64`: Residual variance estimate
- `elbo::Vector{Float64}`: ELBO trace during optimization
- `converged::Bool`: Whether algorithm converged
"""
struct SuSiEResult
    alpha::Matrix{Float64}
    mu::Matrix{Float64}
    mu2::Matrix{Float64}
    pip::Vector{Float64}
    credible_sets::Vector{Vector{Int}}
    cs_coverage::Vector{Float64}
    cs_purity::Vector{Float64}
    sigma2::Float64
    elbo::Vector{Float64}
    converged::Bool
end

"""
    susie(X::Matrix{Float64}, y::Vector{Float64}; kwargs...) -> SuSiEResult

Run SuSiE (Sum of Single Effects) fine-mapping algorithm.

# Arguments
- `X`: Standardized genotype matrix (n × p)
- `y`: Phenotype vector (n × 1)

# Keyword Arguments
- `L::Int=10`: Maximum number of causal effects
- `prior_variance::Float64=0.2`: Prior variance for effect sizes
- `residual_variance::Union{Float64, Nothing}=nothing`: Fix residual variance
- `prior_weights::Union{Vector{Float64}, Nothing}=nothing`: Prior inclusion weights
- `max_iter::Int=100`: Maximum iterations
- `tol::Float64=1e-3`: Convergence tolerance
- `coverage::Float64=0.95`: Target credible set coverage
- `min_purity::Float64=0.5`: Minimum purity for valid CS

# Mathematical Model
SuSiE models the phenotype as:
y = Σ_{l=1}^L X b_l + ε, ε ~ N(0, σ² I)

where each b_l is a "single effect":
b_l = γ_l ⊙ β_l
γ_l ~ Multinomial(1, π)
β_l | γ_l ~ N(0, σ²_0)

Key features:
- Each single effect has exactly one non-zero component
- Multiple single effects allow multiple causal variants
- Posterior inclusion probability: PIP_j = 1 - Π_l (1 - α_lj)
- Credible sets: smallest set with coverage ≥ target

# Algorithm
Uses coordinate ascent variational inference (CAVI):
1. Initialize all effects
2. For each effect l:
   a. Compute residuals excluding effect l
   b. Update variational parameters (α_l, μ_l, σ²_l)
3. Update residual variance σ²
4. Check convergence via ELBO
5. Construct credible sets from converged α

# Example
```julia
# Fine-map a GWAS region
result = susie(X_region, y; L=5, coverage=0.95)

# Get credible sets
for (i, cs) in enumerate(result.credible_sets)
    snps = variant_ids[cs]
    println("Credible set \$i: \$snps (coverage: \$(result.cs_coverage[i]))")
end

# Per-SNP PIP
high_pip = findall(result.pip .> 0.5)
```

# References
- Wang et al. (2020) J. R. Stat. Soc. B
- Zou et al. (2022) PLoS Genet. (SuSiE RSS)
"""
function susie(
    X::Matrix{Float64},
    y::Vector{Float64};
    L::Int=10,
    prior_variance::Float64=0.2,
    residual_variance::Union{Float64, Nothing}=nothing,
    prior_weights::Union{Vector{Float64}, Nothing}=nothing,
    max_iter::Int=100,
    tol::Float64=1e-3,
    coverage::Float64=0.95,
    min_purity::Float64=0.5,
    verbose::Bool=true
)
    n, p = size(X)

    # Standardize X if not already done
    X_mean = mean(X, dims=1)
    X_std = std(X, dims=1)
    X_std[X_std .== 0] .= 1  # Avoid division by zero
    X_scaled = (X .- X_mean) ./ X_std

    # Center y
    y_mean = mean(y)
    y_centered = y .- y_mean

    # Prior inclusion weights
    if prior_weights === nothing
        prior_weights = fill(1.0 / p, p)
    end
    prior_weights = prior_weights ./ sum(prior_weights)

    # Initialize residual variance
    if residual_variance === nothing
        σ2 = var(y_centered)
    else
        σ2 = residual_variance
    end

    # Precompute X'X diagonal and X'y
    XtX_diag = vec(sum(X_scaled.^2, dims=1))
    Xty = X_scaled' * y_centered

    # Initialize variational parameters
    α = ones(L, p) ./ p  # Posterior inclusion probabilities
    μ = zeros(L, p)      # Posterior means
    μ2 = zeros(L, p)     # Posterior second moments

    # Prior variance for effects
    σ2_0 = prior_variance * var(y_centered)

    # ELBO history
    elbo_history = Float64[]

    # Current fitted values
    Xb = zeros(n)  # Sum of X * (α_l ⊙ μ_l)

    if verbose
        prog = Progress(max_iter; desc="SuSiE: ")
    end

    converged = false

    for iter in 1:max_iter
        # Update each single effect
        for l in 1:L
            # Compute residuals excluding effect l
            Xb_l = X_scaled * (α[l, :] .* μ[l, :])
            r_l = y_centered - Xb + Xb_l

            # Update variational parameters for effect l
            α[l, :], μ[l, :], μ2[l, :] = update_single_effect(
                X_scaled, r_l, XtX_diag, σ2, σ2_0, prior_weights
            )

            # Update Xb
            Xb = Xb - Xb_l + X_scaled * (α[l, :] .* μ[l, :])
        end

        # Update residual variance if not fixed
        if residual_variance === nothing
            σ2 = update_residual_variance(y_centered, X_scaled, α, μ, μ2, n)
            σ2 = max(σ2, 1e-10)  # Prevent numerical issues
        end

        # Compute ELBO
        elbo = compute_susie_elbo(y_centered, X_scaled, α, μ, μ2, σ2, σ2_0, prior_weights)
        push!(elbo_history, elbo)

        # Check convergence
        if iter > 1
            elbo_change = abs(elbo_history[end] - elbo_history[end-1])
            if elbo_change < tol
                converged = true
                if verbose
                    println("\nConverged at iteration $iter (ELBO change: $elbo_change)")
                end
                break
            end
        end

        if verbose
            next!(prog)
        end
    end

    # Compute per-SNP PIP
    pip = compute_pip(α)

    # Construct credible sets
    credible_sets, cs_coverage_vals, cs_purity_vals = construct_credible_sets(
        α, X_scaled; coverage=coverage, min_purity=min_purity
    )

    return SuSiEResult(
        α,
        μ,
        μ2,
        pip,
        credible_sets,
        cs_coverage_vals,
        cs_purity_vals,
        σ2,
        elbo_history,
        converged
    )
end

"""
    susie_rss(z::Vector{Float64}, R::Matrix{Float64}, n::Int; kwargs...) -> SuSiEResult

Run SuSiE using GWAS summary statistics (RSS format).

# Arguments
- `z`: Z-scores from GWAS
- `R`: LD correlation matrix
- `n`: Sample size

# Keyword Arguments
Same as `susie` plus:
- `var_y::Float64=1.0`: Variance of y (assumed standardized)

# Mathematical Model
With summary statistics, SuSiE models:
ẑ = R β̂ + ε

where β̂ ~ N(β, R⁻¹/n) under the null.

The sufficient statistics are:
- X'X ≈ n R
- X'y ≈ √n ẑ (when y has variance 1)

# Example
```julia
# From GWAS summary statistics
result = susie_rss(z_scores, ld_matrix, n; L=10)
```

# References
- Zou et al. (2022) PLoS Genet.
"""
function susie_rss(
    z::Vector{Float64},
    R::Matrix{Float64},
    n::Int;
    L::Int=10,
    prior_variance::Float64=0.2,
    residual_variance::Union{Float64, Nothing}=nothing,
    prior_weights::Union{Vector{Float64}, Nothing}=nothing,
    max_iter::Int=100,
    tol::Float64=1e-3,
    coverage::Float64=0.95,
    min_purity::Float64=0.5,
    var_y::Float64=1.0,
    verbose::Bool=true
)
    p = length(z)

    # Convert to sufficient statistics
    # X'X = n * R
    # X'y = sqrt(n) * z * sqrt(var_y)
    XtX = n * R
    Xty = sqrt(n) * z * sqrt(var_y)
    XtX_diag = diag(XtX)

    # Prior weights
    if prior_weights === nothing
        prior_weights = fill(1.0 / p, p)
    end
    prior_weights = prior_weights ./ sum(prior_weights)

    # Initialize
    if residual_variance === nothing
        σ2 = var_y
    else
        σ2 = residual_variance
    end

    σ2_0 = prior_variance * var_y

    # Initialize variational parameters
    α = ones(L, p) ./ p
    μ = zeros(L, p)
    μ2 = zeros(L, p)

    elbo_history = Float64[]
    converged = false

    if verbose
        prog = Progress(max_iter; desc="SuSiE-RSS: ")
    end

    for iter in 1:max_iter
        # Update each single effect
        for l in 1:L
            # Effective Xty excluding effect l
            b_minus_l = sum(α[setdiff(1:L, l), :] .* μ[setdiff(1:L, l), :], dims=1)[:]
            Xty_l = Xty - XtX * b_minus_l

            # Update
            α[l, :], μ[l, :], μ2[l, :] = update_single_effect_rss(
                XtX, Xty_l, XtX_diag, σ2, σ2_0, prior_weights
            )
        end

        # Update residual variance
        if residual_variance === nothing
            σ2 = update_residual_variance_rss(Xty, XtX, α, μ, μ2, n, var_y)
            σ2 = max(σ2, 1e-10)
        end

        # ELBO
        elbo = compute_susie_elbo_rss(Xty, XtX, α, μ, μ2, σ2, σ2_0, prior_weights, n, var_y)
        push!(elbo_history, elbo)

        if iter > 1
            if abs(elbo_history[end] - elbo_history[end-1]) < tol
                converged = true
                if verbose
                    println("\nConverged at iteration $iter")
                end
                break
            end
        end

        if verbose
            next!(prog)
        end
    end

    pip = compute_pip(α)

    # For credible sets, need correlation matrix
    credible_sets, cs_coverage_vals, cs_purity_vals = construct_credible_sets_rss(
        α, R; coverage=coverage, min_purity=min_purity
    )

    return SuSiEResult(
        α,
        μ,
        μ2,
        pip,
        credible_sets,
        cs_coverage_vals,
        cs_purity_vals,
        σ2,
        elbo_history,
        converged
    )
end

"""
    update_single_effect(X, r, XtX_diag, σ2, σ2_0, π) -> (α, μ, μ2)

Update variational parameters for a single effect.
"""
function update_single_effect(
    X::Matrix{Float64},
    r::Vector{Float64},
    XtX_diag::Vector{Float64},
    σ2::Float64,
    σ2_0::Float64,
    π::Vector{Float64}
)
    p = length(π)

    # Posterior variance for each SNP
    σ2_post = 1.0 ./ (XtX_diag ./ σ2 .+ 1.0 / σ2_0)

    # Posterior mean for each SNP
    Xtr = X' * r
    μ_post = σ2_post .* Xtr ./ σ2

    # Log Bayes factor for each SNP (compared to null)
    log_BF = 0.5 * log.(σ2_post ./ σ2_0) .+ 0.5 * μ_post.^2 ./ σ2_post

    # Posterior inclusion probability (softmax with prior)
    log_α = log_BF .+ log.(π)
    log_α = log_α .- maximum(log_α)  # For numerical stability
    α = exp.(log_α)
    α = α ./ sum(α)

    # Second moment
    μ2 = σ2_post .+ μ_post.^2

    return α, μ_post, μ2
end

"""
    update_single_effect_rss(XtX, Xty, XtX_diag, σ2, σ2_0, π) -> (α, μ, μ2)

Update variational parameters using summary statistics.
"""
function update_single_effect_rss(
    XtX::Matrix{Float64},
    Xty::Vector{Float64},
    XtX_diag::Vector{Float64},
    σ2::Float64,
    σ2_0::Float64,
    π::Vector{Float64}
)
    p = length(π)

    # Posterior variance
    σ2_post = 1.0 ./ (XtX_diag ./ σ2 .+ 1.0 / σ2_0)

    # Posterior mean
    μ_post = σ2_post .* Xty ./ σ2

    # Log Bayes factor
    log_BF = 0.5 * log.(σ2_post ./ σ2_0) .+ 0.5 * μ_post.^2 ./ σ2_post

    # Posterior inclusion
    log_α = log_BF .+ log.(π .+ 1e-300)
    log_α = log_α .- maximum(log_α)
    α = exp.(log_α)
    α = α ./ sum(α)

    μ2 = σ2_post .+ μ_post.^2

    return α, μ_post, μ2
end

"""
    update_residual_variance(y, X, α, μ, μ2, n) -> Float64

Update residual variance estimate.
"""
function update_residual_variance(
    y::Vector{Float64},
    X::Matrix{Float64},
    α::Matrix{Float64},
    μ::Matrix{Float64},
    μ2::Matrix{Float64},
    n::Int
)
    L, p = size(α)

    # Expected residual sum of squares
    # E[(y - Xb)²] = y'y - 2y'X E[b] + E[b'X'Xb]

    # E[b] = Σ_l α_l ⊙ μ_l
    b_mean = vec(sum(α .* μ, dims=1))

    # E[b'X'Xb] is more complex due to cross terms
    # = Σ_l E[b_l' X'X b_l] + Σ_{l≠l'} E[b_l]' X'X E[b_l']

    yty = y' * y
    Xb_mean = X * b_mean
    yXb = y' * Xb_mean

    # First term: Σ_l α_l' diag(X'X) (μ2_l)
    XtX_diag = vec(sum(X.^2, dims=1))
    term1 = sum(α .* μ2 .* XtX_diag', dims=2)

    # Cross terms (simplified)
    term2 = Xb_mean' * Xb_mean

    expected_rss = yty - 2 * yXb + sum(term1) + term2

    # Add variance terms
    # For full computation, need E[b_l² | α_l] for each l

    σ2 = expected_rss / n

    return max(σ2, 1e-10)
end

"""
    update_residual_variance_rss(Xty, XtX, α, μ, μ2, n, var_y) -> Float64

Update residual variance using summary statistics.
"""
function update_residual_variance_rss(
    Xty::Vector{Float64},
    XtX::Matrix{Float64},
    α::Matrix{Float64},
    μ::Matrix{Float64},
    μ2::Matrix{Float64},
    n::Int,
    var_y::Float64
)
    L, p = size(α)

    b_mean = vec(sum(α .* μ, dims=1))

    # y'y = n * var_y
    yty = n * var_y

    # y'Xb = Xty' * b
    yXb = Xty' * b_mean

    # b'X'Xb
    bXtXb = b_mean' * XtX * b_mean

    # Variance correction
    XtX_diag = diag(XtX)
    var_term = sum(α .* μ2 .* XtX_diag', dims=2)

    expected_rss = yty - 2 * yXb + bXtXb + sum(var_term) - b_mean' * Diagonal(XtX_diag) * b_mean

    return max(expected_rss / n, 1e-10)
end

"""
    compute_susie_elbo(y, X, α, μ, μ2, σ2, σ2_0, π) -> Float64

Compute the Evidence Lower Bound (ELBO) for SuSiE.
"""
function compute_susie_elbo(
    y::Vector{Float64},
    X::Matrix{Float64},
    α::Matrix{Float64},
    μ::Matrix{Float64},
    μ2::Matrix{Float64},
    σ2::Float64,
    σ2_0::Float64,
    π::Vector{Float64}
)
    n = length(y)
    L, p = size(α)

    # E[log p(y | b, σ2)]
    b_mean = vec(sum(α .* μ, dims=1))
    residuals = y - X * b_mean

    # Need to include variance of b
    XtX_diag = vec(sum(X.^2, dims=1))
    var_Xb = 0.0
    for l in 1:L
        var_Xb += sum(α[l, :] .* (μ2[l, :] - μ[l, :].^2) .* XtX_diag)
    end

    log_lik = -n/2 * log(2π * σ2) - (sum(residuals.^2) + var_Xb) / (2σ2)

    # E[log p(b | σ2_0)]
    log_prior = 0.0
    for l in 1:L
        # E[β_l² | γ_l] = α_l' μ2_l
        E_b2 = sum(α[l, :] .* μ2[l, :])
        log_prior += -0.5 * log(2π * σ2_0) - E_b2 / (2σ2_0)
    end

    # E[log p(γ | π)]
    log_prior_gamma = 0.0
    for l in 1:L
        log_prior_gamma += sum(α[l, :] .* log.(π .+ 1e-300))
    end

    # E[log q(b, γ)]
    entropy = 0.0
    for l in 1:L
        # Entropy of categorical × Gaussian
        # -E[log q(γ_l)] = -Σ α_lj log α_lj
        entropy -= sum(α[l, :] .* log.(α[l, :] .+ 1e-300))
        # Gaussian entropy per component (weighted)
        σ2_post = μ2[l, :] - μ[l, :].^2
        for j in 1:p
            if α[l, j] > 1e-10
                entropy += α[l, j] * 0.5 * (1 + log(2π * max(σ2_post[j], 1e-10)))
            end
        end
    end

    return log_lik + log_prior + log_prior_gamma + entropy
end

"""
    compute_susie_elbo_rss(Xty, XtX, α, μ, μ2, σ2, σ2_0, π, n, var_y) -> Float64

Compute ELBO using summary statistics.
"""
function compute_susie_elbo_rss(
    Xty::Vector{Float64},
    XtX::Matrix{Float64},
    α::Matrix{Float64},
    μ::Matrix{Float64},
    μ2::Matrix{Float64},
    σ2::Float64,
    σ2_0::Float64,
    π::Vector{Float64},
    n::Int,
    var_y::Float64
)
    L, p = size(α)

    # Sufficient statistics for ELBO
    b_mean = vec(sum(α .* μ, dims=1))

    yty = n * var_y
    yXb = Xty' * b_mean
    bXtXb = b_mean' * XtX * b_mean

    XtX_diag = diag(XtX)
    var_term = sum(sum(α .* μ2 .* XtX_diag', dims=1))

    expected_rss = yty - 2 * yXb + bXtXb + var_term - b_mean' * Diagonal(XtX_diag) * b_mean

    log_lik = -n/2 * log(2π * σ2) - expected_rss / (2σ2)

    # Prior and entropy terms (same as before)
    log_prior = 0.0
    for l in 1:L
        E_b2 = sum(α[l, :] .* μ2[l, :])
        log_prior += -0.5 * log(2π * σ2_0) - E_b2 / (2σ2_0)
    end

    log_prior_gamma = 0.0
    for l in 1:L
        log_prior_gamma += sum(α[l, :] .* log.(π .+ 1e-300))
    end

    entropy = 0.0
    for l in 1:L
        entropy -= sum(α[l, :] .* log.(α[l, :] .+ 1e-300))
        σ2_post = μ2[l, :] - μ[l, :].^2
        for j in 1:p
            if α[l, j] > 1e-10
                entropy += α[l, j] * 0.5 * (1 + log(2π * max(σ2_post[j], 1e-10)))
            end
        end
    end

    return log_lik + log_prior + log_prior_gamma + entropy
end

"""
    compute_pip(α::Matrix{Float64}) -> Vector{Float64}

Compute per-SNP posterior inclusion probability.

PIP_j = 1 - Π_l (1 - α_lj)
"""
function compute_pip(α::Matrix{Float64})
    L, p = size(α)
    pip = ones(p)

    for j in 1:p
        for l in 1:L
            pip[j] *= (1 - α[l, j])
        end
    end

    return 1 .- pip
end

"""
    construct_credible_sets(α, X; coverage, min_purity) -> (sets, coverages, purities)

Construct credible sets from SuSiE posterior.
"""
function construct_credible_sets(
    α::Matrix{Float64},
    X::Matrix{Float64};
    coverage::Float64=0.95,
    min_purity::Float64=0.5
)
    L, p = size(α)

    credible_sets = Vector{Int}[]
    cs_coverage = Float64[]
    cs_purity = Float64[]

    # Compute correlation matrix for purity calculation
    R = cor(X)

    for l in 1:L
        # Check if this effect is "active" (concentrated on few SNPs)
        if maximum(α[l, :]) < 0.1
            continue  # Skip very diffuse effects
        end

        # Sort SNPs by inclusion probability
        sorted_idx = sortperm(α[l, :], rev=true)

        # Find smallest set with coverage ≥ target
        cumsum_alpha = cumsum(α[l, sorted_idx])
        set_size = findfirst(cumsum_alpha .>= coverage)

        if set_size === nothing
            set_size = p
        end

        cs = sorted_idx[1:set_size]

        # Compute purity (minimum pairwise |r|)
        if length(cs) > 1
            purity = minimum(abs.(R[cs, cs]))
        else
            purity = 1.0
        end

        # Only keep sets meeting purity threshold
        if purity >= min_purity
            push!(credible_sets, cs)
            push!(cs_coverage, cumsum_alpha[set_size])
            push!(cs_purity, purity)
        end
    end

    return credible_sets, cs_coverage, cs_purity
end

"""
    construct_credible_sets_rss(α, R; coverage, min_purity) -> (sets, coverages, purities)

Construct credible sets using correlation matrix directly.
"""
function construct_credible_sets_rss(
    α::Matrix{Float64},
    R::Matrix{Float64};
    coverage::Float64=0.95,
    min_purity::Float64=0.5
)
    L, p = size(α)

    credible_sets = Vector{Int}[]
    cs_coverage = Float64[]
    cs_purity = Float64[]

    for l in 1:L
        if maximum(α[l, :]) < 0.1
            continue
        end

        sorted_idx = sortperm(α[l, :], rev=true)
        cumsum_alpha = cumsum(α[l, sorted_idx])
        set_size = findfirst(cumsum_alpha .>= coverage)

        if set_size === nothing
            set_size = p
        end

        cs = sorted_idx[1:set_size]

        if length(cs) > 1
            purity = minimum(abs.(R[cs, cs]))
        else
            purity = 1.0
        end

        if purity >= min_purity
            push!(credible_sets, cs)
            push!(cs_coverage, cumsum_alpha[set_size])
            push!(cs_purity, purity)
        end
    end

    return credible_sets, cs_coverage, cs_purity
end

"""
    susie_get_cs_summary(result::SuSiEResult, variant_ids::Vector{String}) -> DataFrame

Create a summary DataFrame of credible sets.
"""
function susie_get_cs_summary(result::SuSiEResult, variant_ids::Vector{String})
    n_cs = length(result.credible_sets)

    summaries = DataFrame(
        cs_id = Int[],
        size = Int[],
        coverage = Float64[],
        purity = Float64[],
        lead_snp = String[],
        lead_pip = Float64[],
        snps = String[]
    )

    for i in 1:n_cs
        cs = result.credible_sets[i]
        pips = result.pip[cs]

        lead_idx = argmax(pips)
        lead_snp = variant_ids[cs[lead_idx]]
        lead_pip = pips[lead_idx]

        snp_str = join(variant_ids[cs], ";")

        push!(summaries, (
            cs_id = i,
            size = length(cs),
            coverage = result.cs_coverage[i],
            purity = result.cs_purity[i],
            lead_snp = lead_snp,
            lead_pip = lead_pip,
            snps = snp_str
        ))
    end

    return summaries
end
