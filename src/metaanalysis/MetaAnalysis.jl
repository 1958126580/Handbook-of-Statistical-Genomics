# ============================================================================
# MetaAnalysis.jl - Meta-Analysis Methods for Genomics
# ============================================================================
# Comprehensive implementation of fixed-effect and random-effect meta-analysis
# Including heterogeneity testing and trans-ancestry methods
# ============================================================================

"""
    MetaAnalysisResult

Structure containing results from meta-analysis.

# Fields
- `beta::Float64`: Combined effect estimate
- `se::Float64`: Standard error of combined effect
- `z::Float64`: Z-score
- `pvalue::Float64`: P-value
- `q_stat::Float64`: Cochran's Q statistic
- `q_pvalue::Float64`: P-value for heterogeneity
- `i2::Float64`: I² statistic (% variance due to heterogeneity)
- `tau2::Float64`: Between-study variance (random effects)
- `n_studies::Int`: Number of studies
- `method::String`: Method used
"""
struct MetaAnalysisResult
    beta::Float64
    se::Float64
    z::Float64
    pvalue::Float64
    q_stat::Float64
    q_pvalue::Float64
    i2::Float64
    tau2::Float64
    n_studies::Int
    method::String
end

"""
    fixed_effects_meta(betas::Vector{Float64}, ses::Vector{Float64}) -> MetaAnalysisResult

Fixed-effects (inverse-variance weighted) meta-analysis.

# Arguments
- `betas`: Effect estimates from each study
- `ses`: Standard errors from each study

# Mathematical Details
The fixed-effects model assumes a single true effect size β:
β̂_k ~ N(β, se²_k)

Combined estimate:
β̂_FE = Σ_k (w_k * β̂_k) / Σ_k w_k

where w_k = 1/se²_k (inverse variance weights)

Standard error:
SE(β̂_FE) = 1 / √(Σ_k w_k)

# Example
```julia
# Combine GWAS results from 5 cohorts
result = fixed_effects_meta(betas_per_cohort, ses_per_cohort)
println("Combined beta: \$(result.beta) (SE: \$(result.se))")
println("P-value: \$(result.pvalue)")
```

# References
- Borenstein et al. (2009) Introduction to Meta-Analysis
- Evangelou & Ioannidis (2013) Nat. Rev. Genet.
"""
function fixed_effects_meta(betas::Vector{Float64}, ses::Vector{Float64})
    @assert length(betas) == length(ses) "betas and ses must have same length"

    k = length(betas)

    # Inverse variance weights
    weights = 1.0 ./ (ses.^2)

    # Combined estimate
    beta_fe = sum(weights .* betas) / sum(weights)

    # Standard error
    se_fe = 1.0 / sqrt(sum(weights))

    # Z-score and p-value
    z = beta_fe / se_fe
    pvalue = 2 * ccdf(Normal(), abs(z))

    # Heterogeneity statistics
    Q = sum(weights .* (betas .- beta_fe).^2)
    q_pvalue = ccdf(Chisq(k - 1), Q)

    # I² statistic
    i2 = max(0, (Q - (k - 1)) / Q) * 100

    return MetaAnalysisResult(
        beta_fe,
        se_fe,
        z,
        pvalue,
        Q,
        q_pvalue,
        i2,
        0.0,  # No tau² for fixed effects
        k,
        "Fixed Effects (IVW)"
    )
end

"""
    random_effects_meta(betas::Vector{Float64}, ses::Vector{Float64};
                       method=:dl) -> MetaAnalysisResult

Random-effects meta-analysis accounting for between-study heterogeneity.

# Arguments
- `betas`: Effect estimates from each study
- `ses`: Standard errors from each study
- `method`: Method for estimating τ² - :dl (DerSimonian-Laird), :reml, :pm (Paule-Mandel)

# Mathematical Details
The random-effects model allows study-specific true effects:
θ_k ~ N(μ, τ²)
β̂_k | θ_k ~ N(θ_k, se²_k)

Combined estimate:
β̂_RE = Σ_k (w*_k * β̂_k) / Σ_k w*_k

where w*_k = 1/(se²_k + τ²)

τ² is the between-study variance, estimated by DerSimonian-Laird:
τ²_DL = max(0, (Q - (k-1)) / (Σw - Σw²/Σw))

# Example
```julia
result = random_effects_meta(betas, ses; method=:dl)
println("Between-study heterogeneity τ² = \$(result.tau2)")
println("I² = \$(result.i2)%")
```

# References
- DerSimonian & Laird (1986) Control. Clin. Trials
- Veroniki et al. (2016) Res. Synth. Methods (τ² estimators comparison)
"""
function random_effects_meta(
    betas::Vector{Float64},
    ses::Vector{Float64};
    method::Symbol=:dl
)
    @assert length(betas) == length(ses)

    k = length(betas)

    # Fixed effects weights
    w = 1.0 ./ (ses.^2)
    beta_fe = sum(w .* betas) / sum(w)

    # Cochran's Q
    Q = sum(w .* (betas .- beta_fe).^2)

    # Estimate τ² (between-study variance)
    if method == :dl
        # DerSimonian-Laird
        C = sum(w) - sum(w.^2) / sum(w)
        tau2 = max(0, (Q - (k - 1)) / C)
    elseif method == :pm
        # Paule-Mandel
        tau2 = paule_mandel_tau2(betas, ses)
    elseif method == :reml
        # REML
        tau2 = reml_tau2(betas, ses)
    else
        error("Unknown method: $method")
    end

    # Random effects weights
    w_star = 1.0 ./ (ses.^2 .+ tau2)

    # Combined estimate
    beta_re = sum(w_star .* betas) / sum(w_star)
    se_re = 1.0 / sqrt(sum(w_star))

    # Z and p-value
    z = beta_re / se_re
    pvalue = 2 * ccdf(Normal(), abs(z))

    # Q p-value
    q_pvalue = ccdf(Chisq(max(k - 1, 1)), Q)

    # I²
    i2 = max(0, (Q - (k - 1)) / Q) * 100

    return MetaAnalysisResult(
        beta_re,
        se_re,
        z,
        pvalue,
        Q,
        q_pvalue,
        i2,
        tau2,
        k,
        "Random Effects ($method)"
    )
end

"""
    paule_mandel_tau2(betas, ses; max_iter=100, tol=1e-5) -> Float64

Paule-Mandel iterative estimator for τ².
"""
function paule_mandel_tau2(betas::Vector{Float64}, ses::Vector{Float64};
                          max_iter::Int=100, tol::Float64=1e-5)
    k = length(betas)
    tau2 = 0.0

    for _ in 1:max_iter
        w = 1.0 ./ (ses.^2 .+ tau2)
        beta_hat = sum(w .* betas) / sum(w)
        Q = sum(w .* (betas .- beta_hat).^2)

        # Solve Q = k - 1 for tau2
        if Q <= k - 1
            return 0.0
        end

        # Newton step
        dQ = -sum(w.^2 .* (betas .- beta_hat).^2)
        tau2_new = tau2 + (Q - (k - 1)) / (-dQ + 1e-10)
        tau2_new = max(0, tau2_new)

        if abs(tau2_new - tau2) < tol
            return tau2_new
        end
        tau2 = tau2_new
    end

    return tau2
end

"""
    reml_tau2(betas, ses; max_iter=100, tol=1e-5) -> Float64

REML estimator for τ².
"""
function reml_tau2(betas::Vector{Float64}, ses::Vector{Float64};
                  max_iter::Int=100, tol::Float64=1e-5)
    k = length(betas)
    tau2 = var(betas) - mean(ses.^2)
    tau2 = max(0, tau2)

    for _ in 1:max_iter
        w = 1.0 ./ (ses.^2 .+ tau2)
        beta_hat = sum(w .* betas) / sum(w)

        # REML equation
        numerator = sum(w.^2 .* ((betas .- beta_hat).^2 .- ses.^2))
        denominator = sum(w.^2)

        tau2_new = tau2 + numerator / (denominator + 1e-10)
        tau2_new = max(0, tau2_new)

        if abs(tau2_new - tau2) < tol
            return tau2_new
        end
        tau2 = tau2_new
    end

    return tau2
end

"""
    sample_size_weighted_meta(betas::Vector{Float64}, ns::Vector{Int},
                             eafs::Vector{Float64}) -> MetaAnalysisResult

Sample size weighted meta-analysis (METAL scheme 2).

# Arguments
- `betas`: Effect estimates (or Z-scores)
- `ns`: Sample sizes
- `eafs`: Effect allele frequencies

# Algorithm
Weights proportional to √(2 * N * EAF * (1-EAF))
This accounts for allele frequency differences across studies.

# References
- Willer et al. (2010) Bioinformatics (METAL)
"""
function sample_size_weighted_meta(
    zscores::Vector{Float64},
    ns::Vector{Int},
    eafs::Vector{Float64}
)
    k = length(zscores)

    # Weights
    weights = sqrt.(2 .* ns .* eafs .* (1 .- eafs))
    weights = weights ./ sum(weights)

    # Combined z-score
    z_combined = sum(weights .* zscores) / sqrt(sum(weights.^2))

    # P-value
    pvalue = 2 * ccdf(Normal(), abs(z_combined))

    # Heterogeneity (using z-scores)
    z_mean = sum(weights .* zscores)
    Q = sum((zscores .- z_mean).^2)
    q_pvalue = ccdf(Chisq(max(k - 1, 1)), Q)
    i2 = max(0, (Q - (k - 1)) / Q) * 100

    return MetaAnalysisResult(
        z_combined,  # Z-score, not beta
        1.0 / sqrt(sum(ns)),  # Approximate SE
        z_combined,
        pvalue,
        Q,
        q_pvalue,
        i2,
        0.0,
        k,
        "Sample-size weighted"
    )
end

"""
    trans_ancestry_meta(betas::Vector{Float64}, ses::Vector{Float64},
                       ancestries::Vector{String}; kwargs...) -> MetaAnalysisResult

Trans-ancestry meta-analysis with ancestry-aware weighting.

# Arguments
- `betas`: Effect estimates from each study
- `ses`: Standard errors
- `ancestries`: Ancestry labels for each study

# Keyword Arguments
- `method`: :fe (fixed), :re (random), :mantra (MANTRA-like)
- `prior_cor::Float64=0.8`: Prior correlation for effect sizes across ancestries

# Example
```julia
result = trans_ancestry_meta(betas, ses, ["EUR", "EUR", "EAS", "AFR"];
                            method=:mantra)
```

# References
- Morris (2011) Genet. Epidemiol. (MANTRA)
- Mägi et al. (2017) Am. J. Hum. Genet. (MR-MEGA)
"""
function trans_ancestry_meta(
    betas::Vector{Float64},
    ses::Vector{Float64},
    ancestries::Vector{String};
    method::Symbol=:re,
    prior_cor::Float64=0.8
)
    k = length(betas)
    unique_anc = unique(ancestries)
    n_anc = length(unique_anc)

    if method == :fe || method == :re
        # Standard meta-analysis
        if method == :fe
            return fixed_effects_meta(betas, ses)
        else
            return random_effects_meta(betas, ses)
        end
    elseif method == :mantra
        # MANTRA-like: allow ancestry-specific effects with correlation prior

        # Group studies by ancestry
        anc_groups = Dict{String, Vector{Int}}()
        for (i, anc) in enumerate(ancestries)
            if !haskey(anc_groups, anc)
                anc_groups[anc] = Int[]
            end
            push!(anc_groups[anc], i)
        end

        # Estimate ancestry-specific effects
        anc_betas = Dict{String, Float64}()
        anc_ses = Dict{String, Float64}()

        for (anc, indices) in anc_groups
            if length(indices) > 0
                result = fixed_effects_meta(betas[indices], ses[indices])
                anc_betas[anc] = result.beta
                anc_ses[anc] = result.se
            end
        end

        # Combine ancestry-specific effects with correlation prior
        # Using simplified approach: weighted average with shrinkage
        all_anc_betas = [anc_betas[anc] for anc in unique_anc]
        all_anc_ses = [anc_ses[anc] for anc in unique_anc]

        # Apply correlation-based shrinkage
        mean_beta = mean(all_anc_betas)
        shrunk_betas = prior_cor .* mean_beta .+ (1 - prior_cor) .* all_anc_betas

        # Final meta-analysis of shrunk estimates
        return fixed_effects_meta(shrunk_betas, all_anc_ses)
    end
end

"""
    mr_mega(betas::Vector{Float64}, ses::Vector{Float64},
           pc_loadings::Matrix{Float64}; kwargs...) -> NamedTuple

MR-MEGA: Meta-Regression of Multi-AncEstry Genetic Association.

# Arguments
- `betas`: Effect estimates
- `ses`: Standard errors
- `pc_loadings`: Ancestry PC loadings for each study (k × n_pcs)

# Keyword Arguments
- `n_pcs::Int=2`: Number of ancestry PCs to use

# Model
MR-MEGA models effect heterogeneity as a function of ancestry:
β_k = β_0 + Σ_j β_j * PC_{kj} + e_k

This allows testing for:
1. Overall association (β_0)
2. Ancestry-correlated heterogeneity (β_j)
3. Residual heterogeneity (e_k)

# Returns
Named tuple with overall association and heterogeneity components

# References
- Mägi et al. (2017) Am. J. Hum. Genet.
"""
function mr_mega(
    betas::Vector{Float64},
    ses::Vector{Float64},
    pc_loadings::Matrix{Float64};
    n_pcs::Int=2
)
    k = length(betas)
    pcs = pc_loadings[:, 1:min(n_pcs, size(pc_loadings, 2))]

    # Design matrix: [1, PC1, PC2, ...]
    X = hcat(ones(k), pcs)

    # Inverse variance weights
    W = Diagonal(1.0 ./ ses.^2)

    # Weighted least squares
    beta_hat = (X' * W * X) \ (X' * W * betas)
    var_beta = inv(X' * W * X)

    # Overall effect (intercept)
    beta_0 = beta_hat[1]
    se_0 = sqrt(var_beta[1, 1])
    z_0 = beta_0 / se_0
    p_0 = 2 * ccdf(Normal(), abs(z_0))

    # Ancestry-correlated heterogeneity test
    # Test H0: β_1 = β_2 = ... = 0
    if n_pcs > 0
        beta_anc = beta_hat[2:end]
        var_anc = var_beta[2:end, 2:end]
        chi2_anc = beta_anc' * inv(var_anc) * beta_anc
        p_anc = ccdf(Chisq(n_pcs), chi2_anc)
    else
        chi2_anc = 0.0
        p_anc = 1.0
    end

    # Residual heterogeneity
    predicted = X * beta_hat
    residuals = betas - predicted
    Q_res = sum((residuals ./ ses).^2)
    df_res = k - 1 - n_pcs
    p_res = ccdf(Chisq(max(df_res, 1)), Q_res)
    i2_res = max(0, (Q_res - df_res) / Q_res) * 100

    return (
        beta_overall = beta_0,
        se_overall = se_0,
        pvalue_overall = p_0,
        chi2_ancestry = chi2_anc,
        pvalue_ancestry = p_anc,
        chi2_residual = Q_res,
        pvalue_residual = p_res,
        i2_residual = i2_res,
        beta_pcs = beta_hat[2:end],
        n_studies = k
    )
end

"""
    gwas_meta_analysis(summary_stats::Vector{DataFrame};
                      snp_col=:SNP, beta_col=:BETA, se_col=:SE,
                      method=:fixed) -> DataFrame

Run meta-analysis across multiple GWAS summary statistics files.

# Arguments
- `summary_stats`: Vector of DataFrames with summary statistics

# Keyword Arguments
- `snp_col`: Column name for SNP identifiers
- `beta_col`: Column name for effect estimates
- `se_col`: Column name for standard errors
- `method`: Meta-analysis method (:fixed, :random)

# Returns
DataFrame with meta-analysis results for each SNP

# Example
```julia
# Load summary stats from multiple studies
stats = [CSV.read(f, DataFrame) for f in sumstat_files]

# Run meta-analysis
meta_results = gwas_meta_analysis(stats; method=:random)

# Filter significant results
significant = filter(r -> r.pvalue < 5e-8, meta_results)
```
"""
function gwas_meta_analysis(
    summary_stats::Vector{DataFrame};
    snp_col::Symbol=:SNP,
    beta_col::Symbol=:BETA,
    se_col::Symbol=:SE,
    method::Symbol=:fixed,
    verbose::Bool=true
)
    n_studies = length(summary_stats)

    # Find common SNPs
    all_snps = [Set(df[!, snp_col]) for df in summary_stats]
    common_snps = reduce(intersect, all_snps)

    if verbose
        println("Found $(length(common_snps)) SNPs common to all $n_studies studies")
    end

    results = DataFrame(
        SNP = String[],
        BETA = Float64[],
        SE = Float64[],
        Z = Float64[],
        P = Float64[],
        Q = Float64[],
        Q_P = Float64[],
        I2 = Float64[],
        N_STUDIES = Int[]
    )

    # Create lookup tables for each study
    lookups = [Dict(zip(df[!, snp_col], 1:nrow(df))) for df in summary_stats]

    if verbose
        prog = Progress(length(common_snps); desc="Meta-analyzing: ")
    end

    for snp in common_snps
        betas = Float64[]
        ses = Float64[]

        for (i, df) in enumerate(summary_stats)
            if haskey(lookups[i], snp)
                row_idx = lookups[i][snp]
                push!(betas, df[row_idx, beta_col])
                push!(ses, df[row_idx, se_col])
            end
        end

        if length(betas) >= 2 && all(isfinite.(betas)) && all(ses .> 0)
            if method == :fixed
                result = fixed_effects_meta(betas, ses)
            else
                result = random_effects_meta(betas, ses)
            end

            push!(results, (
                SNP = snp,
                BETA = result.beta,
                SE = result.se,
                Z = result.z,
                P = result.pvalue,
                Q = result.q_stat,
                Q_P = result.q_pvalue,
                I2 = result.i2,
                N_STUDIES = length(betas)
            ))
        end

        if verbose
            next!(prog)
        end
    end

    sort!(results, :P)

    return results
end

"""
    forest_plot_data(betas::Vector{Float64}, ses::Vector{Float64},
                    study_names::Vector{String}; method=:random) -> DataFrame

Prepare data for forest plot visualization.

# Arguments
- `betas`: Effect estimates
- `ses`: Standard errors
- `study_names`: Names of studies

# Returns
DataFrame suitable for forest plot with columns:
study, beta, se, lower_ci, upper_ci, weight
"""
function forest_plot_data(
    betas::Vector{Float64},
    ses::Vector{Float64},
    study_names::Vector{String};
    method::Symbol=:random
)
    k = length(betas)

    # Get meta-analysis result
    if method == :fixed
        meta_result = fixed_effects_meta(betas, ses)
    else
        meta_result = random_effects_meta(betas, ses)
    end

    # Compute weights
    if method == :fixed
        weights = 1.0 ./ ses.^2
    else
        weights = 1.0 ./ (ses.^2 .+ meta_result.tau2)
    end
    weights = weights ./ sum(weights) .* 100

    # 95% CI for each study
    z_crit = 1.96
    lower_ci = betas .- z_crit .* ses
    upper_ci = betas .+ z_crit .* ses

    df = DataFrame(
        study = study_names,
        beta = betas,
        se = ses,
        lower_ci = lower_ci,
        upper_ci = upper_ci,
        weight = weights
    )

    # Add meta-analysis summary
    push!(df, (
        study = "Combined ($(method))",
        beta = meta_result.beta,
        se = meta_result.se,
        lower_ci = meta_result.beta - z_crit * meta_result.se,
        upper_ci = meta_result.beta + z_crit * meta_result.se,
        weight = 100.0
    ))

    return df
end

"""
    leave_one_out_analysis(betas::Vector{Float64}, ses::Vector{Float64},
                          study_names::Vector{String}) -> DataFrame

Leave-one-out sensitivity analysis for meta-analysis.

# Arguments
- `betas`: Effect estimates
- `ses`: Standard errors
- `study_names`: Names of studies

# Returns
DataFrame showing meta-analysis result with each study excluded
"""
function leave_one_out_analysis(
    betas::Vector{Float64},
    ses::Vector{Float64},
    study_names::Vector{String}
)
    k = length(betas)

    results = DataFrame(
        excluded_study = String[],
        beta = Float64[],
        se = Float64[],
        pvalue = Float64[],
        i2 = Float64[]
    )

    for i in 1:k
        mask = [j != i for j in 1:k]
        result = random_effects_meta(betas[mask], ses[mask])

        push!(results, (
            excluded_study = study_names[i],
            beta = result.beta,
            se = result.se,
            pvalue = result.pvalue,
            i2 = result.i2
        ))
    end

    return results
end

"""
    publication_bias_test(betas::Vector{Float64}, ses::Vector{Float64}) -> NamedTuple

Test for publication bias using Egger's regression and related methods.

# Arguments
- `betas`: Effect estimates
- `ses`: Standard errors

# Returns
Named tuple with Egger's test results and funnel plot asymmetry

# Tests
1. Egger's regression: regress beta/se on 1/se
   Non-zero intercept suggests bias
2. Begg's rank correlation
3. Trim-and-fill estimate

# References
- Egger et al. (1997) BMJ
- Begg & Mazumdar (1994) Biometrics
"""
function publication_bias_test(betas::Vector{Float64}, ses::Vector{Float64})
    k = length(betas)

    # Egger's regression: Z = a + b * precision
    # where Z = beta/se and precision = 1/se
    z_scores = betas ./ ses
    precision = 1.0 ./ ses

    # Weighted regression
    X = hcat(ones(k), precision)
    W = Diagonal(ses.^2)  # Weight by variance

    beta_egger = (X' * W * X) \ (X' * W * z_scores)
    resid = z_scores - X * beta_egger
    σ2 = sum(resid.^2 .* diag(W)) / (k - 2)
    var_beta = σ2 * inv(X' * W * X)

    intercept = beta_egger[1]
    se_intercept = sqrt(var_beta[1, 1])
    t_stat = intercept / se_intercept
    p_egger = 2 * ccdf(TDist(k - 2), abs(t_stat))

    # Begg's rank correlation
    # Correlation between effect size and variance
    ranks_beta = sortperm(sortperm(betas))
    ranks_var = sortperm(sortperm(ses.^2))
    tau_begg = cor(ranks_beta, ranks_var)

    return (
        egger_intercept = intercept,
        egger_se = se_intercept,
        egger_pvalue = p_egger,
        begg_tau = tau_begg,
        n_studies = k
    )
end
