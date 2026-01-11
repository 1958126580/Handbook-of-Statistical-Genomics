# ============================================================================
# PowerCalculations.jl - Statistical Power and Sample Size Calculations
# ============================================================================
# Comprehensive power calculations for various genetic study designs
# Including GWAS, rare variant, heritability, and PRS studies
# ============================================================================

"""
    PowerResult

Structure containing power calculation results.

# Fields
- `power::Float64`: Statistical power (0-1)
- `sample_size::Int`: Sample size used
- `alpha::Float64`: Significance level
- `effect_size::Float64`: Effect size
- `maf::Float64`: Minor allele frequency
- `study_type::String`: Type of study
- `parameters::Dict`: Additional parameters
"""
struct PowerResult
    power::Float64
    sample_size::Int
    alpha::Float64
    effect_size::Float64
    maf::Float64
    study_type::String
    parameters::Dict{Symbol, Any}
end

"""
    gwas_power(n::Int, maf::Float64, beta::Float64, sigma::Float64;
              alpha::Float64=5e-8) -> Float64

Calculate power for single-variant GWAS test.

# Arguments
- `n`: Sample size
- `maf`: Minor allele frequency
- `beta`: Effect size (regression coefficient)
- `sigma`: Residual standard deviation
- `alpha`: Significance threshold (genome-wide: 5e-8)

# Mathematical Details
For quantitative trait GWAS with additive model:
y = β₀ + β₁G + ε, ε ~ N(0, σ²)

The non-centrality parameter for the test of β₁ = 0:
λ = n × β² × 2p(1-p) / σ²

where p is the MAF.

Power = P(χ²₁(λ) > χ²_{1,1-α})

# Example
```julia
# Power for detecting effect β=0.1 with MAF=0.2 in 10,000 samples
power = gwas_power(10000, 0.2, 0.1, 1.0; alpha=5e-8)
println("Power: \$(round(power*100, digits=1))%")
```

# References
- Sham & Purcell (2014) Nat. Rev. Genet.
"""
function gwas_power(
    n::Int,
    maf::Float64,
    beta::Float64,
    sigma::Float64;
    alpha::Float64=5e-8
)
    # Variance explained by SNP
    var_g = 2 * maf * (1 - maf) * beta^2

    # Non-centrality parameter
    ncp = n * var_g / sigma^2

    # Critical value
    crit = quantile(Chisq(1), 1 - alpha)

    # Power
    power = ccdf(NoncentralChisq(1, ncp), crit)

    return power
end

"""
    gwas_sample_size(power::Float64, maf::Float64, beta::Float64, sigma::Float64;
                    alpha::Float64=5e-8) -> Int

Calculate required sample size for GWAS.

# Arguments
- `power`: Desired power (0-1)
- `maf`: Minor allele frequency
- `beta`: Effect size
- `sigma`: Residual SD
- `alpha`: Significance level

# Returns
Required sample size

# Example
```julia
# Sample size for 80% power
n = gwas_sample_size(0.8, 0.2, 0.1, 1.0; alpha=5e-8)
println("Required N = \$n")
```
"""
function gwas_sample_size(
    power::Float64,
    maf::Float64,
    beta::Float64,
    sigma::Float64;
    alpha::Float64=5e-8
)
    # Binary search for sample size
    n_low = 100
    n_high = 10_000_000

    while n_high - n_low > 10
        n_mid = div(n_low + n_high, 2)
        current_power = gwas_power(n_mid, maf, beta, sigma; alpha=alpha)

        if current_power < power
            n_low = n_mid
        else
            n_high = n_mid
        end
    end

    return n_high
end

"""
    case_control_power(n_cases::Int, n_controls::Int, maf::Float64, or::Float64;
                      alpha::Float64=5e-8, prevalence::Float64=0.1) -> Float64

Calculate power for case-control GWAS.

# Arguments
- `n_cases`: Number of cases
- `n_controls`: Number of controls
- `maf`: Minor allele frequency (in population)
- `or`: Odds ratio for risk allele
- `alpha`: Significance threshold
- `prevalence`: Disease prevalence

# Mathematical Details
Under multiplicative model:
P(disease | G=g) = K × OR^g / [1 - K + K × OR^g × (1 + p(OR-1))^(-1)]

For large samples, the test statistic follows:
χ² ~ χ²₁(λ) where λ = n × (p₁ - p₀)² × r(1-r) / [p̄(1-p̄)]

with p₁, p₀ being MAF in cases and controls, r = n_cases/n.

# Example
```julia
power = case_control_power(5000, 5000, 0.2, 1.3; prevalence=0.05)
```

# References
- Skol et al. (2006) Nat. Genet.
"""
function case_control_power(
    n_cases::Int,
    n_controls::Int,
    maf::Float64,
    or::Float64;
    alpha::Float64=5e-8,
    prevalence::Float64=0.1
)
    p = maf  # Population MAF
    K = prevalence

    # MAF in cases and controls under multiplicative model
    # P(G=1 | case) = P(case | G=1) × P(G=1) / P(case)
    # P(case | G=g) ∝ K × OR^g

    # Genotype frequencies in population
    p_aa = (1 - p)^2
    p_ab = 2 * p * (1 - p)
    p_bb = p^2

    # Penetrances
    f0 = K  # Baseline penetrance (approximation)
    f1 = K * or / (p_aa + p_ab * or + p_bb * or^2)
    f2 = K * or^2 / (p_aa + p_ab * or + p_bb * or^2)

    # Normalize
    mean_f = f0 * p_aa + f1 * p_ab + f2 * p_bb
    f0 = f0 * K / mean_f
    f1 = f1 * K / mean_f
    f2 = f2 * K / mean_f

    # Allele frequency in cases
    p_case = (f1 * p_ab + 2 * f2 * p_bb) / (2 * K)

    # Allele frequency in controls
    p_ctrl = ((1 - f0) * (1 - p)^2 * 2 * (1 - p) +
              (1 - f1) * 2 * p * (1 - p) * (1 + (1-p)) +
              (1 - f2) * p^2 * 2 * p) / (2 * (1 - K))
    p_ctrl = (p - K * p_case) / (1 - K)

    # Non-centrality parameter
    n_total = n_cases + n_controls
    r = n_cases / n_total
    p_bar = r * p_case + (1 - r) * p_ctrl

    ncp = n_total * (p_case - p_ctrl)^2 * r * (1 - r) / (p_bar * (1 - p_bar) + 1e-10)

    # Power
    crit = quantile(Chisq(1), 1 - alpha)
    power = ccdf(NoncentralChisq(1, max(0, ncp)), crit)

    return power
end

"""
    rare_variant_power(n::Int, n_variants::Int, maf_mean::Float64,
                      effect_mean::Float64; kwargs...) -> Float64

Calculate power for rare variant burden/SKAT tests.

# Arguments
- `n`: Sample size
- `n_variants`: Number of rare variants in region
- `maf_mean`: Mean MAF of variants
- `effect_mean`: Mean effect size of causal variants

# Keyword Arguments
- `causal_proportion::Float64=0.2`: Proportion of variants that are causal
- `effect_sd::Float64=0`: SD of effect sizes (0 = fixed effect)
- `test_type::Symbol=:burden`: Test type - :burden, :skat, :skat_o
- `alpha::Float64=2.5e-6`: Significance level (gene-level)

# Mathematical Details
For burden test:
- Aggregate variants: B = Σ_j w_j G_j
- Test variance: Var(B) = n × Σ_j w_j² × 2p_j(1-p_j)
- NCP depends on total causal signal

For SKAT:
- Uses mixture of chi-squared distribution
- Power depends on eigenvalues of variance component

# Example
```julia
# Power for gene with 20 rare variants
power = rare_variant_power(5000, 20, 0.005, 0.3; causal_proportion=0.3)
```

# References
- Wu et al. (2011) Am. J. Hum. Genet.
- Lee et al. (2012) Am. J. Hum. Genet. (power calculations)
"""
function rare_variant_power(
    n::Int,
    n_variants::Int,
    maf_mean::Float64,
    effect_mean::Float64;
    causal_proportion::Float64=0.2,
    effect_sd::Float64=0.0,
    test_type::Symbol=:burden,
    alpha::Float64=2.5e-6,
    sigma::Float64=1.0
)
    # Generate variant-specific MAFs (from truncated exponential)
    mafs = [maf_mean * (1 + 0.5 * randn()) for _ in 1:n_variants]
    mafs = clamp.(mafs, 1e-5, 0.05)

    # Determine causal variants
    n_causal = round(Int, n_variants * causal_proportion)
    causal_idx = randperm(n_variants)[1:n_causal]

    # Generate effect sizes for causal variants
    effects = zeros(n_variants)
    for j in causal_idx
        if effect_sd > 0
            effects[j] = effect_mean + effect_sd * randn()
        else
            effects[j] = effect_mean
        end
    end

    if test_type == :burden
        # Burden test power
        # Total variance explained
        var_g = sum(2 .* mafs .* (1 .- mafs) .* effects.^2)

        # NCP
        ncp = n * var_g / sigma^2

        crit = quantile(Chisq(1), 1 - alpha)
        power = ccdf(NoncentralChisq(1, ncp), crit)

    elseif test_type == :skat
        # SKAT power (simplified)
        # Uses sum of independent chi-squared

        var_components = 2 .* mafs .* (1 .- mafs) .* effects.^2
        ncp = n * sum(var_components) / sigma^2

        # SKAT statistic approximation
        crit = quantile(Chisq(n_causal), 1 - alpha)
        power = ccdf(NoncentralChisq(n_causal, ncp), crit)

    elseif test_type == :skat_o
        # SKAT-O: combination of burden and SKAT
        # Use optimistic estimate
        power_burden = rare_variant_power(n, n_variants, maf_mean, effect_mean;
                                          causal_proportion=causal_proportion,
                                          test_type=:burden, alpha=alpha)
        power_skat = rare_variant_power(n, n_variants, maf_mean, effect_mean;
                                        causal_proportion=causal_proportion,
                                        test_type=:skat, alpha=alpha)
        power = max(power_burden, power_skat) * 0.95  # Slight penalty for multiple testing
    end

    return power
end

"""
    heritability_power(n::Int, h2::Float64, n_snps::Int;
                      alpha::Float64=0.05, method=:greml) -> Float64

Calculate power to detect heritability.

# Arguments
- `n`: Sample size
- `h2`: True heritability
- `n_snps`: Number of SNPs used
- `alpha`: Significance level
- `method`: Estimation method - :greml, :ldsc

# Mathematical Details
For GREML:
SE(h²) ≈ √(2/n) × (1 + h² × (n/M))

where M is effective number of independent SNPs.

Power = Φ(h²/SE - z_{α/2})

# Example
```julia
power = heritability_power(10000, 0.3, 500000)
```

# References
- Visscher et al. (2014) PLoS Genet. (GREML power)
"""
function heritability_power(
    n::Int,
    h2::Float64,
    n_snps::Int;
    alpha::Float64=0.05,
    method::Symbol=:greml
)
    if method == :greml
        # GREML SE approximation
        # Account for effective number of independent markers
        M_eff = min(n_snps, n - 1)  # Can't have more independent markers than samples

        se_h2 = sqrt(2 / n) * (1 + h2 * n / M_eff)

    elseif method == :ldsc
        # LDSC SE is typically larger
        se_h2 = sqrt(2 / n) * 1.5  # Approximate inflation factor
    end

    # Z-test for h² > 0
    z = h2 / se_h2
    z_crit = quantile(Normal(), 1 - alpha/2)

    power = cdf(Normal(), z - z_crit) + cdf(Normal(), -z - z_crit)

    return power
end

"""
    prs_power(n_gwas::Int, n_target::Int, h2::Float64, r2_prs::Float64;
             alpha::Float64=0.05) -> Float64

Calculate power for PRS prediction in target sample.

# Arguments
- `n_gwas`: Sample size of discovery GWAS
- `n_target`: Sample size of target sample
- `h2`: Heritability of trait
- `r2_prs`: Expected R² of PRS (from h² and GWAS power)
- `alpha`: Significance level

# Mathematical Details
The expected R² of PRS depends on:
- Discovery GWAS sample size (determines precision of weights)
- Number of causal variants
- LD structure

Power to detect PRS association:
NCP = n_target × r2_prs / (1 - r2_prs)

For incremental R²:
F-statistic with df = (1, n_target - p - 1)

# Example
```julia
power = prs_power(100000, 5000, 0.5, 0.1)
```

# References
- Dudbridge (2013) PLoS Genet.
"""
function prs_power(
    n_gwas::Int,
    n_target::Int,
    h2::Float64,
    r2_prs::Float64;
    alpha::Float64=0.05,
    n_covariates::Int=5
)
    # Non-centrality parameter for F-test
    df1 = 1  # PRS adds 1 predictor
    df2 = n_target - n_covariates - 2

    # F statistic under alternative
    # E[F] = (df2 × r2_prs) / (df1 × (1 - r2_prs)) + 1
    ncp = df2 * r2_prs / (1 - r2_prs + 1e-10)

    # Critical value
    f_crit = quantile(FDist(df1, df2), 1 - alpha)

    # Power
    power = ccdf(NoncentralF(df1, df2, ncp), f_crit)

    return power
end

"""
    expected_prs_r2(n_gwas::Int, h2::Float64, n_causal::Int;
                  p_threshold::Float64=5e-8) -> Float64

Calculate expected PRS R² given GWAS and trait parameters.

# Arguments
- `n_gwas`: Discovery GWAS sample size
- `h2`: Trait heritability
- `n_causal`: Number of causal variants
- `p_threshold`: P-value threshold for SNP selection

# Mathematical Details
R²_PRS = h² × [power × (1 - FDR)]

where power is the average power to detect causal variants
and FDR accounts for false discoveries.

# References
- Wray et al. (2013) Nat. Rev. Genet.
"""
function expected_prs_r2(
    n_gwas::Int,
    h2::Float64,
    n_causal::Int;
    p_threshold::Float64=5e-8,
    n_snps_total::Int=1_000_000
)
    # Average effect size per variant (assuming equal effects)
    var_per_snp = h2 / n_causal

    # Average MAF (assume 0.1-0.5)
    avg_maf = 0.25

    # Effect size
    beta_sq = var_per_snp / (2 * avg_maf * (1 - avg_maf))
    beta = sqrt(beta_sq)

    # Power to detect individual variants
    avg_power = gwas_power(n_gwas, avg_maf, beta, sqrt(1 - h2); alpha=p_threshold)

    # Expected number of true positives
    n_tp = n_causal * avg_power

    # Expected number of false positives (genome-wide)
    n_null = n_snps_total - n_causal
    n_fp = n_null * p_threshold

    # Precision
    precision = n_tp / (n_tp + n_fp + 1e-10)

    # Expected R²
    # True signal captured
    signal_captured = avg_power * h2

    # Noise from false positives dilutes signal
    r2_expected = signal_captured * precision

    return r2_expected
end

"""
    genetic_correlation_power(n1::Int, n2::Int, rg::Float64, h2_1::Float64, h2_2::Float64;
                             alpha::Float64=0.05, method=:ldsc) -> Float64

Calculate power to detect genetic correlation.

# Arguments
- `n1`: Sample size for trait 1
- `n2`: Sample size for trait 2
- `rg`: True genetic correlation
- `h2_1`: Heritability of trait 1
- `h2_2`: Heritability of trait 2
- `method`: Method - :ldsc, :gcta

# Example
```julia
power = genetic_correlation_power(50000, 100000, 0.3, 0.4, 0.5)
```

# References
- Bulik-Sullivan et al. (2015) Nat. Genet.
"""
function genetic_correlation_power(
    n1::Int,
    n2::Int,
    rg::Float64,
    h2_1::Float64,
    h2_2::Float64;
    alpha::Float64=0.05,
    method::Symbol=:ldsc,
    n_snps::Int=1_000_000
)
    if method == :ldsc
        # LDSC SE approximation
        # SE(rg) ≈ √(1 + rg²×n_snps/(n1×n2×h2_1×h2_2)) / √(n_snps)
        se_rg = sqrt(1 / n_snps + rg^2 / (sqrt(n1 * n2) * h2_1 * h2_2 + 1e-10))
    else
        # GCTA bivariate GREML
        se_rg = sqrt(2 / min(n1, n2)) * (1 + abs(rg))
    end

    # Z-test
    z = rg / se_rg
    z_crit = quantile(Normal(), 1 - alpha/2)

    power = cdf(Normal(), z - z_crit) + cdf(Normal(), -z - z_crit)

    return power
end

"""
    finemapping_power(n::Int, causal_idx::Int, r2_with_tag::Float64,
                     pip_threshold::Float64=0.95) -> Float64

Calculate power to fine-map causal variant.

# Arguments
- `n`: Sample size
- `causal_idx`: Index of causal variant in region
- `r2_with_tag`: R² between causal and best tagging variant
- `pip_threshold`: PIP threshold for declaring causal

# Example
```julia
power = finemapping_power(100000, 1, 0.8)
```
"""
function finemapping_power(
    n::Int,
    causal_effect::Float64,
    maf::Float64,
    r2_with_tag::Float64;
    pip_threshold::Float64=0.95,
    n_variants_in_ld::Int=10
)
    # Power depends on ability to distinguish causal from correlated variants
    # In perfect LD (r²=1), can't distinguish
    # As r² decreases, can distinguish better

    # Approximate: PIP concentrates on causal as n → ∞
    # Rate of concentration depends on LD structure

    # NCP for causal variant
    var_g = 2 * maf * (1 - maf) * causal_effect^2
    ncp_causal = n * var_g

    # NCP for correlated variant
    ncp_tag = ncp_causal * r2_with_tag

    # Ability to distinguish: difference in log Bayes factors
    delta_lbf = (ncp_causal - ncp_tag) / 2

    # Probability causal has highest PIP
    # Approximately: proportional to exp(delta_lbf) vs sum over LD variants
    prob_correct = 1 / (1 + (n_variants_in_ld - 1) * exp(-delta_lbf))

    # Power = P(correct) × P(PIP > threshold | correct)
    # Simplified: assume P(PIP > threshold | correct) ≈ 1 for large n
    power = prob_correct

    return power
end

"""
    power_summary_plot_data(n_range::AbstractRange, maf::Float64, beta::Float64;
                          alpha::Float64=5e-8) -> DataFrame

Generate data for power curve visualization.

# Arguments
- `n_range`: Range of sample sizes
- `maf`: Minor allele frequency
- `beta`: Effect size
- `alpha`: Significance level

# Returns
DataFrame with columns: n, power

# Example
```julia
data = power_summary_plot_data(1000:1000:100000, 0.2, 0.05)
# Plot power curve
```
"""
function power_summary_plot_data(
    n_range::AbstractRange,
    maf::Float64,
    beta::Float64;
    alpha::Float64=5e-8,
    sigma::Float64=1.0
)
    powers = [gwas_power(n, maf, beta, sigma; alpha=alpha) for n in n_range]

    return DataFrame(
        n = collect(n_range),
        power = powers
    )
end

"""
    sample_size_table(power_targets::Vector{Float64}, maf_values::Vector{Float64},
                     beta::Float64; alpha::Float64=5e-8) -> DataFrame

Generate sample size requirements table.

# Arguments
- `power_targets`: Target power levels (e.g., [0.5, 0.8, 0.9])
- `maf_values`: MAF values to consider
- `beta`: Effect size
- `alpha`: Significance level

# Returns
DataFrame with sample sizes for each power/MAF combination
"""
function sample_size_table(
    power_targets::Vector{Float64},
    maf_values::Vector{Float64},
    beta::Float64;
    alpha::Float64=5e-8,
    sigma::Float64=1.0
)
    results = DataFrame(MAF = maf_values)

    for power in power_targets
        col_name = "Power_$(round(Int, power*100))%"
        sample_sizes = [gwas_sample_size(power, maf, beta, sigma; alpha=alpha) for maf in maf_values]
        results[!, Symbol(col_name)] = sample_sizes
    end

    return results
end
