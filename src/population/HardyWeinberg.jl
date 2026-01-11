# ============================================================================
# HardyWeinberg.jl - Hardy-Weinberg Equilibrium Analysis
# ============================================================================
# Implements Hardy-Weinberg equilibrium testing and related statistics
# as described in the population genetics chapters of the Handbook.
# ============================================================================

"""
    allele_frequencies(genotypes::AbstractVector)

Calculate allele frequencies from genotype data.

# Arguments
- `genotypes`: Vector of genotype values (0, 1, 2, or missing)

# Returns
- NamedTuple with `p` (reference allele freq) and `q` (alternate allele freq)

# Example
```julia
genos = [0, 1, 2, 1, 0, 2]
freqs = allele_frequencies(genos)
# freqs.p ≈ 0.5, freqs.q ≈ 0.5
```
"""
function allele_frequencies(genotypes::AbstractVector)
    valid_genos = collect(skipmissing(genotypes))
    n = length(valid_genos)
    
    if n == 0
        return (p=NaN, q=NaN, n=0)
    end
    
    # Count alternate alleles (genotype value = count of alt allele)
    alt_count = sum(valid_genos)
    total_alleles = 2 * n
    
    q = alt_count / total_alleles  # Alternate allele frequency
    p = 1 - q                       # Reference allele frequency
    
    return (p=p, q=q, n=n)
end

"""
    genotype_frequencies(genotypes::AbstractVector)

Calculate observed genotype frequencies.

# Returns
- NamedTuple with frequencies of AA (f_aa), Aa (f_Aa), and aa (f_aa)
"""
function genotype_frequencies(genotypes::AbstractVector)
    valid_genos = collect(skipmissing(genotypes))
    n = length(valid_genos)
    
    if n == 0
        return (f_AA=NaN, f_Aa=NaN, f_aa=NaN, n_AA=0, n_Aa=0, n_aa=0, n=0)
    end
    
    n_AA = count(==(0), valid_genos)
    n_Aa = count(==(1), valid_genos)
    n_aa = count(==(2), valid_genos)
    
    return (
        f_AA = n_AA / n,
        f_Aa = n_Aa / n,
        f_aa = n_aa / n,
        n_AA = n_AA,
        n_Aa = n_Aa,
        n_aa = n_aa,
        n = n
    )
end

"""
    hwe_expected_frequencies(p::Real)

Calculate expected genotype frequencies under Hardy-Weinberg equilibrium.

# Arguments
- `p`: Reference allele frequency

# Returns
- NamedTuple with expected frequencies
"""
function hwe_expected_frequencies(p::Real)
    q = 1 - p
    return (
        f_AA = p^2,
        f_Aa = 2 * p * q,
        f_aa = q^2
    )
end

"""
    hwe_test(genotypes::AbstractVector; method::Symbol=:chisq)

Test for Hardy-Weinberg equilibrium.

# Arguments
- `genotypes`: Vector of genotypes (0, 1, 2)
- `method`: Test method (:chisq, :exact, or :likelihood)

# Returns
- StatisticalTestResult with test statistic and p-value

# Example
```julia
genos = [0, 1, 2, 1, 0, 2, 1, 1, 0, 1]
result = hwe_test(genos)
# result.pvalue > 0.05 suggests HWE
```
"""
function hwe_test(genotypes::AbstractVector; method::Symbol=:chisq)
    obs = genotype_frequencies(genotypes)
    
    if obs.n < 5
        @warn "Sample size too small for reliable HWE test"
    end
    
    # Calculate allele frequencies
    af = allele_frequencies(genotypes)
    exp_freq = hwe_expected_frequencies(af.p)
    
    if method == :chisq
        return hwe_chisq_test(obs, exp_freq)
    elseif method == :exact
        return hwe_exact_test(obs.n_AA, obs.n_Aa, obs.n_aa)
    elseif method == :likelihood
        return hwe_likelihood_ratio_test(obs, af)
    else
        throw(ArgumentError("Unknown method: $method"))
    end
end

"""
    hwe_chisq_test(obs::NamedTuple, exp_freq::NamedTuple)

Chi-squared test for Hardy-Weinberg equilibrium.
"""
function hwe_chisq_test(obs::NamedTuple, exp_freq::NamedTuple)
    n = obs.n
    
    # Expected counts
    e_AA = exp_freq.f_AA * n
    e_Aa = exp_freq.f_Aa * n
    e_aa = exp_freq.f_aa * n
    
    # Check for valid expected counts
    if any(x -> x < 5, [e_AA, e_Aa, e_aa])
        @warn "Expected counts < 5, consider using exact test"
    end
    
    # Chi-squared statistic
    chi2 = (obs.n_AA - e_AA)^2 / e_AA +
           (obs.n_Aa - e_Aa)^2 / e_Aa +
           (obs.n_aa - e_aa)^2 / e_aa
    
    # 1 degree of freedom (3 categories - 1 estimated parameter - 1)
    df = 1
    pval = ccdf(Chisq(df), chi2)
    
    StatisticalTestResult(chi2, pval, df, "HWE Chi-squared test")
end

"""
    hwe_exact_test(n_AA::Int, n_Aa::Int, n_aa::Int)

Exact test for Hardy-Weinberg equilibrium using complete enumeration.

This implements the exact test described by Wigginton et al. (2005).
"""
function hwe_exact_test(n_AA::Int, n_Aa::Int, n_aa::Int)
    n = n_AA + n_Aa + n_aa
    n_A = 2 * n_AA + n_Aa  # Count of A alleles
    n_a = 2 * n_aa + n_Aa  # Count of a alleles
    
    # For exact test, enumerate all possible heterozygote counts
    # given the observed allele counts
    
    # Probability of observed configuration
    function log_hwe_prob(n_het::Int)
        # Given n_het heterozygotes, calculate homozygote counts
        n_hom_A = (n_A - n_het) ÷ 2
        n_hom_a = (n_a - n_het) ÷ 2
        
        # Log probability under HWE
        log_num = logfactorial(n) + log(2) * n_het
        log_denom = (logfactorial(n_hom_A) + logfactorial(n_het) + 
                    logfactorial(n_hom_a) + logfactorial(2*n))
        
        return log_num - log_denom + logfactorial(n_A) + logfactorial(n_a)
    end
    
    # Helper for log factorial
    logfactorial(x::Int) = x <= 1 ? 0.0 : sum(log(i) for i in 2:x)
    
    # Observed probability
    obs_log_prob = log_hwe_prob(n_Aa)
    
    # Enumerate all possible heterozygote counts
    # n_het must have same parity as min(n_A, n_a)
    min_het = (n_A % 2 == 0) ? 0 : 1
    max_het = min(n_A, n_a)
    
    # Sum probabilities for configurations as or more extreme
    total_log_prob = -Inf
    extreme_log_prob = -Inf
    
    for n_het in min_het:2:max_het
        lp = log_hwe_prob(n_het)
        total_log_prob = logaddexp(total_log_prob, lp)
        
        if lp <= obs_log_prob + 1e-10
            extreme_log_prob = logaddexp(extreme_log_prob, lp)
        end
    end
    
    pval = exp(extreme_log_prob - total_log_prob)
    pval = clamp(pval, 0.0, 1.0)
    
    # Calculate heterozygosity excess/deficit as test statistic
    expected_het = 2 * (n_A / (2*n)) * (n_a / (2*n)) * n
    het_stat = (n_Aa - expected_het) / sqrt(expected_het)
    
    StatisticalTestResult(het_stat, pval, nothing, "HWE Exact test")
end

"""Log-sum-exp helper for numerical stability."""
function logaddexp(a::Float64, b::Float64)
    if a == -Inf
        return b
    elseif b == -Inf
        return a
    elseif a > b
        return a + log1p(exp(b - a))
    else
        return b + log1p(exp(a - b))
    end
end

"""
    hwe_likelihood_ratio_test(obs::NamedTuple, af::NamedTuple)

Likelihood ratio test for Hardy-Weinberg equilibrium.
"""
function hwe_likelihood_ratio_test(obs::NamedTuple, af::NamedTuple)
    n = obs.n
    
    # Log-likelihood under HWE
    exp_freq = hwe_expected_frequencies(af.p)
    ll_hwe = obs.n_AA * log(exp_freq.f_AA + 1e-10) +
             obs.n_Aa * log(exp_freq.f_Aa + 1e-10) +
             obs.n_aa * log(exp_freq.f_aa + 1e-10)
    
    # Log-likelihood under observed frequencies (saturated model)
    ll_sat = obs.n_AA * log(obs.f_AA + 1e-10) +
             obs.n_Aa * log(obs.f_Aa + 1e-10) +
             obs.n_aa * log(obs.f_aa + 1e-10)
    
    # Likelihood ratio statistic
    G = 2 * (ll_sat - ll_hwe)
    G = max(G, 0.0)  # Ensure non-negative
    
    df = 1
    pval = ccdf(Chisq(df), G)
    
    StatisticalTestResult(G, pval, df, "HWE Likelihood ratio test")
end

"""
    inbreeding_coefficient(genotypes::AbstractVector)

Calculate the inbreeding coefficient (F_IS) from genotype data.

F_IS = 1 - (observed heterozygosity / expected heterozygosity)

# Returns
- Float64: Inbreeding coefficient (-1 to 1)
"""
function inbreeding_coefficient(genotypes::AbstractVector)
    obs = genotype_frequencies(genotypes)
    af = allele_frequencies(genotypes)
    
    if af.n == 0 || af.p == 0 || af.p == 1
        return NaN
    end
    
    exp_het = 2 * af.p * af.q
    obs_het = obs.f_Aa
    
    f_is = 1 - (obs_het / exp_het)
    
    return f_is
end

"""
    hwe_test_matrix(gm::GenotypeMatrix; method::Symbol=:chisq)

Perform HWE test for all variants in a genotype matrix.

# Returns
- Vector of StatisticalTestResult for each variant
"""
function hwe_test_matrix(gm::GenotypeMatrix; method::Symbol=:chisq)
    n_vars = n_variants(gm)
    results = Vector{StatisticalTestResult}(undef, n_vars)
    
    for j in 1:n_vars
        results[j] = hwe_test(gm.data[:, j]; method=method)
    end
    
    return results
end

"""
    filter_hwe(gm::GenotypeMatrix, threshold::Float64=1e-6)

Filter variants failing Hardy-Weinberg equilibrium test.

# Arguments
- `gm`: GenotypeMatrix
- `threshold`: P-value threshold for filtering

# Returns
- Indices of variants passing HWE filter
"""
function filter_hwe(gm::GenotypeMatrix, threshold::Float64=1e-6)
    results = hwe_test_matrix(gm; method=:exact)
    passing_idx = findall(r -> r.pvalue >= threshold, results)
    return passing_idx
end
