# ============================================================================
# Drift.jl - Genetic Drift Analysis
# ============================================================================

"""
    genetic_drift_simulate(N::Int, n_generations::Int, n_loci::Int;
                          initial_freq::Float64=0.5)

Simulate genetic drift for multiple independent loci.

# Returns
- Matrix of allele frequencies (generations × loci)
"""
function genetic_drift_simulate(N::Int, n_generations::Int, n_loci::Int;
                               initial_freq::Float64=0.5)
    freqs = Matrix{Float64}(undef, n_generations + 1, n_loci)
    
    for locus in 1:n_loci
        freqs[:, locus] = wright_fisher_simulate(N, initial_freq, n_generations)
    end
    
    return freqs
end

"""
    effective_population_size(freqs_t1::AbstractVector, freqs_t2::AbstractVector, 
                             generations::Int)

Estimate effective population size using temporal method.

Based on Nei & Tajima (1981) temporal variance method.

# Arguments
- `freqs_t1`: Allele frequencies at time 1
- `freqs_t2`: Allele frequencies at time 2
- `generations`: Number of generations between samples
"""
function effective_population_size(freqs_t1::AbstractVector, freqs_t2::AbstractVector,
                                  generations::Int)
    @assert length(freqs_t1) == length(freqs_t2)
    
    F_values = Float64[]
    
    for i in eachindex(freqs_t1)
        p1, p2 = freqs_t1[i], freqs_t2[i]
        
        # Skip fixed or nearly fixed loci
        if p1 < 0.05 || p1 > 0.95 || p2 < 0.05 || p2 > 0.95
            continue
        end
        
        # Standardized frequency variance
        p_mean = (p1 + p2) / 2
        F = (p1 - p2)^2 / (2 * p_mean * (1 - p_mean))
        push!(F_values, F)
    end
    
    if isempty(F_values)
        return NaN
    end
    
    F_mean = mean(F_values)
    F_se = std(F_values) / sqrt(length(F_values))
    
    # Ne = t / (2 * (F - 1/(2S)))  where S is sample size
    # Simplified: Ne ≈ t / (2 * F)
    Ne = generations / (2 * F_mean)
    Ne_lower = generations / (2 * (F_mean + 1.96 * F_se))
    Ne_upper = generations / (2 * max(F_mean - 1.96 * F_se, 0.01))
    
    return (Ne=Ne, lower=Ne_lower, upper=Ne_upper, n_loci=length(F_values))
end

"""
    bottleneck_detect(heterozygosity_history::AbstractVector; window::Int=10)

Detect population bottlenecks from heterozygosity time series.

# Returns
- Indices of detected bottleneck events
"""
function bottleneck_detect(heterozygosity_history::AbstractVector; window::Int=10)
    n = length(heterozygosity_history)
    if n < 2 * window
        return Int[]
    end
    
    bottlenecks = Int[]
    
    for i in (window+1):(n-window)
        # Compare heterozygosity before and at this point
        h_before = mean(heterozygosity_history[(i-window):(i-1)])
        h_at = heterozygosity_history[i]
        h_after = mean(heterozygosity_history[(i+1):(i+window)])
        
        # Bottleneck: sharp drop followed by recovery
        drop_ratio = h_before > 0 ? h_at / h_before : 1.0
        recovery_ratio = h_at > 0 ? h_after / h_at : 1.0
        
        if drop_ratio < 0.7 && recovery_ratio > 1.1
            push!(bottlenecks, i)
        end
    end
    
    return bottlenecks
end

"""
    coalescent_Ne_estimate(sample_size::Int, n_segregating_sites::Int, 
                          sequence_length::Int)

Estimate effective population size from sequence data using Watterson's estimator.

θ_W = S / a_n, where a_n = Σ(1/i) for i=1 to n-1
Ne = θ_W / (4μ)

# Returns
- Estimate of θ (given unknown μ, this is proportional to Ne)
"""
function coalescent_Ne_estimate(sample_size::Int, n_segregating_sites::Int,
                               sequence_length::Int)
    n = sample_size
    if n <= 1
        return (theta_w=NaN, se=NaN)
    end
    
    # Harmonic number
    a_n = sum(1.0 / i for i in 1:(n-1))
    
    # Watterson's estimator of θ per site
    theta_w = n_segregating_sites / (a_n * sequence_length)
    
    # Variance (for SE calculation)
    b_n = sum(1.0 / i^2 for i in 1:(n-1))
    var_S = a_n * theta_w + b_n * theta_w^2
    se_theta = sqrt(var_S) / (a_n * sequence_length)
    
    return (theta_w=theta_w, se=se_theta)
end

"""
    tajima_D(n_samples::Int, n_segregating_sites::Int, mean_pairwise_diff::Float64)

Calculate Tajima's D statistic for testing neutral evolution.

D = (π - θ_W) / sqrt(Var(π - θ_W))

Positive D: Balancing selection or population contraction
Negative D: Positive selection or population expansion
Zero: Neutral evolution
"""
function tajima_D(n_samples::Int, n_segregating_sites::Int, mean_pairwise_diff::Float64)
    n = n_samples
    S = n_segregating_sites
    π = mean_pairwise_diff
    
    if S == 0
        return (D=0.0, pvalue=1.0)
    end
    
    # Harmonic numbers
    a1 = sum(1.0 / i for i in 1:(n-1))
    a2 = sum(1.0 / i^2 for i in 1:(n-1))
    
    # Coefficients for variance calculation
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n^2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1^2
    e1 = c1 / a1
    e2 = c2 / (a1^2 + a2)
    
    # Watterson's theta
    theta_W = S / a1
    
    # Variance of D
    var_d = e1 * S + e2 * S * (S - 1)
    
    if var_d <= 0
        return (D=0.0, pvalue=1.0)
    end
    
    # Tajima's D
    D = (π - theta_W) / sqrt(var_d)
    
    # Approximate p-value using beta distribution (Tajima 1989)
    # Under neutrality, D follows approximately beta distribution
    pvalue = 2 * min(ccdf(Normal(), abs(D)), 0.5)  # Two-tailed
    
    return (D=D, pvalue=pvalue)
end

"""
    founder_effect(ancestral_freqs::AbstractVector, n_founders::Int)

Simulate founder effect by sampling from ancestral population.

# Arguments
- `ancestral_freqs`: Allele frequencies in source population
- `n_founders`: Number of founding individuals

# Returns
- New allele frequencies after founder event
"""
function founder_effect(ancestral_freqs::AbstractVector, n_founders::Int)
    n_alleles = 2 * n_founders
    
    new_freqs = similar(ancestral_freqs, Float64)
    
    for i in eachindex(ancestral_freqs)
        # Sample founder alleles
        n_derived = rand(Binomial(n_alleles, ancestral_freqs[i]))
        new_freqs[i] = n_derived / n_alleles
    end
    
    return new_freqs
end

"""
    heterozygosity_loss_rate(Ne::Int)

Calculate expected rate of heterozygosity loss per generation.

H_t = H_0 * (1 - 1/(2Ne))^t
Rate = 1/(2Ne)
"""
heterozygosity_loss_rate(Ne::Int) = 1 / (2 * Ne)

"""
    time_to_fixation(Ne::Int; p0::Float64=0.5)

Expected time to fixation for a neutral allele.
"""
function time_to_fixation(Ne::Int; p0::Float64=0.5)
    # Expected time conditional on fixation (in generations)
    if p0 == 0.0 || p0 == 1.0
        return 0.0
    end
    
    return -4 * Ne * (1 - p0) * log(1 - p0) / p0
end

"""
    variance_effective_size(population_sizes::AbstractVector)

Calculate variance effective population size from fluctuating census sizes.

1/Ne = (1/t) * Σ(1/Ni)

This accounts for the disproportionate effect of small population sizes.
"""
function variance_effective_size(population_sizes::AbstractVector)
    t = length(population_sizes)
    harmonic_mean = t / sum(1.0 / N for N in population_sizes)
    return harmonic_mean
end

"""
    inbreeding_effective_size(population_sizes::AbstractVector, variance_sizes::AbstractVector)

Estimate effective size accounting for variance in reproductive success.

Ne = (N * k̄ - 1) / (k̄ - 1 + Vk/k̄)

where k̄ = mean offspring number, Vk = variance in offspring number
"""
function inbreeding_effective_size(N::Int, mean_offspring::Float64, var_offspring::Float64)
    if mean_offspring == 0
        return 0.0
    end
    
    Ne = (N * mean_offspring - 1) / (mean_offspring - 1 + var_offspring / mean_offspring)
    return Ne
end
