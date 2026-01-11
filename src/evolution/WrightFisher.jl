# ============================================================================
# WrightFisher.jl - Wright-Fisher Population Model
# ============================================================================

"""
    wright_fisher_simulate(N::Int, p0::Float64, n_generations::Int; 
                          μ::Float64=0.0, s::Float64=0.0)

Simulate allele frequency trajectory under Wright-Fisher model.

# Arguments
- `N`: Effective population size
- `p0`: Initial allele frequency
- `n_generations`: Number of generations to simulate
- `μ`: Mutation rate (per generation)
- `s`: Selection coefficient (fitness = 1 + s for selected allele)

# Returns
- Vector of allele frequencies over generations
"""
function wright_fisher_simulate(N::Int, p0::Float64, n_generations::Int;
                               μ::Float64=0.0, s::Float64=0.0)
    @assert 0 <= p0 <= 1 "Initial frequency must be in [0,1]"
    @assert N > 0 "Population size must be positive"
    
    frequencies = Vector{Float64}(undef, n_generations + 1)
    frequencies[1] = p0
    p = p0
    
    for gen in 1:n_generations
        # Apply selection (if any)
        if s != 0
            # Marginal fitness: w̄ = p²(1+s) + 2p(1-p)(1+s/2) + (1-p)²
            # Simplified: p' = p(1+s) / (1 + p*s) for dominant selection
            w_bar = p^2 * (1 + s) + 2 * p * (1 - p) + (1 - p)^2
            p = p * (1 + s * p + s/2 * (1 - p)) / w_bar
        end
        
        # Apply mutation (if any)
        if μ > 0
            p = p * (1 - μ) + (1 - p) * μ
        end
        
        # Genetic drift: binomial sampling
        n_copies = rand(Binomial(2 * N, p))
        p = n_copies / (2 * N)
        
        frequencies[gen + 1] = p
        
        # Stop if fixed
        if p == 0.0 || p == 1.0
            frequencies[(gen + 1):end] .= p
            break
        end
    end
    
    return frequencies
end

"""
    wright_fisher_trajectory(N::Int, p0::Float64, n_generations::Int, n_replicates::Int)

Simulate multiple Wright-Fisher trajectories.

# Returns
- Matrix of frequencies (n_generations+1 × n_replicates)
"""
function wright_fisher_trajectory(N::Int, p0::Float64, n_generations::Int, 
                                 n_replicates::Int; kwargs...)
    trajectories = Matrix{Float64}(undef, n_generations + 1, n_replicates)
    
    for rep in 1:n_replicates
        trajectories[:, rep] = wright_fisher_simulate(N, p0, n_generations; kwargs...)
    end
    
    return trajectories
end

"""
    fixation_probability(N::Int, p0::Float64; s::Float64=0.0, n_simulations::Int=1000)

Estimate fixation probability by simulation.
"""
function fixation_probability(N::Int, p0::Float64; s::Float64=0.0, 
                             n_simulations::Int=1000)
    n_fixed = 0
    max_gen = 10 * N  # Sufficient time for fixation
    
    for _ in 1:n_simulations
        traj = wright_fisher_simulate(N, p0, max_gen; s=s)
        if traj[end] == 1.0
            n_fixed += 1
        end
    end
    
    return n_fixed / n_simulations
end

"""
    fixation_probability_theory(N::Int, p0::Float64, s::Float64=0.0)

Theoretical fixation probability (Kimura's formula).

For neutral alleles: π = p0
For selected alleles: π = (1 - exp(-4Ns*p0)) / (1 - exp(-4Ns))
"""
function fixation_probability_theory(N::Int, p0::Float64, s::Float64=0.0)
    if s == 0.0
        return p0
    else
        if abs(4 * N * s) < 1e-6
            return p0  # Nearly neutral
        end
        num = 1 - exp(-4 * N * s * p0)
        denom = 1 - exp(-4 * N * s)
        return num / denom
    end
end

"""
    expected_fixation_time(N::Int, p0::Float64; conditional::Bool=true)

Expected time to fixation under neutral Wright-Fisher model.

# Arguments
- `N`: Effective population size
- `p0`: Initial frequency
- `conditional`: If true, time conditional on fixation; else unconditional
"""
function expected_fixation_time(N::Int, p0::Float64; conditional::Bool=true)
    if conditional
        # Conditional on fixation (Ewens 1979)
        return -4 * N * (1 - p0) * log(1 - p0) / p0
    else
        # Unconditional (weighted average)
        t_fix = -4 * N * (1 - p0) * log(1 - p0) / p0
        t_loss = -4 * N * p0 * log(p0) / (1 - p0)
        return p0 * t_fix + (1 - p0) * t_loss
    end
end

"""
    heterozygosity_decay(N::Int, H0::Float64, n_generations::Int)

Expected heterozygosity decay under drift.

H_t = H_0 * (1 - 1/(2N))^t
"""
function heterozygosity_decay(N::Int, H0::Float64, n_generations::Int)
    t = 0:n_generations
    decay_factor = (1 - 1 / (2 * N)) .^ t
    return H0 .* decay_factor
end

"""
    simulate_multilocus(N::Int, n_loci::Int, n_generations::Int;
                       initial_freqs::Vector{Float64}=Float64[])

Simulate Wright-Fisher model for multiple independent loci.
"""
function simulate_multilocus(N::Int, n_loci::Int, n_generations::Int;
                            initial_freqs::Vector{Float64}=Float64[])
    if isempty(initial_freqs)
        initial_freqs = rand(n_loci)
    end
    
    @assert length(initial_freqs) == n_loci
    
    frequencies = Matrix{Float64}(undef, n_generations + 1, n_loci)
    
    for j in 1:n_loci
        frequencies[:, j] = wright_fisher_simulate(N, initial_freqs[j], n_generations)
    end
    
    return frequencies
end

"""
    coalescent_effective_size(sample_frequencies::AbstractMatrix, generations::AbstractVector)

Estimate effective population size from temporal allele frequency changes.
Uses Nei & Tajima (1981) temporal method.
"""
function coalescent_effective_size(freq_t1::AbstractVector, freq_t2::AbstractVector, 
                                  t::Int)
    @assert length(freq_t1) == length(freq_t2)
    
    # Calculate F (standardized variance)
    F_values = Float64[]
    
    for i in eachindex(freq_t1)
        p1, p2 = freq_t1[i], freq_t2[i]
        
        # Skip fixed loci
        if p1 == 0 || p1 == 1 || p2 == 0 || p2 == 1
            continue
        end
        
        # F = (p1 - p2)² / ((p1 + p2)/2 - p1*p2)
        p_mean = (p1 + p2) / 2
        F = (p1 - p2)^2 / (p_mean * (1 - p_mean))
        push!(F_values, F)
    end
    
    if isempty(F_values)
        return NaN
    end
    
    F_mean = mean(F_values)
    
    # Ne = t / (2 * F)
    Ne = t / (2 * F_mean)
    
    return Ne
end
