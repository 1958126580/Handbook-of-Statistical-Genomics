# ============================================================================
# BasicCoalescent.jl - Kingman's Coalescent
# ============================================================================

"""
    CoalescentTree

Represents a coalescent genealogy.
"""
struct CoalescentTree
    n_samples::Int
    coalescence_times::Vector{Float64}  # Times of coalescence events
    tree_height::Float64                 # Total height (TMRCA)
    total_branch_length::Float64         # Sum of all branch lengths
end

"""
    coalescent_simulate(n::Int; Ne::Float64=1.0)

Simulate a coalescent tree for n samples.

# Arguments
- `n`: Number of samples
- `Ne`: Effective population size (scales time)

# Returns
- CoalescentTree with coalescence times
"""
function coalescent_simulate(n::Int; Ne::Float64=1.0)
    @assert n >= 2 "Need at least 2 samples"
    
    coal_times = Float64[]
    total_branch = 0.0
    current_time = 0.0
    k = n  # Current number of lineages
    
    while k > 1
        # Rate of coalescence
        rate = k * (k - 1) / 2
        
        # Time to next coalescence (exponential)
        wait_time = rand(Exponential(2 * Ne / rate))
        
        current_time += wait_time
        push!(coal_times, current_time)
        
        # Branch length contribution
        total_branch += k * wait_time
        
        k -= 1
    end
    
    tmrca = current_time
    
    CoalescentTree(n, coal_times, tmrca, total_branch)
end

"""
    time_to_mrca(n::Int; Ne::Float64=1.0, n_simulations::Int=10000)

Estimate expected time to most recent common ancestor.

Theoretical expectation: E[TMRCA] = 2Ne(1 - 1/n)
"""
function time_to_mrca(n::Int; Ne::Float64=1.0, n_simulations::Int=10000)
    tmrca_values = Float64[]
    
    for _ in 1:n_simulations
        tree = coalescent_simulate(n; Ne=Ne)
        push!(tmrca_values, tree.tree_height)
    end
    
    theoretical = 2 * Ne * (1 - 1/n)
    
    (empirical_mean=mean(tmrca_values), 
     empirical_sd=std(tmrca_values),
     theoretical_mean=theoretical)
end

"""
    expected_branch_lengths(n::Int; Ne::Float64=1.0)

Calculate expected total branch length for coalescent tree.

E[L] = 2Ne * Σ(1/i) for i=1 to n-1
"""
function expected_branch_lengths(n::Int; Ne::Float64=1.0)
    harmonic = sum(1.0 / i for i in 1:(n-1))
    return 2 * Ne * harmonic
end

"""
    expected_pairwise_coalescence(; Ne::Float64=1.0)

Expected coalescence time for two random lineages.

E[T2] = 2Ne
"""
expected_pairwise_coalescence(; Ne::Float64=1.0) = 2 * Ne

"""
    simulate_mutations_on_tree(tree::CoalescentTree, θ::Float64)

Place mutations on coalescent tree according to infinite sites model.

# Arguments
- `tree`: CoalescentTree
- `θ`: Population-scaled mutation rate (4Neμ)

# Returns
- Number of segregating sites and their frequencies
"""
function simulate_mutations_on_tree(tree::CoalescentTree, θ::Float64)
    # Expected number of mutations
    expected_muts = θ * tree.total_branch_length / 2
    n_mutations = rand(Poisson(expected_muts))
    
    # For each mutation, sample branch length uniformly
    # and determine which samples carry the mutation
    mutation_freqs = Float64[]
    
    n = tree.n_samples
    
    for _ in 1:n_mutations
        # Sample position on tree
        pos = rand() * tree.total_branch_length
        
        # Determine number of descendants
        cumulative = 0.0
        n_descendants = n
        k = n
        
        for (i, coal_time) in enumerate(tree.coalescence_times)
            prev_time = i == 1 ? 0.0 : tree.coalescence_times[i-1]
            branch_length = (coal_time - prev_time) * k
            
            if cumulative + branch_length >= pos
                # Mutation on branch with k lineages
                # Random number of descendants (simplified)
                n_descendants = rand(1:k)
                break
            end
            
            cumulative += branch_length
            k -= 1
        end
        
        push!(mutation_freqs, n_descendants / n)
    end
    
    return (n_mutations=n_mutations, frequencies=mutation_freqs)
end

"""
    site_frequency_spectrum(n::Int, θ::Float64; n_simulations::Int=1000)

Simulate site frequency spectrum under coalescent.

# Returns
- Expected counts in each frequency class (singletons, doubletons, etc.)
"""
function site_frequency_spectrum(n::Int, θ::Float64; n_simulations::Int=1000)
    # SFS bins: 1, 2, ..., n-1 derived alleles
    sfs = zeros(n - 1)
    
    for _ in 1:n_simulations
        tree = coalescent_simulate(n)
        muts = simulate_mutations_on_tree(tree, θ)
        
        for freq in muts.frequencies
            count = round(Int, freq * n)
            if 1 <= count <= n - 1
                sfs[count] += 1
            end
        end
    end
    
    # Expected SFS under neutrality: θ/i for i derived alleles
    expected_sfs = [θ / i for i in 1:(n-1)]
    
    return (observed=sfs / n_simulations, expected=expected_sfs)
end

"""
    coalescent_likelihood(tree::CoalescentTree, Ne::Float64)

Calculate likelihood of coalescent tree given effective population size.
"""
function coalescent_likelihood(tree::CoalescentTree, Ne::Float64)
    log_lik = 0.0
    n = tree.n_samples
    k = n
    prev_time = 0.0
    
    for coal_time in tree.coalescence_times
        rate = k * (k - 1) / (4 * Ne)
        wait_time = coal_time - prev_time
        
        # Exponential density
        log_lik += log(rate) - rate * wait_time
        
        prev_time = coal_time
        k -= 1
    end
    
    return log_lik
end
