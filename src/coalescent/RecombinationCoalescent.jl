# ============================================================================
# RecombinationCoalescent.jl - ARG and Recombination
# ============================================================================

"""
    ARGNode

Node in an Ancestral Recombination Graph.
"""
struct ARGNode
    id::Int
    parent_ids::Vector{Int}     # Multiple parents possible (after recombination)
    children_ids::Vector{Int}
    time::Float64
    node_type::Symbol           # :sample, :coalescent, :recombination
    breakpoint::Union{Float64, Nothing}  # For recombination nodes
end

"""
    AncestralRecombinationGraph

Represents an ARG with recombination events.
"""
struct AncestralRecombinationGraph
    nodes::Vector{ARGNode}
    n_samples::Int
    n_recombinations::Int
    sequence_length::Float64
end

"""
    coalescent_with_recombination(n::Int, ρ::Float64, L::Float64=1.0; Ne::Float64=1.0)

Simulate coalescent with recombination.

# Arguments
- `n`: Number of samples
- `ρ`: Population-scaled recombination rate (4Ne*r*L)
- `L`: Sequence length
- `Ne`: Effective population size
"""
function coalescent_with_recombination(n::Int, ρ::Float64, L::Float64=1.0; Ne::Float64=1.0)
    nodes = ARGNode[]
    node_id = 0
    
    # Initialize sample nodes
    active_lineages = Int[]
    for i in 1:n
        node_id += 1
        push!(nodes, ARGNode(node_id, Int[], Int[], 0.0, :sample, nothing))
        push!(active_lineages, node_id)
    end
    
    current_time = 0.0
    n_recombs = 0
    
    while length(active_lineages) > 1
        k = length(active_lineages)
        
        # Rate of coalescence
        coal_rate = k * (k - 1) / (4 * Ne)
        
        # Rate of recombination (proportional to number of lineages and sequence length)
        recomb_rate = k * ρ / (4 * Ne)
        
        total_rate = coal_rate + recomb_rate
        
        # Time to next event
        wait_time = rand(Exponential(1 / total_rate))
        current_time += wait_time
        
        # Determine event type
        if rand() < coal_rate / total_rate
            # Coalescence event
            node_id += 1
            
            # Choose two lineages to coalesce
            idx1, idx2 = sample(1:k, 2, replace=false)
            lin1, lin2 = active_lineages[idx1], active_lineages[idx2]
            
            push!(nodes, ARGNode(node_id, Int[], [lin1, lin2], current_time, 
                                :coalescent, nothing))
            
            # Update active lineages
            deleteat!(active_lineages, sort([idx1, idx2], rev=true))
            push!(active_lineages, node_id)
        else
            # Recombination event
            n_recombs += 1
            
            # Choose lineage to recombine
            idx = rand(1:k)
            lin = active_lineages[idx]
            
            # Random breakpoint
            breakpoint = rand() * L
            
            # Create two new lineages (ancestral to left and right of breakpoint)
            node_id += 1
            left_id = node_id
            push!(nodes, ARGNode(node_id, Int[], [lin], current_time, 
                                :recombination, breakpoint))
            
            node_id += 1
            right_id = node_id
            push!(nodes, ARGNode(node_id, Int[], [lin], current_time,
                                :recombination, breakpoint))
            
            # Update active lineages
            active_lineages[idx] = left_id
            push!(active_lineages, right_id)
        end
    end
    
    AncestralRecombinationGraph(nodes, n, n_recombs, L)
end

"""
    arg_simulate(n::Int; ρ::Float64=10.0, L::Float64=1000.0, Ne::Float64=1.0)

Wrapper for ARG simulation with sensible defaults.
"""
function arg_simulate(n::Int; ρ::Float64=10.0, L::Float64=1000.0, Ne::Float64=1.0)
    coalescent_with_recombination(n, ρ, L; Ne=Ne)
end

"""
    extract_local_tree(arg::AncestralRecombinationGraph, position::Float64)

Extract the local tree at a specific genomic position from the ARG.
"""
function extract_local_tree(arg::AncestralRecombinationGraph, position::Float64)
    @assert 0 <= position <= arg.sequence_length
    
    # Track which nodes are relevant for this position
    # (simplified implementation - full ARG traversal is complex)
    
    # For now, return coalescence times from relevant nodes
    coal_times = Float64[]
    
    for node in arg.nodes
        if node.node_type == :coalescent
            push!(coal_times, node.time)
        end
    end
    
    sort!(coal_times)
    
    # Return as simplified CoalescentTree
    if isempty(coal_times)
        return CoalescentTree(arg.n_samples, Float64[], 0.0, 0.0)
    end
    
    tmrca = coal_times[end]
    total_branch = sum(coal_times)  # Simplified
    
    CoalescentTree(arg.n_samples, coal_times, tmrca, total_branch)
end

"""
    recombination_rate_estimate(haplotypes::Matrix{Int8}, positions::Vector{Float64})

Estimate recombination rate from haplotype data using composite likelihood.
"""
function recombination_rate_estimate(haplotypes::Matrix{Int8}, positions::Vector{Float64})
    n_haps, n_vars = size(haplotypes)
    
    if n_vars < 2
        return (rho=NaN, se=NaN)
    end
    
    # Use LD decay to estimate recombination
    ld_values = Float64[]
    distances = Float64[]
    
    for i in 1:(n_vars-1)
        for j in (i+1):min(i+50, n_vars)  # Limit range for efficiency
            dist = positions[j] - positions[i]
            
            # Calculate r² from haplotype data
            h1 = haplotypes[:, i]
            h2 = haplotypes[:, j]
            
            p1 = mean(h1)
            p2 = mean(h2)
            
            if p1 > 0 && p1 < 1 && p2 > 0 && p2 < 1
                D = mean(h1 .* h2) - p1 * p2
                r2 = D^2 / (p1 * (1 - p1) * p2 * (1 - p2))
                
                push!(ld_values, r2)
                push!(distances, dist)
            end
        end
    end
    
    if length(ld_values) < 10
        return (rho=NaN, se=NaN)
    end
    
    # Fit LD decay model: E[r²] ≈ 1/(1 + 4ρd)
    # Use log transformation for regression
    
    # Simple regression: log(1/r² - 1) ≈ log(4ρ) + log(d)
    valid_idx = findall(r -> 0 < r < 1, ld_values)
    
    if length(valid_idx) < 10
        return (rho=NaN, se=NaN)
    end
    
    y = log.(1.0 ./ ld_values[valid_idx] .- 1)
    x = log.(distances[valid_idx])
    
    # Simple linear model
    n = length(y)
    x_mean = mean(x)
    y_mean = mean(y)
    
    slope = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
    intercept = y_mean - slope * x_mean
    
    # ρ ≈ exp(intercept) / 4
    rho = exp(intercept) / 4
    
    # Standard error (simplified)
    residuals = y .- (intercept .+ slope .* x)
    se_resid = sqrt(sum(residuals.^2) / (n - 2))
    se_rho = rho * se_resid  # Approximate
    
    return (rho=rho, se=se_rho)
end

"""
    hudson_estimator(n_haps::Int, n_segregating::Int, min_n_recomb::Int)

Hudson's estimator of recombination rate.
R_M = min number of recombination events (four-gamete test)
"""
function hudson_estimator(n_haps::Int, n_segregating::Int, min_n_recomb::Int)
    if n_segregating <= 1
        return 0.0
    end
    
    # ρ ≈ R_M / expected coalescent tree length
    # Simplified: ρ ≈ R_M / (4 * harmonic_number)
    a_n = sum(1.0 / i for i in 1:(n_haps - 1))
    
    return min_n_recomb / (4 * a_n)
end

"""
    minimum_recombination_events(haplotypes::Matrix{Int8})

Count minimum number of recombination events using four-gamete test.
"""
function minimum_recombination_events(haplotypes::Matrix{Int8})
    n_haps, n_vars = size(haplotypes)
    
    min_recombs = 0
    
    for i in 1:(n_vars-1)
        for j in (i+1):n_vars
            # Check for all four gametes
            gametes = Set{Tuple{Int8, Int8}}()
            
            for h in 1:n_haps
                push!(gametes, (haplotypes[h, i], haplotypes[h, j]))
            end
            
            if length(gametes) == 4
                min_recombs += 1
                break  # Count each position once
            end
        end
    end
    
    return min_recombs
end
