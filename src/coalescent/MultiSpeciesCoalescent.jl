# ============================================================================
# MultiSpeciesCoalescent.jl - Gene Tree / Species Tree
# ============================================================================

"""
    SpeciesTree

Representation of a species tree with population sizes.
"""
struct SpeciesTree
    n_species::Int
    species_names::Vector{String}
    divergence_times::Vector{Float64}
    population_sizes::Vector{Float64}  # Ne for each branch
    tree_structure::Dict{Int, Vector{Int}}  # Parent -> children mapping
end

"""
    multispecies_coalescent(species_tree::SpeciesTree, samples_per_species::Vector{Int})

Simulate gene tree under multispecies coalescent model.

# Arguments
- `species_tree`: Species tree with divergence times and population sizes
- `samples_per_species`: Number of samples from each species
"""
function multispecies_coalescent(species_tree::SpeciesTree, 
                                samples_per_species::Vector{Int})
    @assert length(samples_per_species) == species_tree.n_species
    
    n_total = sum(samples_per_species)
    
    # Gene tree coalescence times
    coal_times = Float64[]
    lineage_assignments = Dict{Int, Int}()  # Sample -> current species
    
    # Initialize: each sample belongs to its species
    sample_id = 0
    for (sp, n_samp) in enumerate(samples_per_species)
        for _ in 1:n_samp
            sample_id += 1
            lineage_assignments[sample_id] = sp
        end
    end
    
    current_time = 0.0
    n_lineages = n_total
    
    # Process time intervals between species divergences
    div_times = sort(species_tree.divergence_times)
    push!(div_times, Inf)  # Add infinite time at end
    
    for (interval_idx, next_div_time) in enumerate(div_times)
        # Coalescence within each species until next divergence
        while n_lineages > 1 && current_time < next_div_time
            # Count lineages per species
            lineages_per_species = Dict{Int, Vector{Int}}()
            for (lin, sp) in lineage_assignments
                if !haskey(lineages_per_species, sp)
                    lineages_per_species[sp] = Int[]
                end
                push!(lineages_per_species[sp], lin)
            end
            
            # Calculate total coalescence rate
            total_rate = 0.0
            for (sp, lins) in lineages_per_species
                k = length(lins)
                if k >= 2
                    Ne = species_tree.population_sizes[min(sp, length(species_tree.population_sizes))]
                    total_rate += k * (k - 1) / (4 * Ne)
                end
            end
            
            if total_rate == 0
                break
            end
            
            # Time to next event
            wait_time = rand(Exponential(1 / total_rate))
            
            if current_time + wait_time > next_div_time
                current_time = next_div_time
                break
            end
            
            current_time += wait_time
            push!(coal_times, current_time)
            
            # Choose which species has coalescence (proportional to rate)
            r = rand() * total_rate
            cumulative = 0.0
            chosen_sp = 1
            
            for (sp, lins) in lineages_per_species
                k = length(lins)
                if k >= 2
                    Ne = species_tree.population_sizes[min(sp, length(species_tree.population_sizes))]
                    rate = k * (k - 1) / (4 * Ne)
                    cumulative += rate
                    if cumulative >= r
                        chosen_sp = sp
                        break
                    end
                end
            end
            
            # Coalesce two lineages in chosen species
            lins = lineages_per_species[chosen_sp]
            idx1, idx2 = sample(1:length(lins), 2, replace=false)
            lin1, lin2 = lins[idx1], lins[idx2]
            
            # Remove one lineage (merge into the other)
            delete!(lineage_assignments, lin2)
            n_lineages -= 1
        end
        
        # At species divergence, merge populations
        # (in forward time, populations split; in backward time, they merge)
        # This is handled by the tree structure
    end
    
    # Final coalescence for remaining lineages
    while n_lineages > 1
        rate = n_lineages * (n_lineages - 1) / (4 * species_tree.population_sizes[1])
        wait_time = rand(Exponential(1 / rate))
        current_time += wait_time
        push!(coal_times, current_time)
        n_lineages -= 1
    end
    
    sort!(coal_times)
    tmrca = isempty(coal_times) ? 0.0 : coal_times[end]
    
    CoalescentTree(n_total, coal_times, tmrca, sum(coal_times))
end

"""
    gene_tree_species_tree(species_tree::SpeciesTree, n_gene_trees::Int;
                          samples_per_species::Int=2)

Simulate multiple gene trees and analyze discordance with species tree.
"""
function gene_tree_species_tree(species_tree::SpeciesTree, n_gene_trees::Int;
                               samples_per_species::Int=2)
    samp_vec = fill(samples_per_species, species_tree.n_species)
    
    gene_trees = [multispecies_coalescent(species_tree, samp_vec) 
                  for _ in 1:n_gene_trees]
    
    # Calculate statistics
    tmrca_values = [gt.tree_height for gt in gene_trees]
    branch_lengths = [gt.total_branch_length for gt in gene_trees]
    
    (gene_trees=gene_trees,
     mean_tmrca=mean(tmrca_values),
     sd_tmrca=std(tmrca_values),
     mean_branch_length=mean(branch_lengths))
end

"""
    incomplete_lineage_sorting_probability(τ::Float64, Ne::Float64)

Calculate probability of ILS for a species divergence.

P(ILS) = 2/3 * exp(-τ/(2Ne))
"""
function incomplete_lineage_sorting_probability(τ::Float64, Ne::Float64)
    return (2/3) * exp(-τ / (2 * Ne))
end

"""
    ancestral_polymorphism_probability(τ::Float64, Ne::Float64)

Probability that a polymorphism predates a species divergence.
"""
function ancestral_polymorphism_probability(τ::Float64, Ne::Float64)
    return exp(-τ / (2 * Ne))
end
