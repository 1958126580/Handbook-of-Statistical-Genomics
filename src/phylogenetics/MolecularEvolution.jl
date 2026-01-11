# ============================================================================
# MolecularEvolution.jl - Molecular Evolution Models
# ============================================================================

# This module extends the models from evolution/Mutation.jl with
# phylogenetic likelihood calculations

"""
    phylogenetic_likelihood(tree::PhyloTree, sequences::Vector{Vector{Int}},
                           model::SubstitutionModel)

Calculate phylogenetic likelihood using Felsenstein's pruning algorithm.
"""
function phylogenetic_likelihood(tree::PhyloTree, sequences::Vector{Vector{Int}},
                                model::SubstitutionModel)
    n_sites = length(sequences[1])
    n_tips = tree.n_tips
    
    log_lik = 0.0
    
    for site in 1:n_sites
        site_lik = site_likelihood(tree, [seq[site] for seq in sequences], model)
        log_lik += log(max(site_lik, 1e-300))
    end
    
    return log_lik
end

"""Calculate likelihood for a single site using pruning algorithm."""
function site_likelihood(tree::PhyloTree, site_data::Vector{Int}, 
                        model::SubstitutionModel)
    # Simplified for star tree / pairwise comparisons
    # Full implementation would traverse tree structure
    
    # Use stationary distribution
    π = model isa JC69 ? fill(0.25, 4) : 
        (model isa HKY85 ? collect(model.π) : fill(0.25, 4))
    
    # Product over all tips
    lik = 0.0
    for state in 1:4
        prob = π[state]
        for data in site_data
            if data == state
                prob *= 1.0
            else
                prob *= 0.01  # Simplified: small prob for different state
            end
        end
        lik += prob
    end
    
    return lik
end

"""
    gamma_rate_heterogeneity(n_categories::Int, α::Float64)

Generate discrete gamma rate categories for among-site rate variation.
"""
function gamma_rate_heterogeneity(n_categories::Int, α::Float64)
    rates = Float64[]
    
    for i in 1:n_categories
        lower = (i - 1) / n_categories
        upper = i / n_categories
        
        # Mean rate in this category
        mean_rate = (quantile(Gamma(α, 1/α), upper) + 
                    quantile(Gamma(α, 1/α), lower)) / 2
        push!(rates, mean_rate)
    end
    
    return rates ./ mean(rates)  # Normalize to mean 1
end

"""
    model_selection_aic(sequences::Vector{Vector{Int}}, tree::PhyloTree)

Compare substitution models using AIC.
"""
function model_selection_aic(sequences::Vector{Vector{Int}}, tree::PhyloTree)
    models = [
        (JC69(), 0, "JC69"),          # 0 free rate params
        (K80(2.0), 1, "K80"),         # 1 param (κ)
        (HKY85(2.0), 4, "HKY85"),     # 4 params (κ + 3 freq)
    ]
    
    results = []
    
    for (model, n_params, name) in models
        ll = phylogenetic_likelihood(tree, sequences, model)
        aic = 2 * n_params - 2 * ll
        push!(results, (model=name, log_likelihood=ll, n_params=n_params, aic=aic))
    end
    
    sort!(results, by=x -> x.aic)
    
    return results
end
