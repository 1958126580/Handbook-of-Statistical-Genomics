# ============================================================================
# DiversityIndices.jl - Microbial Diversity Measures
# ============================================================================

"""
    shannon_diversity(abundance::Vector{Float64})

Calculate Shannon diversity index H' = -Σ pᵢ log(pᵢ)
"""
function shannon_diversity(abundance::Vector{Float64})
    total = sum(abundance)
    if total == 0 return 0.0 end
    p = abundance ./ total
    p = p[p .> 0]
    return -sum(p .* log.(p))
end

"""
    simpson_diversity(abundance::Vector{Float64})

Calculate Simpson diversity index 1 - Σ pᵢ²
"""
function simpson_diversity(abundance::Vector{Float64})
    total = sum(abundance)
    if total == 0 return 0.0 end
    p = abundance ./ total
    return 1 - sum(p.^2)
end

"""
    chao1(abundance::Vector{Float64})

Chao1 richness estimator.
"""
function chao1(abundance::Vector{Float64})
    S_obs = count(x -> x > 0, abundance)
    n1 = count(==(1), abundance)  # Singletons
    n2 = max(count(==(2), abundance), 1)  # Doubletons
    return S_obs + (n1 * (n1 - 1)) / (2 * (n2 + 1))
end

"""
    beta_diversity(comm1::Vector{Float64}, comm2::Vector{Float64}; method::Symbol=:bray_curtis)

Calculate beta diversity between communities.
"""
function beta_diversity(comm1::Vector{Float64}, comm2::Vector{Float64}; 
                       method::Symbol=:bray_curtis)
    if method == :bray_curtis
        num = sum(abs.(comm1 .- comm2))
        denom = sum(comm1) + sum(comm2)
        return denom > 0 ? num / denom : 0.0
    elseif method == :jaccard
        shared = sum((comm1 .> 0) .& (comm2 .> 0))
        union = sum((comm1 .> 0) .| (comm2 .> 0))
        return union > 0 ? 1 - shared / union : 0.0
    else
        throw(ArgumentError("Unknown method: $method"))
    end
end
