# ============================================================================
# MethylationProcessing.jl - Methylation Data Processing
# ============================================================================

"""Beta to M-value conversion."""
beta_to_m_value(beta::Float64) = log2((beta + 0.001) / (1 - beta + 0.001))
m_to_beta_value(m::Float64) = 2^m / (1 + 2^m)

"""
    normalize_methylation(beta_values::Matrix{Float64}; method::Symbol=:quantile)

Normalize methylation beta values.
"""
function normalize_methylation(beta_values::Matrix{Float64}; method::Symbol=:quantile)
    n_sites, n_samples = size(beta_values)
    
    if method == :quantile
        # Quantile normalization
        sorted_data = sort(beta_values, dims=2)
        row_means = mean(sorted_data, dims=2)
        
        normalized = similar(beta_values)
        for j in 1:n_samples
            order = sortperm(beta_values[:, j])
            for (rank, idx) in enumerate(order)
                normalized[idx, j] = row_means[rank]
            end
        end
        return normalized
        
    elseif method == :bmiq
        # Simplified BMIQ-like normalization
        return beta_values  # Placeholder
        
    else
        return beta_values
    end
end
