# ============================================================================
# MetagenomeAssociation.jl - Microbiome-Phenotype Association
# ============================================================================

"""
    metagenome_association(abundance::Matrix{Float64}, phenotype::Vector{Float64})

Test association between microbial taxa and phenotype.
"""
function metagenome_association(abundance::Matrix{Float64}, phenotype::Vector{Float64};
                               transform::Symbol=:clr)
    n_taxa, n_samples = size(abundance)
    
    # Compositional transformation
    if transform == :clr
        # Centered log-ratio
        log_ab = log.(abundance .+ 0.5)
        geo_mean = mean(log_ab, dims=1)
        transformed = log_ab .- geo_mean
    else
        transformed = abundance
    end
    
    pvalues = Vector{Float64}(undef, n_taxa)
    betas = Vector{Float64}(undef, n_taxa)
    
    for t in 1:n_taxa
        complete = findall(i -> !isnan(phenotype[i]), 1:n_samples)
        X = hcat(ones(length(complete)), transformed[t, complete])
        y = phenotype[complete]
        
        try
            result = linear_regression(X, y)
            betas[t] = result.coefficients[2]
            pvalues[t] = result.pvalues[2]
        catch
            betas[t] = pvalues[t] = NaN
        end
    end
    
    fdr_result = fdr_correction(pvalues)
    return (betas=betas, pvalues=pvalues, qvalues=fdr_result.qvalues)
end

"""
    compositional_analysis(abundance::Matrix{Float64})

Analyze microbial compositional data using proper compositional methods.
"""
function compositional_analysis(abundance::Matrix{Float64})
    n_taxa, n_samples = size(abundance)
    
    # Add pseudocount
    ab_pseudo = abundance .+ 0.5
    
    # CLR transformation
    log_ab = log.(ab_pseudo)
    geo_mean = mean(log_ab, dims=1)
    clr = log_ab .- geo_mean
    
    # Aitchison distance matrix
    dist_matrix = zeros(n_samples, n_samples)
    for i in 1:n_samples
        for j in (i+1):n_samples
            d = sqrt(sum((clr[:, i] .- clr[:, j]).^2))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
        end
    end
    
    return (clr_transformed=clr, distance_matrix=dist_matrix)
end
