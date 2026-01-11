# ============================================================================
# PCA.jl - Principal Component Analysis for Population Structure
# ============================================================================

"""
    PCAResult

Result of genetic PCA analysis.
"""
struct PCAResult
    scores::Matrix{Float64}          # Sample scores (n_samples × n_components)
    loadings::Matrix{Float64}        # Variant loadings
    variance_explained::Vector{Float64}
    eigenvalues::Vector{Float64}
    sample_ids::Vector{String}
end

"""
    genetic_pca(gm::GenotypeMatrix; n_components::Int=10)

Perform PCA on genotype data for population structure analysis.
Uses Patterson et al. (2006) method.
"""
function genetic_pca(gm::GenotypeMatrix; n_components::Int=10)
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    n_components = min(n_components, n_samp - 1, n_var)
    
    # Standardize genotype matrix
    Z = Matrix{Float64}(undef, n_samp, n_var)
    valid_vars = Int[]
    
    for j in 1:n_var
        genos = gm.data[:, j]
        valid = collect(skipmissing(genos))
        
        if length(valid) < n_samp * 0.9
            continue
        end
        
        p = sum(valid) / (2 * length(valid))
        
        if p < 0.01 || p > 0.99
            continue
        end
        
        push!(valid_vars, j)
        
        for i in 1:n_samp
            g = gm.data[i, j]
            if ismissing(g)
                Z[i, j] = 0.0  # Mean imputation
            else
                Z[i, j] = (g - 2*p) / sqrt(2*p*(1-p))
            end
        end
    end
    
    # Use only valid variants
    Z_valid = Z[:, valid_vars]
    
    # Compute GRM and extract eigenvectors
    GRM = (Z_valid * Z_valid') ./ length(valid_vars)
    
    eigen_result = eigen(Symmetric(GRM))
    
    # Sort by eigenvalue (descending)
    order = sortperm(eigen_result.values, rev=true)
    eigenvalues = eigen_result.values[order]
    eigenvectors = eigen_result.vectors[:, order]
    
    # Extract top components
    scores = eigenvectors[:, 1:n_components]
    
    # Variance explained
    total_var = sum(max.(eigenvalues, 0))
    var_explained = max.(eigenvalues[1:n_components], 0) ./ total_var
    
    # Loadings (simplified: correlation with PCs)
    loadings = Z_valid' * scores / n_samp
    
    PCAResult(scores, loadings, var_explained, eigenvalues[1:n_components], gm.sample_ids)
end

"""
    pca_projection(pca_result::PCAResult, new_gm::GenotypeMatrix)

Project new samples onto existing PC space.
"""
function pca_projection(pca_result::PCAResult, new_gm::GenotypeMatrix)
    # Standardize new genotypes using same parameters
    # Then project onto loadings
    
    n_new = n_samples(new_gm)
    n_comp = size(pca_result.loadings, 2)
    
    projections = zeros(n_new, n_comp)
    
    # Simplified projection
    for i in 1:n_new
        for comp in 1:n_comp
            for j in 1:n_variants(new_gm)
                g = new_gm.data[i, j]
                if !ismissing(g)
                    projections[i, comp] += Float64(g) * pca_result.loadings[j, comp]
                end
            end
        end
    end
    
    return projections
end

"""
    tracy_widom_test(eigenvalues::Vector{Float64}, n_samples::Int, n_variants::Int)

Test significance of principal components using Tracy-Widom distribution.
"""
function tracy_widom_test(eigenvalues::Vector{Float64}, n_samples::Int, n_variants::Int)
    n_test = min(length(eigenvalues), 20)
    
    pvalues = Vector{Float64}(undef, n_test)
    
    for k in 1:n_test
        # Standardize eigenvalue using Tracy-Widom
        μ = (sqrt(n_samples - 0.5) + sqrt(n_variants - 0.5))^2 / n_variants
        σ = (sqrt(n_samples - 0.5) + sqrt(n_variants - 0.5)) * 
            (1/sqrt(n_samples - 0.5) + 1/sqrt(n_variants - 0.5))^(1/3) / n_variants
        
        tw_stat = (eigenvalues[k] - μ) / σ
        
        # Approximate Tracy-Widom with standard normal (simplified)
        pvalues[k] = ccdf(Normal(), tw_stat)
    end
    
    # Significant PCs are those with small p-values
    n_significant = count(p -> p < 0.05, pvalues)
    
    return (pvalues=pvalues, n_significant=n_significant)
end
