# ============================================================================
# EWAS.jl - Epigenome-Wide Association Studies
# ============================================================================

"""
    ewas_association(beta::Matrix{Float64}, phenotype::Vector{Float64};
                    covariates::Union{Matrix{Float64}, Nothing}=nothing)

Epigenome-wide association study.
"""
function ewas_association(beta::Matrix{Float64}, phenotype::Vector{Float64};
                         covariates::Union{Matrix{Float64}, Nothing}=nothing)
    n_sites = size(beta, 1)
    n_samples = size(beta, 2)
    
    betas_out = Vector{Float64}(undef, n_sites)
    ses = Vector{Float64}(undef, n_sites)
    pvalues = Vector{Float64}(undef, n_sites)
    
    for s in 1:n_sites
        meth = beta[s, :]
        
        complete = findall(i -> !isnan(meth[i]) && !isnan(phenotype[i]), 1:n_samples)
        if length(complete) < 20
            betas_out[s] = ses[s] = pvalues[s] = NaN
            continue
        end
        
        if covariates === nothing
            X = hcat(ones(length(complete)), meth[complete])
        else
            X = hcat(ones(length(complete)), meth[complete], covariates[complete, :])
        end
        y = phenotype[complete]
        
        try
            result = linear_regression(X, y)
            betas_out[s] = result.coefficients[2]
            ses[s] = result.se[2]
            pvalues[s] = result.pvalues[2]
        catch
            betas_out[s] = ses[s] = pvalues[s] = NaN
        end
    end
    
    fdr_result = fdr_correction(pvalues)
    return (betas=betas_out, se=ses, pvalues=pvalues, qvalues=fdr_result.qvalues)
end

"""Adjust for cell type composition (simplified)."""
function cell_type_adjust(beta::Matrix{Float64}, cell_props::Matrix{Float64})
    n_sites, n_samples = size(beta)
    adjusted = similar(beta)
    
    for s in 1:n_sites
        X = hcat(ones(n_samples), cell_props)
        result = linear_regression(X, beta[s, :])
        adjusted[s, :] = result.residuals .+ mean(beta[s, :])
    end
    
    return adjusted
end
