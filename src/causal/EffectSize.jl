# ============================================================================
# EffectSize.jl - Effect Size Estimation
# ============================================================================

"""
    heritability_estimate(gm::GenotypeMatrix, phenotype::AbstractPhenotype)

Estimate heritability using GREML-like approach.
"""
function heritability_estimate(gm::GenotypeMatrix, phenotype::AbstractPhenotype)
    n_samp = n_samples(gm)
    
    # Get phenotype vector
    y = Float64[]
    for i in 1:n_samp
        val = get_value(phenotype, i)
        push!(y, ismissing(val) ? NaN : Float64(val))
    end
    
    # Complete cases
    complete = findall(!isnan, y)
    y_c = y[complete]
    
    # Calculate GRM
    K = grm_matrix(gm)
    K_c = K[complete, complete]
    
    # REML estimation
    var_comp = emma_reml(y_c, K_c)
    
    return (h2=var_comp.heritability,
            sigma2_g=var_comp.sigma2_g,
            sigma2_e=var_comp.sigma2_e,
            n_samples=length(complete))
end

"""
    genetic_variance(gm::GenotypeMatrix, phenotype::AbstractPhenotype)

Partition phenotypic variance into genetic components.
"""
function genetic_variance(gm::GenotypeMatrix, phenotype::AbstractPhenotype)
    result = heritability_estimate(gm, phenotype)
    
    # Total variance
    y_vals = [get_value(phenotype, i) for i in 1:n_samples(phenotype)]
    y_clean = collect(skipmissing(y_vals))
    var_p = var(Float64.(y_clean))
    
    return (V_G=result.sigma2_g,
            V_E=result.sigma2_e,
            V_P=var_p,
            h2=result.h2)
end

"""
    polygenic_score(gm::GenotypeMatrix, weights::Vector{Float64})

Calculate polygenic risk/trait scores.
"""
function polygenic_score(gm::GenotypeMatrix, weights::Vector{Float64})
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    @assert length(weights) == n_var "Weights must match number of variants"
    
    scores = Vector{Float64}(undef, n_samp)
    
    for i in 1:n_samp
        score = 0.0
        n_valid = 0
        
        for j in 1:n_var
            g = gm.data[i, j]
            if !ismissing(g)
                score += Float64(g) * weights[j]
                n_valid += 1
            end
        end
        
        scores[i] = n_valid > 0 ? score : NaN
    end
    
    # Standardize
    valid_scores = filter(!isnan, scores)
    if !isempty(valid_scores)
        μ = mean(valid_scores)
        σ = std(valid_scores)
        scores = [(isnan(s) ? NaN : (s - μ) / σ) for s in scores]
    end
    
    return scores
end

"""
    prs_from_gwas(gwas_result::GWASResult; pvalue_threshold::Float64=5e-8)

Create polygenic score weights from GWAS results.
"""
function prs_from_gwas(gwas_result::GWASResult; pvalue_threshold::Float64=5e-8)
    n_var = length(gwas_result.betas)
    weights = zeros(n_var)
    
    for j in 1:n_var
        if !isnan(gwas_result.pvalues[j]) && gwas_result.pvalues[j] < pvalue_threshold
            weights[j] = gwas_result.betas[j]
        end
    end
    
    n_included = count(!=(0), weights)
    
    return (weights=weights, n_variants=n_included, threshold=pvalue_threshold)
end
