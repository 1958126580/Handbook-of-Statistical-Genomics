# ============================================================================
# GxE.jl - Gene-Environment Interaction Analysis
# ============================================================================

"""
    gxe_interaction(gm::GenotypeMatrix, phenotype::AbstractPhenotype, 
                   environment::AbstractVector; method::Symbol=:full)

Test for gene-environment interaction.

# Arguments
- `gm`: GenotypeMatrix
- `phenotype`: Phenotype vector
- `environment`: Environmental exposure vector
- `method`: :full (full model), :product (product term), or :stratified

# Returns
- GWASResult with interaction effects
"""
function gxe_interaction(gm::GenotypeMatrix, phenotype::AbstractPhenotype,
                        environment::AbstractVector; method::Symbol=:full)
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    # Prepare phenotype
    y = Float64[]
    for i in 1:n_samp
        val = get_value(phenotype, i)
        push!(y, ismissing(val) ? NaN : Float64(val))
    end
    
    # Prepare environment
    e = Float64[]
    for i in 1:n_samp
        val = environment[i]
        push!(e, ismissing(val) ? NaN : Float64(val))
    end
    
    # Results
    betas_g = Vector{Float64}(undef, n_var)
    betas_gxe = Vector{Float64}(undef, n_var)
    ses_gxe = Vector{Float64}(undef, n_var)
    pvals_gxe = Vector{Float64}(undef, n_var)
    pvals_joint = Vector{Float64}(undef, n_var)
    
    for j in 1:n_var
        g = [ismissing(gm.data[i, j]) ? NaN : Float64(gm.data[i, j]) for i in 1:n_samp]
        
        # Complete cases
        complete = findall(i -> !isnan(y[i]) && !isnan(g[i]) && !isnan(e[i]), 1:n_samp)
        
        if length(complete) < 30
            betas_g[j] = betas_gxe[j] = ses_gxe[j] = NaN
            pvals_gxe[j] = pvals_joint[j] = NaN
            continue
        end
        
        y_c = y[complete]
        g_c = g[complete]
        e_c = e[complete]
        n_c = length(complete)
        
        if method == :full || method == :product
            # Full model: Y = β0 + β_g*G + β_e*E + β_gxe*G*E + ε
            gxe = g_c .* e_c
            X = hcat(ones(n_c), g_c, e_c, gxe)
            
            try
                result = linear_regression(X, y_c)
                
                betas_g[j] = result.coefficients[2]
                betas_gxe[j] = result.coefficients[4]
                ses_gxe[j] = result.se[4]
                pvals_gxe[j] = result.pvalues[4]
                
                # Joint test for G and GxE
                # Compare full model vs E-only model
                X_reduced = hcat(ones(n_c), e_c)
                result_reduced = linear_regression(X_reduced, y_c)
                
                # F-test for additional terms
                ss_full = sum(result.residuals.^2)
                ss_reduced = sum(result_reduced.residuals.^2)
                df_diff = 2  # Two additional parameters
                df_full = n_c - 4
                
                F_stat = ((ss_reduced - ss_full) / df_diff) / (ss_full / df_full)
                pvals_joint[j] = ccdf(FDist(df_diff, df_full), F_stat)
            catch
                betas_g[j] = betas_gxe[j] = ses_gxe[j] = NaN
                pvals_gxe[j] = pvals_joint[j] = NaN
            end
        end
    end
    
    # Return as modified GWAS result
    GWASResult(gm.variant_ids, gm.chromosomes, gm.positions,
               betas_gxe, ses_gxe, betas_gxe ./ ses_gxe, pvals_gxe,
               minor_allele_frequency(gm), fill(n_samp, n_var),
               "GxE interaction")
end

"""
    stratified_gwas(gm::GenotypeMatrix, phenotype::AbstractPhenotype,
                   strata::AbstractVector)

Perform GWAS stratified by environmental or categorical variable.

# Returns
- Dictionary of GWASResults for each stratum
"""
function stratified_gwas(gm::GenotypeMatrix, phenotype::AbstractPhenotype,
                        strata::AbstractVector)
    unique_strata = unique(filter(!ismissing, strata))
    results = Dict{Any, GWASResult}()
    
    for stratum in unique_strata
        # Get indices for this stratum
        stratum_idx = findall(i -> !ismissing(strata[i]) && strata[i] == stratum, 
                             1:n_samples(gm))
        
        if length(stratum_idx) < 30
            continue
        end
        
        # Subset genotype matrix
        sub_genos = gm.data[stratum_idx, :]
        sub_gm = GenotypeMatrix(sub_genos, 
                               gm.sample_ids[stratum_idx],
                               gm.variant_ids,
                               gm.chromosomes,
                               gm.positions,
                               gm.ref_alleles,
                               gm.alt_alleles)
        
        # Subset phenotype
        if phenotype isa ContinuousPhenotype
            sub_pheno = ContinuousPhenotype(phenotype.values[stratum_idx])
        else
            sub_pheno = BinaryPhenotype([phenotype.values[i] for i in stratum_idx])
        end
        
        # Run GWAS
        results[stratum] = gwas_single_variant(sub_gm, sub_pheno)
    end
    
    return results
end

"""
    heterogeneity_test(effects::Vector{Float64}, ses::Vector{Float64})

Test for heterogeneity of effects across strata using Cochran's Q.
"""
function heterogeneity_test(effects::Vector{Float64}, ses::Vector{Float64})
    k = length(effects)
    @assert length(ses) == k
    
    # Weights
    w = 1.0 ./ (ses.^2)
    
    # Fixed effect estimate
    β_fixed = sum(w .* effects) / sum(w)
    
    # Cochran's Q
    Q = sum(w .* (effects .- β_fixed).^2)
    
    # Degrees of freedom
    df = k - 1
    
    # P-value
    pval = ccdf(Chisq(df), Q)
    
    # I² (heterogeneity measure)
    I2 = max(0.0, (Q - df) / Q)
    
    return (Q=Q, pvalue=pval, I_squared=I2, df=df, fixed_effect=β_fixed)
end

"""
    meta_analysis_gxe(strat_results::Dict{Any, GWASResult}, variant_idx::Int)

Meta-analyze GxE effects across strata.
"""
function meta_analysis_gxe(strat_results::Dict{Any, GWASResult}, variant_idx::Int)
    effects = Float64[]
    ses = Float64[]
    
    for (stratum, result) in strat_results
        if variant_idx <= length(result.betas) && !isnan(result.betas[variant_idx])
            push!(effects, result.betas[variant_idx])
            push!(ses, result.standard_errors[variant_idx])
        end
    end
    
    if length(effects) < 2
        return (meta_beta=NaN, meta_se=NaN, meta_pvalue=NaN, het_pvalue=NaN)
    end
    
    # Fixed-effect meta-analysis
    w = 1.0 ./ (ses.^2)
    meta_beta = sum(w .* effects) / sum(w)
    meta_se = sqrt(1.0 / sum(w))
    meta_z = meta_beta / meta_se
    meta_pval = 2 * ccdf(Normal(), abs(meta_z))
    
    # Heterogeneity test
    het = heterogeneity_test(effects, ses)
    
    return (meta_beta=meta_beta, meta_se=meta_se, meta_pvalue=meta_pval, 
            het_pvalue=het.pvalue, I_squared=het.I_squared)
end
