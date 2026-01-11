# ============================================================================
# SingleVariant.jl - GWAS Single-Variant Association Tests
# ============================================================================

"""
    GWASResult

Container for GWAS association results.
"""
struct GWASResult <: AbstractAssociationResult
    variant_ids::Vector{String}
    chromosomes::Vector{Chromosome}
    positions::Vector{Position}
    betas::Vector{Float64}
    standard_errors::Vector{Float64}
    statistics::Vector{Float64}
    pvalues::Vector{Float64}
    mafs::Vector{Float64}
    n_samples::Vector{Int}
    test_type::String
end

"""
    gwas_single_variant(gm::GenotypeMatrix, phenotype::AbstractPhenotype;
                       covariates::Union{CovariateMatrix, Nothing}=nothing,
                       test::Symbol=:auto)

Perform genome-wide association study using single-variant tests.

# Arguments
- `gm`: GenotypeMatrix with genotype data
- `phenotype`: Phenotype vector
- `covariates`: Optional covariate matrix
- `test`: Test type (:linear, :logistic, or :auto)

# Returns
- GWASResult with association statistics
"""
function gwas_single_variant(gm::GenotypeMatrix, phenotype::AbstractPhenotype;
                            covariates::Union{CovariateMatrix, Nothing}=nothing,
                            test::Symbol=:auto)
    # Determine test type
    if test == :auto
        test = phenotype_type(phenotype) == :binary ? :logistic : :linear
    end
    
    if test == :linear
        return gwas_linear(gm, phenotype; covariates=covariates)
    else
        return gwas_logistic(gm, phenotype; covariates=covariates)
    end
end

"""
    gwas_linear(gm::GenotypeMatrix, phenotype::AbstractPhenotype;
               covariates::Union{CovariateMatrix, Nothing}=nothing)

Linear regression GWAS for quantitative traits.
"""
function gwas_linear(gm::GenotypeMatrix, phenotype::AbstractPhenotype;
                    covariates::Union{CovariateMatrix, Nothing}=nothing)
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    # Get phenotype values
    y = Float64[]
    for i in 1:n_samp
        val = get_value(phenotype, i)
        push!(y, ismissing(val) ? NaN : Float64(val))
    end
    
    # Prepare covariates (add intercept)
    if covariates === nothing
        base_X = ones(n_samp, 1)  # Intercept only
    else
        base_X = hcat(ones(n_samp), covariates.data)
    end
    
    # Results storage
    betas = Vector{Float64}(undef, n_var)
    ses = Vector{Float64}(undef, n_var)
    tstats = Vector{Float64}(undef, n_var)
    pvals = Vector{Float64}(undef, n_var)
    mafs = Vector{Float64}(undef, n_var)
    ns = Vector{Int}(undef, n_var)
    
    for j in 1:n_var
        # Get genotypes for this variant
        g = Vector{Float64}(undef, n_samp)
        for i in 1:n_samp
            geno = gm.data[i, j]
            g[i] = ismissing(geno) ? NaN : Float64(geno)
        end
        
        # Find complete cases
        complete = findall(i -> !isnan(y[i]) && !isnan(g[i]), 1:n_samp)
        
        if length(complete) < 10
            betas[j] = NaN
            ses[j] = NaN
            tstats[j] = NaN
            pvals[j] = NaN
            mafs[j] = NaN
            ns[j] = length(complete)
            continue
        end
        
        # Design matrix with genotype
        X = hcat(base_X[complete, :], g[complete])
        y_sub = y[complete]
        
        # MAF
        maf = mean(g[complete]) / 2
        maf = min(maf, 1 - maf)
        mafs[j] = maf
        ns[j] = length(complete)
        
        if maf < 0.001
            # Too rare for reliable testing
            betas[j] = NaN
            ses[j] = NaN
            tstats[j] = NaN
            pvals[j] = NaN
            continue
        end
        
        # OLS regression
        try
            result = linear_regression(X, y_sub)
            
            # Genotype effect is last coefficient
            geno_idx = size(X, 2)
            betas[j] = result.coefficients[geno_idx]
            ses[j] = result.se[geno_idx]
            tstats[j] = result.t_statistics[geno_idx]
            pvals[j] = result.pvalues[geno_idx]
        catch
            betas[j] = NaN
            ses[j] = NaN
            tstats[j] = NaN
            pvals[j] = NaN
        end
    end
    
    GWASResult(gm.variant_ids, gm.chromosomes, gm.positions,
               betas, ses, tstats, pvals, mafs, ns, "Linear regression")
end

"""
    gwas_logistic(gm::GenotypeMatrix, phenotype::AbstractPhenotype;
                 covariates::Union{CovariateMatrix, Nothing}=nothing)

Logistic regression GWAS for binary traits.
"""
function gwas_logistic(gm::GenotypeMatrix, phenotype::AbstractPhenotype;
                      covariates::Union{CovariateMatrix, Nothing}=nothing)
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    # Get phenotype values (convert to 0/1)
    y = Float64[]
    for i in 1:n_samp
        val = get_value(phenotype, i)
        if ismissing(val)
            push!(y, NaN)
        else
            push!(y, val ? 1.0 : 0.0)
        end
    end
    
    # Prepare covariates
    if covariates === nothing
        base_X = ones(n_samp, 1)
    else
        base_X = hcat(ones(n_samp), covariates.data)
    end
    
    # Results storage
    betas = Vector{Float64}(undef, n_var)
    ses = Vector{Float64}(undef, n_var)
    zstats = Vector{Float64}(undef, n_var)
    pvals = Vector{Float64}(undef, n_var)
    mafs = Vector{Float64}(undef, n_var)
    ns = Vector{Int}(undef, n_var)
    
    for j in 1:n_var
        g = Vector{Float64}(undef, n_samp)
        for i in 1:n_samp
            geno = gm.data[i, j]
            g[i] = ismissing(geno) ? NaN : Float64(geno)
        end
        
        complete = findall(i -> !isnan(y[i]) && !isnan(g[i]), 1:n_samp)
        
        if length(complete) < 20
            betas[j] = NaN
            ses[j] = NaN
            zstats[j] = NaN
            pvals[j] = NaN
            mafs[j] = NaN
            ns[j] = length(complete)
            continue
        end
        
        X = hcat(base_X[complete, :], g[complete])
        y_sub = y[complete]
        
        maf = mean(g[complete]) / 2
        maf = min(maf, 1 - maf)
        mafs[j] = maf
        ns[j] = length(complete)
        
        if maf < 0.01
            betas[j] = NaN
            ses[j] = NaN
            zstats[j] = NaN
            pvals[j] = NaN
            continue
        end
        
        try
            result = logistic_regression(X, y_sub)
            
            geno_idx = size(X, 2)
            betas[j] = result.coefficients[geno_idx]
            ses[j] = result.se[geno_idx]
            zstats[j] = result.z_statistics[geno_idx]
            pvals[j] = result.pvalues[geno_idx]
        catch
            betas[j] = NaN
            ses[j] = NaN
            zstats[j] = NaN
            pvals[j] = NaN
        end
    end
    
    GWASResult(gm.variant_ids, gm.chromosomes, gm.positions,
               betas, ses, zstats, pvals, mafs, ns, "Logistic regression")
end

"""
    score_test(gm::GenotypeMatrix, phenotype::AbstractPhenotype, variant_idx::Int)

Perform score test for a single variant.
More efficient than Wald test for scanning.
"""
function score_test(gm::GenotypeMatrix, phenotype::AbstractPhenotype, variant_idx::Int)
    n_samp = n_samples(gm)
    
    # Get data
    g = [ismissing(gm.data[i, variant_idx]) ? NaN : Float64(gm.data[i, variant_idx]) 
         for i in 1:n_samp]
    y = [ismissing(get_value(phenotype, i)) ? NaN : Float64(get_value(phenotype, i))
         for i in 1:n_samp]
    
    complete = findall(i -> !isnan(y[i]) && !isnan(g[i]), 1:n_samp)
    
    if length(complete) < 10
        return (score=NaN, pvalue=NaN)
    end
    
    g_sub = g[complete]
    y_sub = y[complete]
    
    # Null model: y = μ
    μ = mean(y_sub)
    
    # Score statistic: U = Σ g_i(y_i - μ)
    U = sum(g_sub .* (y_sub .- μ))
    
    # Variance of score
    var_U = sum(g_sub.^2) * var(y_sub)
    
    # Score test statistic
    if var_U > 0
        score_stat = U^2 / var_U
        pval = ccdf(Chisq(1), score_stat)
    else
        score_stat = NaN
        pval = NaN
    end
    
    return (score=score_stat, pvalue=pval)
end

"""
    gwas_to_dataframe(result::GWASResult)

Convert GWAS results to DataFrame for easy manipulation.
"""
function gwas_to_dataframe(result::GWASResult)
    DataFrame(
        variant_id = result.variant_ids,
        chromosome = result.chromosomes,
        position = result.positions,
        beta = result.betas,
        se = result.standard_errors,
        statistic = result.statistics,
        pvalue = result.pvalues,
        maf = result.mafs,
        n = result.n_samples
    )
end

"""
    filter_gwas_results(result::GWASResult; min_maf::Float64=0.01, 
                       max_pvalue::Float64=1.0)

Filter GWAS results by MAF and p-value.
"""
function filter_gwas_results(result::GWASResult; min_maf::Float64=0.01,
                            max_pvalue::Float64=1.0)
    keep = findall(i -> result.mafs[i] >= min_maf && 
                       !isnan(result.pvalues[i]) &&
                       result.pvalues[i] <= max_pvalue, 
                  1:length(result.pvalues))
    
    GWASResult(
        result.variant_ids[keep],
        result.chromosomes[keep],
        result.positions[keep],
        result.betas[keep],
        result.standard_errors[keep],
        result.statistics[keep],
        result.pvalues[keep],
        result.mafs[keep],
        result.n_samples[keep],
        result.test_type
    )
end
