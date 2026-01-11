# ============================================================================
# MixedModels.jl - Linear Mixed Model GWAS
# ============================================================================

"""
    grm_matrix(gm::GenotypeMatrix; method::Symbol=:standard)

Calculate Genetic Relationship Matrix (GRM).

# Arguments
- `gm`: GenotypeMatrix
- `method`: :standard (Yang et al.) or :ibs (identity by state)

# Returns
- Symmetric GRM matrix (n_samples × n_samples)
"""
function grm_matrix(gm::GenotypeMatrix; method::Symbol=:standard)
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    # Standardized genotype matrix
    Z = Matrix{Float64}(undef, n_samp, n_var)
    
    valid_vars = 0
    for j in 1:n_var
        genos = gm.data[:, j]
        valid_genos = collect(skipmissing(genos))
        
        if length(valid_genos) < n_samp * 0.9
            continue  # Skip variants with > 10% missing
        end
        
        p = sum(valid_genos) / (2 * length(valid_genos))
        
        if p < 0.01 || p > 0.99
            continue  # Skip rare variants
        end
        
        valid_vars += 1
        
        for i in 1:n_samp
            g = gm.data[i, j]
            if ismissing(g)
                Z[i, j] = 0.0  # Mean imputation
            else
                if method == :standard
                    # Yang et al. standardization
                    Z[i, j] = (g - 2*p) / sqrt(2*p*(1-p))
                else
                    Z[i, j] = Float64(g)
                end
            end
        end
    end
    
    # GRM = ZZ' / n_variants
    if method == :standard
        G = (Z * Z') ./ valid_vars
    else
        # IBS-based
        G = Matrix{Float64}(undef, n_samp, n_samp)
        for i in 1:n_samp
            for j in i:n_samp
                ibs_sum = 0.0
                n_compared = 0
                for v in 1:n_var
                    if !ismissing(gm.data[i, v]) && !ismissing(gm.data[j, v])
                        ibs_sum += 2 - abs(gm.data[i, v] - gm.data[j, v])
                        n_compared += 1
                    end
                end
                G[i, j] = n_compared > 0 ? ibs_sum / (2 * n_compared) : 0.5
                G[j, i] = G[i, j]
            end
        end
    end
    
    return G
end

"""
    mixed_model_gwas(gm::GenotypeMatrix, phenotype::AbstractPhenotype, K::Matrix{Float64};
                    covariates::Union{CovariateMatrix, Nothing}=nothing)

Mixed model GWAS accounting for population structure and relatedness.

Model: y = Xβ + Zg + u + ε
where u ~ N(0, σ²_g * K) and ε ~ N(0, σ²_e * I)
"""
function mixed_model_gwas(gm::GenotypeMatrix, phenotype::AbstractPhenotype, K::Matrix{Float64};
                         covariates::Union{CovariateMatrix, Nothing}=nothing)
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    # Phenotype vector
    y = Float64[]
    for i in 1:n_samp
        val = get_value(phenotype, i)
        push!(y, ismissing(val) ? NaN : Float64(val))
    end
    
    # Complete cases
    complete = findall(!isnan, y)
    y_c = y[complete]
    K_c = K[complete, complete]
    n_c = length(complete)
    
    # Estimate variance components using null model
    # Using spectral decomposition for efficiency
    var_comp = emma_reml(y_c, K_c)
    
    δ = var_comp.delta  # σ²_e / σ²_g
    
    # Transform data for efficient testing
    # (I + δ⁻¹K)⁻¹ = δ/(δ+λ) * eigenvectors
    eigen_K = eigen(Symmetric(K_c))
    λ = eigen_K.values
    U = eigen_K.vectors
    
    # Transformed y
    y_star = U' * y_c
    
    # Pre-calculate weights
    weights = 1.0 ./ (1.0 .+ λ ./ δ)
    
    # Results storage
    betas = Vector{Float64}(undef, n_var)
    ses = Vector{Float64}(undef, n_var)
    tstats = Vector{Float64}(undef, n_var)
    pvals = Vector{Float64}(undef, n_var)
    mafs = Vector{Float64}(undef, n_var)
    
    for j in 1:n_var
        # Genotype for this variant
        g = [gm.data[i, j] for i in complete]
        valid_g = findall(!ismissing, g)
        
        if length(valid_g) < n_c * 0.9
            betas[j] = ses[j] = tstats[j] = NaN
            pvals[j] = NaN
            mafs[j] = NaN
            continue
        end
        
        # Mean impute missing
        g_float = Float64.([ismissing(gi) ? mean(skipmissing(g)) : Float64(gi) for gi in g])
        
        maf = mean(g_float) / 2
        maf = min(maf, 1 - maf)
        mafs[j] = maf
        
        if maf < 0.01
            betas[j] = ses[j] = tstats[j] = pvals[j] = NaN
            continue
        end
        
        # Transform genotype
        g_star = U' * g_float
        
        # Weighted least squares
        # β = (g'Wg)⁻¹ g'Wy
        gWg = sum(weights .* g_star.^2)
        gWy = sum(weights .* g_star .* y_star)
        
        if gWg > 0
            beta = gWy / gWg
            
            # Residual variance
            resid = y_star .- beta .* g_star
            sigma2 = sum(weights .* resid.^2) / (n_c - 2)
            
            # Standard error
            se = sqrt(sigma2 / gWg)
            
            betas[j] = beta
            ses[j] = se
            tstats[j] = beta / se
            pvals[j] = 2 * ccdf(TDist(n_c - 2), abs(beta / se))
        else
            betas[j] = ses[j] = tstats[j] = pvals[j] = NaN
        end
    end
    
    GWASResult(gm.variant_ids, gm.chromosomes, gm.positions,
               betas, ses, tstats, pvals, mafs, fill(length(complete), n_var),
               "Mixed model (EMMA)")
end

"""
    emma_reml(y::Vector{Float64}, K::Matrix{Float64})

EMMA REML estimation of variance components.
Returns σ²_g, σ²_e, and δ = σ²_e/σ²_g.
"""
function emma_reml(y::Vector{Float64}, K::Matrix{Float64})
    n = length(y)
    
    # Spectral decomposition of K
    eigen_K = eigen(Symmetric(K))
    λ = eigen_K.values
    U = eigen_K.vectors
    
    # Transform y
    y_star = U' * y
    
    # Grid search for δ (log scale)
    best_ll = -Inf
    best_delta = 1.0
    
    for log_delta in range(-5, 5, length=50)
        δ = exp(log_delta)
        
        # REML log-likelihood
        denominator = λ .+ δ
        
        # σ²_e estimate given δ
        σ2_e = sum(y_star.^2 ./ denominator) / n
        
        # Log-likelihood
        ll = -0.5 * n * log(2π * σ2_e)
        ll -= 0.5 * sum(log.(denominator))
        ll -= 0.5 * n
        
        if ll > best_ll
            best_ll = ll
            best_delta = δ
        end
    end
    
    # Final variance estimates
    δ = best_delta
    denominator = λ .+ δ
    σ2_e = sum(y_star.^2 ./ denominator) / n
    σ2_g = σ2_e / δ
    
    # Heritability
    h2 = σ2_g / (σ2_g + σ2_e)
    
    return (sigma2_g=σ2_g, sigma2_e=σ2_e, delta=δ, heritability=h2, log_likelihood=best_ll)
end

"""
    kinship_adjustment(gm::GenotypeMatrix, phenotype::AbstractPhenotype)

Quick kinship adjustment using principal components.
Alternative to full mixed model for initial screening.
"""
function kinship_adjustment(gm::GenotypeMatrix, phenotype::AbstractPhenotype; n_pcs::Int=10)
    # Calculate principal components from GRM
    K = grm_matrix(gm)
    eigen_K = eigen(Symmetric(K))
    
    # Top PCs
    n_pcs = min(n_pcs, n_samples(gm) - 1)
    pcs = eigen_K.vectors[:, end:-1:end-n_pcs+1]
    
    # Create covariate matrix from PCs
    covariates = CovariateMatrix(pcs, ["PC$i" for i in 1:n_pcs])
    
    # Run GWAS with PC covariates
    return gwas_single_variant(gm, phenotype; covariates=covariates)
end
