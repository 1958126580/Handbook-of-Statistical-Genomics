# ============================================================================
# Imputation.jl - Genotype Imputation Methods
# ============================================================================

"""
    ImputationResult

Container for imputation results.
"""
struct ImputationResult
    imputed::Matrix{Union{Float64, Missing}}  # Imputed dosages
    info_scores::Vector{Float64}              # Imputation quality (R²)
    n_imputed::Vector{Int}                    # Count of imputed values per variant
end

"""
    impute_genotypes(gm::GenotypeMatrix; method::Symbol=:mean)

Impute missing genotypes.

# Arguments
- `gm`: GenotypeMatrix with missing values
- `method`: Imputation method (:mean, :mode, :knn, :em)

# Returns
- ImputationResult with imputed data
"""
function impute_genotypes(gm::GenotypeMatrix; method::Symbol=:mean)
    if method == :mean
        return mean_imputation(gm)
    elseif method == :mode
        return mode_imputation(gm)
    elseif method == :knn
        return knn_impute(gm)
    elseif method == :em
        return em_imputation(gm)
    else
        throw(ArgumentError("Unknown imputation method: $method"))
    end
end

"""
    mean_imputation(gm::GenotypeMatrix)

Simple mean imputation: replace missing with variant mean (dosage).
"""
function mean_imputation(gm::GenotypeMatrix)
    n_samp, n_var = size(gm)
    imputed = Matrix{Union{Float64, Missing}}(undef, n_samp, n_var)
    info_scores = Vector{Float64}(undef, n_var)
    n_imputed = Vector{Int}(undef, n_var)
    
    for j in 1:n_var
        genos = gm.data[:, j]
        valid_genos = collect(skipmissing(genos))
        
        if isempty(valid_genos)
            # All missing - use population average (1.0)
            variant_mean = 1.0
            info_scores[j] = 0.0
        else
            variant_mean = mean(valid_genos)
            # Info score: proportion non-missing × variance ratio
            info_scores[j] = length(valid_genos) / n_samp
        end
        
        n_miss = 0
        for i in 1:n_samp
            if ismissing(genos[i])
                imputed[i, j] = variant_mean
                n_miss += 1
            else
                imputed[i, j] = Float64(genos[i])
            end
        end
        n_imputed[j] = n_miss
    end
    
    ImputationResult(imputed, info_scores, n_imputed)
end

"""
    mode_imputation(gm::GenotypeMatrix)

Mode imputation: replace missing with most common genotype.
"""
function mode_imputation(gm::GenotypeMatrix)
    n_samp, n_var = size(gm)
    imputed = Matrix{Union{Float64, Missing}}(undef, n_samp, n_var)
    info_scores = Vector{Float64}(undef, n_var)
    n_imputed = Vector{Int}(undef, n_var)
    
    for j in 1:n_var
        genos = gm.data[:, j]
        valid_genos = collect(skipmissing(genos))
        
        if isempty(valid_genos)
            variant_mode = 1  # Default
            info_scores[j] = 0.0
        else
            # Find mode
            counts = [count(==(g), valid_genos) for g in 0:2]
            variant_mode = argmax(counts) - 1
            info_scores[j] = length(valid_genos) / n_samp
        end
        
        n_miss = 0
        for i in 1:n_samp
            if ismissing(genos[i])
                imputed[i, j] = Float64(variant_mode)
                n_miss += 1
            else
                imputed[i, j] = Float64(genos[i])
            end
        end
        n_imputed[j] = n_miss
    end
    
    ImputationResult(imputed, info_scores, n_imputed)
end

"""
    knn_impute(gm::GenotypeMatrix; k::Int=10)

K-nearest neighbors imputation using sample similarity.
"""
function knn_impute(gm::GenotypeMatrix; k::Int=10)
    n_samp, n_var = size(gm)
    imputed = Matrix{Union{Float64, Missing}}(undef, n_samp, n_var)
    info_scores = Vector{Float64}(undef, n_var)
    n_imputed = zeros(Int, n_var)
    
    # Calculate sample similarity matrix (on non-missing data)
    # Use IBS (identity by state) similarity
    similarity = calculate_ibs_matrix(gm)
    
    for j in 1:n_var
        genos = gm.data[:, j]
        
        for i in 1:n_samp
            if ismissing(genos[i])
                # Find k nearest neighbors with non-missing genotype
                neighbor_scores = Float64[]
                neighbor_genos = Int[]
                
                for other in 1:n_samp
                    if other != i && !ismissing(genos[other])
                        push!(neighbor_scores, similarity[i, other])
                        push!(neighbor_genos, genos[other])
                    end
                end
                
                if length(neighbor_genos) == 0
                    # No valid neighbors, use mean
                    valid = collect(skipmissing(genos))
                    imputed[i, j] = isempty(valid) ? 1.0 : mean(valid)
                else
                    # Select top k neighbors
                    k_actual = min(k, length(neighbor_genos))
                    top_k_idx = partialsortperm(neighbor_scores, 1:k_actual, rev=true)
                    
                    # Weighted average of neighbor genotypes
                    weights = neighbor_scores[top_k_idx]
                    geno_vals = neighbor_genos[top_k_idx]
                    
                    if sum(weights) > 0
                        imputed[i, j] = sum(weights .* geno_vals) / sum(weights)
                    else
                        imputed[i, j] = mean(geno_vals)
                    end
                end
                n_imputed[j] += 1
            else
                imputed[i, j] = Float64(genos[i])
            end
        end
        
        # Info score
        info_scores[j] = 1.0 - n_imputed[j] / n_samp
    end
    
    ImputationResult(imputed, info_scores, n_imputed)
end

"""Calculate IBS (identity by state) similarity matrix."""
function calculate_ibs_matrix(gm::GenotypeMatrix)
    n_samp = n_samples(gm)
    sim = ones(n_samp, n_samp)
    
    for i in 1:n_samp
        for j in (i+1):n_samp
            # Calculate IBS across all variants
            ibs_sum = 0.0
            n_compared = 0
            
            for v in 1:n_variants(gm)
                g1, g2 = gm.data[i, v], gm.data[j, v]
                if !ismissing(g1) && !ismissing(g2)
                    ibs_sum += 2 - abs(g1 - g2)
                    n_compared += 1
                end
            end
            
            if n_compared > 0
                sim[i, j] = ibs_sum / (2 * n_compared)
                sim[j, i] = sim[i, j]
            end
        end
    end
    
    return sim
end

"""
    em_imputation(gm::GenotypeMatrix; maxiter::Int=50, tol::Float64=1e-4)

EM-based imputation using LD structure.
"""
function em_imputation(gm::GenotypeMatrix; maxiter::Int=50, tol::Float64=1e-4)
    n_samp, n_var = size(gm)
    
    # Initialize with mean imputation
    init_result = mean_imputation(gm)
    imputed = copy(init_result.imputed)
    
    # Iterate EM
    for iter in 1:maxiter
        prev_imputed = copy(imputed)
        
        for j in 1:n_var
            if init_result.n_imputed[j] == 0
                continue  # No missing values
            end
            
            # Find neighboring variants in LD
            neighbors = find_ld_neighbors(gm, j)
            
            if isempty(neighbors)
                continue
            end
            
            # Build regression model on complete cases
            complete_idx = findall(i -> !ismissing(gm.data[i, j]), 1:n_samp)
            
            if length(complete_idx) < 10
                continue
            end
            
            # Design matrix from neighbor genotypes
            X = hcat(ones(length(complete_idx)), 
                    [imputed[complete_idx, n] for n in neighbors]...)
            y = [imputed[i, j] for i in complete_idx]
            
            # Fit linear model
            try
                β = X \ y  # OLS
                
                # Impute missing values
                for i in 1:n_samp
                    if ismissing(gm.data[i, j])
                        x_i = vcat(1.0, [imputed[i, n] for n in neighbors])
                        pred = dot(β, x_i)
                        imputed[i, j] = clamp(pred, 0.0, 2.0)
                    end
                end
            catch
                # Singular matrix, skip this iteration
                continue
            end
        end
        
        # Check convergence
        max_change = maximum(abs.(imputed - prev_imputed))
        if max_change < tol
            break
        end
    end
    
    # Calculate info scores
    info_scores = Vector{Float64}(undef, n_var)
    for j in 1:n_var
        observed = findall(i -> !ismissing(gm.data[i, j]), 1:n_samp)
        if length(observed) < 2
            info_scores[j] = 0.0
        else
            obs_var = var([gm.data[i, j] for i in observed])
            imp_var = var(imputed[:, j])
            info_scores[j] = imp_var > 0 ? min(1.0, obs_var / imp_var) : 0.0
        end
    end
    
    ImputationResult(imputed, info_scores, init_result.n_imputed)
end

"""Find variants in LD with the target variant."""
function find_ld_neighbors(gm::GenotypeMatrix, target_j::Int; 
                          r2_threshold::Float64=0.3, max_neighbors::Int=5)
    neighbors = Int[]
    r2_values = Float64[]
    
    target_chr = gm.chromosomes[target_j]
    target_pos = gm.positions[target_j]
    
    for j in 1:n_variants(gm)
        if j == target_j
            continue
        end
        
        # Only consider same chromosome and nearby variants
        if gm.chromosomes[j] != target_chr
            continue
        end
        if abs(gm.positions[j] - target_pos) > 500_000  # 500kb window
            continue
        end
        
        r2 = calculate_ld(gm.data[:, target_j], gm.data[:, j]; measure=:r2)
        if !isnan(r2) && r2 >= r2_threshold
            push!(neighbors, j)
            push!(r2_values, r2)
        end
    end
    
    # Return top neighbors by r²
    if length(neighbors) > max_neighbors
        top_idx = partialsortperm(r2_values, 1:max_neighbors, rev=true)
        return neighbors[top_idx]
    end
    
    return neighbors
end

"""
    imputation_quality(observed::AbstractVector, imputed::AbstractVector)

Calculate imputation quality metrics.

# Returns
- NamedTuple with R², concordance, and RMSE
"""
function imputation_quality(observed::AbstractVector, imputed::AbstractVector)
    valid_idx = findall(i -> !ismissing(observed[i]) && !ismissing(imputed[i]),
                       1:length(observed))
    
    if length(valid_idx) < 2
        return (r_squared=NaN, concordance=NaN, rmse=NaN, n=0)
    end
    
    obs = [observed[i] for i in valid_idx]
    imp = [imputed[i] for i in valid_idx]
    
    # R²
    r = cor(Float64.(obs), Float64.(imp))
    r_squared = r^2
    
    # Concordance (for hard calls)
    obs_hard = round.(Int, obs)
    imp_hard = round.(Int, clamp.(imp, 0, 2))
    concordance = mean(obs_hard .== imp_hard)
    
    # RMSE
    rmse = sqrt(mean((Float64.(obs) .- Float64.(imp)).^2))
    
    (r_squared=r_squared, concordance=concordance, rmse=rmse, n=length(valid_idx))
end
