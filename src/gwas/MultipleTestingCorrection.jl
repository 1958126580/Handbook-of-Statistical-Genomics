# ============================================================================
# MultipleTestingCorrection.jl - Multiple Testing Adjustment
# ============================================================================

"""
    bonferroni_correction(pvalues::AbstractVector; alpha::Float64=0.05)

Apply Bonferroni correction for multiple testing.

# Arguments
- `pvalues`: Vector of p-values
- `alpha`: Family-wise error rate to control

# Returns
- NamedTuple with adjusted p-values and significance threshold
"""
function bonferroni_correction(pvalues::AbstractVector; alpha::Float64=0.05)
    valid_p = filter(!isnan, pvalues)
    n_tests = length(valid_p)
    
    threshold = alpha / n_tests
    adjusted = [isnan(p) ? NaN : min(p * n_tests, 1.0) for p in pvalues]
    significant = [!isnan(p) && p < threshold for p in pvalues]
    
    return (adjusted_pvalues=adjusted,
            threshold=threshold,
            n_significant=sum(significant),
            n_tests=n_tests)
end

"""
    fdr_correction(pvalues::AbstractVector; alpha::Float64=0.05, method::Symbol=:bh)

Apply FDR correction using Benjamini-Hochberg or Benjamini-Yekutieli.

# Arguments
- `pvalues`: Vector of p-values
- `alpha`: FDR level to control
- `method`: :bh (Benjamini-Hochberg) or :by (Benjamini-Yekutieli)

# Returns
- NamedTuple with q-values and significant variants
"""
function fdr_correction(pvalues::AbstractVector; alpha::Float64=0.05, 
                       method::Symbol=:bh)
    n = length(pvalues)
    
    # Get valid p-values with indices
    valid_indices = findall(!isnan, pvalues)
    valid_p = pvalues[valid_indices]
    m = length(valid_p)
    
    if m == 0
        return (qvalues=fill(NaN, n), n_significant=0, threshold=NaN)
    end
    
    # Sort p-values
    sorted_idx = sortperm(valid_p)
    sorted_p = valid_p[sorted_idx]
    
    # Calculate adjusted p-values (q-values)
    qvalues_sorted = similar(sorted_p, Float64)
    
    # Correction factor for BY method
    c_m = method == :by ? sum(1.0/i for i in 1:m) : 1.0
    
    # Calculate from largest to smallest
    qvalues_sorted[m] = min(sorted_p[m] * m * c_m, 1.0)
    
    for i in (m-1):-1:1
        adjusted = sorted_p[i] * m * c_m / i
        qvalues_sorted[i] = min(adjusted, qvalues_sorted[i+1])
    end
    
    # Unsort q-values
    qvalues_valid = similar(valid_p, Float64)
    qvalues_valid[sorted_idx] = qvalues_sorted
    
    # Map back to original indices
    qvalues = fill(NaN, n)
    for (orig_idx, q) in zip(valid_indices, qvalues_valid)
        qvalues[orig_idx] = q
    end
    
    # Count significant
    n_significant = count(q -> !isnan(q) && q < alpha, qvalues)
    
    # Find threshold (largest p-value still significant)
    threshold_idx = findfirst(q -> q >= alpha, qvalues_sorted)
    threshold = threshold_idx === nothing ? sorted_p[end] : sorted_p[max(1, threshold_idx-1)]
    
    return (qvalues=qvalues, n_significant=n_significant, threshold=threshold)
end

"""
    genomic_control(pvalues::AbstractVector)

Calculate genomic control inflation factor (λ_GC).

λ_GC > 1 indicates inflation (possible population stratification)
λ_GC < 1 indicates deflation (possible over-correction)
"""
function genomic_control(pvalues::AbstractVector)
    valid_p = filter(p -> !isnan(p) && p > 0 && p <= 1, pvalues)
    
    if isempty(valid_p)
        return (lambda_gc=NaN, corrected_pvalues=pvalues)
    end
    
    # Convert to chi-squared statistics
    chi2_stats = [quantile(Chisq(1), 1 - p) for p in valid_p]
    
    # Lambda GC = median(chi2) / 0.4549 (median of chi2(1))
    median_chi2 = median(chi2_stats)
    lambda_gc = median_chi2 / 0.4549
    
    # Correct p-values if lambda > 1
    if lambda_gc > 1
        corrected = similar(pvalues, Float64)
        for (i, p) in enumerate(pvalues)
            if isnan(p) || p <= 0 || p > 1
                corrected[i] = p
            else
                chi2 = quantile(Chisq(1), 1 - p)
                corrected_chi2 = chi2 / lambda_gc
                corrected[i] = ccdf(Chisq(1), corrected_chi2)
            end
        end
    else
        corrected = pvalues
    end
    
    return (lambda_gc=lambda_gc, corrected_pvalues=corrected)
end

"""
    permutation_threshold(gm::GenotypeMatrix, phenotype::AbstractPhenotype;
                         n_permutations::Int=1000, alpha::Float64=0.05)

Estimate significance threshold using permutation.

# Arguments
- `gm`: GenptypeMatrix
- `phenotype`: Phenotype vector
- `n_permutations`: Number of permutations
- `alpha`: Significance level

# Returns
- Permutation-based significance threshold
"""
function permutation_threshold(gm::GenotypeMatrix, phenotype::AbstractPhenotype;
                              n_permutations::Int=1000, alpha::Float64=0.05)
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    # Store minimum p-values from each permutation
    min_pvalues = Float64[]
    
    for perm in 1:n_permutations
        # Permute phenotype
        perm_order = shuffle(1:n_samp)
        
        perm_pheno = if phenotype isa ContinuousPhenotype
            ContinuousPhenotype(phenotype.values[perm_order], phenotype.name)
        else
            BinaryPhenotype([phenotype.values[i] for i in perm_order], phenotype.name)
        end
        
        # Run quick GWAS
        result = gwas_single_variant(gm, perm_pheno)
        
        # Get minimum p-value
        valid_p = filter(!isnan, result.pvalues)
        if !isempty(valid_p)
            push!(min_pvalues, minimum(valid_p))
        end
    end
    
    # Threshold = (1-alpha) quantile of min p-values
    sort!(min_pvalues)
    threshold_idx = round(Int, alpha * length(min_pvalues))
    threshold = min_pvalues[max(1, threshold_idx)]
    
    return (threshold=threshold, n_permutations=length(min_pvalues))
end

"""
    effective_number_of_tests(gm::GenotypeMatrix; r2_threshold::Float64=0.2)

Estimate effective number of independent tests accounting for LD.
Uses eigenvalue decomposition of correlation matrix.
"""
function effective_number_of_tests(gm::GenotypeMatrix; max_variants::Int=5000)
    n_var = min(n_variants(gm), max_variants)
    
    # Calculate genotype correlation matrix
    geno_data = Matrix{Float64}(undef, n_samples(gm), n_var)
    for j in 1:n_var
        for i in 1:n_samples(gm)
            g = gm.data[i, j]
            geno_data[i, j] = ismissing(g) ? 0.0 : Float64(g)
        end
    end
    
    # Center and scale
    for j in 1:n_var
        μ = mean(geno_data[:, j])
        σ = std(geno_data[:, j])
        if σ > 0
            geno_data[:, j] = (geno_data[:, j] .- μ) ./ σ
        end
    end
    
    # Correlation matrix
    R = cor(geno_data)
    
    # Eigenvalue decomposition
    eigenvalues = eigvals(R)
    eigenvalues = max.(eigenvalues, 0)  # Ensure non-negative
    
    # Effective number of tests (Li and Ji, 2005)
    # M_eff = sum of f(eigenvalues) where f constrains to [0,1]
    M_eff = sum(min.(1.0, eigenvalues))
    
    # Alternative: count eigenvalues > 1
    M_alt = count(e -> e > 1, eigenvalues)
    
    return (M_eff=M_eff, M_alt=M_alt, n_variants=n_var)
end

"""
    sidak_correction(pvalues::AbstractVector; n_tests::Union{Int, Nothing}=nothing)

Apply Sidak correction.
"""
function sidak_correction(pvalues::AbstractVector; n_tests::Union{Int, Nothing}=nothing)
    if n_tests === nothing
        n_tests = count(!isnan, pvalues)
    end
    
    adjusted = [isnan(p) ? NaN : 1 - (1 - p)^n_tests for p in pvalues]
    adjusted = min.(adjusted, 1.0)
    
    return adjusted
end
