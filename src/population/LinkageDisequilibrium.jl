# ============================================================================
# LinkageDisequilibrium.jl - LD Analysis and Haplotype Block Detection
# ============================================================================

"""
    calculate_ld(geno1::AbstractVector, geno2::AbstractVector; measure::Symbol=:r2)

Calculate linkage disequilibrium between two variants.

# Arguments
- `geno1`, `geno2`: Genotype vectors for each variant
- `measure`: LD measure (:r2, :d, :dprime)

# Returns
- LD value
"""
function calculate_ld(geno1::AbstractVector, geno2::AbstractVector; 
                     measure::Symbol=:r2)
    # Get complete cases
    valid_idx = findall(i -> !ismissing(geno1[i]) && !ismissing(geno2[i]), 
                       1:length(geno1))
    
    if length(valid_idx) < 10
        return NaN
    end
    
    g1 = [geno1[i] for i in valid_idx]
    g2 = [geno2[i] for i in valid_idx]
    n = length(valid_idx)
    
    # Allele frequencies
    p1 = sum(g1) / (2 * n)  # Freq of alt allele at locus 1
    p2 = sum(g2) / (2 * n)  # Freq of alt allele at locus 2
    
    if p1 == 0 || p1 == 1 || p2 == 0 || p2 == 1
        return NaN  # Monomorphic
    end
    
    # Estimate haplotype frequency using EM
    # For genotype data, we observe: g1*g2 products but need haplotypes
    # Use composite LD estimator
    
    # Calculate D using Rogers & Huff (2008) estimator
    # Covariance of genotype dosages
    cov_g = cov(Float64.(g1), Float64.(g2))
    
    # D = cov(g1, g2) / 2
    D = cov_g / 2
    
    if measure == :d
        return D
    elseif measure == :dprime
        # D' = D / D_max
        if D > 0
            D_max = min(p1 * (1 - p2), (1 - p1) * p2)
        else
            D_max = min(p1 * p2, (1 - p1) * (1 - p2))
        end
        return D_max > 0 ? D / D_max : NaN
    else  # :r2
        # r² = D² / (p1 * (1-p1) * p2 * (1-p2))
        r2 = D^2 / (p1 * (1 - p1) * p2 * (1 - p2))
        return r2
    end
end

"""
    ld_matrix(gm::GenotypeMatrix; measure::Symbol=:r2, 
             variant_indices::Union{Vector{Int}, Nothing}=nothing)

Calculate pairwise LD matrix for variants.

# Arguments
- `gm`: GenotypeMatrix
- `measure`: LD measure (:r2, :d, :dprime)
- `variant_indices`: Optional subset of variant indices

# Returns
- Symmetric LD matrix
"""
function ld_matrix(gm::GenotypeMatrix; measure::Symbol=:r2,
                  variant_indices::Union{Vector{Int}, Nothing}=nothing)
    
    indices = variant_indices === nothing ? (1:n_variants(gm)) : variant_indices
    n = length(indices)
    
    ld_mat = Matrix{Float64}(undef, n, n)
    
    for i in 1:n
        ld_mat[i, i] = 1.0
        for j in (i+1):n
            ld_val = calculate_ld(gm.data[:, indices[i]], gm.data[:, indices[j]]; 
                                  measure=measure)
            ld_mat[i, j] = ld_val
            ld_mat[j, i] = ld_val
        end
    end
    
    return ld_mat
end

"""
    ld_prune(gm::GenotypeMatrix; r2_threshold::Float64=0.2, 
            window_size::Int=50, step_size::Int=5)

Prune variants based on LD using sliding window approach.

# Arguments
- `gm`: GenotypeMatrix
- `r2_threshold`: Maximum r² allowed between retained variants
- `window_size`: Number of variants in each window
- `step_size`: Step size for sliding window

# Returns
- Vector of indices of variants to keep
"""
function ld_prune(gm::GenotypeMatrix; r2_threshold::Float64=0.2,
                 window_size::Int=50, step_size::Int=5)
    
    n_vars = n_variants(gm)
    keep = trues(n_vars)
    
    # Sort variants by position within each chromosome
    chr_order = sortperm(1:n_vars, by=j -> (gm.chromosomes[j], gm.positions[j]))
    
    # Process each chromosome separately
    unique_chrs = unique(gm.chromosomes)
    
    for chr in unique_chrs
        chr_mask = gm.chromosomes .== chr
        chr_indices = findall(chr_mask)
        
        if length(chr_indices) < 2
            continue
        end
        
        # Sort by position
        sort!(chr_indices, by=j -> gm.positions[j])
        
        # Sliding window LD pruning
        window_start = 1
        while window_start <= length(chr_indices)
            window_end = min(window_start + window_size - 1, length(chr_indices))
            window_indices = chr_indices[window_start:window_end]
            
            # Only consider variants still marked for keeping
            active = [idx for idx in window_indices if keep[idx]]
            
            if length(active) > 1
                # Calculate LD for all pairs in window
                for i in 1:length(active)
                    if !keep[active[i]]
                        continue
                    end
                    for j in (i+1):length(active)
                        if !keep[active[j]]
                            continue
                        end
                        
                        r2 = calculate_ld(gm.data[:, active[i]], 
                                         gm.data[:, active[j]]; 
                                         measure=:r2)
                        
                        if !isnan(r2) && r2 > r2_threshold
                            # Remove variant with lower MAF
                            maf_i = minor_allele_frequency_single(gm.data[:, active[i]])
                            maf_j = minor_allele_frequency_single(gm.data[:, active[j]])
                            
                            if maf_i < maf_j
                                keep[active[i]] = false
                            else
                                keep[active[j]] = false
                            end
                        end
                    end
                end
            end
            
            window_start += step_size
        end
    end
    
    return findall(keep)
end

"""Calculate MAF for a single variant."""
function minor_allele_frequency_single(genotypes::AbstractVector)
    valid = collect(skipmissing(genotypes))
    if isempty(valid)
        return 0.0
    end
    p = sum(valid) / (2 * length(valid))
    return min(p, 1 - p)
end

"""
    find_haplotype_blocks(gm::GenotypeMatrix; method::Symbol=:gabriel,
                         min_maf::Float64=0.05)

Identify haplotype blocks using specified method.

# Arguments
- `gm`: GenotypeMatrix
- `method`: Block detection method (:gabriel, :fourGamete, :solid_spine)
- `min_maf`: Minimum MAF for variants to consider

# Returns
- Vector of (start_idx, end_idx) tuples defining blocks
"""
function find_haplotype_blocks(gm::GenotypeMatrix; method::Symbol=:gabriel,
                              min_maf::Float64=0.05)
    
    # Filter by MAF
    mafs = minor_allele_frequency(gm)
    valid_vars = findall(m -> m >= min_maf, mafs)
    
    if length(valid_vars) < 2
        return Tuple{Int, Int}[]
    end
    
    blocks = Tuple{Int, Int}[]
    
    if method == :gabriel
        blocks = gabriel_blocks(gm, valid_vars)
    elseif method == :fourGamete
        blocks = four_gamete_blocks(gm, valid_vars)
    elseif method == :solid_spine
        blocks = solid_spine_blocks(gm, valid_vars)
    else
        throw(ArgumentError("Unknown method: $method"))
    end
    
    return blocks
end

"""
    gabriel_blocks(gm::GenotypeMatrix, variant_indices::Vector{Int})

Gabriel et al. (2002) block detection method.
Blocks are regions where at least 95% of informative pairs show strong LD.
"""
function gabriel_blocks(gm::GenotypeMatrix, variant_indices::Vector{Int})
    n = length(variant_indices)
    blocks = Tuple{Int, Int}[]
    
    # Calculate D' for all pairs
    dprime_matrix = zeros(n, n)
    for i in 1:n
        dprime_matrix[i, i] = 1.0
        for j in (i+1):n
            dprime = calculate_ld(gm.data[:, variant_indices[i]], 
                                 gm.data[:, variant_indices[j]]; 
                                 measure=:dprime)
            dprime_matrix[i, j] = isnan(dprime) ? 0.0 : abs(dprime)
            dprime_matrix[j, i] = dprime_matrix[i, j]
        end
    end
    
    # Find blocks: regions where >= 95% pairs have |D'| > 0.7
    block_start = 1
    
    while block_start < n
        block_end = block_start
        
        for end_candidate in (block_start + 1):n
            # Check if adding this variant maintains block criteria
            strong_ld_count = 0
            total_pairs = 0
            
            for i in block_start:end_candidate
                for j in (i+1):end_candidate
                    total_pairs += 1
                    if dprime_matrix[i, j] > 0.7
                        strong_ld_count += 1
                    end
                end
            end
            
            if total_pairs > 0 && strong_ld_count / total_pairs >= 0.95
                block_end = end_candidate
            else
                break
            end
        end
        
        if block_end > block_start
            push!(blocks, (variant_indices[block_start], variant_indices[block_end]))
        end
        
        block_start = block_end + 1
    end
    
    return blocks
end

"""Four-gamete test block detection."""
function four_gamete_blocks(gm::GenotypeMatrix, variant_indices::Vector{Int})
    n = length(variant_indices)
    blocks = Tuple{Int, Int}[]
    
    block_start = 1
    
    while block_start < n
        block_end = block_start
        
        for end_candidate in (block_start + 1):n
            # Check four-gamete rule between all pairs in potential block
            four_gamete_violated = false
            
            for i in block_start:(end_candidate-1)
                if four_gamete_violated
                    break
                end
                for j in (i+1):end_candidate
                    if has_four_gametes(gm.data[:, variant_indices[i]], 
                                       gm.data[:, variant_indices[j]])
                        four_gamete_violated = true
                        break
                    end
                end
            end
            
            if !four_gamete_violated
                block_end = end_candidate
            else
                break
            end
        end
        
        if block_end > block_start
            push!(blocks, (variant_indices[block_start], variant_indices[block_end]))
        end
        
        block_start = block_end + 1
    end
    
    return blocks
end

"""Check if two variants exhibit all four gametes (evidence of recombination)."""
function has_four_gametes(geno1::AbstractVector, geno2::AbstractVector)
    gametes = Set{Tuple{Int, Int}}()
    
    for i in eachindex(geno1)
        if ismissing(geno1[i]) || ismissing(geno2[i])
            continue
        end
        
        g1, g2 = geno1[i], geno2[i]
        
        # For homozygotes, we know the haplotypes
        if g1 == 0 && g2 == 0
            push!(gametes, (0, 0))
        elseif g1 == 0 && g2 == 2
            push!(gametes, (0, 1))
        elseif g1 == 2 && g2 == 0
            push!(gametes, (1, 0))
        elseif g1 == 2 && g2 == 2
            push!(gametes, (1, 1))
        end
        # Skip heterozygotes as they're ambiguous
    end
    
    return length(gametes) == 4
end

"""Solid spine of LD block detection."""
function solid_spine_blocks(gm::GenotypeMatrix, variant_indices::Vector{Int})
    n = length(variant_indices)
    blocks = Tuple{Int, Int}[]
    
    block_start = 1
    
    while block_start < n
        block_end = block_start
        
        for end_candidate in (block_start + 1):n
            # Check if first variant has strong LD with all others in block
            all_strong = true
            for j in (block_start + 1):end_candidate
                dprime = calculate_ld(gm.data[:, variant_indices[block_start]], 
                                     gm.data[:, variant_indices[j]]; 
                                     measure=:dprime)
                if isnan(dprime) || abs(dprime) < 0.8
                    all_strong = false
                    break
                end
            end
            
            if all_strong
                block_end = end_candidate
            else
                break
            end
        end
        
        if block_end > block_start
            push!(blocks, (variant_indices[block_start], variant_indices[block_end]))
        end
        
        block_start = block_end + 1
    end
    
    return blocks
end
