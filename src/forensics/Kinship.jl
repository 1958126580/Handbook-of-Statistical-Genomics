# ============================================================================
# Kinship.jl - Kinship and Relatedness Estimation
# ============================================================================

"""
    kinship_coefficient(gm::GenotypeMatrix, i::Int, j::Int)

Estimate kinship coefficient between two individuals using KING-robust.
"""
function kinship_coefficient(gm::GenotypeMatrix, i::Int, j::Int)
    n_var = n_variants(gm)
    
    n_aa_Aa = 0  # IBS0 where one is hom and other is het
    n_Aa_AA = 0
    n_het_het = 0  # Both heterozygous
    n_valid = 0
    
    for v in 1:n_var
        g1, g2 = gm.data[i, v], gm.data[j, v]
        if ismissing(g1) || ismissing(g2)
            continue
        end
        n_valid += 1
        
        if g1 == 1 && g2 == 1
            n_het_het += 1
        elseif (g1 == 0 && g2 == 2) || (g1 == 2 && g2 == 0)
            n_aa_Aa += 1  # IBS0
        end
    end
    
    if n_valid == 0
        return NaN
    end
    
    # KING-robust estimator
    kinship = (n_het_het - 2 * n_aa_Aa) / (n_het_het + 1e-10) / 4
    
    return kinship
end

"""
    ibd_estimation(gm::GenotypeMatrix, i::Int, j::Int)

Estimate IBD sharing proportions (Z0, Z1, Z2).
"""
function ibd_estimation(gm::GenotypeMatrix, i::Int, j::Int)
    n_var = n_variants(gm)
    
    ibs_counts = zeros(Int, 3)  # IBS0, IBS1, IBS2
    
    for v in 1:n_var
        g1, g2 = gm.data[i, v], gm.data[j, v]
        if ismissing(g1) || ismissing(g2)
            continue
        end
        
        ibs = 2 - abs(g1 - g2)
        ibs_counts[ibs + 1] += 1
    end
    
    total = sum(ibs_counts)
    if total == 0
        return (Z0=NaN, Z1=NaN, Z2=NaN)
    end
    
    # Simplified IBD estimation from IBS
    ibs_props = ibs_counts ./ total
    
    # These are approximate; full method requires allele frequencies
    Z2 = ibs_props[3] - 0.25  # Approximate
    Z0 = ibs_props[1]
    Z1 = 1 - Z0 - Z2
    
    return (Z0=max(0, Z0), Z1=max(0, Z1), Z2=max(0, min(1, Z2)))
end

"""
    relatedness_matrix(gm::GenotypeMatrix)

Calculate pairwise relatedness matrix for all samples.
"""
function relatedness_matrix(gm::GenotypeMatrix)
    n = n_samples(gm)
    K = Matrix{Float64}(undef, n, n)
    
    for i in 1:n
        K[i, i] = 0.5  # Self-kinship
        for j in (i+1):n
            k = kinship_coefficient(gm, i, j)
            K[i, j] = k
            K[j, i] = k
        end
    end
    
    return K
end
