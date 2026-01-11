# ============================================================================
# NaturalSelection.jl - Selection Detection in Phylogenetics
# ============================================================================

"""
    dn_ds_ratio(seq1::Vector{Int}, seq2::Vector{Int}; genetic_code::Symbol=:standard)

Calculate dN/dS (Ka/Ks) ratio between two coding sequences.

# Arguments
- `seq1`, `seq2`: Coding DNA sequences (as codon indices or nucleotides)
- `genetic_code`: Genetic code to use

# Returns
- NamedTuple with dN, dS, and omega (dN/dS)
"""
function dn_ds_ratio(codons1::Vector{Int}, codons2::Vector{Int})
    @assert length(codons1) == length(codons2)
    
    n_syn = 0      # Synonymous differences
    n_nonsyn = 0   # Non-synonymous differences
    s_sites = 0.0  # Synonymous sites
    n_sites = 0.0  # Non-synonymous sites
    
    for i in eachindex(codons1)
        if codons1[i] != codons2[i]
            # Count difference type (simplified: assume single nucleotide change)
            if is_synonymous_change(codons1[i], codons2[i])
                n_syn += 1
            else
                n_nonsyn += 1
            end
        end
        
        # Count site types for this codon
        syn, nonsyn = count_site_types(codons1[i])
        s_sites += syn
        n_sites += nonsyn
    end
    
    # Calculate rates
    if s_sites > 0
        p_s = n_syn / s_sites
        dS = p_s < 0.75 ? -0.75 * log(1 - 4*p_s/3) : Inf  # JC correction
    else
        dS = 0.0
    end
    
    if n_sites > 0
        p_n = n_nonsyn / n_sites
        dN = p_n < 0.75 ? -0.75 * log(1 - 4*p_n/3) : Inf
    else
        dN = 0.0
    end
    
    omega = dS > 0 ? dN / dS : NaN
    
    return (dN=dN, dS=dS, omega=omega, n_syn=n_syn, n_nonsyn=n_nonsyn)
end

"""Check if codon change is synonymous (simplified)."""
function is_synonymous_change(codon1::Int, codon2::Int)
    # Simplified: map codons to amino acids
    # In practice, would use full genetic code
    aa1 = mod(codon1, 21) + 1  # Placeholder mapping
    aa2 = mod(codon2, 21) + 1
    return aa1 == aa2
end

"""Count synonymous and non-synonymous sites in a codon."""
function count_site_types(codon::Int)
    # Each codon has ~1 synonymous and ~2 non-synonymous sites on average
    return (syn=1.0, nonsyn=2.0)
end

"""
    site_selection_test(alignment::Matrix{Int}, tree::PhyloTree)

Test for selection at each site using site-specific models.
"""
function site_selection_test(alignment::Matrix{Int}, tree::PhyloTree)
    n_sites = size(alignment, 2)
    
    site_omega = Vector{Float64}(undef, n_sites)
    site_pvalue = Vector{Float64}(undef, n_sites)
    
    for site in 1:n_sites
        # Calculate omega for this site across all sequence pairs
        omegas = Float64[]
        
        for i in 1:(size(alignment, 1) - 1)
            for j in (i+1):size(alignment, 1)
                result = dn_ds_ratio([alignment[i, site]], [alignment[j, site]])
                if !isnan(result.omega) && !isinf(result.omega)
                    push!(omegas, result.omega)
                end
            end
        end
        
        if !isempty(omegas)
            site_omega[site] = mean(omegas)
            # Test if omega significantly > 1 (positive selection)
            if length(omegas) > 2
                t_stat = (mean(omegas) - 1) / (std(omegas) / sqrt(length(omegas)))
                site_pvalue[site] = ccdf(TDist(length(omegas) - 1), t_stat)
            else
                site_pvalue[site] = 1.0
            end
        else
            site_omega[site] = NaN
            site_pvalue[site] = NaN
        end
    end
    
    return (omega=site_omega, pvalue=site_pvalue)
end

"""
    branch_site_model(tree::PhyloTree, alignment::Matrix{Int}, foreground_branch::Int)

Test for positive selection on a specific branch.
"""
function branch_site_model(tree::PhyloTree, alignment::Matrix{Int}, foreground_branch::Int)
    # Simplified: compare omega on foreground vs background
    n_sites = size(alignment, 2)
    
    # This is a placeholder for the full branch-site model
    # Full implementation would use ML optimization
    
    foreground_omega = 1.5  # Placeholder
    background_omega = 0.2
    
    # Likelihood ratio test
    ll_alternative = -1000.0  # Placeholder
    ll_null = -1050.0
    
    lr_stat = 2 * (ll_alternative - ll_null)
    pvalue = ccdf(Chisq(1), lr_stat)
    
    return (foreground_omega=foreground_omega, background_omega=background_omega,
            lr_statistic=lr_stat, pvalue=pvalue)
end
