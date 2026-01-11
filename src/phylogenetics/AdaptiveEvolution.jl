# ============================================================================
# AdaptiveEvolution.jl - Tests for Adaptive Evolution
# ============================================================================

"""
    mcdonald_kreitman_test(polymorphic_syn::Int, polymorphic_nonsyn::Int,
                          divergent_syn::Int, divergent_nonsyn::Int)

McDonald-Kreitman test for adaptive evolution.
Compares within-species polymorphism to between-species divergence.
"""
function mcdonald_kreitman_test(polymorphic_syn::Int, polymorphic_nonsyn::Int,
                               divergent_syn::Int, divergent_nonsyn::Int)
    # 2x2 contingency table
    #                 Synonymous  Non-synonymous
    # Polymorphic        Ps            Pn
    # Divergent          Ds            Dn
    
    Ps, Pn = polymorphic_syn, polymorphic_nonsyn
    Ds, Dn = divergent_syn, divergent_nonsyn
    
    # Neutrality index (NI)
    NI = (Pn * Ds) / (Ps * Dn + 1e-10)
    
    # Direction of selection (DoS)
    DoS = Dn / (Dn + Ds + 1e-10) - Pn / (Pn + Ps + 1e-10)
    
    # Fisher's exact test (simplified chi-squared)
    n = Ps + Pn + Ds + Dn
    expected_Ps = (Ps + Pn) * (Ps + Ds) / n
    expected_Pn = (Ps + Pn) * (Pn + Dn) / n
    expected_Ds = (Ds + Dn) * (Ps + Ds) / n
    expected_Dn = (Ds + Dn) * (Pn + Dn) / n
    
    chi2 = (Ps - expected_Ps)^2 / expected_Ps +
           (Pn - expected_Pn)^2 / expected_Pn +
           (Ds - expected_Ds)^2 / expected_Ds +
           (Dn - expected_Dn)^2 / expected_Dn
    
    pvalue = ccdf(Chisq(1), chi2)
    
    # Alpha: proportion of substitutions fixed by positive selection
    alpha = 1 - (Ds * Pn) / (Dn * Ps + 1e-10)
    alpha = clamp(alpha, 0.0, 1.0)
    
    return (neutrality_index=NI, direction_of_selection=DoS,
            alpha=alpha, chi_squared=chi2, pvalue=pvalue)
end

"""
    positive_selection_sites(alignment::Matrix{Int}; 
                            method::Symbol=:bayes_empirical_bayes)

Identify sites under positive selection.
"""
function positive_selection_sites(alignment::Matrix{Int}; method::Symbol=:simple)
    n_sequences, n_sites = size(alignment)
    
    site_scores = Vector{Float64}(undef, n_sites)
    
    for site in 1:n_sites
        site_data = alignment[:, site]
        
        # Calculate rate of nonsynonymous change at this site
        n_changes = 0
        for i in 1:(n_sequences-1)
            for j in (i+1):n_sequences
                if site_data[i] != site_data[j]
                    n_changes += 1
                end
            end
        end
        
        # Normalize by number of comparisons
        n_comparisons = n_sequences * (n_sequences - 1) / 2
        site_scores[site] = n_changes / n_comparisons
    end
    
    # Identify sites with high rate (potential positive selection)
    mean_score = mean(site_scores)
    std_score = std(site_scores)
    
    selected_sites = findall(s -> s > mean_score + 2 * std_score, site_scores)
    
    return (scores=site_scores, selected_sites=selected_sites,
            threshold=mean_score + 2*std_score)
end

"""
    adaptive_evolution_test(polymorphism_data::Vector{Float64}, 
                           divergence_data::Vector{Float64})

Test for adaptive evolution using polymorphism-divergence correlation.
"""
function adaptive_evolution_test(polymorphism_data::Vector{Float64},
                                divergence_data::Vector{Float64})
    @assert length(polymorphism_data) == length(divergence_data)
    
    # Correlation between polymorphism and divergence
    r = cor(polymorphism_data, divergence_data)
    
    n = length(polymorphism_data)
    t_stat = r * sqrt(n - 2) / sqrt(1 - r^2)
    pvalue = 2 * ccdf(TDist(n - 2), abs(t_stat))
    
    # Under neutrality, expect positive correlation
    # Negative correlation suggests selection
    selection_signal = r < 0 ? :positive_selection : :neutral_or_negative
    
    return (correlation=r, t_statistic=t_stat, pvalue=pvalue,
            interpretation=selection_signal)
end
