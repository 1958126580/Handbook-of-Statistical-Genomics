# ============================================================================
# ForensicDNA.jl - Forensic DNA Analysis
# ============================================================================

"""
    match_probability(profile1::Vector{String}, profile2::Vector{String},
                     allele_freqs::Dict{String, Dict{String, Float64}})

Calculate DNA profile match probability.
"""
function match_probability(profile1::Vector{String}, profile2::Vector{String},
                          allele_freqs::Dict{String, Dict{String, Float64}})
    if profile1 != profile2
        return 0.0
    end
    
    prob = 1.0
    for (locus, freqs) in allele_freqs
        # Assuming homozygous at each locus for simplicity
        p = get(freqs, profile1[1], 0.01)
        prob *= p^2
    end
    
    return prob
end

"""
    likelihood_ratio(evidence_profile::Vector{String}, suspect_profile::Vector{String},
                    allele_freqs::Dict{String, Float64})

Calculate likelihood ratio for forensic DNA comparison.
"""
function likelihood_ratio(evidence_profile::Vector{String}, 
                         suspect_profile::Vector{String},
                         allele_freqs::Dict{String, Float64})
    # Hp: Suspect is source
    # Hd: Random individual is source
    
    if evidence_profile != suspect_profile
        return 0.0  # Exclusion
    end
    
    # LR = 1 / P(profile | random)
    random_prob = 1.0
    for allele in evidence_profile
        p = get(allele_freqs, allele, 0.01)
        random_prob *= p
    end
    
    lr = 1 / (random_prob + 1e-20)
    return lr
end

"""
    str_analysis(str_data::Dict{String, Tuple{Int, Int}}, 
                population_freqs::Dict{String, Dict{Int, Float64}})

Analyze STR (Short Tandem Repeat) profile.
"""
function str_analysis(str_data::Dict{String, Tuple{Int, Int}},
                     population_freqs::Dict{String, Dict{Int, Float64}})
    profile_prob = 1.0
    
    for (locus, (a1, a2)) in str_data
        if !haskey(population_freqs, locus)
            continue
        end
        freqs = population_freqs[locus]
        p1 = get(freqs, a1, 0.01)
        p2 = get(freqs, a2, 0.01)
        
        if a1 == a2
            locus_prob = p1^2
        else
            locus_prob = 2 * p1 * p2
        end
        profile_prob *= locus_prob
    end
    
    rmp = profile_prob
    lr = 1 / rmp
    
    return (random_match_probability=rmp, likelihood_ratio=lr)
end
