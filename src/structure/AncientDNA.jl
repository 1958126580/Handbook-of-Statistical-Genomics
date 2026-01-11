# ============================================================================
# AncientDNA.jl - Ancient DNA Analysis Methods
# ============================================================================

"""
    ancient_dna_damage(sequences::Vector{String}; reference::String="")

Estimate ancient DNA damage patterns (C→T and G→A substitutions).
"""
function ancient_dna_damage(sequences::Vector{String}; reference::String="")
    damage_patterns = Dict{String, Vector{Float64}}(
        "C_to_T_5prime" => Float64[],
        "G_to_A_3prime" => Float64[],
        "misincorporation" => Float64[]
    )
    
    for seq in sequences
        L = length(seq)
        if L < 10 || (length(reference) > 0 && length(reference) != L)
            continue
        end
        
        # Check 5' end C→T damage (first 10 bp)
        ct_5 = 0
        c_5 = 0
        for i in 1:min(10, L)
            if length(reference) > 0
                if reference[i] == 'C'
                    c_5 += 1
                    if seq[i] == 'T'
                        ct_5 += 1
                    end
                end
            end
        end
        
        if c_5 > 0
            push!(damage_patterns["C_to_T_5prime"], ct_5 / c_5)
        end
        
        # Check 3' end G→A damage (last 10 bp)
        ga_3 = 0
        g_3 = 0
        for i in max(1, L-9):L
            if length(reference) > 0
                if reference[i] == 'G'
                    g_3 += 1
                    if seq[i] == 'A'
                        ga_3 += 1
                    end
                end
            end
        end
        
        if g_3 > 0
            push!(damage_patterns["G_to_A_3prime"], ga_3 / g_3)
        end
    end
    
    return (
        C_to_T_5prime = isempty(damage_patterns["C_to_T_5prime"]) ? NaN : 
                        mean(damage_patterns["C_to_T_5prime"]),
        G_to_A_3prime = isempty(damage_patterns["G_to_A_3prime"]) ? NaN :
                        mean(damage_patterns["G_to_A_3prime"])
    )
end

"""
    contamination_estimate(gm::GenotypeMatrix, sample_idx::Int;
                          known_contaminant::Union{Vector{Int}, Nothing}=nothing)

Estimate contamination in ancient DNA sample.
Uses heterozygosity-based method for mtDNA or X chromosome in males.
"""
function contamination_estimate(gm::GenotypeMatrix, sample_idx::Int;
                               expected_het::Float64=0.0)
    # For haploid loci (mtDNA, male X), heterozygosity indicates contamination
    
    genos = gm.data[sample_idx, :]
    valid_genos = collect(skipmissing(genos))
    
    if isempty(valid_genos)
        return (contamination=NaN, se=NaN)
    end
    
    # Count heterozygotes
    n_het = count(==(1), valid_genos)
    n_total = length(valid_genos)
    
    # Observed heterozygosity rate
    obs_het = n_het / n_total
    
    # Expected het under no contamination is 0 for haploid
    # Contamination rate ≈ 2 * observed_het (simplified)
    if expected_het == 0.0
        contamination = 2 * obs_het
    else
        contamination = max(0, 2 * (obs_het - expected_het))
    end
    
    # Standard error (binomial)
    se = sqrt(obs_het * (1 - obs_het) / n_total) * 2
    
    return (contamination=min(contamination, 1.0), se=se, n_sites=n_total)
end

"""
    archaic_introgression(modern_gm::GenotypeMatrix, modern_idx::Vector{Int},
                         archaic_gm::GenotypeMatrix, archaic_idx::Int,
                         outgroup_idx::Vector{Int})

Detect archaic introgression using S* statistic approach.
"""
function archaic_introgression(modern_gm::GenotypeMatrix, modern_idx::Vector{Int},
                              archaic_idx::Int, outgroup_idx::Vector{Int})
    n_var = n_variants(modern_gm)
    
    introgression_score = Vector{Float64}(undef, n_var)
    
    for j in 1:n_var
        # Get alleles
        modern_alleles = [modern_gm.data[i, j] for i in modern_idx]
        modern_alleles = filter(!ismissing, modern_alleles)
        
        archaic_allele = modern_gm.data[archaic_idx, j]
        
        outgroup_alleles = [modern_gm.data[i, j] for i in outgroup_idx]
        outgroup_alleles = filter(!ismissing, outgroup_alleles)
        
        if isempty(modern_alleles) || ismissing(archaic_allele) || isempty(outgroup_alleles)
            introgression_score[j] = NaN
            continue
        end
        
        # Archaic-specific alleles: present in archaic but rare in outgroup
        archaic_derived = archaic_allele > 0
        outgroup_freq = sum(outgroup_alleles) / (2 * length(outgroup_alleles))
        
        if archaic_derived && outgroup_freq < 0.05
            # Check if this allele is at intermediate frequency in modern (potential introgression)
            modern_freq = sum(modern_alleles) / (2 * length(modern_alleles))
            introgression_score[j] = modern_freq
        else
            introgression_score[j] = 0.0
        end
    end
    
    # Summarize across genome
    valid_scores = filter(!isnan, introgression_score)
    
    return (mean_score=mean(valid_scores),
            n_archaic_sites=count(s -> s > 0.05, valid_scores),
            scores=introgression_score)
end
