# ============================================================================
# Selection.jl - Natural Selection Models
# ============================================================================

"""
    FitnessModel

Abstract type for fitness models.
"""
abstract type FitnessModel end

"""
    AdditiveSelection <: FitnessModel

Additive (codominant) selection model.
Fitness: AA=1, Aa=1+s, aa=1+2s
"""
struct AdditiveSelection <: FitnessModel
    s::Float64  # Selection coefficient per allele
    AdditiveSelection(s::Float64) = new(s)
end

"""
    DominantSelection <: FitnessModel

Dominant selection model.
Fitness: AA=1, Aa=1+s, aa=1+s
"""
struct DominantSelection <: FitnessModel
    s::Float64
    DominantSelection(s::Float64) = new(s)
end

"""
    RecessiveSelection <: FitnessModel

Recessive selection model.
Fitness: AA=1, Aa=1, aa=1+s
"""
struct RecessiveSelection <: FitnessModel
    s::Float64
    RecessiveSelection(s::Float64) = new(s)
end

"""
    OverdominantSelection <: FitnessModel

Overdominant (heterozygote advantage) selection.
Fitness: AA=1-s1, Aa=1, aa=1-s2
"""
struct OverdominantSelection <: FitnessModel
    s1::Float64  # Fitness reduction for AA
    s2::Float64  # Fitness reduction for aa
    OverdominantSelection(s1::Float64, s2::Float64) = new(s1, s2)
end

"""
    fitness(model::FitnessModel, genotype::Int)

Calculate fitness for a genotype (0=AA, 1=Aa, 2=aa).
"""
function fitness(model::AdditiveSelection, genotype::Int)
    return 1.0 + model.s * genotype
end

function fitness(model::DominantSelection, genotype::Int)
    return genotype == 0 ? 1.0 : 1.0 + model.s
end

function fitness(model::RecessiveSelection, genotype::Int)
    return genotype == 2 ? 1.0 + model.s : 1.0
end

function fitness(model::OverdominantSelection, genotype::Int)
    if genotype == 0
        return 1.0 - model.s1
    elseif genotype == 2
        return 1.0 - model.s2
    else
        return 1.0
    end
end

"""
    selection_coefficient(observed_freq::Float64, expected_freq::Float64, generations::Int)

Estimate selection coefficient from frequency change over time.
Uses Haldane's formula for single-generation change.
"""
function selection_coefficient(freq_initial::Float64, freq_final::Float64, 
                              generations::Int)
    if freq_initial <= 0 || freq_initial >= 1 || 
       freq_final <= 0 || freq_final >= 1
        return NaN
    end
    
    # For additive selection:  s ≈ (1/t) * ln(p_t(1-p_0) / (p_0(1-p_t)))
    s = (1 / generations) * log((freq_final * (1 - freq_initial)) / 
                                (freq_initial * (1 - freq_final)))
    return s
end

"""
    mean_fitness(model::FitnessModel, p::Float64)

Calculate population mean fitness given allele frequency p.
"""
function mean_fitness(model::AdditiveSelection, p::Float64)
    q = 1 - p
    w_AA = 1.0
    w_Aa = 1.0 + model.s
    w_aa = 1.0 + 2 * model.s
    return p^2 * w_AA + 2*p*q * w_Aa + q^2 * w_aa
end

function mean_fitness(model::OverdominantSelection, p::Float64)
    q = 1 - p
    return p^2 * (1 - model.s1) + 2*p*q * 1.0 + q^2 * (1 - model.s2)
end

"""
    equilibrium_frequency(model::OverdominantSelection)

Calculate equilibrium allele frequency for overdominant selection.
"""
function equilibrium_frequency(model::OverdominantSelection)
    return model.s2 / (model.s1 + model.s2)
end

"""
    selective_sweep_detect(allele_freqs::AbstractMatrix; window_size::Int=50)

Detect potential selective sweeps using extended haplotype homozygosity patterns.

# Arguments
- `allele_freqs`: Matrix of allele frequencies or haplotypes
- `window_size`: Size of sliding window

# Returns
- Vector of sweep scores for each position
"""
function selective_sweep_detect(gm::GenotypeMatrix; window_size::Int=50)
    n_var = n_variants(gm)
    scores = Vector{Float64}(undef, n_var)
    
    for j in 1:n_var
        # Calculate local statistics
        start_idx = max(1, j - window_size ÷ 2)
        end_idx = min(n_var, j + window_size ÷ 2)
        
        # Heterozygosity in window
        het_values = Float64[]
        for k in start_idx:end_idx
            genos = collect(skipmissing(gm.data[:, k]))
            if !isempty(genos)
                het = count(==(1), genos) / length(genos)
                push!(het_values, het)
            end
        end
        
        # Sweep signature: reduced heterozygosity
        if !isempty(het_values)
            scores[j] = -mean(het_values)  # Negative: low het = high score
        else
            scores[j] = 0.0
        end
    end
    
    # Standardize scores
    μ = mean(scores)
    σ = std(scores)
    if σ > 0
        scores = (scores .- μ) ./ σ
    end
    
    return scores
end

"""
    ihh(haplotypes::Matrix{Int8}, core_idx::Int; max_extend::Int=500)

Calculate integrated haplotype homozygosity (iHH) around a core SNP.
"""
function ihh(haplotypes::Matrix{Int8}, core_idx::Int; max_extend::Int=500)
    n_haps = size(haplotypes, 1)
    n_vars = size(haplotypes, 2)
    
    # Separate by core allele
    core_0_idx = findall(h -> haplotypes[h, core_idx] == 0, 1:n_haps)
    core_1_idx = findall(h -> haplotypes[h, core_idx] == 1, 1:n_haps)
    
    function calc_ehh(hap_indices::Vector{Int}, direction::Int)
        if length(hap_indices) <= 1
            return 0.0
        end
        
        ihh_val = 0.0
        current_idx = core_idx
        
        # Track which haplotypes are still homozygous
        active = Set(hap_indices)
        
        for step in 1:max_extend
            next_idx = current_idx + direction
            
            if next_idx < 1 || next_idx > n_vars
                break
            end
            
            # Group by allele at this position
            groups = Dict{Int8, Vector{Int}}()
            for h in active
                allele = haplotypes[h, next_idx]
                if !haskey(groups, allele)
                    groups[allele] = Int[]
                end
                push!(groups[allele], h)
            end
            
            # EHH = sum of (n_i choose 2) / (n choose 2)
            n_total = length(active)
            if n_total <= 1
                break
            end
            
            ehh = 0.0
            for (_, group) in groups
                n_g = length(group)
                if n_g > 1
                    ehh += n_g * (n_g - 1) / (n_total * (n_total - 1))
                end
            end
            
            # Integrate EHH
            ihh_val += ehh
            
            # Update active set (only keep largest group)
            if isempty(groups)
                break
            end
            max_group = argmax(length, values(groups))
            active = Set(max_group)
            current_idx = next_idx
            
            if ehh < 0.05
                break
            end
        end
        
        return ihh_val
    end
    
    # Calculate iHH for both alleles in both directions
    ihh_0 = calc_ehh(core_0_idx, 1) + calc_ehh(core_0_idx, -1)
    ihh_1 = calc_ehh(core_1_idx, 1) + calc_ehh(core_1_idx, -1)
    
    return (ihh_0=ihh_0, ihh_1=ihh_1)
end

"""
    ihs_score(gm::GenotypeMatrix)

Calculate integrated haplotype score (iHS) for each variant.
"""
function ihs_score(gm::GenotypeMatrix)
    n_var = n_variants(gm)
    ihs_values = Vector{Float64}(undef, n_var)
    
    # Convert genotypes to pseudo-haplotypes (using allele count)
    haplotypes = Matrix{Int8}(undef, 2 * n_samples(gm), n_var)
    for i in 1:n_samples(gm)
        for j in 1:n_var
            g = gm.data[i, j]
            if ismissing(g)
                haplotypes[2*i-1, j] = 0
                haplotypes[2*i, j] = 0
            elseif g == 0
                haplotypes[2*i-1, j] = 0
                haplotypes[2*i, j] = 0
            elseif g == 2
                haplotypes[2*i-1, j] = 1
                haplotypes[2*i, j] = 1
            else
                haplotypes[2*i-1, j] = 0
                haplotypes[2*i, j] = 1
            end
        end
    end
    
    for j in 1:n_var
        result = ihh(haplotypes, j)
        
        if result.ihh_0 > 0 && result.ihh_1 > 0
            ihs_values[j] = log(result.ihh_1 / result.ihh_0)
        else
            ihs_values[j] = 0.0
        end
    end
    
    # Standardize
    μ = mean(filter(!isnan, ihs_values))
    σ = std(filter(!isnan, ihs_values))
    if σ > 0
        ihs_values = (ihs_values .- μ) ./ σ
    end
    
    return ihs_values
end
