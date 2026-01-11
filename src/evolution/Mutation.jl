# ============================================================================
# Mutation.jl - Mutation Models
# ============================================================================

"""
    SubstitutionModel

Abstract type for nucleotide substitution models.
"""
abstract type SubstitutionModel <: AbstractEvolutionaryModel end

"""
    JC69 <: SubstitutionModel

Jukes-Cantor (1969) substitution model.
Equal rates for all substitutions, equal base frequencies.
"""
struct JC69 <: SubstitutionModel
    μ::Float64  # Overall mutation rate
    
    JC69(μ::Float64=1.0) = new(μ)
end

"""
    K80 <: SubstitutionModel

Kimura (1980) two-parameter model.
Distinguishes transitions from transversions.
"""
struct K80 <: SubstitutionModel
    κ::Float64  # Transition/transversion ratio
    μ::Float64  # Overall rate
    
    K80(κ::Float64=2.0, μ::Float64=1.0) = new(κ, μ)
end

"""
    HKY85 <: SubstitutionModel

Hasegawa-Kishino-Yano (1985) model.
Transitions/transversions + unequal base frequencies.
"""
struct HKY85 <: SubstitutionModel
    κ::Float64                 # Ti/Tv ratio
    π::NTuple{4, Float64}      # Base frequencies (A, C, G, T)
    μ::Float64                 # Overall rate
    
    function HKY85(κ::Float64=2.0, π::NTuple{4, Float64}=(0.25, 0.25, 0.25, 0.25),
                   μ::Float64=1.0)
        @assert sum(π) ≈ 1.0 "Base frequencies must sum to 1"
        new(κ, π, μ)
    end
end

"""
    GTR <: SubstitutionModel

General Time Reversible model.
Most general reversible model with 6 rate parameters.
"""
struct GTR <: SubstitutionModel
    rates::NTuple{6, Float64}  # Relative rates: AC, AG, AT, CG, CT, GT
    π::NTuple{4, Float64}      # Base frequencies
    μ::Float64                 # Overall rate
    
    function GTR(rates::NTuple{6, Float64}=(1.0,1.0,1.0,1.0,1.0,1.0),
                π::NTuple{4, Float64}=(0.25,0.25,0.25,0.25), μ::Float64=1.0)
        new(rates, π, μ)
    end
end

"""
    rate_matrix(model::SubstitutionModel)

Get the instantaneous rate matrix (Q matrix) for a substitution model.
Rows/columns ordered: A, C, G, T (indices 1-4)
"""
function rate_matrix(model::JC69)
    α = model.μ / 3
    Q = fill(α, 4, 4)
    for i in 1:4
        Q[i, i] = -3α
    end
    return Q
end

function rate_matrix(model::K80)
    α = model.μ * model.κ / (model.κ + 2)  # Transition rate
    β = model.μ / (model.κ + 2)             # Transversion rate
    
    # A=1, C=2, G=3, T=4
    Q = [
        -2β-α    β       α       β     ;  # From A
         β      -2β-α    β       α     ;  # From C
         α       β      -2β-α    β     ;  # From G
         β       α       β      -2β-α     # From T
    ]
    return Q
end

function rate_matrix(model::HKY85)
    κ, π, μ = model.κ, model.π, model.μ
    πA, πC, πG, πT = π
    
    Q = zeros(4, 4)
    # Q[i,j] = π[j] for transversions, κ*π[j] for transitions
    
    # Transitions: A↔G (purines), C↔T (pyrimidines)
    Q[1, 3] = κ * πG  # A→G
    Q[3, 1] = κ * πA  # G→A
    Q[2, 4] = κ * πT  # C→T
    Q[4, 2] = κ * πC  # T→C
    
    # Transversions
    Q[1, 2] = πC; Q[1, 4] = πT
    Q[2, 1] = πA; Q[2, 3] = πG
    Q[3, 2] = πC; Q[3, 4] = πT
    Q[4, 1] = πA; Q[4, 3] = πG
    
    # Diagonal: negative sum of row
    for i in 1:4
        Q[i, i] = -sum(Q[i, :])
    end
    
    # Scale to desired rate
    avg_rate = -sum(π[i] * Q[i, i] for i in 1:4)
    Q *= μ / avg_rate
    
    return Q
end

function rate_matrix(model::GTR)
    r, π = model.rates, model.π
    rAC, rAG, rAT, rCG, rCT, rGT = r
    πA, πC, πG, πT = π
    
    Q = [
        0       rAC*πC  rAG*πG  rAT*πT;
        rAC*πA  0       rCG*πG  rCT*πT;
        rAG*πA  rCG*πC  0       rGT*πT;
        rAT*πA  rCT*πC  rGT*πG  0
    ]
    
    for i in 1:4
        Q[i, i] = -sum(Q[i, :])
    end
    
    avg_rate = -sum(π[i] * Q[i, i] for i in 1:4)
    Q *= model.μ / avg_rate
    
    return Q
end

"""
    transition_probability_matrix(model::SubstitutionModel, t::Float64)

Calculate transition probability matrix P(t) = exp(Q*t).
"""
function transition_probability_matrix(model::SubstitutionModel, t::Float64)
    Q = rate_matrix(model)
    return exp(Q * t)
end

"""
    simulate_sequence(model::SubstitutionModel, ancestor::Vector{Int}, t::Float64)

Simulate sequence evolution from ancestor over time t.

# Arguments
- `model`: Substitution model
- `ancestor`: Ancestral sequence (1=A, 2=C, 3=G, 4=T)
- `t`: Evolutionary time (branch length)

# Returns
- Descendant sequence
"""
function simulate_sequence(model::SubstitutionModel, ancestor::Vector{Int}, t::Float64)
    P = transition_probability_matrix(model, t)
    descendant = similar(ancestor)
    
    for i in eachindex(ancestor)
        old_state = ancestor[i]
        # Sample new state according to transition probabilities
        probs = P[old_state, :]
        descendant[i] = sample(1:4, Weights(probs))
    end
    
    return descendant
end

"""
    mutation_rate_estimate(sequences::Vector{Vector{Int}})

Estimate mutation rate from aligned sequences using pairwise differences.
"""
function mutation_rate_estimate(sequences::Vector{Vector{Int}})
    n = length(sequences)
    L = length(sequences[1])
    
    total_diff = 0
    n_pairs = 0
    
    for i in 1:n
        for j in (i+1):n
            diff = sum(sequences[i] .!= sequences[j])
            total_diff += diff
            n_pairs += 1
        end
    end
    
    # Average pairwise difference
    π = total_diff / (n_pairs * L)
    
    return π
end

"""
    infinite_sites_model(n::Int, θ::Float64)

Simulate under infinite sites model using coalescent.

# Arguments
- `n`: Sample size
- `θ`: Population-scaled mutation rate (4Nμ)

# Returns
- Matrix of mutations (samples × segregating sites)
"""
function infinite_sites_model(n::Int, θ::Float64)
    # Generate coalescent tree
    n_lineages = n
    total_branch_length = 0.0
    branch_lengths = Float64[]
    branch_sizes = Int[]  # Number of descendants for each branch
    
    current_n = n
    while current_n > 1
        # Time to next coalescence
        rate = current_n * (current_n - 1) / 2
        t = rand(Exponential(1 / rate))
        
        # Add branch lengths
        for _ in 1:current_n
            push!(branch_lengths, t)
            push!(branch_sizes, current_n)
        end
        total_branch_length += current_n * t
        
        current_n -= 1
    end
    
    # Number of mutations (Poisson)
    expected_mutations = θ * total_branch_length / 2
    n_mutations = rand(Poisson(expected_mutations))
    
    if n_mutations == 0
        return zeros(Int8, n, 0)
    end
    
    # Place mutations on branches
    mutations = zeros(Int8, n, n_mutations)
    
    for m in 1:n_mutations
        # Choose random position on tree (proportional to branch length)
        pos = rand() * total_branch_length
        cumulative = 0.0
        
        for (i, bl) in enumerate(branch_lengths)
            cumulative += bl
            if cumulative >= pos
                # This mutation affects first branch_sizes[i] samples
                mutations[1:min(branch_sizes[i], n), m] .= 1
                break
            end
        end
    end
    
    return mutations
end

"""
    substitution_matrix(seq1::Vector{Int}, seq2::Vector{Int})

Calculate substitution count matrix between two sequences.
"""
function substitution_matrix(seq1::Vector{Int}, seq2::Vector{Int})
    @assert length(seq1) == length(seq2)
    
    counts = zeros(Int, 4, 4)
    for i in eachindex(seq1)
        counts[seq1[i], seq2[i]] += 1
    end
    
    return counts
end

"""
    jukes_cantor_distance(seq1::Vector{Int}, seq2::Vector{Int})

Calculate Jukes-Cantor corrected evolutionary distance.
"""
function jukes_cantor_distance(seq1::Vector{Int}, seq2::Vector{Int})
    p = sum(seq1 .!= seq2) / length(seq1)
    
    if p >= 0.75
        return Inf  # Saturation
    end
    
    return -0.75 * log(1 - 4*p/3)
end

"""
    kimura_distance(seq1::Vector{Int}, seq2::Vector{Int})

Calculate Kimura 2-parameter distance.
"""
function kimura_distance(seq1::Vector{Int}, seq2::Vector{Int})
    n_trans = 0  # Transitions
    n_transv = 0  # Transversions
    L = length(seq1)
    
    # Purines: A(1), G(3), Pyrimidines: C(2), T(4)
    is_purine(x) = x == 1 || x == 3
    
    for i in 1:L
        a, b = seq1[i], seq2[i]
        if a != b
            if is_purine(a) == is_purine(b)
                n_trans += 1
            else
                n_transv += 1
            end
        end
    end
    
    P = n_trans / L
    Q = n_transv / L
    
    if 1 - 2*P - Q <= 0 || 1 - 2*Q <= 0
        return Inf
    end
    
    return -0.5 * log((1 - 2*P - Q) * sqrt(1 - 2*Q))
end
