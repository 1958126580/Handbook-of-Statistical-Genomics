# ============================================================================
# Mutation.jl - Nucleotide Substitution Models
# ============================================================================
# This module implements the major nucleotide substitution models used in
# molecular evolution and phylogenetics, as described in the Handbook of
# Statistical Genomics (4th Edition), Chapter on Molecular Evolution.
#
# Implemented models:
# - JC69: Jukes-Cantor (1969) - equal rates
# - K80:  Kimura (1980) - transition/transversion bias
# - F81:  Felsenstein (1981) - unequal base frequencies
# - HKY85: Hasegawa-Kishino-Yano (1985) - ts/tv + unequal frequencies
# - TN93: Tamura-Nei (1993) - two transition rates
# - GTR:  General Time Reversible - fully parameterized
# ============================================================================

"""
    SubstitutionModel

Abstract base type for all nucleotide substitution models.
All models must implement rate_matrix() and transition_probability_matrix().
"""
abstract type SubstitutionModel <: AbstractEvolutionaryModel end

# ============================================================================
# Jukes-Cantor Model (JC69)
# ============================================================================

"""
    JC69 <: SubstitutionModel

Jukes-Cantor (1969) substitution model.

The simplest substitution model assuming:
- Equal base frequencies (πA = πC = πG = πT = 0.25)
- Equal substitution rates for all changes

The rate matrix Q has the form:
    Q_ij = μ for i ≠ j
    Q_ii = -3μ

# Fields
- `μ::Float64`: Substitution rate (default 1.0 for normalized model)

# Example
```julia
model = JC69()
Q = rate_matrix(model)
P = transition_probability_matrix(model, 0.1)
```

# Reference
Jukes, T.H. and Cantor, C.R. (1969). Evolution of Protein Molecules.
Academic Press, New York.
"""
struct JC69 <: SubstitutionModel
    μ::Float64

    function JC69(μ::Float64=1.0)
        @assert μ > 0 "Substitution rate must be positive"
        new(μ)
    end
end

"""
    rate_matrix(model::JC69)

Return the instantaneous rate matrix Q for the JC69 model.

The matrix is normalized so that the average substitution rate is 1 per unit time,
meaning -Σ_i π_i Q_ii = 1 where π_i = 0.25 for all bases.

# Returns
- 4×4 Matrix{Float64} with rows/columns ordered A, C, G, T
"""
function rate_matrix(model::JC69)
    μ = model.μ
    # For a normalized JC69, μ = 1/3 gives average rate of 1
    # Q_ii = -3μ, so -0.25 * 4 * (-3μ) = 3μ = 1 when μ = 1/3
    α = μ / 3
    Q = fill(α, 4, 4)
    for i in 1:4
        Q[i, i] = -3α
    end
    return Q
end

"""
    stationary_distribution(model::JC69)

Return the stationary (equilibrium) distribution for JC69.
Always returns uniform distribution [0.25, 0.25, 0.25, 0.25].
"""
stationary_distribution(::JC69) = fill(0.25, 4)

"""
    transition_probability_matrix(model::JC69, t::Float64)

Calculate the transition probability matrix P(t) = exp(Qt) for JC69.

Uses the analytical solution:
- P_ii(t) = 1/4 + 3/4 * exp(-4μt/3)
- P_ij(t) = 1/4 - 1/4 * exp(-4μt/3) for i ≠ j

# Arguments
- `model`: JC69 model instance
- `t`: Branch length (evolutionary time)

# Returns
- 4×4 transition probability matrix
"""
function transition_probability_matrix(model::JC69, t::Float64)
    @assert t >= 0 "Time must be non-negative"
    μ = model.μ

    # Analytical solution derived from eigendecomposition
    e_term = exp(-4 * μ * t / 3)
    p_same = 0.25 + 0.75 * e_term
    p_diff = 0.25 - 0.25 * e_term

    P = fill(p_diff, 4, 4)
    for i in 1:4
        P[i, i] = p_same
    end

    return P
end

"""
    jukes_cantor_distance(p::Float64)

Calculate evolutionary distance using Jukes-Cantor correction.

Given observed proportion of differences p, the corrected distance is:
d = -3/4 * ln(1 - 4p/3)

This accounts for multiple hits (back mutations and parallel mutations)
that obscure the true evolutionary distance.

# Arguments
- `p`: Proportion of differing sites (0 ≤ p < 0.75)

# Returns
- Corrected evolutionary distance

# Notes
- Returns Inf if p ≥ 0.75 (saturation - too many differences to correct)
"""
function jukes_cantor_distance(p::Float64)
    if p < 0
        throw(ArgumentError("Proportion must be non-negative"))
    end
    if p >= 0.75
        return Inf  # Saturation - sequences too divergent
    end
    return -0.75 * log(1 - 4*p/3)
end

"""
    jukes_cantor_distance(seq1::Vector{Int}, seq2::Vector{Int})

Calculate Jukes-Cantor corrected evolutionary distance between two sequences.

# Arguments
- `seq1`, `seq2`: Sequences encoded as integers (1=A, 2=C, 3=G, 4=T)

# Returns
- JC69-corrected evolutionary distance
"""
function jukes_cantor_distance(seq1::Vector{Int}, seq2::Vector{Int})
    @assert length(seq1) == length(seq2) "Sequences must have equal length"
    p = sum(seq1 .!= seq2) / length(seq1)
    return jukes_cantor_distance(p)
end

# ============================================================================
# Kimura 2-Parameter Model (K80)
# ============================================================================

"""
    K80 <: SubstitutionModel

Kimura (1980) 2-parameter model.

Distinguishes between:
- Transitions (A↔G, C↔T): rate α
- Transversions (A↔C, A↔T, G↔C, G↔T): rate β

The transition/transversion ratio κ = α/β typically ranges from 2-10 for
most organisms, reflecting the biochemical differences between these
types of mutations.

# Fields
- `κ::Float64`: Transition/transversion ratio (default 2.0)
- `μ::Float64`: Overall substitution rate (default 1.0)

# Example
```julia
model = K80(2.0)  # κ = 2, transitions twice as frequent as transversions
Q = rate_matrix(model)
d = kimura_distance(0.1, 0.05)  # 10% transitions, 5% transversions
```

# Reference
Kimura, M. (1980). A simple method for estimating evolutionary rates
of base substitutions through comparative studies of nucleotide sequences.
Journal of Molecular Evolution, 16:111-120.
"""
struct K80 <: SubstitutionModel
    κ::Float64  # Transition/transversion ratio
    μ::Float64  # Overall rate

    function K80(κ::Float64=2.0, μ::Float64=1.0)
        @assert κ > 0 "κ must be positive"
        @assert μ > 0 "μ must be positive"
        new(κ, μ)
    end
end

function rate_matrix(model::K80)
    κ, μ = model.κ, model.μ

    # Normalize rates so average substitution rate equals μ
    # With equal frequencies: avg_rate = 0.25 * (α + 2β) * 4 = α + 2β
    # We want α + 2β = μ, with κ = α/β
    # Solving: α = μκ/(κ+2), β = μ/(κ+2)
    α = μ * κ / (κ + 2)  # Transition rate
    β = μ / (κ + 2)       # Transversion rate

    # Order: A=1, C=2, G=3, T=4
    # Transitions: A↔G (1↔3), C↔T (2↔4)
    # Transversions: all others
    Q = [
        -(α + 2β)  β          α          β;
        β          -(α + 2β)  β          α;
        α          β          -(α + 2β)  β;
        β          α          β          -(α + 2β)
    ]

    return Q
end

stationary_distribution(::K80) = fill(0.25, 4)

function transition_probability_matrix(model::K80, t::Float64)
    @assert t >= 0 "Time must be non-negative"
    κ, μ = model.κ, model.μ

    α = μ * κ / (κ + 2)
    β = μ / (κ + 2)

    # Analytical solutions from eigendecomposition
    e1 = exp(-4β * t)
    e2 = exp(-2(α + β) * t)

    p0 = 0.25 + 0.25 * e1 + 0.5 * e2      # Same nucleotide
    p1 = 0.25 + 0.25 * e1 - 0.5 * e2      # Transition
    p2 = 0.25 - 0.25 * e1                  # Transversion

    # Order: A=1, C=2, G=3, T=4
    P = [
        p0 p2 p1 p2;
        p2 p0 p2 p1;
        p1 p2 p0 p2;
        p2 p1 p2 p0
    ]

    return P
end

"""
    kimura_distance(P::Float64, Q::Float64)

Calculate evolutionary distance using Kimura 2-parameter correction.

# Arguments
- `P`: Proportion of transitions
- `Q`: Proportion of transversions

# Returns
- Corrected evolutionary distance

# Notes
- Returns Inf if correction formula becomes undefined (saturation)
"""
function kimura_distance(P::Float64, Q::Float64)
    # Check for saturation
    if 1 - 2*P - Q <= 0 || 1 - 2*Q <= 0
        return Inf
    end

    # Kimura's formula
    d = -0.5 * log((1 - 2*P - Q) * sqrt(1 - 2*Q))
    return d
end

"""
    kimura_distance(seq1::Vector{Int}, seq2::Vector{Int})

Calculate Kimura 2-parameter distance between two sequences.
"""
function kimura_distance(seq1::Vector{Int}, seq2::Vector{Int})
    @assert length(seq1) == length(seq2) "Sequences must have equal length"
    L = length(seq1)

    n_trans = 0   # Transitions
    n_transv = 0  # Transversions

    # Purines: A(1), G(3), Pyrimidines: C(2), T(4)
    is_purine(x) = x == 1 || x == 3

    for i in 1:L
        a, b = seq1[i], seq2[i]
        if a != b
            if is_purine(a) == is_purine(b)
                n_trans += 1  # Both purines or both pyrimidines = transition
            else
                n_transv += 1  # One purine, one pyrimidine = transversion
            end
        end
    end

    P = n_trans / L   # Proportion of transitions
    Q = n_transv / L  # Proportion of transversions

    return kimura_distance(P, Q)
end

# ============================================================================
# Felsenstein 1981 Model (F81)
# ============================================================================

"""
    F81 <: SubstitutionModel

Felsenstein (1981) model with unequal base frequencies.

Like JC69, assumes all substitutions occur at the same rate, but accounts for
unequal equilibrium base frequencies. This is important when analyzing
genomes with biased composition (e.g., AT-rich or GC-rich).

# Fields
- `π::NTuple{4, Float64}`: Base frequencies [πA, πC, πG, πT]
- `μ::Float64`: Overall substitution rate

# Example
```julia
model = F81((0.3, 0.2, 0.2, 0.3))  # AT-rich genome
```

# Reference
Felsenstein, J. (1981). Evolutionary trees from DNA sequences:
a maximum likelihood approach. Journal of Molecular Evolution, 17:368-376.
"""
struct F81 <: SubstitutionModel
    π::NTuple{4, Float64}
    μ::Float64

    function F81(π::NTuple{4, Float64}=(0.25, 0.25, 0.25, 0.25), μ::Float64=1.0)
        @assert all(x -> x >= 0, π) "Frequencies must be non-negative"
        @assert abs(sum(π) - 1.0) < 1e-10 "Frequencies must sum to 1"
        @assert μ > 0 "Rate must be positive"
        new(π, μ)
    end
end

# Constructor from Vector
F81(π::Vector{Float64}, μ::Float64=1.0) = F81(Tuple(π), μ)

function rate_matrix(model::F81)
    π = collect(model.π)
    μ = model.μ

    Q = zeros(4, 4)

    # Off-diagonal: Q_ij = μ * π_j
    for i in 1:4
        for j in 1:4
            if i != j
                Q[i, j] = π[j]
            end
        end
        Q[i, i] = -sum(Q[i, :])
    end

    # Normalize to desired rate
    avg_rate = -sum(π[i] * Q[i, i] for i in 1:4)
    Q .*= μ / avg_rate

    return Q
end

stationary_distribution(model::F81) = collect(model.π)

function transition_probability_matrix(model::F81, t::Float64)
    π = collect(model.π)
    μ = model.μ

    # β = 1 / (1 - Σπ²)
    β = 1.0 / (1.0 - sum(π .^ 2))
    e_term = exp(-β * μ * t)

    P = zeros(4, 4)
    for i in 1:4
        for j in 1:4
            if i == j
                P[i, j] = e_term + π[j] * (1 - e_term)
            else
                P[i, j] = π[j] * (1 - e_term)
            end
        end
    end

    return P
end

# ============================================================================
# HKY85 Model
# ============================================================================

"""
    HKY85 <: SubstitutionModel

Hasegawa-Kishino-Yano (1985) model.

The most commonly used substitution model, combining:
- Transition/transversion bias (like K80)
- Unequal base frequencies (like F81)

This captures the two most important aspects of molecular evolution:
1. Transitions are more common than transversions
2. Base composition varies between genomes

# Fields
- `κ::Float64`: Transition/transversion ratio
- `π::NTuple{4, Float64}`: Base frequencies [πA, πC, πG, πT]
- `μ::Float64`: Overall substitution rate

# Example
```julia
model = HKY85(2.0, (0.3, 0.2, 0.2, 0.3))
```

# Reference
Hasegawa, M., Kishino, H., and Yano, T. (1985). Dating of the human-ape
splitting by a molecular clock of mitochondrial DNA in humans and chimpanzees.
Journal of Molecular Evolution, 22:160-174.
"""
struct HKY85 <: SubstitutionModel
    κ::Float64                  # Transition/transversion ratio
    π::NTuple{4, Float64}       # Base frequencies (A, C, G, T)
    μ::Float64                  # Overall substitution rate

    function HKY85(κ::Float64=2.0, π::NTuple{4, Float64}=(0.25, 0.25, 0.25, 0.25),
                   μ::Float64=1.0)
        @assert κ > 0 "κ must be positive"
        @assert all(x -> x >= 0, π) "Frequencies must be non-negative"
        @assert abs(sum(π) - 1.0) < 1e-10 "Frequencies must sum to 1"
        @assert μ > 0 "Rate must be positive"
        new(κ, π, μ)
    end
end

# Constructor from Vector
HKY85(κ::Float64, π::Vector{Float64}, μ::Float64=1.0) = HKY85(κ, Tuple(π), μ)

function rate_matrix(model::HKY85)
    κ, μ = model.κ, model.μ
    πA, πC, πG, πT = model.π

    Q = zeros(4, 4)

    # Transitions: A↔G (purines), C↔T (pyrimidines)
    Q[1, 3] = κ * πG  # A→G
    Q[3, 1] = κ * πA  # G→A
    Q[2, 4] = κ * πT  # C→T
    Q[4, 2] = κ * πC  # T→C

    # Transversions
    Q[1, 2] = πC      # A→C
    Q[1, 4] = πT      # A→T
    Q[2, 1] = πA      # C→A
    Q[2, 3] = πG      # C→G
    Q[3, 2] = πC      # G→C
    Q[3, 4] = πT      # G→T
    Q[4, 1] = πA      # T→A
    Q[4, 3] = πG      # T→G

    # Diagonal: negative sum of row
    for i in 1:4
        Q[i, i] = -sum(Q[i, :])
    end

    # Normalize to desired rate
    π = collect(model.π)
    avg_rate = -sum(π[i] * Q[i, i] for i in 1:4)
    Q .*= μ / avg_rate

    return Q
end

stationary_distribution(model::HKY85) = collect(model.π)

function transition_probability_matrix(model::HKY85, t::Float64)
    # Use matrix exponential for accurate computation
    Q = rate_matrix(model)
    return exp(Q * t)
end

# ============================================================================
# Tamura-Nei 1993 Model (TN93)
# ============================================================================

"""
    TN93 <: SubstitutionModel

Tamura-Nei (1993) model.

Extends HKY85 by allowing different rates for:
- Purine transitions (A↔G): rate α₁
- Pyrimidine transitions (C↔T): rate α₂
- All transversions: rate β

This is particularly useful for mitochondrial DNA where the two types
of transitions often occur at different rates.

# Fields
- `α1::Float64`: Purine transition rate parameter
- `α2::Float64`: Pyrimidine transition rate parameter
- `β::Float64`: Transversion rate parameter (often set to 1)
- `π::NTuple{4, Float64}`: Base frequencies

# Example
```julia
model = TN93(2.0, 3.0, 1.0, (0.25, 0.25, 0.25, 0.25))
```

# Reference
Tamura, K. and Nei, M. (1993). Estimation of the number of nucleotide
substitutions in the control region of mitochondrial DNA in humans
and chimpanzees. Molecular Biology and Evolution, 10:512-526.
"""
struct TN93 <: SubstitutionModel
    α1::Float64  # Purine transition rate
    α2::Float64  # Pyrimidine transition rate
    β::Float64   # Transversion rate
    π::NTuple{4, Float64}
    μ::Float64   # Overall rate

    function TN93(α1::Float64=2.0, α2::Float64=2.0, β::Float64=1.0,
                  π::NTuple{4, Float64}=(0.25, 0.25, 0.25, 0.25),
                  μ::Float64=1.0)
        @assert α1 > 0 && α2 > 0 && β > 0 "Rate parameters must be positive"
        @assert all(x -> x >= 0, π) && abs(sum(π) - 1) < 1e-10
        @assert μ > 0 "Rate must be positive"
        new(α1, α2, β, π, μ)
    end
end

function rate_matrix(model::TN93)
    α1, α2, β, μ = model.α1, model.α2, model.β, model.μ
    πA, πC, πG, πT = model.π

    Q = zeros(4, 4)

    # Purine transitions (A↔G)
    Q[1, 3] = α1 * πG
    Q[3, 1] = α1 * πA

    # Pyrimidine transitions (C↔T)
    Q[2, 4] = α2 * πT
    Q[4, 2] = α2 * πC

    # All transversions
    Q[1, 2] = β * πC
    Q[1, 4] = β * πT
    Q[2, 1] = β * πA
    Q[2, 3] = β * πG
    Q[3, 2] = β * πC
    Q[3, 4] = β * πT
    Q[4, 1] = β * πA
    Q[4, 3] = β * πG

    # Diagonal
    for i in 1:4
        Q[i, i] = -sum(Q[i, :])
    end

    # Normalize
    π = collect(model.π)
    avg_rate = -sum(π[i] * Q[i, i] for i in 1:4)
    Q .*= μ / avg_rate

    return Q
end

stationary_distribution(model::TN93) = collect(model.π)

function transition_probability_matrix(model::TN93, t::Float64)
    Q = rate_matrix(model)
    return exp(Q * t)
end

# ============================================================================
# General Time Reversible Model (GTR)
# ============================================================================

"""
    GTR <: SubstitutionModel

General Time Reversible model.

The most general time-reversible nucleotide substitution model with:
- 4 base frequency parameters (3 free after constraint Σπᵢ = 1)
- 6 exchangeability parameters (5 free after normalization)

The rate matrix satisfies detailed balance: πᵢ Qᵢⱼ = πⱼ Qⱼᵢ

This model includes all other time-reversible models as special cases:
- JC69: all rates equal, equal frequencies
- K80: rAG = rCT ≠ others, equal frequencies
- F81: all rates equal, unequal frequencies
- HKY85: rAG = rCT ≠ others, unequal frequencies

# Fields
- `rates::NTuple{6, Float64}`: Exchangeability parameters [rAC, rAG, rAT, rCG, rCT, rGT]
- `π::NTuple{4, Float64}`: Base frequencies [πA, πC, πG, πT]
- `μ::Float64`: Overall substitution rate

# Rate Matrix Construction
Q_ij = r_ij * π_j for i ≠ j (where r_ij = r_ji)

# Example
```julia
# Create GTR with higher transition rates
rates = (1.0, 4.0, 1.0, 1.0, 4.0, 1.0)  # rAG and rCT are 4× transversion rates
π = (0.3, 0.2, 0.2, 0.3)
model = GTR(rates, π)
```

# Reference
Tavaré, S. (1986). Some probabilistic and statistical problems in the
analysis of DNA sequences. Lectures on Mathematics in the Life Sciences, 17:57-86.
"""
struct GTR <: SubstitutionModel
    rates::NTuple{6, Float64}  # Relative rates: AC, AG, AT, CG, CT, GT
    π::NTuple{4, Float64}      # Base frequencies
    μ::Float64                 # Overall rate

    function GTR(rates::NTuple{6, Float64}=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                 π::NTuple{4, Float64}=(0.25, 0.25, 0.25, 0.25),
                 μ::Float64=1.0)
        @assert all(x -> x > 0, rates) "Rates must be positive"
        @assert all(x -> x >= 0, π) && abs(sum(π) - 1) < 1e-10
        @assert μ > 0 "Rate must be positive"
        new(rates, π, μ)
    end
end

# Constructor from Vectors
GTR(rates::Vector{Float64}, π::Vector{Float64}, μ::Float64=1.0) =
    GTR(Tuple(rates), Tuple(π), μ)

function rate_matrix(model::GTR)
    rAC, rAG, rAT, rCG, rCT, rGT = model.rates
    πA, πC, πG, πT = model.π
    μ = model.μ

    # Build Q matrix
    # Q[i,j] = r[i,j] * π[j] for i ≠ j
    Q = [
        0       rAC*πC  rAG*πG  rAT*πT;
        rAC*πA  0       rCG*πG  rCT*πT;
        rAG*πA  rCG*πC  0       rGT*πT;
        rAT*πA  rCT*πC  rGT*πG  0
    ]

    # Diagonal
    for i in 1:4
        Q[i, i] = -sum(Q[i, :])
    end

    # Normalize
    π = collect(model.π)
    avg_rate = -sum(π[i] * Q[i, i] for i in 1:4)
    Q .*= μ / avg_rate

    return Q
end

stationary_distribution(model::GTR) = collect(model.π)

function transition_probability_matrix(model::GTR, t::Float64)
    Q = rate_matrix(model)
    return exp(Q * t)
end

# ============================================================================
# Sequence Simulation
# ============================================================================

"""
    generate_random_sequence(n::Int, π::Vector{Float64}=fill(0.25, 4))

Generate a random nucleotide sequence from given base frequencies.

# Arguments
- `n`: Sequence length
- `π`: Base frequencies [πA, πC, πG, πT] (default: uniform)

# Returns
- Vector{Int} with values 1-4 representing A, C, G, T
"""
function generate_random_sequence(n::Int, π::Vector{Float64}=fill(0.25, 4))
    @assert length(π) == 4 && abs(sum(π) - 1) < 1e-10
    return sample(1:4, Weights(π), n)
end

"""
    simulate_sequence(model::SubstitutionModel, ancestor::Vector{Int}, t::Float64)

Simulate sequence evolution from ancestor over evolutionary time t.

# Arguments
- `model`: Substitution model
- `ancestor`: Ancestral sequence (1=A, 2=C, 3=G, 4=T)
- `t`: Evolutionary time (branch length)

# Returns
- Evolved descendant sequence

# Example
```julia
model = HKY85(2.0)
ancestor = generate_random_sequence(1000)
descendant = simulate_sequence(model, ancestor, 0.1)
```
"""
function simulate_sequence(model::SubstitutionModel, ancestor::Vector{Int}, t::Float64)
    @assert all(x -> 1 <= x <= 4, ancestor) "Sequence must contain values 1-4"
    @assert t >= 0 "Time must be non-negative"

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
    simulate_alignment(model::SubstitutionModel, n_taxa::Int, seq_length::Int;
                      tree_height::Float64=1.0, topology::Symbol=:star)

Simulate a sequence alignment by evolving sequences along a tree.

# Arguments
- `model`: Substitution model
- `n_taxa`: Number of taxa (sequences to generate)
- `seq_length`: Length of each sequence
- `tree_height`: Height of the tree
- `topology`: Tree topology (:star for star tree, :random for random coalescent tree)

# Returns
- Dict mapping taxon names to sequences
"""
function simulate_alignment(model::SubstitutionModel, n_taxa::Int, seq_length::Int;
                           tree_height::Float64=1.0, topology::Symbol=:star)
    # Generate root sequence from stationary distribution
    π = stationary_distribution(model)
    root_seq = generate_random_sequence(seq_length, π)

    sequences = Dict{String, Vector{Int}}()

    if topology == :star
        # Star tree: all taxa directly connected to root
        for i in 1:n_taxa
            branch_length = tree_height
            evolved_seq = simulate_sequence(model, root_seq, branch_length)
            sequences["Taxon_$i"] = evolved_seq
        end
    else
        # Random coalescent-like tree
        current_seqs = [root_seq]

        for i in 1:n_taxa
            # Pick a random ancestor and evolve from it
            ancestor = current_seqs[rand(1:length(current_seqs))]
            branch_length = tree_height * rand()
            evolved_seq = simulate_sequence(model, ancestor, branch_length)
            push!(current_seqs, evolved_seq)
            sequences["Taxon_$i"] = evolved_seq
        end
    end

    return sequences
end

# ============================================================================
# Infinite Sites Model
# ============================================================================

"""
    infinite_sites_model(n::Int, θ::Float64)

Simulate under the infinite sites mutation model using a coalescent tree.

Under the infinite sites model, each mutation occurs at a unique site
(no recurrent mutations). This is a good approximation when:
- Sequence length >> number of mutations
- θ = 4Neμ is small

# Arguments
- `n`: Sample size (number of sequences)
- `θ`: Population-scaled mutation rate (θ = 4Neμ for diploids)

# Returns
- Matrix{Int8} of mutations (samples × segregating sites)
  - 0 = ancestral allele
  - 1 = derived allele
"""
function infinite_sites_model(n::Int, θ::Float64)
    @assert n >= 2 "Need at least 2 samples"
    @assert θ > 0 "θ must be positive"

    # Generate coalescent tree and track lineage membership
    # Each branch is characterized by: start time, end time, lineage size
    branches = Vector{Tuple{Float64, Float64, Int}}()
    total_branch_length = 0.0

    current_n = n
    current_time = 0.0

    # Lineage tracking: which samples descend from each branch
    # Start with n singleton lineages
    lineage_members = [Set([i]) for i in 1:n]

    while current_n > 1
        # Time to next coalescence
        rate = current_n * (current_n - 1) / 2
        wait_time = rand(Exponential(1 / rate))

        # Record branches for this interval
        for i in 1:current_n
            push!(branches, (current_time, current_time + wait_time, length(lineage_members[i])))
        end

        total_branch_length += current_n * wait_time
        current_time += wait_time

        # Randomly choose two lineages to coalesce
        idx = sample(1:current_n, 2, replace=false)
        sort!(idx)

        # Merge lineages
        merged = union(lineage_members[idx[1]], lineage_members[idx[2]])
        deleteat!(lineage_members, idx[2])
        lineage_members[idx[1]] = merged

        current_n -= 1
    end

    # Number of mutations (Poisson with mean θ * total_length / 2)
    expected_mutations = θ * total_branch_length / 2
    n_mutations = rand(Poisson(expected_mutations))

    if n_mutations == 0
        return zeros(Int8, n, 0)
    end

    # Place mutations on branches
    mutations = zeros(Int8, n, n_mutations)

    # Cumulative branch lengths for uniform sampling
    cum_lengths = cumsum([b[2] - b[1] for b in branches])
    total_length = cum_lengths[end]

    for m in 1:n_mutations
        # Choose random position on tree
        pos = rand() * total_length
        branch_idx = searchsortedfirst(cum_lengths, pos)
        branch_idx = min(branch_idx, length(branches))

        # Determine which samples carry this mutation
        # (simplified: use branch size as approximation)
        n_carriers = min(branches[branch_idx][3], n)
        carriers = sample(1:n, n_carriers, replace=false)
        mutations[carriers, m] .= 1
    end

    return mutations
end

# ============================================================================
# Mutation Rate Estimation
# ============================================================================

"""
    estimate_mutation_rate(sequences::Vector{Vector{Int}})

Estimate mutation rate from aligned sequences using average pairwise differences.

This estimates π (nucleotide diversity), which equals θ = 4Neμ under neutrality.

# Arguments
- `sequences`: Vector of aligned sequences

# Returns
- Estimated π (nucleotide diversity)
"""
function estimate_mutation_rate(sequences::Vector{Vector{Int}})
    n = length(sequences)
    if n < 2
        return 0.0
    end

    L = length(sequences[1])
    @assert all(s -> length(s) == L, sequences) "All sequences must have equal length"

    total_diff = 0
    n_pairs = 0

    for i in 1:n
        for j in (i+1):n
            diff = sum(sequences[i] .!= sequences[j])
            total_diff += diff
            n_pairs += 1
        end
    end

    # Average pairwise difference per site
    π = total_diff / (n_pairs * L)

    return π
end

"""
    watterson_theta(S::Int, n::Int, L::Int)

Calculate Watterson's estimator of θ from segregating sites.

θ̂_W = S / (a_n × L)

where a_n = Σᵢ₌₁ⁿ⁻¹ 1/i

# Arguments
- `S`: Number of segregating sites
- `n`: Sample size
- `L`: Sequence length
"""
function watterson_theta(S::Int, n::Int, L::Int)
    if n < 2
        return 0.0
    end
    a_n = sum(1.0 / i for i in 1:(n-1))
    return S / (a_n * L)
end

"""
    substitution_matrix(seq1::Vector{Int}, seq2::Vector{Int})

Calculate the 4×4 substitution count matrix between two sequences.

Entry (i,j) counts the number of positions where seq1 has base i and seq2 has base j.

# Returns
- 4×4 count matrix
"""
function substitution_matrix(seq1::Vector{Int}, seq2::Vector{Int})
    @assert length(seq1) == length(seq2) "Sequences must have equal length"

    counts = zeros(Int, 4, 4)
    for i in eachindex(seq1)
        counts[seq1[i], seq2[i]] += 1
    end

    return counts
end

# ============================================================================
# Genetic Code and Synonymous/Nonsynonymous Substitutions
# ============================================================================

"""
    GENETIC_CODE

Standard genetic code mapping codons to amino acids.
Uses single-letter amino acid codes. '*' represents stop codons.
"""
const GENETIC_CODE = Dict{String, Char}(
    "TTT" => 'F', "TTC" => 'F', "TTA" => 'L', "TTG" => 'L',
    "TCT" => 'S', "TCC" => 'S', "TCA" => 'S', "TCG" => 'S',
    "TAT" => 'Y', "TAC" => 'Y', "TAA" => '*', "TAG" => '*',
    "TGT" => 'C', "TGC" => 'C', "TGA" => '*', "TGG" => 'W',
    "CTT" => 'L', "CTC" => 'L', "CTA" => 'L', "CTG" => 'L',
    "CCT" => 'P', "CCC" => 'P', "CCA" => 'P', "CCG" => 'P',
    "CAT" => 'H', "CAC" => 'H', "CAA" => 'Q', "CAG" => 'Q',
    "CGT" => 'R', "CGC" => 'R', "CGA" => 'R', "CGG" => 'R',
    "ATT" => 'I', "ATC" => 'I', "ATA" => 'I', "ATG" => 'M',
    "ACT" => 'T', "ACC" => 'T', "ACA" => 'T', "ACG" => 'T',
    "AAT" => 'N', "AAC" => 'N', "AAA" => 'K', "AAG" => 'K',
    "AGT" => 'S', "AGC" => 'S', "AGA" => 'R', "AGG" => 'R',
    "GTT" => 'V', "GTC" => 'V', "GTA" => 'V', "GTG" => 'V',
    "GCT" => 'A', "GCC" => 'A', "GCA" => 'A', "GCG" => 'A',
    "GAT" => 'D', "GAC" => 'D', "GAA" => 'E', "GAG" => 'E',
    "GGT" => 'G', "GGC" => 'G', "GGA" => 'G', "GGG" => 'G'
)

"""
    translate_codon(codon::String)

Translate a DNA codon to its amino acid.

# Returns
- Single character amino acid code, or 'X' for invalid codons
"""
function translate_codon(codon::String)
    return get(GENETIC_CODE, uppercase(codon), 'X')
end

"""
    dn_ds_ratio(seq1::String, seq2::String)

Calculate dN/dS (ω) ratio between two coding sequences.

This is a simplified implementation using the Nei-Gojobori method.

# Arguments
- `seq1`, `seq2`: Aligned coding DNA sequences (length must be multiple of 3)

# Returns
- NamedTuple with:
  - `dN`: Nonsynonymous substitution rate
  - `dS`: Synonymous substitution rate
  - `omega`: dN/dS ratio
  - `n_syn`: Number of synonymous differences
  - `n_nonsyn`: Number of nonsynonymous differences

# Interpretation of ω
- ω < 1: Purifying (negative) selection
- ω ≈ 1: Neutral evolution
- ω > 1: Positive (diversifying) selection
"""
function dn_ds_ratio(seq1::String, seq2::String)
    @assert length(seq1) == length(seq2) "Sequences must have equal length"
    @assert length(seq1) % 3 == 0 "Sequence length must be multiple of 3"

    n_codons = length(seq1) ÷ 3
    n_syn = 0
    n_nonsyn = 0

    for i in 1:n_codons
        start_idx = (i - 1) * 3 + 1
        end_idx = start_idx + 2

        codon1 = uppercase(seq1[start_idx:end_idx])
        codon2 = uppercase(seq2[start_idx:end_idx])

        if codon1 == codon2
            continue
        end

        aa1 = translate_codon(codon1)
        aa2 = translate_codon(codon2)

        if aa1 == 'X' || aa2 == 'X'
            continue  # Skip invalid codons
        end

        if aa1 == aa2
            n_syn += 1
        else
            n_nonsyn += 1
        end
    end

    # Approximate number of synonymous and nonsynonymous sites
    # Using a rough approximation (proper calculation is more complex)
    syn_sites = n_codons * 0.75
    nonsyn_sites = n_codons * 2.25

    # Apply JC correction
    dS = n_syn > 0 && syn_sites > 0 ? jukes_cantor_distance(n_syn / syn_sites) : 0.0
    dN = n_nonsyn > 0 && nonsyn_sites > 0 ? jukes_cantor_distance(n_nonsyn / nonsyn_sites) : 0.0

    # Handle edge cases for omega
    omega = if dS > 0
        dN / dS
    elseif dN > 0
        Inf
    else
        0.0
    end

    return (dN=dN, dS=dS, omega=omega, n_syn=n_syn, n_nonsyn=n_nonsyn)
end

"""
    count_synonymous_sites(codon::String)

Count the number of synonymous sites in a codon.

A site is synonymous if some mutations at that site are synonymous.
Uses averaging over all possible single-nucleotide changes.

# Returns
- Float64: Expected number of synonymous sites (0 to 3)
"""
function count_synonymous_sites(codon::String)
    codon = uppercase(codon)
    aa = translate_codon(codon)

    if aa == 'X'
        return 0.0
    end

    n_syn = 0.0
    nucleotides = ['A', 'C', 'G', 'T']

    for pos in 1:3
        n_syn_at_pos = 0
        for nuc in nucleotides
            if nuc != codon[pos]
                # Create mutant codon
                mutant = codon[1:pos-1] * string(nuc) * codon[pos+1:end]
                mutant_aa = translate_codon(mutant)
                if mutant_aa == aa
                    n_syn_at_pos += 1
                end
            end
        end
        n_syn += n_syn_at_pos / 3  # Average over 3 possible changes
    end

    return n_syn
end
