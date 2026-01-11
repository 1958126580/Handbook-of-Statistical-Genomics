# ============================================================================
# Types.jl - Core Abstract Types and Interfaces
# ============================================================================
# This file defines the fundamental abstract type hierarchy for the
# StatisticalGenomics package based on the Handbook of Statistical Genomics
# (4th Edition). All concrete implementations inherit from these abstract
# types, enabling polymorphism, extensibility, and type-safe dispatch.
# ============================================================================

# ============================================================================
# Primary Abstract Type Hierarchy
# ============================================================================

"""
    AbstractGenotype

Abstract supertype for all genotype representations in the package.

Subtypes must implement the following interface methods:
- `n_samples(g)::Int`: Return the number of samples
- `n_variants(g)::Int`: Return the number of variants
- `get_genotype(g, i, j)`: Get genotype for sample i at variant j

# Extended Interface (optional)
- `missing_rate(g)`: Return proportion of missing data
- `minor_allele_frequency(g)`: Return MAF for each variant
"""
abstract type AbstractGenotype end

"""
    AbstractPhenotype

Abstract supertype for all phenotype representations.

Subtypes must implement:
- `n_samples(p)::Int`: Return the number of samples
- `phenotype_type(p)::Symbol`: Return the type (:continuous, :binary, :categorical, :survival)
- `get_value(p, i)`: Get phenotype value for sample i

# Extended Interface
- `standardize(p)`: Return standardized phenotype
- `inverse_normal_transform(p)`: Apply rank-based inverse normal transformation
"""
abstract type AbstractPhenotype end

"""
    AbstractPopulation

Abstract supertype for population-level data structures.

Subtypes should contain both genotype and phenotype information,
along with metadata about the population such as:
- Sample identifiers
- Population labels
- Geographic or temporal information
"""
abstract type AbstractPopulation end

"""
    AbstractVariant

Abstract supertype for genetic variant representations.

Subtypes must implement:
- `chromosome(v)`: Return the chromosome identifier
- `position(v)`: Return the genomic position
- `reference_allele(v)`: Return the reference allele
- `alternate_allele(v)`: Return the alternate allele(s)
- `variant_id(v)`: Return the variant identifier (e.g., rsID)
"""
abstract type AbstractVariant end

"""
    AbstractAllele

Abstract supertype for allele representations.

This is the most basic unit representing a single allelic state,
which can be a nucleotide (A, C, G, T) or a structural variant.
"""
abstract type AbstractAllele end

"""
    AbstractPhylogeneticTree

Abstract supertype for phylogenetic tree structures.

Subtypes must implement tree traversal and manipulation methods:
- `n_tips(t)`: Number of terminal nodes
- `n_nodes(t)`: Total number of nodes
- `tree_height(t)`: Height of the tree
- `branch_lengths(t)`: Vector of branch lengths
"""
abstract type AbstractPhylogeneticTree end

"""
    AbstractCoalescentTree <: AbstractPhylogeneticTree

Abstract supertype for coalescent genealogy representations.

Extends tree concepts with coalescent-specific timing information:
- `coalescence_times(t)`: Times of coalescent events
- `time_to_mrca(t)`: Time to most recent common ancestor
"""
abstract type AbstractCoalescentTree <: AbstractPhylogeneticTree end

"""
    AbstractEvolutionaryModel

Abstract supertype for models of molecular evolution.

Subtypes must implement:
- `rate_matrix(m)`: Return the instantaneous rate matrix Q
- `stationary_distribution(m)`: Return equilibrium base frequencies
- `transition_probability_matrix(m, t)`: Return P(t) = exp(Qt)
"""
abstract type AbstractEvolutionaryModel end

"""
    AbstractSubstitutionModel <: AbstractEvolutionaryModel

Abstract type for nucleotide/amino acid substitution models.

Standard substitution models (JC69, K80, HKY85, GTR) implement this interface.
"""
abstract type AbstractSubstitutionModel <: AbstractEvolutionaryModel end

"""
    AbstractAssociationResult

Abstract supertype for association test results.

Subtypes store test statistics, p-values, effect sizes, and related metadata.
Common fields include:
- Effect estimates (beta, odds ratio)
- Standard errors
- Test statistics
- P-values
- Sample sizes
"""
abstract type AbstractAssociationResult end

"""
    AbstractGeneticDistance

Abstract supertype for genetic distance/similarity measures.

Subtypes must implement pairwise distance calculation:
- `distance(d, seq1, seq2)`: Calculate distance between two sequences
- `distance_matrix(d, seqs)`: Calculate all pairwise distances
"""
abstract type AbstractGeneticDistance end

"""
    AbstractDemographicModel

Abstract type for demographic models used in population genetics.

Models can represent:
- Constant population size
- Exponential growth/decline
- Bottlenecks
- Population splits and admixture
"""
abstract type AbstractDemographicModel end

"""
    AbstractSelectionModel

Abstract type for models of natural selection.

Subtypes represent different selection regimes:
- Directional selection
- Balancing selection
- Purifying selection
- Frequency-dependent selection
"""
abstract type AbstractSelectionModel end

# ============================================================================
# Common Interface Functions (to be implemented by subtypes)
# ============================================================================

"""
    n_samples(x)

Return the number of samples in the data structure.
This is a generic function that subtypes of AbstractGenotype,
AbstractPhenotype, and AbstractPopulation must implement.
"""
function n_samples end

"""
    n_variants(x)

Return the number of genetic variants in the data structure.
"""
function n_variants end

"""
    get_genotype(g::AbstractGenotype, sample_idx::Int, variant_idx::Int)

Return the genotype value for a specific sample at a specific variant.
Returns missing for missing data.
"""
function get_genotype end

"""
    get_value(p::AbstractPhenotype, sample_idx::Int)

Return the phenotype value for a specific sample.
Returns missing for missing data.
"""
function get_value end

"""
    phenotype_type(p::AbstractPhenotype)

Return the type of phenotype as a Symbol.
Valid return values:
- `:continuous`: Quantitative traits (height, BMI, etc.)
- `:binary`: Case/control phenotypes
- `:categorical`: Multi-class phenotypes
- `:survival`: Time-to-event data
- `:ordinal`: Ordered categorical data
"""
function phenotype_type end

"""
    chromosome(v::AbstractVariant)

Return the chromosome identifier for a variant.
"""
function chromosome end

"""
    position(v::AbstractVariant)

Return the genomic position (1-based) for a variant.
"""
function position end

"""
    reference_allele(v::AbstractVariant)

Return the reference allele for a variant.
"""
function reference_allele end

"""
    alternate_allele(v::AbstractVariant)

Return the alternate allele(s) for a variant.
"""
function alternate_allele end

"""
    variant_id(v::AbstractVariant)

Return the identifier (e.g., rsID) for a variant.
"""
function variant_id end

# ============================================================================
# Type Aliases for Convenience and Clarity
# ============================================================================

"""
    GenotypeValue = Union{Int8, Missing}

Type alias for single genotype values.
Values: 0 (homozygous reference), 1 (heterozygous), 2 (homozygous alternate), missing
"""
const GenotypeValue = Union{Int8, Missing}

"""
    DosageValue = Union{Float64, Missing}

Type alias for genotype dosage values (continuous representation).
Values range from 0.0 to 2.0, representing expected allele counts.
Used for imputed genotypes where uncertainty exists.
"""
const DosageValue = Union{Float64, Missing}

"""
    AlleleFrequency = Float64

Type alias for allele frequency values (0.0 to 1.0).
Represents the proportion of a specific allele in a population.
"""
const AlleleFrequency = Float64

"""
    ChromosomeID = Union{Int, String}

Type alias for chromosome identifiers.
Can be numeric (1-22) or string ("X", "Y", "MT", "chrX", etc.).
"""
const ChromosomeID = Union{Int, String}

"""
    GenomicPosition = Int64

Type alias for genomic positions (1-based coordinates).
Uses Int64 to accommodate large genome positions.
"""
const GenomicPosition = Int64

"""
    PValue = Float64

Type alias for p-values from statistical tests.
Always in range [0.0, 1.0].
"""
const PValue = Float64

"""
    EffectSize = Float64

Type alias for effect size estimates (beta coefficients, log odds ratios, etc.).
"""
const EffectSize = Float64

# ============================================================================
# Result Container Types
# ============================================================================

"""
    StatisticalTestResult

Immutable struct containing results from a statistical test.

# Fields
- `statistic::Float64`: The test statistic value
- `pvalue::Float64`: The p-value
- `df::Union{Int, Float64, Nothing}`: Degrees of freedom (if applicable)
- `method::String`: Name of the statistical test used
- `alternative::Symbol`: Type of alternative hypothesis (:two_sided, :greater, :less)

# Example
```julia
result = StatisticalTestResult(2.5, 0.01, 98, "t-test", :two_sided)
```
"""
struct StatisticalTestResult
    statistic::Float64
    pvalue::Float64
    df::Union{Int, Float64, Nothing}
    method::String
    alternative::Symbol

    function StatisticalTestResult(
        statistic::Float64,
        pvalue::Float64,
        df::Union{Int, Float64, Nothing},
        method::String,
        alternative::Symbol=:two_sided
    )
        @assert 0.0 <= pvalue <= 1.0 || isnan(pvalue) "P-value must be in [0, 1]"
        @assert alternative in (:two_sided, :greater, :less) "Invalid alternative"
        new(statistic, pvalue, df, method, alternative)
    end
end

# Convenience constructor without alternative
StatisticalTestResult(stat::Float64, pval::Float64, df, method::String) =
    StatisticalTestResult(stat, pval, df, method, :two_sided)

"""
    ConfidenceInterval

Immutable struct representing a confidence interval.

# Fields
- `lower::Float64`: Lower bound of the interval
- `upper::Float64`: Upper bound of the interval
- `level::Float64`: Confidence level (e.g., 0.95 for 95% CI)
- `estimate::Float64`: Point estimate (optional, defaults to midpoint)

# Example
```julia
ci = ConfidenceInterval(0.1, 0.5, 0.95)
width(ci)  # Returns 0.4
contains(ci, 0.3)  # Returns true
```
"""
struct ConfidenceInterval
    lower::Float64
    upper::Float64
    level::Float64
    estimate::Float64

    function ConfidenceInterval(
        lower::Float64,
        upper::Float64,
        level::Float64=0.95;
        estimate::Float64=(lower + upper) / 2
    )
        @assert 0.0 < level < 1.0 "Confidence level must be in (0, 1)"
        @assert lower <= upper || (isnan(lower) && isnan(upper)) "Lower bound must not exceed upper bound"
        new(lower, upper, level, estimate)
    end
end

# Helper functions for ConfidenceInterval
Base.in(x::Real, ci::ConfidenceInterval) = ci.lower <= x <= ci.upper
width(ci::ConfidenceInterval) = ci.upper - ci.lower
midpoint(ci::ConfidenceInterval) = (ci.lower + ci.upper) / 2

"""
    EffectEstimate

Immutable struct containing an effect size estimate with uncertainty measures.

# Fields
- `estimate::Float64`: Point estimate of the effect
- `se::Float64`: Standard error of the estimate
- `ci::ConfidenceInterval`: Confidence interval
- `pvalue::Float64`: P-value for testing effect ≠ 0

# Example
```julia
ee = EffectEstimate(0.25, 0.05, ConfidenceInterval(0.15, 0.35, 0.95), 0.001)
is_significant(ee)  # Returns true
```
"""
struct EffectEstimate
    estimate::Float64
    se::Float64
    ci::ConfidenceInterval
    pvalue::Float64

    function EffectEstimate(
        estimate::Float64,
        se::Float64,
        ci::ConfidenceInterval,
        pvalue::Float64=NaN
    )
        @assert se >= 0 || isnan(se) "Standard error must be non-negative"
        new(estimate, se, ci, pvalue)
    end
end

# Convenience constructor computing CI from SE
function EffectEstimate(estimate::Float64, se::Float64; level::Float64=0.95)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    ci = ConfidenceInterval(estimate - z * se, estimate + z * se, level; estimate=estimate)
    pvalue = 2 * ccdf(Normal(), abs(estimate / se))
    EffectEstimate(estimate, se, ci, pvalue)
end

is_significant(ee::EffectEstimate; α::Float64=0.05) = ee.pvalue < α

"""
    GenomicRegion

Represents a genomic region (interval) on a chromosome.

# Fields
- `chromosome::ChromosomeID`: Chromosome identifier
- `start_pos::GenomicPosition`: Start position (1-based, inclusive)
- `end_pos::GenomicPosition`: End position (1-based, inclusive)
- `name::String`: Optional name for the region

# Example
```julia
region = GenomicRegion(1, 1000000, 2000000, "Gene_XYZ")
length(region)  # Returns 1000001
overlaps(region1, region2)  # Check for overlap
```
"""
struct GenomicRegion
    chromosome::ChromosomeID
    start_pos::GenomicPosition
    end_pos::GenomicPosition
    name::String

    function GenomicRegion(
        chromosome::ChromosomeID,
        start_pos::Integer,
        end_pos::Integer,
        name::String=""
    )
        @assert start_pos <= end_pos "Start position must not exceed end position"
        @assert start_pos > 0 "Positions must be positive"
        new(chromosome, GenomicPosition(start_pos), GenomicPosition(end_pos), name)
    end
end

Base.length(r::GenomicRegion) = r.end_pos - r.start_pos + 1

function overlaps(r1::GenomicRegion, r2::GenomicRegion)
    r1.chromosome == r2.chromosome && r1.start_pos <= r2.end_pos && r2.start_pos <= r1.end_pos
end

function Base.in(pos::Integer, r::GenomicRegion)
    r.start_pos <= pos <= r.end_pos
end

"""
    VarianceComponents

Container for variance component estimates from mixed models.

# Fields
- `genetic::Float64`: Genetic variance (σ²_g)
- `environmental::Float64`: Environmental/residual variance (σ²_e)
- `total::Float64`: Total phenotypic variance
- `heritability::Float64`: Narrow-sense heritability (h² = σ²_g / σ²_total)
- `se_heritability::Float64`: Standard error of heritability estimate
"""
struct VarianceComponents
    genetic::Float64
    environmental::Float64
    total::Float64
    heritability::Float64
    se_heritability::Float64

    function VarianceComponents(genetic::Float64, environmental::Float64;
                                se_heritability::Float64=NaN)
        @assert genetic >= 0 "Genetic variance must be non-negative"
        @assert environmental >= 0 "Environmental variance must be non-negative"
        total = genetic + environmental
        h2 = total > 0 ? genetic / total : NaN
        new(genetic, environmental, total, h2, se_heritability)
    end
end

# ============================================================================
# Utility Functions for Type Checking and Validation
# ============================================================================

"""
    is_valid_genotype(g::Integer)

Check if a genotype value is valid (0, 1, or 2).

# Arguments
- `g`: Integer genotype value

# Returns
- `Bool`: true if valid, false otherwise
"""
is_valid_genotype(g::Integer) = g in (0, 1, 2)
is_valid_genotype(::Missing) = true

"""
    is_valid_allele_frequency(f::Real)

Check if a value is a valid allele frequency (between 0 and 1, inclusive).

# Arguments
- `f`: Numeric value to check

# Returns
- `Bool`: true if valid, false otherwise
"""
is_valid_allele_frequency(f::Real) = 0.0 <= f <= 1.0

"""
    is_valid_chromosome(chr::Union{Int, String})

Check if a chromosome identifier is valid for human genomes.

# Arguments
- `chr`: Chromosome identifier

# Returns
- `Bool`: true if valid, false otherwise

# Examples
```julia
is_valid_chromosome(1)     # true
is_valid_chromosome(22)    # true
is_valid_chromosome("X")   # true
is_valid_chromosome("MT")  # true
is_valid_chromosome(25)    # false
```
"""
function is_valid_chromosome(chr::Union{Int, String})
    if chr isa Int
        return 1 <= chr <= 22
    else
        valid_strings = Set(["X", "Y", "MT", "M", "XY", "chrX", "chrY", "chrM", "chrMT"])
        chr_upper = uppercase(string(chr))
        # Also check for chr1-chr22 format
        return chr_upper in valid_strings ||
               (startswith(chr_upper, "CHR") && tryparse(Int, chr_upper[4:end]) in 1:22)
    end
end

"""
    is_polymorphic(genotypes::AbstractVector)

Check if a variant is polymorphic (has at least 2 different alleles).

# Arguments
- `genotypes`: Vector of genotype values

# Returns
- `Bool`: true if polymorphic, false if monomorphic
"""
function is_polymorphic(genotypes::AbstractVector)
    unique_genos = Set{Int}()
    for g in genotypes
        if !ismissing(g)
            push!(unique_genos, g)
            if length(unique_genos) > 1
                return true
            end
        end
    end
    return false
end

"""
    nucleotide_to_int(base::Char)

Convert nucleotide character to integer encoding.
A=1, C=2, G=3, T=4, other=-1
"""
function nucleotide_to_int(base::Char)
    base = uppercase(base)
    return base == 'A' ? 1 : base == 'C' ? 2 : base == 'G' ? 3 : base == 'T' ? 4 : -1
end

"""
    int_to_nucleotide(i::Int)

Convert integer encoding to nucleotide character.
1=A, 2=C, 3=G, 4=T
"""
function int_to_nucleotide(i::Int)
    return i == 1 ? 'A' : i == 2 ? 'C' : i == 3 ? 'G' : i == 4 ? 'T' : 'N'
end

"""
    complement(base::Char)

Return the complement of a nucleotide.
"""
function complement(base::Char)
    base = uppercase(base)
    return base == 'A' ? 'T' : base == 'T' ? 'A' : base == 'C' ? 'G' : base == 'G' ? 'C' : 'N'
end

"""
    reverse_complement(seq::AbstractString)

Return the reverse complement of a DNA sequence.
"""
function reverse_complement(seq::AbstractString)
    return String([complement(base) for base in reverse(seq)])
end

# ============================================================================
# Iteration and Collection Protocols
# ============================================================================

# Enable iteration over statistical test results for multiple testing
struct MultipleTestResults
    results::Vector{StatisticalTestResult}
    ids::Vector{String}
end

Base.length(mtr::MultipleTestResults) = length(mtr.results)
Base.iterate(mtr::MultipleTestResults) = iterate(mtr.results)
Base.iterate(mtr::MultipleTestResults, state) = iterate(mtr.results, state)
Base.getindex(mtr::MultipleTestResults, i::Int) = mtr.results[i]
Base.getindex(mtr::MultipleTestResults, id::String) = mtr.results[findfirst(==(id), mtr.ids)]

pvalues(mtr::MultipleTestResults) = [r.pvalue for r in mtr.results]
statistics(mtr::MultipleTestResults) = [r.statistic for r in mtr.results]
