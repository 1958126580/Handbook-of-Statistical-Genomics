# ============================================================================
# Types.jl - Core Abstract Types and Interfaces
# ============================================================================
# This file defines the fundamental abstract type hierarchy for the
# StatisticalGenomics package. All concrete implementations inherit from
# these abstract types, enabling polymorphism and extensibility.
# ============================================================================

"""
    AbstractGenotype

Abstract supertype for all genotype representations in the package.

Subtypes must implement:
- `n_samples(g)`: Return the number of samples
- `n_variants(g)`: Return the number of variants
- `get_genotype(g, i, j)`: Get genotype for sample i at variant j
"""
abstract type AbstractGenotype end

"""
    AbstractPhenotype

Abstract supertype for all phenotype representations.

Subtypes must implement:
- `n_samples(p)`: Return the number of samples
- `phenotype_type(p)`: Return the type (:continuous, :binary, :categorical)
- `get_value(p, i)`: Get phenotype value for sample i
"""
abstract type AbstractPhenotype end

"""
    AbstractPopulation

Abstract supertype for population-level data structures.

Subtypes should contain both genotype and phenotype information,
along with metadata about the population.
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
"""
abstract type AbstractVariant end

"""
    AbstractAllele

Abstract supertype for allele representations.

This is the most basic unit representing a single allelic state.
"""
abstract type AbstractAllele end

"""
    AbstractPhylogeneticTree

Abstract supertype for phylogenetic tree structures.

Subtypes must implement tree traversal and manipulation methods.
"""
abstract type AbstractPhylogeneticTree end

"""
    AbstractCoalescentTree

Abstract supertype for coalescent genealogy representations.

Extends tree concepts with coalescent-specific timing information.
"""
abstract type AbstractCoalescentTree <: AbstractPhylogeneticTree end

"""
    AbstractEvolutionaryModel

Abstract supertype for models of molecular evolution.

Subtypes must implement rate matrix generation and likelihood calculation.
"""
abstract type AbstractEvolutionaryModel end

"""
    AbstractAssociationResult

Abstract supertype for association test results.

Subtypes store test statistics, p-values, and effect sizes.
"""
abstract type AbstractAssociationResult end

"""
    AbstractGeneticDistance

Abstract supertype for genetic distance/similarity measures.

Subtypes must implement pairwise distance calculation methods.
"""
abstract type AbstractGeneticDistance end

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
"""
function get_genotype end

"""
    get_value(p::AbstractPhenotype, sample_idx::Int)

Return the phenotype value for a specific sample.
"""
function get_value end

"""
    phenotype_type(p::AbstractPhenotype)

Return the type of phenotype (:continuous, :binary, :categorical, :survival).
"""
function phenotype_type end

# ============================================================================
# Type Aliases for Convenience
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
"""
const DosageValue = Union{Float64, Missing}

"""
    AlleleFrequency = Float64

Type alias for allele frequency values (0.0 to 1.0).
"""
const AlleleFrequency = Float64

"""
    Chromosome = Union{Int, String}

Type alias for chromosome identifiers.
Can be numeric (1-22) or string ("X", "Y", "MT").
"""
const Chromosome = Union{Int, String}

"""
    Position = Int64

Type alias for genomic positions (1-based).
"""
const Position = Int64

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
"""
struct StatisticalTestResult
    statistic::Float64
    pvalue::Float64
    df::Union{Int, Float64, Nothing}
    method::String
end

"""
    ConfidenceInterval

Immutable struct representing a confidence interval.

# Fields
- `lower::Float64`: Lower bound of the interval
- `upper::Float64`: Upper bound of the interval
- `level::Float64`: Confidence level (e.g., 0.95 for 95% CI)
"""
struct ConfidenceInterval
    lower::Float64
    upper::Float64
    level::Float64
    
    function ConfidenceInterval(lower::Float64, upper::Float64, level::Float64=0.95)
        @assert 0.0 < level < 1.0 "Confidence level must be between 0 and 1"
        @assert lower <= upper "Lower bound must not exceed upper bound"
        new(lower, upper, level)
    end
end

"""
    EffectEstimate

Immutable struct containing an effect size estimate with uncertainty.

# Fields
- `estimate::Float64`: Point estimate of the effect
- `se::Float64`: Standard error of the estimate
- `ci::ConfidenceInterval`: Confidence interval
"""
struct EffectEstimate
    estimate::Float64
    se::Float64
    ci::ConfidenceInterval
end

# ============================================================================
# Utility Functions for Type Checking
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

"""
    is_valid_allele_frequency(f::Real)

Check if a value is a valid allele frequency (between 0 and 1).

# Arguments
- `f`: Numeric value to check

# Returns
- `Bool`: true if valid, false otherwise
"""
is_valid_allele_frequency(f::Real) = 0.0 <= f <= 1.0

"""
    is_valid_chromosome(chr::Union{Int, String})

Check if a chromosome identifier is valid.

# Arguments
- `chr`: Chromosome identifier

# Returns
- `Bool`: true if valid, false otherwise
"""
function is_valid_chromosome(chr::Union{Int, String})
    if chr isa Int
        return 1 <= chr <= 22
    else
        return chr in ("X", "Y", "MT", "M", "XY")
    end
end
