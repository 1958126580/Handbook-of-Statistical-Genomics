# ============================================================================
# Phenotypes.jl - Phenotype Data Structures
# ============================================================================

"""
    ContinuousPhenotype <: AbstractPhenotype

Continuous/quantitative phenotype representation.

# Fields
- `values::Vector{Union{Float64, Missing}}`: Phenotype values
- `name::String`: Phenotype name
- `sample_ids::Vector{String}`: Sample identifiers
"""
struct ContinuousPhenotype <: AbstractPhenotype
    values::Vector{Union{Float64, Missing}}
    name::String
    sample_ids::Vector{String}
    
    function ContinuousPhenotype(
        values::AbstractVector,
        name::String="Phenotype",
        sample_ids::Vector{String}=String[]
    )
        n = length(values)
        sample_ids = isempty(sample_ids) ? ["S$i" for i in 1:n] : sample_ids
        vals = [ismissing(v) ? missing : Float64(v) for v in values]
        new(vals, name, sample_ids)
    end
end

n_samples(p::ContinuousPhenotype) = length(p.values)
phenotype_type(::ContinuousPhenotype) = :continuous
get_value(p::ContinuousPhenotype, i::Int) = p.values[i]
Base.length(p::ContinuousPhenotype) = length(p.values)
Base.getindex(p::ContinuousPhenotype, i::Int) = p.values[i]
Base.mean(p::ContinuousPhenotype) = mean(skipmissing(p.values))
Base.var(p::ContinuousPhenotype) = var(skipmissing(p.values))

"""Standardize phenotype to zero mean and unit variance."""
function standardize(p::ContinuousPhenotype)
    μ, σ = mean(p), std(skipmissing(p.values))
    vals = [ismissing(v) ? missing : (v - μ) / σ for v in p.values]
    ContinuousPhenotype(vals, p.name * "_std", p.sample_ids)
end

"""Apply inverse normal transformation."""
function inverse_normal_transform(p::ContinuousPhenotype)
    idx = findall(!ismissing, p.values)
    vals_nm = [p.values[i] for i in idx]
    ranks = ordinalrank(vals_nm)
    n = length(vals_nm)
    transformed = [quantile(Normal(), (r - 0.375)/(n + 0.25)) for r in ranks]
    new_vals = Vector{Union{Float64,Missing}}(fill(missing, length(p.values)))
    for (i, j) in enumerate(idx)
        new_vals[j] = transformed[i]
    end
    ContinuousPhenotype(new_vals, p.name * "_int", p.sample_ids)
end

"""
    BinaryPhenotype <: AbstractPhenotype

Binary (case/control) phenotype representation.
"""
struct BinaryPhenotype <: AbstractPhenotype
    values::Vector{Union{Bool, Missing}}
    name::String
    sample_ids::Vector{String}
    
    function BinaryPhenotype(values::AbstractVector, name::String="Binary", 
                            sample_ids::Vector{String}=String[])
        n = length(values)
        sample_ids = isempty(sample_ids) ? ["S$i" for i in 1:n] : sample_ids
        vals = [ismissing(v) ? missing : Bool(v != 0) for v in values]
        new(vals, name, sample_ids)
    end
end

n_samples(p::BinaryPhenotype) = length(p.values)
phenotype_type(::BinaryPhenotype) = :binary
get_value(p::BinaryPhenotype, i::Int) = p.values[i]
case_count(p::BinaryPhenotype) = sum(skipmissing(p.values))
control_count(p::BinaryPhenotype) = count(x -> !ismissing(x) && !x, p.values)

"""
    PhenotypeVector

Generic phenotype container.
"""
struct PhenotypeVector <: AbstractPhenotype
    data::Vector{Union{Float64, Missing}}
    name::String
    sample_ids::Vector{String}
    type::Symbol
end

n_samples(p::PhenotypeVector) = length(p.data)
phenotype_type(p::PhenotypeVector) = p.type
get_value(p::PhenotypeVector, i::Int) = p.data[i]

"""
    CovariateMatrix

Container for covariate data.
"""
struct CovariateMatrix
    data::Matrix{Float64}
    names::Vector{String}
    sample_ids::Vector{String}
    
    function CovariateMatrix(data::AbstractMatrix, names::Vector{String}=String[],
                            sample_ids::Vector{String}=String[])
        n, p = size(data)
        names = isempty(names) ? ["X$i" for i in 1:p] : names
        sample_ids = isempty(sample_ids) ? ["S$i" for i in 1:n] : sample_ids
        new(Matrix{Float64}(data), names, sample_ids)
    end
end

Base.size(c::CovariateMatrix) = size(c.data)
Base.getindex(c::CovariateMatrix, I...) = c.data[I...]
add_intercept(c::CovariateMatrix) = CovariateMatrix(
    hcat(ones(size(c.data,1)), c.data), 
    vcat(["Intercept"], c.names), c.sample_ids)
