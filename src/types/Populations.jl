# ============================================================================
# Populations.jl - Population Data Structures
# ============================================================================

"""
    Population <: AbstractPopulation

Container for population genetic data combining genotypes, phenotypes, and metadata.

# Fields
- `genotypes::GenotypeMatrix`: Genotype data
- `phenotypes::Dict{String, AbstractPhenotype}`: Named phenotypes
- `covariates::Union{CovariateMatrix, Nothing}`: Optional covariates
- `name::String`: Population name
- `metadata::Dict{String, Any}`: Additional metadata
"""
struct Population <: AbstractPopulation
    genotypes::GenotypeMatrix
    phenotypes::Dict{String, AbstractPhenotype}
    covariates::Union{CovariateMatrix, Nothing}
    name::String
    metadata::Dict{String, Any}
    
    function Population(
        genotypes::GenotypeMatrix;
        phenotypes::Dict{String, AbstractPhenotype}=Dict{String, AbstractPhenotype}(),
        covariates::Union{CovariateMatrix, Nothing}=nothing,
        name::String="Population",
        metadata::Dict{String, Any}=Dict{String, Any}()
    )
        new(genotypes, phenotypes, covariates, name, metadata)
    end
end

n_samples(pop::Population) = n_samples(pop.genotypes)
n_variants(pop::Population) = n_variants(pop.genotypes)
n_phenotypes(pop::Population) = length(pop.phenotypes)
get_phenotype(pop::Population, name::String) = pop.phenotypes[name]
has_phenotype(pop::Population, name::String) = haskey(pop.phenotypes, name)

"""Add a phenotype to the population."""
function add_phenotype!(pop::Population, name::String, pheno::AbstractPhenotype)
    @assert n_samples(pheno) == n_samples(pop) "Sample count mismatch"
    pop.phenotypes[name] = pheno
    return pop
end

"""
    PopulationSample

A subset of samples from a population.
"""
struct PopulationSample <: AbstractPopulation
    parent::Population
    sample_indices::Vector{Int}
    name::String
end

n_samples(ps::PopulationSample) = length(ps.sample_indices)
n_variants(ps::PopulationSample) = n_variants(ps.parent)

"""Create a random sample from a population."""
function sample_population(pop::Population, n::Int; replace::Bool=false)
    indices = sample(1:n_samples(pop), n; replace=replace)
    PopulationSample(pop, sort(indices), "Sample_of_$(pop.name)")
end

"""
    SubPopulation

Represents distinct subpopulations within a larger population.
"""
struct SubPopulation
    population::Population
    assignments::Vector{Int}  # Subpopulation assignment for each sample
    k::Int  # Number of subpopulations
    names::Vector{String}
    
    function SubPopulation(pop::Population, assignments::Vector{Int}; 
                          names::Vector{String}=String[])
        k = maximum(assignments)
        names = isempty(names) ? ["Pop$i" for i in 1:k] : names
        new(pop, assignments, k, names)
    end
end

"""Get samples belonging to a specific subpopulation."""
function get_subpop_samples(sp::SubPopulation, subpop_idx::Int)
    findall(==(subpop_idx), sp.assignments)
end

"""Calculate Fst between subpopulations."""
function calculate_fst(sp::SubPopulation)
    n_vars = n_variants(sp.population)
    fst_values = Vector{Float64}(undef, n_vars)
    
    for j in 1:n_vars
        # Calculate allele frequencies per subpopulation
        freqs = Float64[]
        ns = Int[]
        for k in 1:sp.k
            idx = get_subpop_samples(sp, k)
            genos = skipmissing(sp.population.genotypes.data[idx, j])
            if !isempty(genos)
                push!(freqs, sum(genos) / (2 * length(collect(genos))))
                push!(ns, length(collect(genos)))
            end
        end
        
        if length(freqs) < 2
            fst_values[j] = NaN
            continue
        end
        
        # Weir & Cockerham Fst estimator
        p_bar = sum(ns .* freqs) / sum(ns)
        msc = sum(ns .* (freqs .- p_bar).^2) / (length(freqs) - 1)
        p_bar_het = p_bar * (1 - p_bar)
        
        if p_bar_het > 0
            fst_values[j] = msc / p_bar_het
        else
            fst_values[j] = 0.0
        end
    end
    
    return fst_values
end
