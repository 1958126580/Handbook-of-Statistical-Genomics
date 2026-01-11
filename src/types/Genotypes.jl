# ============================================================================
# Genotypes.jl - Genotype Data Structures and Operations
# ============================================================================
# This file implements concrete genotype representations including:
# - SNPGenotype: Single genotype at a biallelic SNP
# - GenotypeMatrix: Matrix of genotypes (samples × variants)
# - DosageMatrix: Matrix of genotype dosages for imputed data
# ============================================================================

"""
    SNPGenotype

Representation of a single genotype at a biallelic SNP.

# Fields
- `value::Union{Int8, Missing}`: Genotype value (0, 1, 2, or missing)

# Encoding
- 0: Homozygous for reference allele (AA)
- 1: Heterozygous (Aa)
- 2: Homozygous for alternate allele (aa)
- missing: Missing genotype

# Example
```julia
g = SNPGenotype(1)  # Heterozygous genotype
is_het(g)           # Returns true
```
"""
struct SNPGenotype <: AbstractGenotype
    value::Union{Int8, Missing}
    
    function SNPGenotype(v::Union{Integer, Missing})
        if !ismissing(v) && !(v in (0, 1, 2))
            throw(ArgumentError("Genotype value must be 0, 1, 2, or missing"))
        end
        new(ismissing(v) ? missing : Int8(v))
    end
end

# Convenience constructors
SNPGenotype() = SNPGenotype(missing)

"""
    is_missing(g::SNPGenotype)

Check if a genotype is missing.
"""
is_missing(g::SNPGenotype) = ismissing(g.value)

"""
    is_homozygous_ref(g::SNPGenotype)

Check if genotype is homozygous for the reference allele.
"""
is_homozygous_ref(g::SNPGenotype) = !ismissing(g.value) && g.value == 0

"""
    is_heterozygous(g::SNPGenotype)

Check if genotype is heterozygous.
"""
is_heterozygous(g::SNPGenotype) = !ismissing(g.value) && g.value == 1

"""
    is_het(g::SNPGenotype)

Alias for `is_heterozygous`.
"""
is_het(g::SNPGenotype) = is_heterozygous(g)

"""
    is_homozygous_alt(g::SNPGenotype)

Check if genotype is homozygous for the alternate allele.
"""
is_homozygous_alt(g::SNPGenotype) = !ismissing(g.value) && g.value == 2

"""
    alternate_allele_count(g::SNPGenotype)

Return the count of alternate alleles (0, 1, or 2).
Returns missing if genotype is missing.
"""
alternate_allele_count(g::SNPGenotype) = g.value

# ============================================================================
# GenotypeMatrix - Matrix representation of genotypes
# ============================================================================

"""
    GenotypeMatrix <: AbstractGenotype

Matrix representation of genotypes for multiple samples and variants.

# Fields
- `data::Matrix{Union{Int8, Missing}}`: Genotype matrix (samples × variants)
- `sample_ids::Vector{String}`: Sample identifiers
- `variant_ids::Vector{String}`: Variant identifiers (e.g., rsIDs)
- `chromosomes::Vector{Chromosome}`: Chromosome for each variant
- `positions::Vector{Position}`: Genomic position for each variant
- `ref_alleles::Vector{String}`: Reference allele for each variant
- `alt_alleles::Vector{String}`: Alternate allele for each variant

# Example
```julia
# Create a genotype matrix for 100 samples and 1000 variants
gm = GenotypeMatrix(rand(0:2, 100, 1000))

# Access genotype for sample 1 at variant 5
g = gm[1, 5]
```
"""
struct GenotypeMatrix <: AbstractGenotype
    data::Matrix{Union{Int8, Missing}}
    sample_ids::Vector{String}
    variant_ids::Vector{String}
    chromosomes::Vector{Chromosome}
    positions::Vector{Position}
    ref_alleles::Vector{String}
    alt_alleles::Vector{String}
    
    function GenotypeMatrix(
        data::AbstractMatrix,
        sample_ids::Vector{String}=String[],
        variant_ids::Vector{String}=String[],
        chromosomes::Vector{<:Chromosome}=Int[],
        positions::Vector{<:Integer}=Int64[],
        ref_alleles::Vector{String}=String[],
        alt_alleles::Vector{String}=String[]
    )
        n_samples, n_variants = size(data)
        
        # Validate genotype values
        for i in eachindex(data)
            if !ismissing(data[i]) && !(data[i] in (0, 1, 2))
                throw(ArgumentError("Invalid genotype value at index $i: $(data[i])"))
            end
        end
        
        # Generate default IDs if not provided
        if isempty(sample_ids)
            sample_ids = ["SAMPLE_$i" for i in 1:n_samples]
        end
        if isempty(variant_ids)
            variant_ids = ["VAR_$i" for i in 1:n_variants]
        end
        if isempty(chromosomes)
            chromosomes = ones(Int, n_variants)
        end
        if isempty(positions)
            positions = collect(1:n_variants)
        end
        if isempty(ref_alleles)
            ref_alleles = fill("A", n_variants)
        end
        if isempty(alt_alleles)
            alt_alleles = fill("T", n_variants)
        end
        
        # Validate dimensions
        @assert length(sample_ids) == n_samples "Sample IDs must match data rows"
        @assert length(variant_ids) == n_variants "Variant IDs must match data columns"
        @assert length(chromosomes) == n_variants "Chromosomes must match variants"
        @assert length(positions) == n_variants "Positions must match variants"
        @assert length(ref_alleles) == n_variants "Reference alleles must match variants"
        @assert length(alt_alleles) == n_variants "Alternate alleles must match variants"
        
        # Convert to appropriate types
        data_int8 = Matrix{Union{Int8, Missing}}(undef, n_samples, n_variants)
        for i in eachindex(data)
            data_int8[i] = ismissing(data[i]) ? missing : Int8(data[i])
        end
        
        new(data_int8, sample_ids, variant_ids, 
            Vector{Chromosome}(chromosomes), Vector{Position}(positions),
            ref_alleles, alt_alleles)
    end
end

# Interface implementations
n_samples(gm::GenotypeMatrix) = size(gm.data, 1)
n_variants(gm::GenotypeMatrix) = size(gm.data, 2)
get_genotype(gm::GenotypeMatrix, i::Int, j::Int) = gm.data[i, j]

# Array-like interface
Base.size(gm::GenotypeMatrix) = size(gm.data)
Base.size(gm::GenotypeMatrix, d::Int) = size(gm.data, d)
Base.getindex(gm::GenotypeMatrix, i::Int, j::Int) = gm.data[i, j]
Base.getindex(gm::GenotypeMatrix, I...) = gm.data[I...]

"""
    missing_rate(gm::GenotypeMatrix, dim::Symbol=:overall)

Calculate the missing rate in the genotype matrix.

# Arguments
- `gm`: GenotypeMatrix
- `dim`: :overall, :sample, or :variant for different dimensions

# Returns
- Missing rate as Float64 or Vector{Float64}
"""
function missing_rate(gm::GenotypeMatrix, dim::Symbol=:overall)
    if dim == :overall
        return sum(ismissing.(gm.data)) / length(gm.data)
    elseif dim == :sample
        return [sum(ismissing.(gm.data[i, :])) / n_variants(gm) for i in 1:n_samples(gm)]
    elseif dim == :variant
        return [sum(ismissing.(gm.data[:, j])) / n_samples(gm) for j in 1:n_variants(gm)]
    else
        throw(ArgumentError("dim must be :overall, :sample, or :variant"))
    end
end

"""
    minor_allele_frequency(gm::GenotypeMatrix)

Calculate minor allele frequency for each variant.

# Arguments
- `gm`: GenotypeMatrix

# Returns
- Vector{Float64} of MAFs for each variant
"""
function minor_allele_frequency(gm::GenotypeMatrix)
    mafs = Vector{Float64}(undef, n_variants(gm))
    
    for j in 1:n_variants(gm)
        genotypes = skipmissing(gm.data[:, j])
        if isempty(genotypes)
            mafs[j] = NaN
        else
            # Calculate allele frequency
            p = sum(genotypes) / (2 * count(!ismissing, gm.data[:, j]))
            mafs[j] = min(p, 1 - p)  # Convert to MAF
        end
    end
    
    return mafs
end

# ============================================================================
# DosageMatrix - Continuous genotype representation
# ============================================================================

"""
    DosageMatrix <: AbstractGenotype

Matrix representation of genotype dosages (continuous values 0-2).

Dosage values represent the expected count of alternate alleles,
allowing for genotype uncertainty from imputation.

# Fields
- `data::Matrix{Union{Float64, Missing}}`: Dosage matrix (samples × variants)
- `sample_ids::Vector{String}`: Sample identifiers
- `variant_ids::Vector{String}`: Variant identifiers
- `info_scores::Vector{Float64}`: Imputation quality scores (R²)

# Example
```julia
# Create dosage matrix
dm = DosageMatrix(rand(100, 1000) .* 2)

# Get expected heterozygosity
het = expected_heterozygosity(dm)
```
"""
struct DosageMatrix <: AbstractGenotype
    data::Matrix{Union{Float64, Missing}}
    sample_ids::Vector{String}
    variant_ids::Vector{String}
    info_scores::Vector{Float64}
    
    function DosageMatrix(
        data::AbstractMatrix,
        sample_ids::Vector{String}=String[],
        variant_ids::Vector{String}=String[],
        info_scores::Vector{Float64}=Float64[]
    )
        n_samples, n_variants = size(data)
        
        # Validate dosage values are in [0, 2]
        for i in eachindex(data)
            if !ismissing(data[i]) && (data[i] < 0 || data[i] > 2)
                @warn "Dosage value outside [0,2] at index $i: $(data[i])"
            end
        end
        
        # Generate default IDs if not provided
        if isempty(sample_ids)
            sample_ids = ["SAMPLE_$i" for i in 1:n_samples]
        end
        if isempty(variant_ids)
            variant_ids = ["VAR_$i" for i in 1:n_variants]
        end
        if isempty(info_scores)
            info_scores = fill(1.0, n_variants)  # Assume perfect imputation
        end
        
        @assert length(sample_ids) == n_samples
        @assert length(variant_ids) == n_variants
        @assert length(info_scores) == n_variants
        
        # Convert to Float64
        data_f64 = Matrix{Union{Float64, Missing}}(undef, n_samples, n_variants)
        for i in eachindex(data)
            data_f64[i] = ismissing(data[i]) ? missing : Float64(data[i])
        end
        
        new(data_f64, sample_ids, variant_ids, info_scores)
    end
end

# Interface implementations
n_samples(dm::DosageMatrix) = size(dm.data, 1)
n_variants(dm::DosageMatrix) = size(dm.data, 2)
get_genotype(dm::DosageMatrix, i::Int, j::Int) = dm.data[i, j]

# Array-like interface
Base.size(dm::DosageMatrix) = size(dm.data)
Base.size(dm::DosageMatrix, d::Int) = size(dm.data, d)
Base.getindex(dm::DosageMatrix, i::Int, j::Int) = dm.data[i, j]
Base.getindex(dm::DosageMatrix, I...) = dm.data[I...]

"""
    to_hard_calls(dm::DosageMatrix, threshold::Float64=0.9)

Convert dosage matrix to hard genotype calls.

# Arguments
- `dm`: DosageMatrix to convert
- `threshold`: Probability threshold for calling (default 0.9)

# Returns
- GenotypeMatrix with hard calls (missing for uncertain calls)
"""
function to_hard_calls(dm::DosageMatrix, threshold::Float64=0.9)
    n_s, n_v = size(dm)
    hard_calls = Matrix{Union{Int8, Missing}}(undef, n_s, n_v)
    
    for j in 1:n_v
        for i in 1:n_s
            d = dm.data[i, j]
            if ismissing(d)
                hard_calls[i, j] = missing
            else
                # Convert dosage to genotype probabilities
                # Assuming HWE: P(het) = 2p(1-p), P(hom) = p² or (1-p)²
                if d < 0.5 && (1 - d/2) >= threshold
                    hard_calls[i, j] = Int8(0)  # Likely homozygous ref
                elseif d >= 0.5 && d <= 1.5
                    # Check if heterozygous is most likely
                    hard_calls[i, j] = Int8(1)  # Heterozygous
                elseif d > 1.5 && (d/2) >= threshold
                    hard_calls[i, j] = Int8(2)  # Likely homozygous alt
                else
                    hard_calls[i, j] = missing  # Uncertain
                end
            end
        end
    end
    
    return GenotypeMatrix(hard_calls, dm.sample_ids, dm.variant_ids)
end

"""
    allele_frequency(dm::DosageMatrix)

Calculate allele frequency from dosage data.

# Arguments
- `dm`: DosageMatrix

# Returns
- Vector{Float64} of allele frequencies
"""
function allele_frequency(dm::DosageMatrix)
    freqs = Vector{Float64}(undef, n_variants(dm))
    
    for j in 1:n_variants(dm)
        dosages = skipmissing(dm.data[:, j])
        if isempty(dosages)
            freqs[j] = NaN
        else
            freqs[j] = mean(dosages) / 2
        end
    end
    
    return freqs
end

# ============================================================================
# Variant Information Structures
# ============================================================================

"""
    VariantInfo

Detailed information about a genetic variant.

# Fields
- `id::String`: Variant identifier (e.g., rsID)
- `chromosome::Chromosome`: Chromosome location
- `position::Position`: Genomic position (1-based)
- `ref::String`: Reference allele
- `alt::String`: Alternate allele
- `qual::Float64`: Quality score
- `filter::String`: Filter status (PASS, etc.)
- `info::Dict{String, Any}`: Additional INFO fields
"""
struct VariantInfo <: AbstractVariant
    id::String
    chromosome::Chromosome
    position::Position
    ref::String
    alt::String
    qual::Float64
    filter::String
    info::Dict{String, Any}
    
    function VariantInfo(
        id::String,
        chromosome::Chromosome,
        position::Position,
        ref::String,
        alt::String;
        qual::Float64=NaN,
        filter::String=".",
        info::Dict{String, Any}=Dict{String, Any}()
    )
        new(id, chromosome, position, ref, alt, qual, filter, info)
    end
end

# Interface implementations
chromosome(v::VariantInfo) = v.chromosome
position(v::VariantInfo) = v.position
reference_allele(v::VariantInfo) = v.ref
alternate_allele(v::VariantInfo) = v.alt

"""
    is_snp(v::VariantInfo)

Check if variant is a single nucleotide polymorphism.
"""
is_snp(v::VariantInfo) = length(v.ref) == 1 && length(v.alt) == 1

"""
    is_indel(v::VariantInfo)

Check if variant is an insertion or deletion.
"""
is_indel(v::VariantInfo) = length(v.ref) != length(v.alt)

"""
    is_transition(v::VariantInfo)

Check if SNP is a transition (purine↔purine or pyrimidine↔pyrimidine).
"""
function is_transition(v::VariantInfo)
    if !is_snp(v)
        return false
    end
    purines = Set(['A', 'G'])
    pyrimidines = Set(['C', 'T'])
    r, a = uppercase(v.ref[1]), uppercase(v.alt[1])
    return (r in purines && a in purines) || (r in pyrimidines && a in pyrimidines)
end

"""
    is_transversion(v::VariantInfo)

Check if SNP is a transversion (purine↔pyrimidine).
"""
is_transversion(v::VariantInfo) = is_snp(v) && !is_transition(v)
