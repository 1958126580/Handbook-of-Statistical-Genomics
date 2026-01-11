# ============================================================================
# IO.jl - File Input/Output Functions
# ============================================================================

"""
    read_vcf(filepath::String; samples::Union{Vector{String}, Nothing}=nothing)

Read a VCF (Variant Call Format) file into a GenotypeMatrix.

# Arguments
- `filepath`: Path to VCF file
- `samples`: Optional list of samples to include

# Returns
- GenotypeMatrix with genotype data
"""
function read_vcf(filepath::String; samples::Union{Vector{String}, Nothing}=nothing)
    lines = readlines(filepath)
    
    # Parse header
    header_idx = findfirst(l -> startswith(l, "#CHROM"), lines)
    if header_idx === nothing
        throw(ArgumentError("Invalid VCF: no header line found"))
    end
    
    header = split(lines[header_idx], '\t')
    sample_ids = header[10:end]
    
    # Filter samples if requested
    if samples !== nothing
        sample_mask = [s in samples for s in sample_ids]
        sample_ids = sample_ids[sample_mask]
    else
        sample_mask = trues(length(sample_ids))
    end
    
    # Parse variants
    data_lines = lines[(header_idx+1):end]
    n_variants = length(data_lines)
    n_samples = length(sample_ids)
    
    genotypes = Matrix{Union{Int8, Missing}}(missing, n_samples, n_variants)
    variant_ids = Vector{String}(undef, n_variants)
    chromosomes = Vector{Chromosome}(undef, n_variants)
    positions = Vector{Position}(undef, n_variants)
    ref_alleles = Vector{String}(undef, n_variants)
    alt_alleles = Vector{String}(undef, n_variants)
    
    for (j, line) in enumerate(data_lines)
        fields = split(line, '\t')
        chromosomes[j] = tryparse(Int, fields[1]) !== nothing ? parse(Int, fields[1]) : fields[1]
        positions[j] = parse(Int64, fields[2])
        variant_ids[j] = fields[3] == "." ? "var_$j" : fields[3]
        ref_alleles[j] = fields[4]
        alt_alleles[j] = fields[5]
        
        # Parse genotypes (GT field is first in FORMAT)
        format_fields = split(fields[9], ':')
        gt_idx = findfirst(==("GT"), format_fields)
        gt_idx = gt_idx === nothing ? 1 : gt_idx
        
        sample_data = fields[10:end][sample_mask]
        for (i, sd) in enumerate(sample_data)
            gt_str = split(sd, ':')[gt_idx]
            genotypes[i, j] = parse_genotype(gt_str)
        end
    end
    
    GenotypeMatrix(genotypes, collect(sample_ids), variant_ids, 
                   chromosomes, positions, ref_alleles, alt_alleles)
end

"""Parse VCF genotype string to numeric value."""
function parse_genotype(gt::AbstractString)
    gt = strip(gt)
    if gt in (".", "./.", ".|.")
        return missing
    end
    
    # Handle phased (|) and unphased (/) genotypes
    alleles = split(replace(gt, "|" => "/"), "/")
    if length(alleles) != 2
        return missing
    end
    
    try
        a1 = parse(Int, alleles[1])
        a2 = parse(Int, alleles[2])
        return Int8(a1 + a2)
    catch
        return missing
    end
end

"""
    write_vcf(filepath::String, gm::GenotypeMatrix)

Write a GenotypeMatrix to VCF format.
"""
function write_vcf(filepath::String, gm::GenotypeMatrix)
    open(filepath, "w") do io
        # Write header
        println(io, "##fileformat=VCFv4.2")
        println(io, "##source=StatisticalGenomics.jl")
        println(io, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")
        
        # Column header
        print(io, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
        for sid in gm.sample_ids
            print(io, "\t", sid)
        end
        println(io)
        
        # Write variants
        for j in 1:n_variants(gm)
            print(io, gm.chromosomes[j], "\t")
            print(io, gm.positions[j], "\t")
            print(io, gm.variant_ids[j], "\t")
            print(io, gm.ref_alleles[j], "\t")
            print(io, gm.alt_alleles[j], "\t")
            print(io, ".\t.\t.\tGT")
            
            for i in 1:n_samples(gm)
                g = gm.data[i, j]
                gt_str = ismissing(g) ? "./." : (g == 0 ? "0/0" : (g == 1 ? "0/1" : "1/1"))
                print(io, "\t", gt_str)
            end
            println(io)
        end
    end
end

"""
    read_plink(prefix::String)

Read PLINK binary files (.bed, .bim, .fam).

# Arguments
- `prefix`: Path prefix for PLINK files (without extension)

# Returns
- GenotypeMatrix with genotype data
"""
function read_plink(prefix::String)
    # Read .fam file (sample information)
    fam_file = prefix * ".fam"
    fam_lines = readlines(fam_file)
    n_samples = length(fam_lines)
    sample_ids = String[]
    
    for line in fam_lines
        fields = split(line)
        push!(sample_ids, fields[2])  # Individual ID
    end
    
    # Read .bim file (variant information)
    bim_file = prefix * ".bim"
    bim_lines = readlines(bim_file)
    n_variants = length(bim_lines)
    
    variant_ids = Vector{String}(undef, n_variants)
    chromosomes = Vector{Chromosome}(undef, n_variants)
    positions = Vector{Position}(undef, n_variants)
    ref_alleles = Vector{String}(undef, n_variants)
    alt_alleles = Vector{String}(undef, n_variants)
    
    for (j, line) in enumerate(bim_lines)
        fields = split(line)
        chr_str = fields[1]
        chromosomes[j] = tryparse(Int, chr_str) !== nothing ? parse(Int, chr_str) : chr_str
        variant_ids[j] = fields[2]
        positions[j] = parse(Int64, fields[4])
        ref_alleles[j] = fields[6]
        alt_alleles[j] = fields[5]
    end
    
    # Read .bed file (genotypes in binary format)
    bed_file = prefix * ".bed"
    genotypes = Matrix{Union{Int8, Missing}}(missing, n_samples, n_variants)
    
    open(bed_file, "r") do io
        # Check magic number
        magic = read(io, 3)
        if magic != UInt8[0x6c, 0x1b, 0x01]
            throw(ArgumentError("Invalid PLINK .bed file"))
        end
        
        # Read genotypes (SNP-major mode)
        bytes_per_snp = ceil(Int, n_samples / 4)
        
        for j in 1:n_variants
            bytes = read(io, bytes_per_snp)
            sample_idx = 1
            
            for byte in bytes
                for bit_pair in 0:3
                    if sample_idx > n_samples
                        break
                    end
                    
                    geno_code = (byte >> (2 * bit_pair)) & 0x03
                    genotypes[sample_idx, j] = plink_decode(geno_code)
                    sample_idx += 1
                end
            end
        end
    end
    
    GenotypeMatrix(genotypes, sample_ids, variant_ids, 
                   chromosomes, positions, ref_alleles, alt_alleles)
end

"""Decode PLINK binary genotype."""
function plink_decode(code::UInt8)
    if code == 0x00
        return Int8(2)  # Homozygous for A1
    elseif code == 0x01
        return missing  # Missing
    elseif code == 0x02
        return Int8(1)  # Heterozygous
    else  # code == 0x03
        return Int8(0)  # Homozygous for A2
    end
end

"""
    read_expression_matrix(filepath::String; has_header::Bool=true)

Read a gene expression matrix from a CSV/TSV file.

# Returns
- NamedTuple with expression matrix, gene_ids, and sample_ids
"""
function read_expression_matrix(filepath::String; has_header::Bool=true)
    df = CSV.read(filepath, DataFrame)
    
    gene_ids = string.(df[:, 1])
    sample_ids = names(df)[2:end]
    data = Matrix{Float64}(df[:, 2:end])
    
    (expression=data, gene_ids=gene_ids, sample_ids=sample_ids)
end

"""
    save_results(filepath::String, results::Any)

Save analysis results to JLD2 format.
"""
function save_results(filepath::String, results::Any)
    JLD2.save(filepath, "results", results)
end

"""
    load_results(filepath::String)

Load analysis results from JLD2 format.
"""
function load_results(filepath::String)
    JLD2.load(filepath, "results")
end
