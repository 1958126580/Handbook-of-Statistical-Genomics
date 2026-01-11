# ============================================================================
# EQTL.jl - Expression Quantitative Trait Loci Analysis
# ============================================================================

"""
    eqtl_mapping(gm::GenotypeMatrix, expression::Matrix{Float64}, gene_ids::Vector{String};
                cis_window::Int=1_000_000)

Map eQTLs linking genetic variants to gene expression.
"""
function eqtl_mapping(gm::GenotypeMatrix, expression::Matrix{Float64}, gene_ids::Vector{String};
                     cis_window::Int=1_000_000, gene_positions::Dict{String, Tuple{Any, Int}}=Dict())
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    n_genes = length(gene_ids)
    
    @assert size(expression, 1) == n_samp "Expression matrix samples must match genotypes"
    @assert size(expression, 2) == n_genes "Expression columns must match gene_ids"
    
    results = DataFrame(
        gene_id = String[],
        variant_id = String[],
        chromosome = Chromosome[],
        position = Position[],
        beta = Float64[],
        se = Float64[],
        pvalue = Float64[],
        is_cis = Bool[]
    )
    
    for g in 1:n_genes
        expr = expression[:, g]
        gene_id = gene_ids[g]
        
        for j in 1:n_var
            geno = [ismissing(gm.data[i, j]) ? NaN : Float64(gm.data[i, j]) for i in 1:n_samp]
            
            complete = findall(i -> !isnan(geno[i]) && !isnan(expr[i]), 1:n_samp)
            
            if length(complete) < 20
                continue
            end
            
            X = hcat(ones(length(complete)), geno[complete])
            y = expr[complete]
            
            try
                result = linear_regression(X, y)
                
                # Check if cis
                is_cis = false
                if haskey(gene_positions, gene_id)
                    g_chr, g_pos = gene_positions[gene_id]
                    if g_chr == gm.chromosomes[j]
                        is_cis = abs(gm.positions[j] - g_pos) <= cis_window
                    end
                end
                
                push!(results, (
                    gene_id = gene_id,
                    variant_id = gm.variant_ids[j],
                    chromosome = gm.chromosomes[j],
                    position = gm.positions[j],
                    beta = result.coefficients[2],
                    se = result.se[2],
                    pvalue = result.pvalues[2],
                    is_cis = is_cis
                ))
            catch
                continue
            end
        end
    end
    
    return results
end

"""
    cis_eqtl(gm::GenotypeMatrix, expression::Matrix{Float64}, gene_ids::Vector{String},
            gene_positions::Dict{String, Tuple{Any, Int}}; cis_window::Int=1_000_000)

Map only cis-eQTLs (variants near genes).
"""
function cis_eqtl(gm::GenotypeMatrix, expression::Matrix{Float64}, gene_ids::Vector{String},
                gene_positions::Dict{String, Tuple{Any, Int}}; cis_window::Int=1_000_000)
    results = eqtl_mapping(gm, expression, gene_ids; 
                          cis_window=cis_window, gene_positions=gene_positions)
    return filter(row -> row.is_cis, results)
end

"""
    trans_eqtl(gm::GenotypeMatrix, expression::Matrix{Float64}, gene_ids::Vector{String},
              gene_positions::Dict{String, Tuple{Any, Int}}; cis_window::Int=1_000_000)

Map only trans-eQTLs (variants far from genes).
"""
function trans_eqtl(gm::GenotypeMatrix, expression::Matrix{Float64}, gene_ids::Vector{String},
                   gene_positions::Dict{String, Tuple{Any, Int}}; cis_window::Int=1_000_000)
    results = eqtl_mapping(gm, expression, gene_ids;
                          cis_window=cis_window, gene_positions=gene_positions)
    return filter(row -> !row.is_cis, results)
end
