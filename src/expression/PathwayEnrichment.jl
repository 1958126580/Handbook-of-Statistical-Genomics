# ============================================================================
# PathwayEnrichment.jl - Pathway and Gene Set Enrichment
# ============================================================================

"""
    pathway_enrichment(gene_list::Vector{String}, pathway_db::Dict{String, Vector{String}},
                      background::Vector{String})

Test for pathway enrichment using hypergeometric test.
"""
function pathway_enrichment(gene_list::Vector{String}, 
                           pathway_db::Dict{String, Vector{String}},
                           background::Vector{String})
    n_total = length(background)
    n_sig = length(gene_list)
    
    results = DataFrame(
        pathway = String[],
        n_pathway = Int[],
        n_overlap = Int[],
        expected = Float64[],
        fold_enrichment = Float64[],
        pvalue = Float64[]
    )
    
    for (pathway, genes) in pathway_db
        pathway_genes = intersect(genes, background)
        n_pathway = length(pathway_genes)
        
        if n_pathway < 5
            continue
        end
        
        n_overlap = length(intersect(gene_list, pathway_genes))
        expected = n_pathway * n_sig / n_total
        fold_enrichment = n_overlap / (expected + 1e-10)
        
        # Hypergeometric test
        pval = hypergeometric_test(n_overlap, n_pathway, n_total - n_pathway, n_sig)
        
        push!(results, (
            pathway = pathway,
            n_pathway = n_pathway,
            n_overlap = n_overlap,
            expected = expected,
            fold_enrichment = fold_enrichment,
            pvalue = pval
        ))
    end
    
    # FDR correction
    if nrow(results) > 0
        fdr_result = fdr_correction(results.pvalue)
        results.qvalue = fdr_result.qvalues
    end
    
    return sort(results, :pvalue)
end

"""
    hypergeometric_test(k, K, N_K, n)

Hypergeometric test: P(X >= k) where X ~ Hypergeometric(K, N-K, n)
"""
function hypergeometric_test(k::Int, K::Int, N_K::Int, n::Int)
    # P(X >= k)
    pval = 1 - cdf(Hypergeometric(K, N_K, n), k - 1)
    return clamp(pval, 0.0, 1.0)
end

"""
    gsea(expression::Matrix{Float64}, ranking::Vector{Float64}, 
        gene_set::Vector{String}, gene_ids::Vector{String})

Gene Set Enrichment Analysis.
"""
function gsea(expression::Matrix{Float64}, ranking::Vector{Float64},
             gene_set::Vector{String}, gene_ids::Vector{String})
    n_genes = length(gene_ids)
    
    # Sort by ranking
    order = sortperm(ranking, rev=true)
    sorted_genes = gene_ids[order]
    sorted_ranking = ranking[order]
    
    # Calculate running sum
    in_set = [g in gene_set for g in sorted_genes]
    n_in_set = sum(in_set)
    n_out_set = n_genes - n_in_set
    
    if n_in_set == 0
        return (es=0.0, nes=0.0, pvalue=1.0)
    end
    
    # Weight by absolute ranking
    p_hit = abs.(sorted_ranking) .* in_set
    p_hit ./= sum(p_hit)
    p_miss = (1 .- in_set) ./ n_out_set
    
    running_sum = cumsum(p_hit .- p_miss)
    
    # Enrichment score
    es = maximum(abs.(running_sum)) * sign(running_sum[argmax(abs.(running_sum))])
    
    # Permutation test for significance (simplified)
    null_es = Float64[]
    for _ in 1:1000
        perm_set = shuffle(in_set)
        perm_phit = abs.(sorted_ranking) .* perm_set
        perm_phit ./= sum(perm_phit) + 1e-10
        perm_pmiss = (1 .- perm_set) ./ max(sum(.!perm_set), 1)
        perm_rs = cumsum(perm_phit .- perm_pmiss)
        push!(null_es, maximum(abs.(perm_rs)))
    end
    
    # Normalized ES
    nes = es / (mean(null_es) + 1e-10)
    
    # P-value
    if es > 0
        pval = sum(null_es .>= es) / length(null_es)
    else
        pval = sum(null_es .<= es) / length(null_es)
    end
    
    return (es=es, nes=nes, pvalue=pval, leading_edge_size=n_in_set)
end
