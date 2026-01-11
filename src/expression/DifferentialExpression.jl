# ============================================================================
# DifferentialExpression.jl - Differential Gene Expression Analysis
# ============================================================================

"""
    normalize_expression(counts::Matrix{Int}; method::Symbol=:tmm)

Normalize RNA-seq count data.
"""
function normalize_expression(counts::Matrix{Int}; method::Symbol=:tmm)
    n_genes, n_samples = size(counts)
    
    if method == :cpm
        # Counts per million
        lib_sizes = sum(counts, dims=1)
        return counts ./ lib_sizes .* 1e6
        
    elseif method == :rpkm || method == :fpkm
        # RPKM/FPKM (simplified without gene lengths)
        lib_sizes = sum(counts, dims=1)
        return counts ./ lib_sizes .* 1e6
        
    elseif method == :tpm
        # Transcripts per million
        lib_sizes = sum(counts, dims=1)
        rpk = counts ./ 1000  # Simplified without lengths
        scale_factors = sum(rpk, dims=1)
        return rpk ./ scale_factors .* 1e6
        
    elseif method == :tmm
        # TMM normalization (Robinson & Oshlack)
        return tmm_normalize(counts)
        
    else
        throw(ArgumentError("Unknown normalization method: $method"))
    end
end

"""TMM normalization."""
function tmm_normalize(counts::Matrix{Int})
    n_genes, n_samples = size(counts)
    
    # Use first sample as reference
    ref = counts[:, 1]
    lib_sizes = sum(counts, dims=1)
    ref_lib = lib_sizes[1]
    
    norm_factors = ones(n_samples)
    
    for s in 2:n_samples
        f_s = counts[:, s]
        
        # Log ratios and absolute levels
        valid = (ref .> 0) .& (f_s .> 0)
        if sum(valid) < 10
            continue
        end
        
        M = log2.((f_s[valid] ./ lib_sizes[s]) ./ (ref[valid] ./ ref_lib))
        A = 0.5 .* (log2.(f_s[valid] ./ lib_sizes[s]) .+ log2.(ref[valid] ./ ref_lib))
        
        # Trim extreme values (top/bottom 30%)
        M_trim = 0.3
        A_trim = 0.05
        
        M_low, M_high = quantile(M, M_trim), quantile(M, 1 - M_trim)
        A_low, A_high = quantile(A, A_trim), quantile(A, 1 - A_trim)
        
        keep = (M .>= M_low) .& (M .<= M_high) .& (A .>= A_low) .& (A .<= A_high)
        
        if sum(keep) > 5
            norm_factors[s] = 2^mean(M[keep])
        end
    end
    
    return Float64.(counts) .* norm_factors'
end

"""
    differential_expression(expression::Matrix{Float64}, groups::Vector{Int};
                           method::Symbol=:ttest)

Test for differential expression between groups.
"""
function differential_expression(expression::Matrix{Float64}, groups::Vector{Int};
                                method::Symbol=:ttest)
    n_genes = size(expression, 1)
    unique_groups = sort(unique(groups))
    
    @assert length(unique_groups) == 2 "Currently supports 2-group comparison"
    
    g1_idx = findall(==(unique_groups[1]), groups)
    g2_idx = findall(==(unique_groups[2]), groups)
    
    pvalues = Vector{Float64}(undef, n_genes)
    log2fc = Vector{Float64}(undef, n_genes)
    
    for g in 1:n_genes
        expr1 = expression[g, g1_idx]
        expr2 = expression[g, g2_idx]
        
        if var(expr1) < 1e-10 || var(expr2) < 1e-10
            pvalues[g] = 1.0
            log2fc[g] = 0.0
            continue
        end
        
        result = welch_t_test(expr1, expr2)
        pvalues[g] = result.pvalue
        log2fc[g] = log2(mean(expr2) + 1) - log2(mean(expr1) + 1)
    end
    
    # FDR adjustment
    fdr_result = fdr_correction(pvalues)
    
    return (pvalues=pvalues, qvalues=fdr_result.qvalues, log2fc=log2fc)
end

"""
    deseq_normalize(counts::Matrix{Int})

DESeq2-style median-of-ratios normalization.
"""
function deseq_normalize(counts::Matrix{Int})
    n_genes, n_samples = size(counts)
    
    # Geometric mean per gene
    log_counts = log.(Float64.(counts) .+ 1)
    log_geometric_mean = mean(log_counts, dims=2)
    
    # Size factors
    size_factors = ones(n_samples)
    
    for s in 1:n_samples
        ratios = log_counts[:, s] .- log_geometric_mean
        valid_ratios = filter(!isinf, ratios[:])
        if !isempty(valid_ratios)
            size_factors[s] = exp(median(valid_ratios))
        end
    end
    
    return Float64.(counts) ./ size_factors'
end
