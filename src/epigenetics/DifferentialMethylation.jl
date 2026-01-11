# ============================================================================
# DifferentialMethylation.jl - Differential Methylation Analysis
# ============================================================================

"""
    differential_methylation(beta::Matrix{Float64}, groups::Vector{Int})

Test for differential methylation between groups.
"""
function differential_methylation(beta::Matrix{Float64}, groups::Vector{Int})
    n_sites = size(beta, 1)
    g1_idx = findall(==(unique(groups)[1]), groups)
    g2_idx = findall(==(unique(groups)[2]), groups)
    
    pvalues = Vector{Float64}(undef, n_sites)
    delta_beta = Vector{Float64}(undef, n_sites)
    
    for s in 1:n_sites
        b1, b2 = beta[s, g1_idx], beta[s, g2_idx]
        result = welch_t_test(b1, b2)
        pvalues[s] = result.pvalue
        delta_beta[s] = mean(b2) - mean(b1)
    end
    
    fdr_result = fdr_correction(pvalues)
    return (pvalues=pvalues, qvalues=fdr_result.qvalues, delta_beta=delta_beta)
end

"""
    dmr_detection(positions::Vector{Int}, pvalues::Vector{Float64}; 
                 max_gap::Int=1000, min_cpgs::Int=3)

Detect differentially methylated regions (DMRs).
"""
function dmr_detection(positions::Vector{Int}, pvalues::Vector{Float64};
                      max_gap::Int=1000, min_cpgs::Int=3, pval_threshold::Float64=0.05)
    significant = pvalues .< pval_threshold
    order = sortperm(positions)
    
    dmrs = Vector{Tuple{Int, Int, Int, Float64}}()  # (start, end, n_cpgs, mean_pval)
    
    in_dmr = false
    dmr_start = 0
    dmr_cpgs = Int[]
    dmr_pvals = Float64[]
    
    for i in order
        if significant[i]
            if !in_dmr
                in_dmr = true
                dmr_start = positions[i]
                dmr_cpgs = [i]
                dmr_pvals = [pvalues[i]]
            elseif positions[i] - positions[dmr_cpgs[end]] <= max_gap
                push!(dmr_cpgs, i)
                push!(dmr_pvals, pvalues[i])
            else
                # End current DMR
                if length(dmr_cpgs) >= min_cpgs
                    push!(dmrs, (dmr_start, positions[dmr_cpgs[end]], 
                                length(dmr_cpgs), mean(dmr_pvals)))
                end
                dmr_start = positions[i]
                dmr_cpgs = [i]
                dmr_pvals = [pvalues[i]]
            end
        else
            if in_dmr && length(dmr_cpgs) >= min_cpgs
                push!(dmrs, (dmr_start, positions[dmr_cpgs[end]], 
                            length(dmr_cpgs), mean(dmr_pvals)))
            end
            in_dmr = false
        end
    end
    
    return dmrs
end
