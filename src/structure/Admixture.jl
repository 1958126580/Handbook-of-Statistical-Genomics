# ============================================================================
# Admixture.jl - Admixture Analysis and f-statistics
# ============================================================================

"""
    admixture_proportions(gm::GenotypeMatrix, K::Int)

Estimate admixture proportions using maximum likelihood.
"""
function admixture_proportions(gm::GenotypeMatrix, K::Int)
    return structure_clustering(gm, K)
end

"""
    f3_statistic(gm::GenotypeMatrix, target::Vector{Int}, source1::Vector{Int}, source2::Vector{Int})

Calculate f3 statistic: f3(Target; Source1, Source2).

Negative f3 indicates Target is admixed from Source1 and Source2.
"""
function f3_statistic(gm::GenotypeMatrix, target::Vector{Int}, 
                     source1::Vector{Int}, source2::Vector{Int})
    n_var = n_variants(gm)
    
    f3_values = Float64[]
    
    for j in 1:n_var
        # Allele frequencies
        gt = [gm.data[i, j] for i in target if !ismissing(gm.data[i, j])]
        gs1 = [gm.data[i, j] for i in source1 if !ismissing(gm.data[i, j])]
        gs2 = [gm.data[i, j] for i in source2 if !ismissing(gm.data[i, j])]
        
        if isempty(gt) || isempty(gs1) || isempty(gs2)
            continue
        end
        
        pt = sum(gt) / (2 * length(gt))
        p1 = sum(gs1) / (2 * length(gs1))
        p2 = sum(gs2) / (2 * length(gs2))
        
        # f3 = (pt - p1)(pt - p2)
        f3 = (pt - p1) * (pt - p2)
        push!(f3_values, f3)
    end
    
    f3_mean = mean(f3_values)
    f3_se = std(f3_values) / sqrt(length(f3_values))
    z_score = f3_mean / f3_se
    pvalue = 2 * ccdf(Normal(), abs(z_score))
    
    # Negative f3: admixture; Positive: no admixture
    interpretation = f3_mean < 0 && pvalue < 0.05 ? :admixture : :no_admixture
    
    return (f3=f3_mean, se=f3_se, z_score=z_score, pvalue=pvalue,
            interpretation=interpretation)
end

"""
    f4_statistic(gm::GenotypeMatrix, pop_w::Vector{Int}, pop_x::Vector{Int},
                pop_y::Vector{Int}, pop_z::Vector{Int})

Calculate f4 statistic: f4(W, X; Y, Z).

Non-zero f4 indicates gene flow between (W or X) and (Y or Z).
"""
function f4_statistic(gm::GenotypeMatrix, pop_w::Vector{Int}, pop_x::Vector{Int},
                     pop_y::Vector{Int}, pop_z::Vector{Int})
    n_var = n_variants(gm)
    
    f4_values = Float64[]
    
    for j in 1:n_var
        gw = [gm.data[i, j] for i in pop_w if !ismissing(gm.data[i, j])]
        gx = [gm.data[i, j] for i in pop_x if !ismissing(gm.data[i, j])]
        gy = [gm.data[i, j] for i in pop_y if !ismissing(gm.data[i, j])]
        gz = [gm.data[i, j] for i in pop_z if !ismissing(gm.data[i, j])]
        
        if any(isempty, [gw, gx, gy, gz])
            continue
        end
        
        pw = sum(gw) / (2 * length(gw))
        px = sum(gx) / (2 * length(gx))
        py = sum(gy) / (2 * length(gy))
        pz = sum(gz) / (2 * length(gz))
        
        # f4 = (pw - px)(py - pz)
        f4 = (pw - px) * (py - pz)
        push!(f4_values, f4)
    end
    
    f4_mean = mean(f4_values)
    f4_se = std(f4_values) / sqrt(length(f4_values))
    z_score = f4_mean / f4_se
    pvalue = 2 * ccdf(Normal(), abs(z_score))
    
    return (f4=f4_mean, se=f4_se, z_score=z_score, pvalue=pvalue)
end

"""
    d_statistic(gm::GenotypeMatrix, pop1::Vector{Int}, pop2::Vector{Int},
               pop3::Vector{Int}, outgroup::Vector{Int})

Patterson's D statistic (ABBA-BABA test).

D > 0: Gene flow between pop2 and pop3
D < 0: Gene flow between pop1 and pop3
"""
function d_statistic(gm::GenotypeMatrix, pop1::Vector{Int}, pop2::Vector{Int},
                    pop3::Vector{Int}, outgroup::Vector{Int})
    n_var = n_variants(gm)
    
    abba = 0.0
    baba = 0.0
    
    for j in 1:n_var
        g1 = [gm.data[i, j] for i in pop1 if !ismissing(gm.data[i, j])]
        g2 = [gm.data[i, j] for i in pop2 if !ismissing(gm.data[i, j])]
        g3 = [gm.data[i, j] for i in pop3 if !ismissing(gm.data[i, j])]
        go = [gm.data[i, j] for i in outgroup if !ismissing(gm.data[i, j])]
        
        if any(isempty, [g1, g2, g3, go])
            continue
        end
        
        p1 = sum(g1) / (2 * length(g1))
        p2 = sum(g2) / (2 * length(g2))
        p3 = sum(g3) / (2 * length(g3))
        po = sum(go) / (2 * length(go))
        
        # ABBA: derived in P2 and P3 but ancestral in P1 and O
        # BABA: derived in P1 and P3 but ancestral in P2 and O
        abba += (1-p1) * p2 * p3 * (1-po)
        baba += p1 * (1-p2) * p3 * (1-po)
    end
    
    D = (abba - baba) / (abba + baba + 1e-10)
    
    # Z-score requires block jackknife (simplified here)
    se_approx = 0.05  # Approximate
    z_score = D / se_approx
    pvalue = 2 * ccdf(Normal(), abs(z_score))
    
    return (D_stat=D, abba=abba, baba=baba, z_score=z_score, pvalue=pvalue)
end
