# ============================================================================
# CoExpressionNetworks.jl - Gene Co-expression Network Analysis
# ============================================================================

"""
    coexpression_network(expression::Matrix{Float64}; threshold::Float64=0.5)

Build gene co-expression network using correlation.
"""
function coexpression_network(expression::Matrix{Float64}; threshold::Float64=0.5)
    n_genes = size(expression, 1)
    
    # Calculate correlation matrix
    cor_matrix = cor(expression')
    
    # Apply threshold for adjacency
    adjacency = abs.(cor_matrix) .>= threshold
    for i in 1:n_genes
        adjacency[i, i] = false
    end
    
    return (correlation=cor_matrix, adjacency=adjacency)
end

"""
    module_detection(expression::Matrix{Float64}; min_module_size::Int=30)

Detect co-expression modules using hierarchical clustering.
"""
function module_detection(expression::Matrix{Float64}; min_module_size::Int=30)
    n_genes = size(expression, 1)
    
    # Calculate dissimilarity (1 - |correlation|)
    cor_matrix = cor(expression')
    dissim_matrix = 1 .- abs.(cor_matrix)
    
    # Hierarchical clustering (simple single linkage)
    modules = collect(1:n_genes)  # Start with each gene as own module
    
    # Simplified: use correlation threshold for modules
    high_cor_pairs = findall(x -> x > 0.7, abs.(cor_matrix))
    
    # Union-find for module assignment
    parent = collect(1:n_genes)
    
    function find_root(i)
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        return i
    end
    
    function union!(i, j)
        ri, rj = find_root(i), find_root(j)
        if ri != rj
            parent[ri] = rj
        end
    end
    
    for idx in high_cor_pairs
        i, j = Tuple(idx)
        if i != j
            union!(i, j)
        end
    end
    
    # Assign final modules
    final_modules = [find_root(i) for i in 1:n_genes]
    
    # Renumber modules sequentially
    unique_mods = unique(final_modules)
    mod_map = Dict(m => i for (i, m) in enumerate(unique_mods))
    final_modules = [mod_map[m] for m in final_modules]
    
    return (module_assignments=final_modules, n_modules=length(unique_mods))
end

"""
    module_eigengene(expression::Matrix{Float64}, module_assignments::Vector{Int})

Calculate module eigengenes (first PC of each module).
"""
function module_eigengene(expression::Matrix{Float64}, module_assignments::Vector{Int})
    unique_mods = unique(module_assignments)
    n_samples = size(expression, 2)
    
    eigengenes = Matrix{Float64}(undef, length(unique_mods), n_samples)
    
    for (k, mod) in enumerate(unique_mods)
        mod_genes = findall(==(mod), module_assignments)
        
        if length(mod_genes) < 3
            eigengenes[k, :] = mean(expression[mod_genes, :], dims=1)
        else
            mod_expr = expression[mod_genes, :]
            
            # PCA on module expression
            centered = mod_expr .- mean(mod_expr, dims=2)
            svd_result = svd(centered)
            eigengenes[k, :] = svd_result.V[:, 1]
        end
    end
    
    return eigengenes
end
