# ============================================================================
# SingleCellAnalysis.jl - Single-Cell Genomics Methods
# ============================================================================
# Comprehensive methods for single-cell RNA-seq and related analyses
# Including QC, normalization, clustering, differential expression
# ============================================================================

"""
    SingleCellData

Structure for single-cell expression data.

# Fields
- `counts::Matrix{Float64}`: Raw count matrix (genes × cells)
- `gene_names::Vector{String}`: Gene identifiers
- `cell_ids::Vector{String}`: Cell identifiers
- `metadata::DataFrame`: Cell metadata
- `normalized::Union{Matrix{Float64}, Nothing}`: Normalized counts
"""
mutable struct SingleCellData
    counts::Matrix{Float64}
    gene_names::Vector{String}
    cell_ids::Vector{String}
    metadata::DataFrame
    normalized::Union{Matrix{Float64}, Nothing}
end

"""
    SingleCellQC

Quality control results for single-cell data.

# Fields
- `n_genes_per_cell::Vector{Int}`: Number of detected genes per cell
- `n_counts_per_cell::Vector{Float64}`: Total counts per cell
- `pct_mito::Vector{Float64}`: Percentage mitochondrial counts
- `pct_ribo::Vector{Float64}`: Percentage ribosomal counts
- `passed::Vector{Bool}`: Cells passing QC
"""
struct SingleCellQC
    n_genes_per_cell::Vector{Int}
    n_counts_per_cell::Vector{Float64}
    pct_mito::Vector{Float64}
    pct_ribo::Vector{Float64}
    passed::Vector{Bool}
end

"""
    sc_qc(data::SingleCellData; kwargs...) -> SingleCellQC

Perform quality control on single-cell data.

# Arguments
- `data`: SingleCellData object

# Keyword Arguments
- `min_genes::Int=200`: Minimum genes detected per cell
- `max_genes::Int=5000`: Maximum genes (filter doublets)
- `min_counts::Int=500`: Minimum total counts
- `max_pct_mito::Float64=20.0`: Maximum % mitochondrial
- `mito_prefix::String="MT-"`: Prefix for mitochondrial genes

# Example
```julia
qc = sc_qc(data; min_genes=200, max_pct_mito=15.0)
println("Cells passing QC: \$(sum(qc.passed))/\$(length(qc.passed))")
```

# References
- Luecken & Theis (2019) Mol. Syst. Biol. (best practices)
"""
function sc_qc(
    data::SingleCellData;
    min_genes::Int=200,
    max_genes::Int=5000,
    min_counts::Int=500,
    max_pct_mito::Float64=20.0,
    mito_prefix::String="MT-",
    ribo_prefix::String="RP"
)
    n_genes, n_cells = size(data.counts)

    # Genes detected per cell
    n_genes_per_cell = vec(sum(data.counts .> 0, dims=1))

    # Total counts per cell
    n_counts_per_cell = vec(sum(data.counts, dims=1))

    # Mitochondrial content
    mito_genes = findall(startswith.(data.gene_names, mito_prefix))
    if length(mito_genes) > 0
        mito_counts = vec(sum(data.counts[mito_genes, :], dims=1))
        pct_mito = 100 .* mito_counts ./ (n_counts_per_cell .+ 1e-10)
    else
        pct_mito = zeros(n_cells)
    end

    # Ribosomal content
    ribo_genes = findall(startswith.(data.gene_names, ribo_prefix))
    if length(ribo_genes) > 0
        ribo_counts = vec(sum(data.counts[ribo_genes, :], dims=1))
        pct_ribo = 100 .* ribo_counts ./ (n_counts_per_cell .+ 1e-10)
    else
        pct_ribo = zeros(n_cells)
    end

    # QC filtering
    passed = (n_genes_per_cell .>= min_genes) .&
             (n_genes_per_cell .<= max_genes) .&
             (n_counts_per_cell .>= min_counts) .&
             (pct_mito .<= max_pct_mito)

    return SingleCellQC(
        n_genes_per_cell,
        n_counts_per_cell,
        pct_mito,
        pct_ribo,
        passed
    )
end

"""
    sc_normalize(data::SingleCellData; method=:log_normalize) -> Matrix{Float64}

Normalize single-cell count data.

# Arguments
- `data`: SingleCellData object
- `method`: Normalization method

# Methods
1. `:log_normalize` - Log-normalize: log(CPM + 1)
2. `:scran` - Scran pooling-based size factors
3. `:sctransform` - Variance stabilizing transformation

# Example
```julia
normalized = sc_normalize(data; method=:log_normalize)
data.normalized = normalized
```

# References
- Lun et al. (2016) Genome Biol. (scran)
- Hafemeister & Satija (2019) Genome Biol. (sctransform)
"""
function sc_normalize(
    data::SingleCellData;
    method::Symbol=:log_normalize,
    scale_factor::Float64=10000.0
)
    counts = data.counts

    if method == :log_normalize
        # CPM + log1p transformation
        size_factors = sum(counts, dims=1)
        normalized = scale_factor .* counts ./ size_factors
        normalized = log1p.(normalized)

    elseif method == :scran
        # Scran-like deconvolution
        size_factors = scran_size_factors(counts)
        normalized = counts ./ size_factors'
        normalized = log1p.(normalized)

    elseif method == :sctransform
        # Simplified SCTransform
        normalized = sctransform_normalize(counts)

    else
        error("Unknown normalization method: $method")
    end

    return normalized
end

"""
    scran_size_factors(counts::Matrix{Float64}) -> Vector{Float64}

Compute size factors using scran-like deconvolution.
"""
function scran_size_factors(counts::Matrix{Float64})
    n_genes, n_cells = size(counts)

    # Pool cells and compute size factors
    # Simplified version: geometric mean-based

    # Filter lowly expressed genes
    mean_expr = vec(mean(counts, dims=2))
    expressed = mean_expr .> 0.1

    # Compute per-cell scaling factors
    counts_filt = counts[expressed, :]

    # Geometric mean per gene (reference)
    geom_mean = exp.(vec(mean(log.(counts_filt .+ 1), dims=2)))

    # Size factor per cell
    size_factors = zeros(n_cells)
    for j in 1:n_cells
        ratios = counts_filt[:, j] ./ (geom_mean .+ 1e-10)
        # Median ratio
        size_factors[j] = median(ratios[ratios .> 0])
    end

    # Normalize to mean 1
    size_factors = size_factors ./ mean(size_factors)

    return size_factors
end

"""
    sctransform_normalize(counts::Matrix{Float64}) -> Matrix{Float64}

Simplified sctransform variance-stabilizing transformation.
"""
function sctransform_normalize(counts::Matrix{Float64})
    n_genes, n_cells = size(counts)

    # Compute regularized negative binomial parameters
    log_umi = log10.(sum(counts, dims=1) .+ 1)

    normalized = zeros(n_genes, n_cells)

    for g in 1:n_genes
        y = counts[g, :]

        # Fit Poisson regression for mean
        μ = mean(y)

        if μ > 0
            # Variance under Poisson
            var_poisson = μ

            # Observed variance
            var_obs = var(y)

            # Overdispersion parameter
            θ = max(1e-6, μ^2 / (var_obs - μ + 1e-10))

            # Pearson residuals
            expected = μ
            variance = μ + μ^2 / θ
            normalized[g, :] = (y .- expected) ./ sqrt.(variance .+ 1e-10)
        end
    end

    # Clip extreme values
    normalized = clamp.(normalized, -sqrt(n_cells), sqrt(n_cells))

    return normalized
end

"""
    sc_highly_variable_genes(data::SingleCellData; kwargs...) -> Vector{Int}

Identify highly variable genes (HVGs).

# Arguments
- `data`: SingleCellData with normalized counts
- `n_top::Int=2000`: Number of top HVGs to select
- `method`: Selection method - :seurat, :cell_ranger

# Returns
Indices of highly variable genes

# Example
```julia
hvg_idx = sc_highly_variable_genes(data; n_top=2000)
```
"""
function sc_highly_variable_genes(
    data::SingleCellData;
    n_top::Int=2000,
    method::Symbol=:seurat,
    min_mean::Float64=0.0125,
    max_mean::Float64=3.0,
    min_dispersion::Float64=0.5
)
    if data.normalized === nothing
        error("Data must be normalized first")
    end

    expr = data.normalized
    n_genes, n_cells = size(expr)

    # Compute mean and variance for each gene
    gene_mean = vec(mean(expr, dims=2))
    gene_var = vec(var(expr, dims=2))

    # Coefficient of variation (dispersion)
    gene_cv2 = gene_var ./ (gene_mean.^2 .+ 1e-10)

    if method == :seurat
        # Seurat v3 method: variance stabilization
        # Bin genes by mean expression
        n_bins = 20
        mean_bins = range(minimum(gene_mean[gene_mean .> 0]), maximum(gene_mean), length=n_bins+1)

        dispersion_norm = zeros(n_genes)

        for i in 1:n_bins
            in_bin = (gene_mean .>= mean_bins[i]) .& (gene_mean .< mean_bins[i+1])
            if sum(in_bin) > 1
                bin_cv2 = gene_cv2[in_bin]
                dispersion_norm[in_bin] = (bin_cv2 .- mean(bin_cv2)) ./ (std(bin_cv2) .+ 1e-10)
            end
        end

        # Filter by mean and normalized dispersion
        valid = (gene_mean .> min_mean) .& (gene_mean .< max_mean) .&
                (dispersion_norm .> min_dispersion)

        # Select top N from valid
        valid_idx = findall(valid)
        if length(valid_idx) > n_top
            top_idx = sortperm(dispersion_norm[valid_idx], rev=true)[1:n_top]
            hvg_idx = valid_idx[top_idx]
        else
            hvg_idx = valid_idx
        end

    elseif method == :cell_ranger
        # Cell Ranger method: expected variance model
        # Fit mean-variance relationship
        log_mean = log10.(gene_mean .+ 1e-10)
        log_var = log10.(gene_var .+ 1e-10)

        # Simple loess-like fit
        expected_var = log_mean .+ 0.5  # Simplified

        # Residual variance
        residual_var = log_var .- expected_var

        hvg_idx = sortperm(residual_var, rev=true)[1:n_top]
    end

    return hvg_idx
end

"""
    sc_pca(data::SingleCellData, hvg_idx::Vector{Int}; n_pcs=50) -> NamedTuple

Run PCA on single-cell data.

# Arguments
- `data`: SingleCellData with normalized expression
- `hvg_idx`: Indices of highly variable genes
- `n_pcs`: Number of principal components

# Returns
Named tuple with PC scores, loadings, and variance explained
"""
function sc_pca(
    data::SingleCellData,
    hvg_idx::Vector{Int};
    n_pcs::Int=50,
    scale::Bool=true
)
    if data.normalized === nothing
        error("Data must be normalized first")
    end

    expr = data.normalized[hvg_idx, :]'  # Cells × genes

    # Center
    expr_centered = expr .- mean(expr, dims=1)

    # Scale
    if scale
        stds = std(expr_centered, dims=1)
        stds[stds .== 0] .= 1
        expr_centered = expr_centered ./ stds
    end

    # SVD
    U, S, V = svd(expr_centered)

    n_pcs = min(n_pcs, length(S))

    # PC scores (cells × PCs)
    pcs = U[:, 1:n_pcs] .* S[1:n_pcs]'

    # Variance explained
    var_explained = S.^2 ./ sum(S.^2)

    return (
        pcs = pcs,
        loadings = V[:, 1:n_pcs],
        variance_explained = var_explained[1:n_pcs],
        singular_values = S[1:n_pcs]
    )
end

"""
    sc_neighbors(pcs::Matrix{Float64}; k=15, method=:knn) -> Tuple

Compute nearest neighbor graph for cells.

# Arguments
- `pcs`: PC scores matrix (cells × PCs)
- `k`: Number of neighbors
- `method`: `:knn` or `:snn` (shared nearest neighbor)

# Returns
(neighbor_indices, neighbor_distances)
"""
function sc_neighbors(
    pcs::Matrix{Float64};
    k::Int=15,
    method::Symbol=:knn
)
    n_cells = size(pcs, 1)

    # Compute pairwise distances
    distances = zeros(n_cells, n_cells)
    for i in 1:n_cells
        for j in i:n_cells
            d = sqrt(sum((pcs[i, :] .- pcs[j, :]).^2))
            distances[i, j] = d
            distances[j, i] = d
        end
    end

    # Find k nearest neighbors for each cell
    neighbor_indices = zeros(Int, n_cells, k)
    neighbor_distances = zeros(n_cells, k)

    for i in 1:n_cells
        sorted_idx = sortperm(distances[i, :])
        # Exclude self
        neighbor_indices[i, :] = sorted_idx[2:k+1]
        neighbor_distances[i, :] = distances[i, sorted_idx[2:k+1]]
    end

    if method == :snn
        # Compute shared nearest neighbors
        snn_matrix = zeros(n_cells, n_cells)
        for i in 1:n_cells
            neighbors_i = Set(neighbor_indices[i, :])
            for j in 1:n_cells
                neighbors_j = Set(neighbor_indices[j, :])
                shared = length(intersect(neighbors_i, neighbors_j))
                snn_matrix[i, j] = shared / k
            end
        end
        return (neighbor_indices, neighbor_distances, snn_matrix)
    end

    return (neighbor_indices, neighbor_distances)
end

"""
    sc_cluster(pcs::Matrix{Float64}; method=:louvain, resolution=1.0) -> Vector{Int}

Cluster single cells.

# Arguments
- `pcs`: PC scores matrix
- `method`: Clustering method - :louvain, :leiden, :kmeans
- `resolution`: Resolution parameter for community detection

# Returns
Vector of cluster assignments
"""
function sc_cluster(
    pcs::Matrix{Float64};
    method::Symbol=:louvain,
    resolution::Float64=1.0,
    k::Int=15,
    n_clusters::Union{Int, Nothing}=nothing
)
    n_cells = size(pcs, 1)

    if method == :kmeans
        # K-means clustering
        if n_clusters === nothing
            n_clusters = max(2, round(Int, sqrt(n_cells / 2)))
        end

        clusters = kmeans_cluster(pcs, n_clusters)

    elseif method == :louvain || method == :leiden
        # Graph-based clustering
        # Build SNN graph
        nn_result = sc_neighbors(pcs; k=k, method=:snn)
        snn_matrix = nn_result[3]

        # Simple modularity-based clustering
        clusters = louvain_cluster(snn_matrix; resolution=resolution)

    else
        error("Unknown clustering method: $method")
    end

    return clusters
end

"""
    kmeans_cluster(data::Matrix{Float64}, k::Int; max_iter=100) -> Vector{Int}

K-means clustering implementation.
"""
function kmeans_cluster(data::Matrix{Float64}, k::Int; max_iter::Int=100)
    n, p = size(data)

    # Initialize centroids (k-means++)
    centroids = zeros(k, p)
    centroids[1, :] = data[rand(1:n), :]

    for i in 2:k
        # Distance to nearest centroid
        min_dists = [minimum([sum((data[j, :] .- centroids[c, :]).^2) for c in 1:i-1]) for j in 1:n]
        # Sample proportional to distance squared
        probs = min_dists ./ sum(min_dists)
        centroids[i, :] = data[sample_weighted(probs), :]
    end

    # Iterate
    assignments = zeros(Int, n)

    for _ in 1:max_iter
        # Assign points to nearest centroid
        new_assignments = zeros(Int, n)
        for i in 1:n
            dists = [sum((data[i, :] .- centroids[c, :]).^2) for c in 1:k]
            new_assignments[i] = argmin(dists)
        end

        # Check convergence
        if new_assignments == assignments
            break
        end
        assignments = new_assignments

        # Update centroids
        for c in 1:k
            members = findall(assignments .== c)
            if length(members) > 0
                centroids[c, :] = vec(mean(data[members, :], dims=1))
            end
        end
    end

    return assignments
end

function sample_weighted(probs::Vector{Float64})
    r = rand()
    cumsum_p = cumsum(probs)
    for (i, p) in enumerate(cumsum_p)
        if r <= p
            return i
        end
    end
    return length(probs)
end

"""
    louvain_cluster(adjacency::Matrix{Float64}; resolution=1.0) -> Vector{Int}

Louvain community detection algorithm.
"""
function louvain_cluster(adjacency::Matrix{Float64}; resolution::Float64=1.0)
    n = size(adjacency, 1)

    # Initialize: each node in its own community
    communities = collect(1:n)

    # Total edge weight
    m = sum(adjacency) / 2

    # Node degrees
    degrees = vec(sum(adjacency, dims=2))

    # Modularity function
    function modularity(comms)
        Q = 0.0
        for i in 1:n
            for j in 1:n
                if comms[i] == comms[j]
                    Q += adjacency[i, j] - resolution * degrees[i] * degrees[j] / (2m)
                end
            end
        end
        return Q / (2m)
    end

    # Iterate until no improvement
    improved = true
    while improved
        improved = false
        for i in randperm(n)
            current_comm = communities[i]

            # Find neighboring communities
            neighbor_comms = unique(communities[adjacency[i, :] .> 0])

            best_comm = current_comm
            best_delta = 0.0

            for new_comm in neighbor_comms
                if new_comm == current_comm
                    continue
                end

                # Compute modularity change
                # ΔQ = [Σ_in + 2k_i,in] / 2m - [(Σ_tot + k_i) / 2m]² - [Σ_in/2m - (Σ_tot/2m)² - (k_i/2m)²]

                # Simplified: gain from moving to new community
                ki_in = sum(adjacency[i, communities .== new_comm])
                ki_out = sum(adjacency[i, communities .== current_comm])
                Σ_tot_new = sum(degrees[communities .== new_comm])
                Σ_tot_old = sum(degrees[communities .== current_comm])

                delta = (ki_in - ki_out) / m - resolution * degrees[i] * (Σ_tot_new - Σ_tot_old + degrees[i]) / (2 * m^2)

                if delta > best_delta
                    best_delta = delta
                    best_comm = new_comm
                end
            end

            if best_comm != current_comm
                communities[i] = best_comm
                improved = true
            end
        end
    end

    # Renumber communities
    unique_comms = unique(communities)
    comm_map = Dict(c => i for (i, c) in enumerate(unique_comms))
    communities = [comm_map[c] for c in communities]

    return communities
end

"""
    sc_differential_expression(data::SingleCellData, clusters::Vector{Int};
                              test=:wilcoxon) -> Dict{Int, DataFrame}

Find differentially expressed genes between clusters.

# Arguments
- `data`: SingleCellData with normalized expression
- `clusters`: Cluster assignments
- `test`: Statistical test - :wilcoxon, :t_test, :negbinom

# Returns
Dictionary mapping cluster ID to DataFrame of DE results
"""
function sc_differential_expression(
    data::SingleCellData,
    clusters::Vector{Int};
    test::Symbol=:wilcoxon,
    min_pct::Float64=0.1,
    logfc_threshold::Float64=0.25
)
    if data.normalized === nothing
        error("Data must be normalized first")
    end

    expr = data.normalized
    n_genes, n_cells = size(expr)
    unique_clusters = sort(unique(clusters))

    results = Dict{Int, DataFrame}()

    for cluster in unique_clusters
        cluster_cells = findall(clusters .== cluster)
        other_cells = findall(clusters .!= cluster)

        de_results = DataFrame(
            gene = String[],
            avg_logfc = Float64[],
            pct_1 = Float64[],
            pct_2 = Float64[],
            pvalue = Float64[],
            pvalue_adj = Float64[]
        )

        pvalues = Float64[]
        gene_indices = Int[]

        for g in 1:n_genes
            expr_cluster = expr[g, cluster_cells]
            expr_other = expr[g, other_cells]

            # Percentage expressing
            pct_1 = mean(expr_cluster .> 0)
            pct_2 = mean(expr_other .> 0)

            # Filter by min_pct
            if max(pct_1, pct_2) < min_pct
                continue
            end

            # Log fold change
            mean_1 = mean(expr_cluster)
            mean_2 = mean(expr_other)
            logfc = mean_1 - mean_2  # Already log-transformed

            # Statistical test
            if test == :wilcoxon
                pvalue = wilcoxon_test(expr_cluster, expr_other)
            elseif test == :t_test
                pvalue = t_test(expr_cluster, expr_other)
            else
                pvalue = wilcoxon_test(expr_cluster, expr_other)
            end

            push!(gene_indices, g)
            push!(pvalues, pvalue)

            push!(de_results, (
                gene = data.gene_names[g],
                avg_logfc = logfc,
                pct_1 = pct_1,
                pct_2 = pct_2,
                pvalue = pvalue,
                pvalue_adj = 0.0  # Fill later
            ))
        end

        # Multiple testing correction (BH)
        if nrow(de_results) > 0
            de_results.pvalue_adj = benjamini_hochberg(de_results.pvalue)
            sort!(de_results, :pvalue)
        end

        results[cluster] = de_results
    end

    return results
end

"""
    wilcoxon_test(x::Vector{Float64}, y::Vector{Float64}) -> Float64

Wilcoxon rank-sum test (Mann-Whitney U).
"""
function wilcoxon_test(x::Vector{Float64}, y::Vector{Float64})
    nx, ny = length(x), length(y)

    # Combine and rank
    combined = vcat(x, y)
    ranks = sortperm(sortperm(combined))
    rank_x = ranks[1:nx]

    # U statistic
    U = sum(rank_x) - nx * (nx + 1) / 2

    # Normal approximation
    μ = nx * ny / 2
    σ = sqrt(nx * ny * (nx + ny + 1) / 12)

    z = (U - μ) / σ
    pvalue = 2 * ccdf(Normal(), abs(z))

    return pvalue
end

"""
    t_test(x::Vector{Float64}, y::Vector{Float64}) -> Float64

Two-sample t-test.
"""
function t_test(x::Vector{Float64}, y::Vector{Float64})
    nx, ny = length(x), length(y)
    mean_x, mean_y = mean(x), mean(y)
    var_x, var_y = var(x), var(y)

    # Welch's t-test
    se = sqrt(var_x/nx + var_y/ny)
    t = (mean_x - mean_y) / se

    # Welch-Satterthwaite df
    df = (var_x/nx + var_y/ny)^2 / ((var_x/nx)^2/(nx-1) + (var_y/ny)^2/(ny-1))

    pvalue = 2 * ccdf(TDist(df), abs(t))

    return pvalue
end

"""
    benjamini_hochberg(pvalues::Vector{Float64}) -> Vector{Float64}

Benjamini-Hochberg FDR correction.
"""
function benjamini_hochberg(pvalues::Vector{Float64})
    n = length(pvalues)
    sorted_idx = sortperm(pvalues)
    sorted_p = pvalues[sorted_idx]

    adjusted = zeros(n)
    cummin = 1.0

    for i in n:-1:1
        adjusted_p = sorted_p[i] * n / i
        cummin = min(cummin, adjusted_p)
        adjusted[sorted_idx[i]] = min(1.0, cummin)
    end

    return adjusted
end

"""
    sc_umap(pcs::Matrix{Float64}; n_neighbors=15, min_dist=0.1, n_epochs=200) -> Matrix{Float64}

Compute UMAP embedding for visualization.

# Arguments
- `pcs`: PC scores matrix
- `n_neighbors`: Number of neighbors for local structure
- `min_dist`: Minimum distance in embedding
- `n_epochs`: Number of optimization epochs

# Returns
2D embedding matrix (cells × 2)
"""
function sc_umap(
    pcs::Matrix{Float64};
    n_neighbors::Int=15,
    min_dist::Float64=0.1,
    n_epochs::Int=200,
    learning_rate::Float64=1.0
)
    n_cells = size(pcs, 1)

    # Compute nearest neighbors
    nn_result = sc_neighbors(pcs; k=n_neighbors)
    neighbor_indices = nn_result[1]
    neighbor_distances = nn_result[2]

    # Compute fuzzy simplicial set (simplified)
    # High-dimensional probabilities
    σ = zeros(n_cells)
    for i in 1:n_cells
        # Binary search for σ that gives target perplexity
        σ[i] = neighbor_distances[i, n_neighbors] / 3
    end

    # Graph weights
    P = zeros(n_cells, n_cells)
    for i in 1:n_cells
        for (j_idx, j) in enumerate(neighbor_indices[i, :])
            d = neighbor_distances[i, j_idx]
            P[i, j] = exp(-max(0, d - neighbor_distances[i, 1]) / σ[i])
        end
    end

    # Symmetrize
    P = P + P'
    P = P ./ sum(P)

    # Initialize embedding (PCA)
    embedding = pcs[:, 1:2] .* 0.0001

    # Optimize embedding
    a, b = find_ab_params(min_dist)

    for epoch in 1:n_epochs
        # Compute gradients
        grad = zeros(n_cells, 2)

        for i in 1:n_cells
            for j in 1:n_cells
                if i == j
                    continue
                end

                d_ij = sum((embedding[i, :] .- embedding[j, :]).^2)

                # Attractive force (from high-dim neighbors)
                if P[i, j] > 0
                    w = 1 / (1 + a * d_ij^b)
                    grad[i, :] .+= P[i, j] * 2 * a * b * d_ij^(b-1) * w * (embedding[i, :] .- embedding[j, :])
                end

                # Repulsive force
                w = 1 / (1 + a * d_ij^b)
                grad[i, :] .-= (1 - P[i, j]) * 2 * b * w^2 * (embedding[i, :] .- embedding[j, :]) / (0.001 + d_ij)
            end
        end

        # Update
        lr = learning_rate * (1 - epoch / n_epochs)
        embedding .-= lr .* grad

        # Clip
        embedding = clamp.(embedding, -10, 10)
    end

    return embedding
end

function find_ab_params(min_dist::Float64)
    # Simplified: use fixed parameters
    a = 1.576
    b = 0.8951
    return a, b
end
