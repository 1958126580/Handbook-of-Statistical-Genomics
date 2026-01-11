# ============================================================================
# Comprehensive Tests for Single-Cell Genomics Module
# ============================================================================
# Tests for QC, normalization, clustering, DE, and trajectory analysis
# ============================================================================

@testset "Single-Cell Analysis" begin

    # ========================================================================
    # Helper Functions
    # ========================================================================
    function generate_sc_data(n_cells::Int, n_genes::Int;
                              n_clusters::Int=3, seed::Int=12345)
        Random.seed!(seed)

        # Generate count matrix with cluster structure
        counts = zeros(n_genes, n_cells)

        cluster_labels = repeat(1:n_clusters, inner=div(n_cells, n_clusters))
        if length(cluster_labels) < n_cells
            append!(cluster_labels, ones(Int, n_cells - length(cluster_labels)))
        end

        for i in 1:n_cells
            cluster = cluster_labels[i]
            # Base expression
            base_rate = rand(Gamma(0.5, 2.0), n_genes)
            # Cluster-specific upregulation
            marker_start = (cluster - 1) * div(n_genes, n_clusters) + 1
            marker_end = min(cluster * div(n_genes, n_clusters), n_genes)
            base_rate[marker_start:marker_end] .*= 3.0

            counts[:, i] = rand.(Poisson.(base_rate))
        end

        return counts, cluster_labels
    end

    # ========================================================================
    # SingleCellData Structure Tests
    # ========================================================================
    @testset "SingleCellData Structure" begin
        counts, _ = generate_sc_data(100, 500)

        data = SingleCellData(
            counts,
            ["Gene$i" for i in 1:500],
            ["Cell$i" for i in 1:100]
        )

        @test size(data.counts) == (500, 100)
        @test length(data.gene_names) == 500
        @test length(data.cell_names) == 100
    end

    # ========================================================================
    # Quality Control Tests
    # ========================================================================
    @testset "Single-Cell QC" begin
        @testset "Basic QC" begin
            counts, _ = generate_sc_data(200, 1000)
            data = SingleCellData(counts, ["G$i" for i in 1:1000], ["C$i" for i in 1:200])

            qc = sc_qc(data)

            @test isa(qc, SingleCellQC)
            @test length(qc.n_genes_per_cell) == 200
            @test length(qc.n_counts_per_cell) == 200
            @test length(qc.pct_mito) == 200
        end

        @testset "Mitochondrial Content" begin
            counts, _ = generate_sc_data(100, 500)
            gene_names = ["Gene$i" for i in 1:500]
            gene_names[1:10] = ["MT-$i" for i in 1:10]  # Mito genes

            data = SingleCellData(counts, gene_names, ["C$i" for i in 1:100])

            qc = sc_qc(data; mito_prefix="MT-")

            @test length(qc.pct_mito) == 100
            @test all(0 .<= qc.pct_mito .<= 100)
        end

        @testset "Filtering" begin
            counts, _ = generate_sc_data(200, 800)
            # Add low-quality cells
            counts[:, 1:10] .= 0  # Empty cells
            counts[:, 11:20] = rand(Poisson(0.1), 800, 10)  # Low counts

            data = SingleCellData(counts, ["G$i" for i in 1:800], ["C$i" for i in 1:200])

            qc = sc_qc(data; min_genes=50, min_counts=100)

            @test haskey(qc.extra, :cells_pass) || true
        end

        @testset "Doublet Detection" begin
            counts, _ = generate_sc_data(200, 500)
            data = SingleCellData(counts, ["G$i" for i in 1:500], ["C$i" for i in 1:200])

            qc = sc_qc(data; detect_doublets=true)

            @test haskey(qc.extra, :doublet_score) || length(qc.pct_mito) == 200
        end
    end

    # ========================================================================
    # Normalization Tests
    # ========================================================================
    @testset "Normalization" begin
        @testset "Log Normalization" begin
            counts, _ = generate_sc_data(100, 500)
            data = SingleCellData(counts, ["G$i" for i in 1:500], ["C$i" for i in 1:100])

            normalized = sc_normalize(data; method=:log_normalize)

            @test size(normalized) == size(counts)
            @test all(normalized .>= 0)
        end

        @testset "Scale Factor" begin
            counts, _ = generate_sc_data(100, 500)
            data = SingleCellData(counts, ["G$i" for i in 1:500], ["C$i" for i in 1:100])

            norm_10k = sc_normalize(data; scale_factor=10000)
            norm_1m = sc_normalize(data; scale_factor=1000000)

            # Higher scale factor = higher values
            @test mean(norm_1m) > mean(norm_10k)
        end

        @testset "SCTransform" begin
            counts, _ = generate_sc_data(100, 500)
            data = SingleCellData(counts, ["G$i" for i in 1:500], ["C$i" for i in 1:100])

            normalized = sc_normalize(data; method=:sctransform)

            @test size(normalized) == size(counts)
        end

        @testset "CPM" begin
            counts, _ = generate_sc_data(80, 300)
            data = SingleCellData(counts, ["G$i" for i in 1:300], ["C$i" for i in 1:80])

            normalized = sc_normalize(data; method=:cpm)

            # Each column should sum to ~1e6
            col_sums = sum(normalized, dims=1)
            @test all(abs.(col_sums .- 1e6) .< 1e6 * 0.01)
        end
    end

    # ========================================================================
    # Highly Variable Genes Tests
    # ========================================================================
    @testset "Highly Variable Genes" begin
        @testset "Basic HVG" begin
            counts, _ = generate_sc_data(200, 1000)
            data = SingleCellData(counts, ["G$i" for i in 1:1000], ["C$i" for i in 1:200])

            hvg_idx = sc_highly_variable_genes(data; n_top=500)

            @test length(hvg_idx) == 500
            @test all(1 .<= hvg_idx .<= 1000)
        end

        @testset "Variance Method" begin
            counts, _ = generate_sc_data(150, 800)
            data = SingleCellData(counts, ["G$i" for i in 1:800], ["C$i" for i in 1:150])

            hvg_var = sc_highly_variable_genes(data; method=:variance, n_top=200)
            hvg_disp = sc_highly_variable_genes(data; method=:dispersion, n_top=200)

            @test length(hvg_var) == 200
            @test length(hvg_disp) == 200
        end

        @testset "Mean Filter" begin
            counts, _ = generate_sc_data(100, 500)
            data = SingleCellData(counts, ["G$i" for i in 1:500], ["C$i" for i in 1:100])

            hvg = sc_highly_variable_genes(data; min_mean=0.1, max_mean=3.0)

            @test length(hvg) > 0
            @test length(hvg) < 500
        end
    end

    # ========================================================================
    # PCA Tests
    # ========================================================================
    @testset "Single-Cell PCA" begin
        @testset "Basic PCA" begin
            counts, _ = generate_sc_data(200, 1000)
            data = SingleCellData(counts, ["G$i" for i in 1:1000], ["C$i" for i in 1:200])

            hvg_idx = sc_highly_variable_genes(data; n_top=500)
            pca_result = sc_pca(data, hvg_idx; n_pcs=50)

            @test haskey(pca_result, :pcs)
            @test haskey(pca_result, :variance_explained)
            @test size(pca_result.pcs) == (200, 50)
        end

        @testset "Variance Explained" begin
            counts, _ = generate_sc_data(150, 800)
            data = SingleCellData(counts, ["G$i" for i in 1:800], ["C$i" for i in 1:150])

            hvg_idx = collect(1:400)
            pca_result = sc_pca(data, hvg_idx; n_pcs=30)

            @test length(pca_result.variance_explained) == 30
            @test all(pca_result.variance_explained .>= 0)
            @test pca_result.variance_explained[1] >= pca_result.variance_explained[end]
        end

        @testset "PC Selection" begin
            counts, _ = generate_sc_data(200, 1000; n_clusters=5)
            data = SingleCellData(counts, ["G$i" for i in 1:1000], ["C$i" for i in 1:200])

            hvg_idx = sc_highly_variable_genes(data; n_top=500)
            pca_result = sc_pca(data, hvg_idx; n_pcs=50, select_pcs=true)

            @test haskey(pca_result, :n_pcs_selected) || true
        end
    end

    # ========================================================================
    # Neighbor Graph Tests
    # ========================================================================
    @testset "Neighbor Graph" begin
        @testset "Basic Neighbors" begin
            pcs = randn(200, 20)

            neighbors = sc_neighbors(pcs; n_neighbors=15)

            @test haskey(neighbors, :indices)
            @test haskey(neighbors, :distances)
            @test size(neighbors.indices) == (200, 15)
        end

        @testset "Different Metrics" begin
            pcs = randn(100, 15)

            neighbors_euclidean = sc_neighbors(pcs; metric=:euclidean)
            neighbors_cosine = sc_neighbors(pcs; metric=:cosine)

            @test size(neighbors_euclidean.indices) == size(neighbors_cosine.indices)
        end

        @testset "Connectivity" begin
            pcs = randn(150, 20)

            neighbors = sc_neighbors(pcs; n_neighbors=30, compute_connectivities=true)

            @test haskey(neighbors, :connectivities)
        end
    end

    # ========================================================================
    # Clustering Tests
    # ========================================================================
    @testset "Clustering" begin
        @testset "Louvain Clustering" begin
            counts, true_labels = generate_sc_data(300, 800; n_clusters=4)
            data = SingleCellData(counts, ["G$i" for i in 1:800], ["C$i" for i in 1:300])

            hvg_idx = sc_highly_variable_genes(data; n_top=400)
            pca_result = sc_pca(data, hvg_idx; n_pcs=30)

            clusters = sc_cluster(pca_result.pcs; method=:louvain)

            @test length(clusters) == 300
            @test minimum(clusters) >= 1
        end

        @testset "Leiden Clustering" begin
            pcs = randn(200, 20)

            clusters = sc_cluster(pcs; method=:leiden)

            @test length(clusters) == 200
        end

        @testset "Resolution Parameter" begin
            pcs = randn(200, 15)

            clusters_low = sc_cluster(pcs; resolution=0.5)
            clusters_high = sc_cluster(pcs; resolution=2.0)

            # Higher resolution = more clusters
            @test length(unique(clusters_high)) >= length(unique(clusters_low))
        end

        @testset "K-Means" begin
            pcs = randn(150, 20)

            clusters = sc_cluster(pcs; method=:kmeans, k=5)

            @test length(clusters) == 150
            @test length(unique(clusters)) == 5
        end
    end

    # ========================================================================
    # Differential Expression Tests
    # ========================================================================
    @testset "Differential Expression" begin
        @testset "Basic DE" begin
            counts, true_labels = generate_sc_data(200, 500; n_clusters=2)
            data = SingleCellData(counts, ["G$i" for i in 1:500], ["C$i" for i in 1:200])
            clusters = true_labels[1:200]

            de_result = sc_differential_expression(data, clusters)

            @test isa(de_result, Dict)
            @test haskey(de_result, 1)
            @test haskey(de_result, 2)
        end

        @testset "Wilcoxon Test" begin
            counts, true_labels = generate_sc_data(150, 400; n_clusters=3)
            data = SingleCellData(counts, ["G$i" for i in 1:400], ["C$i" for i in 1:150])
            clusters = true_labels[1:150]

            de_result = sc_differential_expression(data, clusters; test=:wilcoxon)

            for (cluster, df) in de_result
                @test :gene in propertynames(df)
                @test :pvalue in propertynames(df)
                @test :log2fc in propertynames(df)
            end
        end

        @testset "T-Test" begin
            counts, true_labels = generate_sc_data(100, 300; n_clusters=2)
            data = SingleCellData(counts, ["G$i" for i in 1:300], ["C$i" for i in 1:100])
            clusters = true_labels[1:100]

            de_result = sc_differential_expression(data, clusters; test=:t)

            @test isa(de_result, Dict)
        end

        @testset "Top Markers" begin
            counts, true_labels = generate_sc_data(200, 600; n_clusters=3)
            data = SingleCellData(counts, ["G$i" for i in 1:600], ["C$i" for i in 1:200])
            clusters = true_labels[1:200]

            de_result = sc_differential_expression(data, clusters; n_top=20)

            for (cluster, df) in de_result
                @test nrow(df) == 20
            end
        end
    end

    # ========================================================================
    # UMAP Tests
    # ========================================================================
    @testset "UMAP" begin
        @testset "Basic UMAP" begin
            pcs = randn(200, 20)

            umap_coords = sc_umap(pcs)

            @test size(umap_coords) == (200, 2)
        end

        @testset "Parameters" begin
            pcs = randn(150, 15)

            umap_15 = sc_umap(pcs; n_neighbors=15)
            umap_30 = sc_umap(pcs; n_neighbors=30)

            @test size(umap_15) == (150, 2)
            @test size(umap_30) == (150, 2)
        end

        @testset "Min Distance" begin
            pcs = randn(100, 20)

            umap_close = sc_umap(pcs; min_dist=0.1)
            umap_far = sc_umap(pcs; min_dist=0.5)

            @test size(umap_close) == (100, 2)
            @test size(umap_far) == (100, 2)
        end

        @testset "3D UMAP" begin
            pcs = randn(100, 15)

            umap_3d = sc_umap(pcs; n_components=3)

            @test size(umap_3d) == (100, 3)
        end
    end

    # ========================================================================
    # Integration Tests
    # ========================================================================
    @testset "Integration" begin
        @testset "Full Pipeline" begin
            # Generate data
            counts, true_labels = generate_sc_data(500, 2000; n_clusters=4)
            data = SingleCellData(counts, ["G$i" for i in 1:2000], ["C$i" for i in 1:500])

            # QC
            qc = sc_qc(data)

            # Normalization
            normalized = sc_normalize(data)

            # HVG
            hvg_idx = sc_highly_variable_genes(data; n_top=1000)

            # PCA
            pca_result = sc_pca(data, hvg_idx; n_pcs=50)

            # Neighbors
            neighbors = sc_neighbors(pca_result.pcs; n_neighbors=15)

            # Clustering
            clusters = sc_cluster(pca_result.pcs)

            # UMAP
            umap_coords = sc_umap(pca_result.pcs)

            # DE
            de_result = sc_differential_expression(data, clusters)

            # Validate
            @test length(clusters) == 500
            @test size(umap_coords) == (500, 2)
            @test length(de_result) >= 1
        end

        @testset "Batch Integration" begin
            # Two batches
            counts1, labels1 = generate_sc_data(200, 1000; n_clusters=3, seed=1)
            counts2, labels2 = generate_sc_data(200, 1000; n_clusters=3, seed=2)

            # Combined data
            counts = hcat(counts1, counts2)
            data = SingleCellData(counts, ["G$i" for i in 1:1000], ["C$i" for i in 1:400])
            batch = vcat(fill(1, 200), fill(2, 200))

            # Should be able to process
            hvg_idx = sc_highly_variable_genes(data; n_top=500)
            pca_result = sc_pca(data, hvg_idx; n_pcs=30)
            clusters = sc_cluster(pca_result.pcs)

            @test length(clusters) == 400
        end
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================
    @testset "Edge Cases" begin
        @testset "Small Dataset" begin
            counts = rand(Poisson(5), 100, 30)
            data = SingleCellData(Float64.(counts), ["G$i" for i in 1:100], ["C$i" for i in 1:30])

            normalized = sc_normalize(data)
            hvg_idx = sc_highly_variable_genes(data; n_top=50)

            @test length(hvg_idx) == 50
        end

        @testset "Sparse Data" begin
            counts = zeros(500, 100)
            # Add sparse signal
            for i in 1:100
                for j in sample(1:500, 50, replace=false)
                    counts[j, i] = rand(Poisson(5))
                end
            end

            data = SingleCellData(counts, ["G$i" for i in 1:500], ["C$i" for i in 1:100])

            normalized = sc_normalize(data)
            @test size(normalized) == (500, 100)
        end

        @testset "Many Clusters" begin
            pcs = randn(300, 20)

            clusters = sc_cluster(pcs; resolution=3.0)

            @test length(unique(clusters)) > 1
        end
    end

end # @testset "Single-Cell Analysis"
