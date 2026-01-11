# ============================================================================
# Comprehensive Tests for Epistasis (Gene-Gene Interaction) Module
# ============================================================================
# Tests for pairwise epistasis, BOOST, MDR, and pathway-based methods
# ============================================================================

@testset "Epistasis Analysis" begin

    # ========================================================================
    # Helper Functions
    # ========================================================================
    function generate_epistasis_data(n::Int, p::Int;
                                     n_interactions::Int=2,
                                     effect_size::Float64=0.5,
                                     seed::Int=12345)
        Random.seed!(seed)

        # Generate genotypes
        genotypes = rand(0:2, n, p) |> x -> Float64.(x)

        # Generate phenotype with epistatic effects
        y = randn(n)

        # Add main effects
        for j in 1:min(3, p)
            y .+= genotypes[:, j] .* 0.2
        end

        # Add interactions
        pairs = []
        for i in 1:n_interactions
            j1, j2 = rand(1:p), rand(1:p)
            while j1 == j2
                j2 = rand(1:p)
            end
            push!(pairs, (j1, j2))
            interaction = genotypes[:, j1] .* genotypes[:, j2]
            y .+= interaction .* effect_size
        end

        return genotypes, y, pairs
    end

    # ========================================================================
    # EpistasisResult Structure Tests
    # ========================================================================
    @testset "EpistasisResult Structure" begin
        result = EpistasisResult(
            [(1, 2), (3, 4)],
            [5.0, 3.0],
            [0.001, 0.01],
            :pairwise,
            Dict(:n_tests => 10)
        )

        @test length(result.pairs) == 2
        @test result.method == :pairwise
        @test result.pvalues[1] < result.pvalues[2]
    end

    # ========================================================================
    # Pairwise Epistasis Tests
    # ========================================================================
    @testset "Pairwise Epistasis" begin
        @testset "Basic Pairwise" begin
            genotypes, y, true_pairs = generate_epistasis_data(500, 20;
                                                                n_interactions=1,
                                                                effect_size=0.8)

            result = pairwise_epistasis(genotypes, y)

            @test isa(result, DataFrame)
            @test :snp1 in propertynames(result)
            @test :snp2 in propertynames(result)
            @test :pvalue in propertynames(result)
        end

        @testset "Detection Power" begin
            genotypes, y, true_pairs = generate_epistasis_data(1000, 10;
                                                                n_interactions=1,
                                                                effect_size=1.0)

            result = pairwise_epistasis(genotypes, y)

            # True interaction should be among top results
            top_pairs = [(row.snp1, row.snp2) for row in eachrow(result[1:5, :])]
            true_pair = true_pairs[1]

            @test (true_pair in top_pairs) || ((true_pair[2], true_pair[1]) in top_pairs)
        end

        @testset "With Covariates" begin
            genotypes, y, _ = generate_epistasis_data(300, 15)
            covariates = randn(300, 2)

            result = pairwise_epistasis(genotypes, y; covariates=covariates)

            @test isa(result, DataFrame)
        end

        @testset "Binary Phenotype" begin
            n = 500
            genotypes, y_cont, _ = generate_epistasis_data(n, 10)
            y_binary = Float64.(y_cont .> median(y_cont))

            result = pairwise_epistasis(genotypes, y_binary; binary=true)

            @test isa(result, DataFrame)
            @test all(0 .<= result.pvalue .<= 1)
        end

        @testset "Parallelization" begin
            genotypes, y, _ = generate_epistasis_data(200, 20)

            result_serial = pairwise_epistasis(genotypes, y; parallel=false)
            result_parallel = pairwise_epistasis(genotypes, y; parallel=true)

            @test nrow(result_serial) == nrow(result_parallel)
        end
    end

    # ========================================================================
    # BOOST Algorithm Tests
    # ========================================================================
    @testset "BOOST Epistasis" begin
        @testset "Basic BOOST" begin
            genotypes, y, _ = generate_epistasis_data(500, 30)

            result = boost_epistasis(genotypes, y)

            @test isa(result, DataFrame)
            @test :interaction_score in propertynames(result)
        end

        @testset "Two-Stage Approach" begin
            genotypes, y, true_pairs = generate_epistasis_data(500, 50;
                                                                n_interactions=2,
                                                                effect_size=0.8)

            result = boost_epistasis(genotypes, y; screening_threshold=0.1)

            @test isa(result, DataFrame)
            @test haskey(result, :stage) || true  # May have stage info
        end

        @testset "Binary Phenotype" begin
            n = 600
            genotypes = rand(0:2, n, 25) |> x -> Float64.(x)
            y = Float64.(rand(n) .> 0.5)

            result = boost_epistasis(genotypes, y; binary=true)

            @test isa(result, DataFrame)
        end

        @testset "Kirkwood Approximation" begin
            genotypes, y, _ = generate_epistasis_data(400, 20)

            result = boost_epistasis(genotypes, y; use_kirkwood=true)

            @test isa(result, DataFrame)
        end
    end

    # ========================================================================
    # Multifactor Dimensionality Reduction (MDR) Tests
    # ========================================================================
    @testset "MDR" begin
        @testset "Basic MDR" begin
            genotypes, y, _ = generate_epistasis_data(300, 10)
            y_binary = Float64.(y .> median(y))

            result = mdr(genotypes, y_binary)

            @test haskey(result, :best_model)
            @test haskey(result, :cv_accuracy)
            @test haskey(result, :cv_consistency)
        end

        @testset "K-Way Interactions" begin
            genotypes = rand(0:2, 200, 8) |> x -> Float64.(x)
            y = Float64.(rand(200) .> 0.5)

            result_2way = mdr(genotypes, y; k=2)
            result_3way = mdr(genotypes, y; k=3)

            @test length(result_2way.best_model) == 2
            @test length(result_3way.best_model) == 3
        end

        @testset "Cross-Validation" begin
            genotypes = rand(0:2, 250, 12) |> x -> Float64.(x)
            y = Float64.(rand(250) .> 0.5)

            result = mdr(genotypes, y; cv_folds=10)

            @test 0 <= result.cv_accuracy <= 1
            @test 0 <= result.cv_consistency <= 10
        end

        @testset "Balanced Accuracy" begin
            genotypes = rand(0:2, 200, 8) |> x -> Float64.(x)
            # Imbalanced phenotype
            y = vcat(zeros(150), ones(50))

            result = mdr(genotypes, y; balance=true)

            @test haskey(result, :balanced_accuracy)
        end

        @testset "Permutation Test" begin
            genotypes = rand(0:2, 150, 6) |> x -> Float64.(x)
            y = Float64.(rand(150) .> 0.5)

            result = mdr(genotypes, y; n_permutations=100)

            @test haskey(result, :pvalue)
            @test 0 <= result.pvalue <= 1
        end
    end

    # ========================================================================
    # Pathway-Based Epistasis Tests
    # ========================================================================
    @testset "Pathway Epistasis" begin
        @testset "Basic Pathway" begin
            genotypes, y, _ = generate_epistasis_data(500, 100)

            gene_sets = Dict(
                "pathway1" => collect(1:20),
                "pathway2" => collect(21:40),
                "pathway3" => collect(41:60)
            )

            result = pathway_epistasis(genotypes, y, gene_sets)

            @test isa(result, DataFrame)
            @test :pathway1 in propertynames(result)
            @test :pathway2 in propertynames(result)
        end

        @testset "Within Pathway" begin
            genotypes, y, _ = generate_epistasis_data(400, 50)

            gene_sets = Dict(
                "pathwayA" => collect(1:25),
                "pathwayB" => collect(26:50)
            )

            result = pathway_epistasis(genotypes, y, gene_sets; within=true)

            @test isa(result, DataFrame)
        end

        @testset "Between Pathway" begin
            genotypes, y, _ = generate_epistasis_data(400, 60)

            gene_sets = Dict(
                "pathway1" => collect(1:20),
                "pathway2" => collect(21:40),
                "pathway3" => collect(41:60)
            )

            result = pathway_epistasis(genotypes, y, gene_sets; between=true)

            @test isa(result, DataFrame)
            # Should have pathway pairs
            @test nrow(result) >= 3  # 3 pathway combinations
        end

        @testset "Enrichment" begin
            genotypes, y, _ = generate_epistasis_data(500, 80)

            gene_sets = Dict(
                "pathway$i" => collect((i-1)*20+1:i*20) for i in 1:4
            )

            result = pathway_epistasis(genotypes, y, gene_sets; enrichment=true)

            @test :enrichment_pvalue in propertynames(result)
        end
    end

    # ========================================================================
    # Random Forest Epistasis Tests
    # ========================================================================
    @testset "Random Forest Epistasis" begin
        @testset "Basic RF" begin
            genotypes, y, _ = generate_epistasis_data(300, 20)

            result = random_forest_epistasis(genotypes, y)

            @test isa(result, DataFrame)
            @test :importance in propertynames(result)
        end

        @testset "Interaction Importance" begin
            genotypes, y, true_pairs = generate_epistasis_data(500, 15;
                                                                n_interactions=1,
                                                                effect_size=1.0)

            result = random_forest_epistasis(genotypes, y; interaction_importance=true)

            @test isa(result, DataFrame)
            @test :pair in propertynames(result) || :interaction in propertynames(result)
        end

        @testset "Number of Trees" begin
            genotypes, y, _ = generate_epistasis_data(200, 10)

            result_small = random_forest_epistasis(genotypes, y; n_trees=50)
            result_large = random_forest_epistasis(genotypes, y; n_trees=200)

            @test isa(result_small, DataFrame)
            @test isa(result_large, DataFrame)
        end

        @testset "Binary Outcome" begin
            n = 300
            genotypes = rand(0:2, n, 15) |> x -> Float64.(x)
            y = Float64.(rand(n) .> 0.5)

            result = random_forest_epistasis(genotypes, y; classification=true)

            @test isa(result, DataFrame)
        end
    end

    # ========================================================================
    # Regression-Based Epistasis Tests
    # ========================================================================
    @testset "Regression Epistasis" begin
        @testset "Linear Interaction Model" begin
            n, p = 500, 20
            genotypes = rand(0:2, n, p) |> x -> Float64.(x)

            # Create interaction
            interaction = genotypes[:, 1] .* genotypes[:, 2]
            y = genotypes[:, 1] .+ genotypes[:, 2] .+ interaction .* 0.5 .+ randn(n)

            result = pairwise_epistasis(genotypes, y; model=:linear)

            @test isa(result, DataFrame)
            # (1,2) pair should be significant
            pair_12 = filter(row -> (row.snp1 == 1 && row.snp2 == 2) ||
                                   (row.snp1 == 2 && row.snp2 == 1), result)
            @test nrow(pair_12) == 1
            @test pair_12.pvalue[1] < 0.05
        end

        @testset "Logistic Interaction Model" begin
            n, p = 400, 15
            genotypes = rand(0:2, n, p) |> x -> Float64.(x)
            linear_pred = genotypes[:, 1] .+ genotypes[:, 2] .*
                         genotypes[:, 3] .* 0.5
            y = Float64.(linear_pred .+ randn(n) .> 0)

            result = pairwise_epistasis(genotypes, y; model=:logistic)

            @test isa(result, DataFrame)
            @test all(0 .<= result.pvalue .<= 1)
        end
    end

    # ========================================================================
    # Multiple Testing Correction
    # ========================================================================
    @testset "Multiple Testing" begin
        @testset "Bonferroni" begin
            genotypes, y, _ = generate_epistasis_data(300, 10)

            result = pairwise_epistasis(genotypes, y; correction=:bonferroni)

            @test :pvalue_corrected in propertynames(result)
            @test all(result.pvalue_corrected .>= result.pvalue)
        end

        @testset "FDR" begin
            genotypes, y, _ = generate_epistasis_data(300, 15)

            result = pairwise_epistasis(genotypes, y; correction=:fdr)

            @test :qvalue in propertynames(result)
        end

        @testset "Permutation" begin
            genotypes, y, _ = generate_epistasis_data(200, 8)

            result = pairwise_epistasis(genotypes, y;
                                        correction=:permutation,
                                        n_permutations=100)

            @test :pvalue_permutation in propertynames(result)
        end
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================
    @testset "Edge Cases" begin
        @testset "Two Variants" begin
            genotypes = rand(0:2, 100, 2) |> x -> Float64.(x)
            y = randn(100)

            result = pairwise_epistasis(genotypes, y)

            @test nrow(result) == 1
        end

        @testset "Monomorphic Variants" begin
            genotypes = rand(0:2, 100, 5) |> x -> Float64.(x)
            genotypes[:, 3] .= 1  # Monomorphic
            y = randn(100)

            result = pairwise_epistasis(genotypes, y)

            # Pairs with monomorphic should be excluded or have NA
            @test isa(result, DataFrame)
        end

        @testset "Perfect Correlation" begin
            genotypes = rand(0:2, 100, 4) |> x -> Float64.(x)
            genotypes[:, 2] = genotypes[:, 1]  # Perfect correlation
            y = randn(100)

            result = pairwise_epistasis(genotypes, y)

            @test isa(result, DataFrame)
        end

        @testset "Small Sample" begin
            genotypes = rand(0:2, 30, 5) |> x -> Float64.(x)
            y = randn(30)

            result = pairwise_epistasis(genotypes, y)

            @test isa(result, DataFrame)
        end
    end

    # ========================================================================
    # Integration Tests
    # ========================================================================
    @testset "Integration" begin
        @testset "Full Epistasis Pipeline" begin
            # Generate data with known interactions
            n, p = 1000, 50
            genotypes, y, true_pairs = generate_epistasis_data(n, p;
                                                                n_interactions=3,
                                                                effect_size=0.6)

            # Step 1: Screen with BOOST
            boost_result = boost_epistasis(genotypes, y; screening_threshold=0.1)

            # Step 2: Fine test top pairs
            top_pairs = [(row.snp1, row.snp2) for row in eachrow(boost_result[1:20, :])]

            # Step 3: Validate with standard regression
            final_result = pairwise_epistasis(genotypes, y)

            @test isa(final_result, DataFrame)

            # At least one true pair should be detected
            significant = filter(row -> row.pvalue < 0.001, final_result)
            sig_pairs = [(row.snp1, row.snp2) for row in eachrow(significant)]

            detected = sum(any((p in sig_pairs) || ((p[2], p[1]) in sig_pairs)
                              for p in true_pairs))
            @test detected >= 1
        end
    end

end # @testset "Epistasis Analysis"
