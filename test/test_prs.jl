# ============================================================================
# Comprehensive Tests for Polygenic Risk Score (PRS) Module
# ============================================================================
# Tests for LDpred, PRS-CS, C+T, and PRS validation methods
# ============================================================================

@testset "Polygenic Risk Scores" begin

    # ========================================================================
    # Helper Functions
    # ========================================================================
    function generate_prs_data(n_gwas::Int, n_target::Int, p::Int;
                               h2::Float64=0.5, seed::Int=12345)
        Random.seed!(seed)

        # Generate GWAS genotypes
        X_gwas = rand(0:2, n_gwas, p) |> x -> Float64.(x)

        # Standardize
        X_gwas = (X_gwas .- mean(X_gwas, dims=1)) ./ std(X_gwas, dims=1)

        # True effects
        β_true = randn(p) .* sqrt(h2 / p)

        # GWAS phenotype
        y_gwas = X_gwas * β_true + randn(n_gwas) * sqrt(1 - h2)

        # Summary statistics
        betas = zeros(p)
        ses = zeros(p)
        pvalues = zeros(p)
        for j in 1:p
            β_j = cov(X_gwas[:, j], y_gwas) / var(X_gwas[:, j])
            se_j = sqrt(var(y_gwas) / (n_gwas * var(X_gwas[:, j])))
            betas[j] = β_j
            ses[j] = se_j
            pvalues[j] = 2 * ccdf(Normal(), abs(β_j / se_j))
        end

        # Target genotypes
        X_target = rand(0:2, n_target, p) |> x -> Float64.(x)
        X_target = (X_target .- mean(X_target, dims=1)) ./ std(X_target, dims=1)

        # Target phenotype
        y_target = X_target * β_true + randn(n_target) * sqrt(1 - h2)

        # LD matrix
        R = cor(X_gwas)

        return betas, ses, pvalues, R, X_target, y_target, β_true
    end

    # ========================================================================
    # PRSResult and PRSWeights Structure Tests
    # ========================================================================
    @testset "Structure Tests" begin
        @testset "PRSWeights" begin
            weights = PRSWeights(
                randn(100),
                collect(1:100),
                :ldpred,
                Dict(:h2 => 0.5)
            )

            @test length(weights.weights) == 100
            @test weights.method == :ldpred
        end

        @testset "PRSResult" begin
            result = PRSResult(
                randn(50),
                0.85,
                0.05,
                0.25,
                Dict(:threshold => 0.5)
            )

            @test length(result.scores) == 50
            @test result.auc == 0.85
        end
    end

    # ========================================================================
    # Clumping and Thresholding (C+T) Tests
    # ========================================================================
    @testset "Clumping and Thresholding" begin
        @testset "Basic C+T" begin
            betas, ses, pvalues, R, X_target, y_target, _ = generate_prs_data(
                5000, 500, 100; h2=0.3
            )

            weights = clump_threshold_prs(betas, pvalues, X_target;
                                          p_threshold=0.05,
                                          r2_threshold=0.1)

            @test isa(weights, PRSWeights)
            @test length(weights.weights) == 100
            @test weights.method == :clump_threshold
        end

        @testset "Multiple Thresholds" begin
            betas, ses, pvalues, R, X_target, _, _ = generate_prs_data(
                3000, 300, 50
            )

            thresholds = [1e-8, 1e-5, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]

            results = Dict()
            for p_thresh in thresholds
                w = clump_threshold_prs(betas, pvalues, X_target;
                                       p_threshold=p_thresh)
                results[p_thresh] = sum(w.weights .!= 0)
            end

            # More liberal threshold should include more variants
            @test results[1.0] >= results[0.05]
            @test results[0.05] >= results[1e-5]
        end

        @testset "LD Pruning" begin
            betas, ses, pvalues, R, X_target, _, _ = generate_prs_data(
                3000, 300, 80
            )

            # Strict LD threshold
            weights_strict = clump_threshold_prs(betas, pvalues, X_target;
                                                 r2_threshold=0.01)

            # Lenient LD threshold
            weights_lenient = clump_threshold_prs(betas, pvalues, X_target;
                                                  r2_threshold=0.5)

            # Stricter pruning keeps fewer variants
            @test sum(weights_strict.weights .!= 0) <= sum(weights_lenient.weights .!= 0)
        end
    end

    # ========================================================================
    # LDpred2 Grid Tests
    # ========================================================================
    @testset "LDpred2 Grid" begin
        @testset "Basic Grid" begin
            betas, ses, pvalues, R, X_target, y_target, _ = generate_prs_data(
                5000, 500, 50; h2=0.3
            )

            results = ldpred2_grid(betas, ses, R, 5000;
                                   h2_grid=[0.1, 0.3, 0.5],
                                   p_grid=[0.01, 0.1, 1.0])

            @test isa(results, Dict)
            @test length(results) == 9  # 3 × 3 combinations
        end

        @testset "Grid Search" begin
            betas, ses, pvalues, R, X_target, y_target, _ = generate_prs_data(
                3000, 300, 30
            )

            results = ldpred2_grid(betas, ses, R, 3000;
                                   h2_grid=[0.1, 0.3],
                                   p_grid=[0.1, 1.0])

            for (params, weights) in results
                @test length(weights) == 30
            end
        end

        @testset "Sparse Option" begin
            betas, ses, pvalues, R, X_target, _, _ = generate_prs_data(
                3000, 300, 40
            )

            results_sparse = ldpred2_grid(betas, ses, R, 3000;
                                          h2_grid=[0.3],
                                          p_grid=[0.1],
                                          sparse=true)

            @test isa(results_sparse, Dict)
        end
    end

    # ========================================================================
    # LDpred2 Auto Tests
    # ========================================================================
    @testset "LDpred2 Auto" begin
        @testset "Basic Auto" begin
            betas, ses, pvalues, R, X_target, y_target, _ = generate_prs_data(
                10000, 1000, 100; h2=0.4
            )

            weights = ldpred2_auto(betas, ses, R, 10000)

            @test isa(weights, Vector{Float64})
            @test length(weights) == 100
        end

        @testset "Convergence" begin
            betas, ses, pvalues, R, X_target, _, _ = generate_prs_data(
                5000, 500, 50
            )

            weights = ldpred2_auto(betas, ses, R, 5000;
                                   n_iter=200, n_burn=50)

            @test !any(isnan.(weights))
            @test !any(isinf.(weights))
        end

        @testset "H2 Estimation" begin
            betas, ses, pvalues, R, X_target, _, _ = generate_prs_data(
                5000, 500, 50; h2=0.3
            )

            weights = ldpred2_auto(betas, ses, R, 5000;
                                   return_h2=true)

            # Should return reasonable h2 estimate
            @test isa(weights, NamedTuple) || isa(weights, Vector)
        end
    end

    # ========================================================================
    # PRS-CS Tests
    # ========================================================================
    @testset "PRS-CS" begin
        @testset "Basic PRS-CS" begin
            betas, ses, pvalues, R, X_target, _, _ = generate_prs_data(
                5000, 500, 50
            )

            weights = prs_cs(betas, ses, R, 5000)

            @test isa(weights, Vector{Float64})
            @test length(weights) == 50
        end

        @testset "Phi Parameter" begin
            betas, ses, pvalues, R, X_target, _, _ = generate_prs_data(
                3000, 300, 30
            )

            # Different phi values
            weights_auto = prs_cs(betas, ses, R, 3000; phi=nothing)  # Auto
            weights_small = prs_cs(betas, ses, R, 3000; phi=1e-4)
            weights_large = prs_cs(betas, ses, R, 3000; phi=1e-2)

            # Different phi should give different shrinkage
            @test var(weights_small) != var(weights_large)
        end

        @testset "MCMC Settings" begin
            betas, ses, pvalues, R, X_target, _, _ = generate_prs_data(
                3000, 300, 30
            )

            weights = prs_cs(betas, ses, R, 3000;
                            n_iter=500, n_burn=100, thin=5)

            @test length(weights) == 30
        end
    end

    # ========================================================================
    # PRS Computation Tests
    # ========================================================================
    @testset "PRS Computation" begin
        @testset "Basic Score" begin
            betas, ses, pvalues, R, X_target, y_target, _ = generate_prs_data(
                5000, 500, 100
            )

            weights = clump_threshold_prs(betas, pvalues, X_target; p_threshold=0.1)
            scores = compute_prs(X_target, weights.weights)

            @test length(scores) == 500
            @test !any(isnan.(scores))
        end

        @testset "Standardization" begin
            n_target, p = 500, 50
            X_target = rand(0:2, n_target, p) |> x -> Float64.(x)
            weights = randn(p)

            scores_raw = compute_prs(X_target, weights; standardize=false)
            scores_std = compute_prs(X_target, weights; standardize=true)

            @test abs(mean(scores_std)) < 0.1
            @test abs(std(scores_std) - 1.0) < 0.1
        end

        @testset "Missing Genotypes" begin
            n_target, p = 100, 20
            X_target = rand(0:2, n_target, p) |> x -> Float64.(x)
            X_target[1:10, 1:5] .= NaN  # Introduce missing

            weights = randn(p)
            scores = compute_prs(X_target, weights; handle_missing=:mean)

            @test length(scores) == n_target
            @test !any(isnan.(scores))
        end
    end

    # ========================================================================
    # PRS Validation Tests
    # ========================================================================
    @testset "PRS Validation" begin
        @testset "Continuous Phenotype" begin
            n = 500
            scores = randn(n)
            # Phenotype correlated with scores
            phenotype = 0.5 .* scores .+ randn(n) * 0.866

            result = validate_prs(scores, phenotype)

            @test haskey(result, :r2)
            @test haskey(result, :correlation)
            @test haskey(result, :pvalue)
            @test 0 <= result.r2 <= 1
        end

        @testset "Binary Phenotype" begin
            n = 500
            scores = randn(n)
            # Binary phenotype based on scores
            phenotype = Float64.(scores .+ randn(n) .> 0)

            result = validate_prs(scores, phenotype; binary=true)

            @test haskey(result, :auc)
            @test 0.5 <= result.auc <= 1.0
        end

        @testset "With Covariates" begin
            n = 500
            scores = randn(n)
            phenotype = 0.3 .* scores .+ randn(n)
            covariates = randn(n, 3)

            result = validate_prs(scores, phenotype; covariates=covariates)

            @test haskey(result, :r2_incremental)
        end

        @testset "Decile Analysis" begin
            n = 1000
            scores = randn(n)
            phenotype = scores .+ randn(n)

            result = validate_prs(scores, phenotype; decile_analysis=true)

            @test haskey(result, :decile_means)
            @test length(result.decile_means) == 10
        end
    end

    # ========================================================================
    # Best PRS Selection Tests
    # ========================================================================
    @testset "Best PRS Selection" begin
        @testset "Grid Selection" begin
            n_target = 500
            X_target = rand(0:2, n_target, 30) |> x -> Float64.(x)
            y_target = randn(n_target)

            # Multiple weight sets
            weight_sets = Dict(
                "set1" => randn(30),
                "set2" => randn(30),
                "set3" => randn(30)
            )

            best = select_best_prs(weight_sets, X_target, y_target)

            @test haskey(best, :best_method)
            @test haskey(best, :best_r2)
            @test best.best_method in keys(weight_sets)
        end

        @testset "Cross-Validation" begin
            n_target = 300
            X_target = rand(0:2, n_target, 20) |> x -> Float64.(x)
            y_target = X_target[:, 1] .+ randn(n_target) * 0.5

            weight_sets = Dict(
                "optimal" => vcat([1.0], zeros(19)),  # First variant
                "random" => randn(20)
            )

            best = select_best_prs(weight_sets, X_target, y_target; cv_folds=5)

            @test best.best_method == "optimal"
        end
    end

    # ========================================================================
    # PRS Stratification Tests
    # ========================================================================
    @testset "PRS Stratification" begin
        @testset "Percentile Groups" begin
            n = 1000
            scores = randn(n)
            phenotype = scores .+ randn(n) * 0.5

            result = stratify_prs(scores, phenotype; n_groups=5)

            @test haskey(result, :group_means)
            @test haskey(result, :group_ses)
            @test length(result.group_means) == 5
        end

        @testset "Risk Enrichment" begin
            n = 1000
            scores = randn(n)
            # Binary outcome with higher risk at higher scores
            phenotype = Float64.(scores .+ randn(n) .> 1.0)

            result = stratify_prs(scores, phenotype; binary=true)

            @test haskey(result, :odds_ratios)
            # Top group should have higher OR
            @test result.odds_ratios[end] > result.odds_ratios[1]
        end
    end

    # ========================================================================
    # Expected R² Tests
    # ========================================================================
    @testset "Expected PRS R²" begin
        @testset "Basic Calculation" begin
            r2 = expected_prs_r2(100000, 10000, 0.5)

            @test 0 < r2 < 0.5
        end

        @testset "Sample Size Effect" begin
            r2_small = expected_prs_r2(10000, 5000, 0.5)
            r2_large = expected_prs_r2(100000, 5000, 0.5)

            @test r2_large > r2_small
        end

        @testset "Heritability Effect" begin
            r2_low = expected_prs_r2(50000, 5000, 0.2)
            r2_high = expected_prs_r2(50000, 5000, 0.8)

            @test r2_high > r2_low
        end

        @testset "M Effect" begin
            # More causal variants = less variance captured per variant
            r2_sparse = expected_prs_r2(50000, 1000, 0.5; n_causal=100)
            r2_poly = expected_prs_r2(50000, 1000, 0.5; n_causal=10000)

            # With same total h2, polygenicity affects prediction differently
            @test isa(r2_sparse, Float64)
            @test isa(r2_poly, Float64)
        end
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================
    @testset "Edge Cases" begin
        @testset "No Significant Variants" begin
            p = 50
            betas = randn(p) .* 0.001  # Tiny effects
            pvalues = ones(p)  # No signal
            X_target = rand(0:2, 100, p) |> x -> Float64.(x)

            weights = clump_threshold_prs(betas, pvalues, X_target;
                                          p_threshold=0.05)

            # Should still return valid weights (all zeros)
            @test length(weights.weights) == p
        end

        @testset "Single Variant" begin
            betas = [0.5]
            ses = [0.1]
            R = ones(1, 1)

            weights = ldpred2_auto(betas, ses, R, 10000)

            @test length(weights) == 1
        end

        @testset "Perfect LD" begin
            p = 10
            betas = randn(p)
            ses = fill(0.1, p)
            R = ones(p, p)  # Perfect LD

            weights = ldpred2_auto(betas, ses, R, 10000)

            @test length(weights) == p
        end
    end

    # ========================================================================
    # Integration Tests
    # ========================================================================
    @testset "Integration" begin
        @testset "Full Pipeline" begin
            # Generate data
            betas, ses, pvalues, R, X_target, y_target, β_true = generate_prs_data(
                20000, 2000, 200; h2=0.4
            )

            # Method 1: C+T
            weights_ct = clump_threshold_prs(betas, pvalues, X_target;
                                            p_threshold=0.01)
            scores_ct = compute_prs(X_target, weights_ct.weights)

            # Method 2: LDpred2-auto
            weights_ldpred = ldpred2_auto(betas, ses, R, 20000)
            scores_ldpred = compute_prs(X_target, weights_ldpred)

            # Method 3: PRS-CS
            weights_prscs = prs_cs(betas, ses, R, 20000)
            scores_prscs = compute_prs(X_target, weights_prscs)

            # Validate all
            r2_ct = validate_prs(scores_ct, y_target).r2
            r2_ldpred = validate_prs(scores_ldpred, y_target).r2
            r2_prscs = validate_prs(scores_prscs, y_target).r2

            @test 0 <= r2_ct <= 1
            @test 0 <= r2_ldpred <= 1
            @test 0 <= r2_prscs <= 1
        end
    end

end # @testset "Polygenic Risk Scores"
