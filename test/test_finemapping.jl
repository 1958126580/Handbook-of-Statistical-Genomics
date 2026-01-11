# ============================================================================
# Comprehensive Tests for Fine-Mapping Module (SuSiE)
# ============================================================================
# Tests for Sum of Single Effects regression and credible set computation
# ============================================================================

@testset "Fine-Mapping (SuSiE)" begin

    # ========================================================================
    # Helper Functions for Test Data Generation
    # ========================================================================
    function generate_finemapping_data(n::Int, p::Int;
                                       n_causal::Int=3,
                                       h2::Float64=0.3,
                                       seed::Int=12345)
        Random.seed!(seed)

        # Generate genotype matrix
        X = randn(n, p)

        # Standardize
        X = (X .- mean(X, dims=1)) ./ std(X, dims=1)

        # Select causal variants
        causal_idx = sample(1:p, n_causal, replace=false)
        β = zeros(p)
        β[causal_idx] = randn(n_causal)

        # Scale to desired heritability
        var_g = var(X * β)
        var_e = var_g * (1 - h2) / h2
        y = X * β + randn(n) * sqrt(var_e)

        return X, y, causal_idx, β
    end

    # ========================================================================
    # SuSiEResult Structure Tests
    # ========================================================================
    @testset "SuSiEResult Structure" begin
        result = SuSiEResult(
            randn(10, 5),      # alpha
            randn(10, 5),      # mu
            rand(10, 5),       # mu2
            rand(10),          # pip
            [[1, 2], [5, 6]],  # cs
            [0.95, 0.92],      # cs_coverage
            1.0,               # sigma2
            [-100.0, -90.0],   # elbo
            true,              # converged
            50                 # n_iterations
        )

        @test size(result.alpha) == (10, 5)
        @test length(result.pip) == 10
        @test length(result.cs) == 2
        @test result.converged == true
    end

    # ========================================================================
    # Basic SuSiE Tests
    # ========================================================================
    @testset "SuSiE Basic" begin
        @testset "Single Causal Variant" begin
            X, y, causal_idx, _ = generate_finemapping_data(500, 100;
                                                             n_causal=1, h2=0.3)

            result = susie(X, y; L=5)

            @test isa(result, SuSiEResult)
            @test length(result.pip) == 100
            @test all(0 .<= result.pip .<= 1)
            @test sum(result.pip) > 0

            # Causal variant should have high PIP
            @test result.pip[causal_idx[1]] > 0.5
        end

        @testset "Multiple Causal Variants" begin
            X, y, causal_idx, _ = generate_finemapping_data(500, 100;
                                                             n_causal=3, h2=0.4)

            result = susie(X, y; L=10)

            @test isa(result, SuSiEResult)

            # At least some causal variants should be in credible sets
            cs_variants = vcat(result.cs...)
            causal_in_cs = sum(c in cs_variants for c in causal_idx)
            @test causal_in_cs >= 1
        end

        @testset "Convergence" begin
            X, y, _, _ = generate_finemapping_data(300, 50; n_causal=2)

            result = susie(X, y; L=5, max_iter=500, tol=1e-4)

            @test result.converged == true
            @test result.n_iterations < 500
            @test length(result.elbo) > 1
            @test result.elbo[end] >= result.elbo[1]  # ELBO should increase
        end

        @testset "L Parameter" begin
            X, y, _, _ = generate_finemapping_data(200, 30; n_causal=2)

            # Different L values
            result_small = susie(X, y; L=2)
            result_large = susie(X, y; L=10)

            @test length(result_small.cs) <= 2
            @test size(result_small.alpha, 2) == 2
            @test size(result_large.alpha, 2) == 10
        end
    end

    # ========================================================================
    # SuSiE with Summary Statistics
    # ========================================================================
    @testset "SuSiE RSS" begin
        @testset "Basic RSS" begin
            n, p = 10000, 100
            X, y, causal_idx, _ = generate_finemapping_data(n, p; n_causal=2)

            # Compute summary statistics
            z = zeros(p)
            for j in 1:p
                β_j = cov(X[:, j], y) / var(X[:, j])
                se_j = sqrt(var(y - X[:, j] .* β_j) / (n * var(X[:, j])))
                z[j] = β_j / se_j
            end

            # LD matrix
            R = cor(X)

            result = susie_rss(z, R, n; L=5)

            @test isa(result, SuSiEResult)
            @test length(result.pip) == p
            @test all(0 .<= result.pip .<= 1)
        end

        @testset "Comparison with Individual Data" begin
            n, p = 500, 50
            X, y, causal_idx, _ = generate_finemapping_data(n, p; n_causal=1, h2=0.4)

            # Individual-level
            result_ind = susie(X, y; L=5)

            # Summary statistics
            z = zeros(p)
            for j in 1:p
                β_j = cov(X[:, j], y) / var(X[:, j])
                se_j = sqrt(var(y - X[:, j] .* β_j) / (n * var(X[:, j])))
                z[j] = β_j / se_j
            end
            R = cor(X)

            result_rss = susie_rss(z, R, n; L=5)

            # PIPs should be similar
            @test cor(result_ind.pip, result_rss.pip) > 0.8
        end

        @testset "With Prior Variance" begin
            n, p = 1000, 80
            X, y, _, _ = generate_finemapping_data(n, p; n_causal=2)

            z = [cov(X[:, j], y) / var(X[:, j]) /
                 sqrt(var(y - X[:, j] .* cov(X[:, j], y) / var(X[:, j])) / (n * var(X[:, j])))
                 for j in 1:p]
            R = cor(X)

            result_default = susie_rss(z, R, n)
            result_custom = susie_rss(z, R, n; prior_variance=0.5)

            @test isa(result_custom, SuSiEResult)
        end

        @testset "LD Mismatch Handling" begin
            n, p = 500, 40
            X, y, _, _ = generate_finemapping_data(n, p; n_causal=1)

            z = [cov(X[:, j], y) / var(X[:, j]) /
                 sqrt(var(y - X[:, j] .* cov(X[:, j], y) / var(X[:, j])) / (n * var(X[:, j])))
                 for j in 1:p]

            # Add noise to LD matrix (simulate reference mismatch)
            R = cor(X) + randn(p, p) * 0.05
            R = (R + R') / 2
            R[diagind(R)] .= 1.0

            result = susie_rss(z, R, n; check_R=true)

            @test isa(result, SuSiEResult)
        end
    end

    # ========================================================================
    # Credible Set Tests
    # ========================================================================
    @testset "Credible Sets" begin
        @testset "Coverage" begin
            X, y, causal_idx, _ = generate_finemapping_data(500, 100;
                                                             n_causal=2, h2=0.4)

            result = susie(X, y; L=5, min_abs_corr=0.5)

            # Check coverage levels
            for cov in result.cs_coverage
                @test cov >= 0.9  # Should be high coverage
            end
        end

        @testset "Purity" begin
            X, y, _, _ = generate_finemapping_data(300, 60; n_causal=2)

            result = susie(X, y; L=5, min_abs_corr=0.5)

            # Credible sets should have correlated variants
            for cs in result.cs
                if length(cs) > 1
                    R = cor(X[:, cs])
                    min_corr = minimum(abs.(R[R .!= 1.0]))
                    @test min_corr >= 0.5  # Purity threshold
                end
            end
        end

        @testset "CS Summary" begin
            X, y, causal_idx, _ = generate_finemapping_data(400, 80; n_causal=2)

            result = susie(X, y; L=5)

            summary = susie_get_cs_summary(result, X)

            @test isa(summary, DataFrame)
            @test :cs_id in propertynames(summary)
            @test :coverage in propertynames(summary)
            @test :purity in propertynames(summary)
        end
    end

    # ========================================================================
    # PIP Computation Tests
    # ========================================================================
    @testset "PIP Computation" begin
        @testset "From Alpha" begin
            p, L = 50, 5
            alpha = rand(p, L)
            alpha = alpha ./ sum(alpha, dims=1)  # Normalize columns

            pip = compute_pip(alpha)

            @test length(pip) == p
            @test all(0 .<= pip .<= 1)
        end

        @testset "Monotonicity" begin
            X, y, _, _ = generate_finemapping_data(200, 30; n_causal=1)

            result = susie(X, y; L=3)

            # Adding more effects shouldn't decrease max PIP
            result_more = susie(X, y; L=10)

            @test maximum(result_more.pip) >= maximum(result.pip) - 0.1
        end
    end

    # ========================================================================
    # Prior Settings Tests
    # ========================================================================
    @testset "Prior Settings" begin
        @testset "Prior Variance Estimation" begin
            X, y, _, _ = generate_finemapping_data(500, 100; n_causal=2, h2=0.3)

            result = susie(X, y; L=5, estimate_prior_variance=true)

            @test isa(result, SuSiEResult)
            @test result.sigma2 > 0
        end

        @testset "Fixed Prior Variance" begin
            X, y, _, _ = generate_finemapping_data(300, 50; n_causal=1)

            result = susie(X, y; L=5, prior_variance=0.1,
                          estimate_prior_variance=false)

            @test isa(result, SuSiEResult)
        end

        @testset "Prior Weights" begin
            X, y, _, _ = generate_finemapping_data(300, 50; n_causal=1)

            # Uniform prior
            result_uniform = susie(X, y; L=5)

            # Informed prior (upweight first 10 variants)
            prior_weights = ones(50)
            prior_weights[1:10] .= 10.0
            prior_weights ./= sum(prior_weights)

            result_informed = susie(X, y; L=5, prior_weights=prior_weights)

            # Informed prior should change results
            @test cor(result_uniform.pip, result_informed.pip) < 1.0
        end
    end

    # ========================================================================
    # Residual Variance Tests
    # ========================================================================
    @testset "Residual Variance" begin
        @testset "Estimation" begin
            X, y, _, _ = generate_finemapping_data(500, 100; n_causal=2, h2=0.3)

            result = susie(X, y; L=5, estimate_residual_variance=true)

            @test result.sigma2 > 0
            @test result.sigma2 < var(y)  # Should be less than total variance
        end

        @testset "Fixed Variance" begin
            X, y, _, _ = generate_finemapping_data(300, 50; n_causal=1)

            fixed_var = 1.0
            result = susie(X, y; L=5, residual_variance=fixed_var,
                          estimate_residual_variance=false)

            @test result.sigma2 == fixed_var
        end
    end

    # ========================================================================
    # Convergence and Stability Tests
    # ========================================================================
    @testset "Convergence" begin
        @testset "ELBO Increase" begin
            X, y, _, _ = generate_finemapping_data(300, 50; n_causal=2)

            result = susie(X, y; L=5, max_iter=100)

            elbo = result.elbo
            # ELBO should generally increase (allow small decreases due to numerics)
            for i in 2:length(elbo)
                @test elbo[i] >= elbo[i-1] - 1e-6
            end
        end

        @testset "Tolerance" begin
            X, y, _, _ = generate_finemapping_data(200, 30; n_causal=1)

            result_loose = susie(X, y; L=5, tol=1e-2)
            result_tight = susie(X, y; L=5, tol=1e-6)

            @test result_tight.n_iterations >= result_loose.n_iterations
        end

        @testset "Max Iterations" begin
            X, y, _, _ = generate_finemapping_data(100, 20; n_causal=1)

            result = susie(X, y; L=5, max_iter=10, tol=1e-10)

            @test result.n_iterations <= 10
        end
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================
    @testset "Edge Cases" begin
        @testset "No Signal" begin
            X = randn(200, 50)
            y = randn(200)  # Pure noise

            result = susie(X, y; L=5)

            @test isa(result, SuSiEResult)
            @test maximum(result.pip) < 0.5  # No confident signal
            @test length(result.cs) == 0 || all(isempty.(result.cs))
        end

        @testset "Perfect Prediction" begin
            n, p = 100, 10
            X = randn(n, p)
            y = X[:, 1]  # Perfect signal

            result = susie(X, y; L=3)

            @test result.pip[1] ≈ 1.0 atol=0.1
        end

        @testset "Highly Correlated Variants" begin
            n = 200
            X_base = randn(n, 5)
            # Create correlated variants
            X = hcat(X_base, X_base .+ randn(n, 5) * 0.1)
            y = X[:, 1] + randn(n) * 0.5

            result = susie(X, y; L=5, min_abs_corr=0.5)

            @test isa(result, SuSiEResult)
            # Correlated variants should be in same CS
            for cs in result.cs
                if 1 in cs
                    @test 6 in cs  # Correlated variant
                end
            end
        end

        @testset "Single Variant" begin
            X = randn(100, 1)
            y = X[:, 1] + randn(100) * 0.5

            result = susie(X, y; L=1)

            @test length(result.pip) == 1
            @test result.pip[1] > 0.5
        end

        @testset "p > n" begin
            n, p = 50, 100
            X = randn(n, p)
            y = X[:, 1] + randn(n) * 0.5

            result = susie(X, y; L=5)

            @test isa(result, SuSiEResult)
            @test length(result.pip) == p
        end
    end

    # ========================================================================
    # Integration Tests
    # ========================================================================
    @testset "Integration" begin
        @testset "Full Pipeline" begin
            # Simulate GWAS region
            n, p = 1000, 200
            X, y, causal_idx, β_true = generate_finemapping_data(n, p;
                                                                  n_causal=3, h2=0.3)

            # Step 1: Run SuSiE
            result = susie(X, y; L=10, min_abs_corr=0.5)

            # Step 2: Get credible sets
            cs_summary = susie_get_cs_summary(result, X)

            # Step 3: Extract PIPs
            pip = result.pip

            # Validation
            @test length(result.cs) > 0
            @test maximum(pip) > 0.5

            # Check causal variant recovery
            top_pip_idx = sortperm(pip, rev=true)[1:5]
            causal_in_top = sum(c in top_pip_idx for c in causal_idx)
            @test causal_in_top >= 1
        end

        @testset "Summary Statistics Pipeline" begin
            n, p = 5000, 100
            X, y, causal_idx, _ = generate_finemapping_data(n, p; n_causal=2, h2=0.3)

            # Compute summary statistics
            z = zeros(p)
            se = zeros(p)
            for j in 1:p
                β_j = cov(X[:, j], y) / var(X[:, j])
                se_j = sqrt(var(y - X[:, j] .* β_j) / (n * var(X[:, j])))
                z[j] = β_j / se_j
                se[j] = se_j
            end

            # LD from reference panel
            R = cor(X)

            # Run SuSiE-RSS
            result = susie_rss(z, R, n; L=5)

            @test isa(result, SuSiEResult)
            @test any(result.pip[causal_idx] .> 0.3)
        end
    end

end # @testset "Fine-Mapping (SuSiE)"
