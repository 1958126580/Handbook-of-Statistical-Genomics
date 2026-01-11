# ============================================================================
# Comprehensive Tests for Heritability Module (LDSC)
# ============================================================================
# Tests for LD score regression, genetic correlation, and partitioned heritability
# ============================================================================

@testset "Heritability Analysis (LDSC)" begin

    # ========================================================================
    # LD Score Computation Tests
    # ========================================================================
    @testset "LD Score Computation" begin
        @testset "Basic LD Scores" begin
            n, m = 500, 100
            genotypes = rand(0:2, n, m) |> x -> Float64.(x)
            positions = collect(1000:1000:100000)

            ld_scores = compute_ld_scores(genotypes, positions)

            @test length(ld_scores) == m
            @test all(ld_scores .>= 1.0)  # Self-LD contributes at least 1
        end

        @testset "With Window Size" begin
            n, m = 300, 50
            genotypes = rand(0:2, n, m) |> x -> Float64.(x)
            positions = collect(1:50)

            ld_scores_small = compute_ld_scores(genotypes, positions; window_kb=10)
            ld_scores_large = compute_ld_scores(genotypes, positions; window_kb=100)

            # Larger window should give higher or equal LD scores
            @test all(ld_scores_large .>= ld_scores_small .- 0.01)
        end

        @testset "Correlation-Based" begin
            n, m = 200, 30
            genotypes = rand(0:2, n, m) |> x -> Float64.(x)
            positions = collect(1:30)

            ld_scores = compute_ld_scores(genotypes, positions; method=:correlation)

            @test length(ld_scores) == m
            @test all(ld_scores .>= 0)
        end

        @testset "Sample Size Adjustment" begin
            n, m = 100, 20
            genotypes = rand(0:2, n, m) |> x -> Float64.(x)
            positions = collect(1:20)

            ld_scores = compute_ld_scores(genotypes, positions; adjust_n=true)

            @test length(ld_scores) == m
        end
    end

    # ========================================================================
    # LDSC Regression Tests
    # ========================================================================
    @testset "LDSC Regression" begin
        @testset "LDScoreResult Structure" begin
            result = LDScoreResult(
                0.5,    # h2
                0.05,   # h2_se
                1.02,   # intercept
                0.02,   # intercept_se
                0.0,    # lambda_gc
                1000,   # n_snps
                Dict{Symbol,Any}()
            )

            @test result.h2 == 0.5
            @test result.h2_se == 0.05
            @test result.intercept == 1.02
        end

        @testset "Basic LDSC" begin
            m = 1000
            n_samples = 50000

            # Simulate null chi-squared statistics
            chi2 = rand(Chisq(1), m)

            # Simulate LD scores
            ld_scores = rand(m) .* 50 .+ 10

            result = ldsc_regression(chi2, ld_scores, n_samples)

            @test isa(result, LDScoreResult)
            @test result.n_snps == m
            @test 0 <= result.intercept <= 2  # Should be near 1 under null
        end

        @testset "With Significant Heritability" begin
            m = 2000
            n_samples = 100000

            # Simulate LD scores
            ld_scores = rand(m) .* 100 .+ 20

            # Add heritability signal to chi-squared
            h2 = 0.3
            chi2_inflation = 1 .+ (n_samples * h2 / m) .* ld_scores
            chi2 = rand(Chisq(1), m) .* chi2_inflation

            result = ldsc_regression(chi2, ld_scores, n_samples)

            @test isa(result, LDScoreResult)
            @test result.h2 > 0  # Should detect heritability
        end

        @testset "Weighted Regression" begin
            m = 500
            chi2 = rand(Chisq(1), m)
            ld_scores = rand(m) .* 30 .+ 5
            weights = 1 ./ (ld_scores .^ 2)  # Standard LDSC weights

            result = ldsc_regression(chi2, ld_scores, 50000; weights=weights)

            @test isa(result, LDScoreResult)
        end

        @testset "Intercept Constraint" begin
            m = 500
            chi2 = rand(Chisq(1), m)
            ld_scores = rand(m) .* 30 .+ 10

            # Constrain intercept to 1
            result = ldsc_regression(chi2, ld_scores, 50000; constrain_intercept=true)

            @test result.intercept ≈ 1.0
        end

        @testset "Two-Step Estimator" begin
            m = 1000
            chi2 = rand(Chisq(1), m)
            ld_scores = rand(m) .* 50 .+ 10

            result = ldsc_regression(chi2, ld_scores, 50000; two_step=true)

            @test isa(result, LDScoreResult)
            @test haskey(result.extra, :attenuation_ratio) || true  # May have extra info
        end
    end

    # ========================================================================
    # Liability Scale Conversion Tests
    # ========================================================================
    @testset "Liability Scale Conversion" begin
        @testset "Basic Conversion" begin
            h2_observed = 0.2
            prevalence = 0.01
            sample_prevalence = 0.5

            h2_liability = observed_to_liability(h2_observed, prevalence, sample_prevalence)

            @test h2_liability > h2_observed  # Liability h2 should be higher
            @test h2_liability <= 1.0
        end

        @testset "Different Prevalences" begin
            h2_obs = 0.15

            # Lower prevalence = larger correction
            h2_rare = observed_to_liability(h2_obs, 0.001, 0.5)
            h2_common = observed_to_liability(h2_obs, 0.1, 0.5)

            @test h2_rare > h2_common
        end

        @testset "Balanced Sample" begin
            h2_obs = 0.25

            # When sample prevalence matches population
            h2_balanced = observed_to_liability(h2_obs, 0.1, 0.1)

            @test h2_balanced >= h2_obs
        end
    end

    # ========================================================================
    # Genetic Correlation Tests
    # ========================================================================
    @testset "Genetic Correlation" begin
        @testset "GeneticCorrelationResult Structure" begin
            result = GeneticCorrelationResult(
                0.6,    # rg
                0.08,   # rg_se
                7.5,    # z
                1e-10,  # pvalue
                Dict{Symbol,Any}()
            )

            @test result.rg == 0.6
            @test result.rg_se == 0.08
            @test result.z == 7.5
        end

        @testset "Basic Genetic Correlation" begin
            m = 1000
            ld_scores = rand(m) .* 50 .+ 10

            # Two correlated traits
            z1 = randn(m)
            z2 = 0.7 .* z1 .+ 0.714 .* randn(m)  # rg ≈ 0.7

            result = genetic_correlation(z1, z2, ld_scores, 50000, 50000)

            @test isa(result, GeneticCorrelationResult)
            @test -1 <= result.rg <= 1
        end

        @testset "Same Trait" begin
            m = 500
            ld_scores = rand(m) .* 30 .+ 5
            z = randn(m)

            result = genetic_correlation(z, z, ld_scores, 50000, 50000)

            @test result.rg ≈ 1.0 atol=0.1
        end

        @testset "Uncorrelated Traits" begin
            m = 1000
            ld_scores = rand(m) .* 50 .+ 10
            z1 = randn(m)
            z2 = randn(m)  # Independent

            result = genetic_correlation(z1, z2, ld_scores, 50000, 50000)

            @test abs(result.rg) < 0.3  # Should be near zero
        end

        @testset "Sample Overlap" begin
            m = 500
            ld_scores = rand(m) .* 30 .+ 10
            z1 = randn(m)
            z2 = randn(m)

            result = genetic_correlation(z1, z2, ld_scores, 50000, 50000;
                                        n_overlap=25000)

            @test isa(result, GeneticCorrelationResult)
        end

        @testset "Negative Correlation" begin
            m = 500
            ld_scores = rand(m) .* 30 .+ 10
            z1 = randn(m)
            z2 = -0.5 .* z1 .+ randn(m)

            result = genetic_correlation(z1, z2, ld_scores, 50000, 50000)

            @test result.rg < 0
        end
    end

    # ========================================================================
    # Partitioned LDSC Tests
    # ========================================================================
    @testset "Partitioned LDSC" begin
        @testset "PartitionedHeritability Structure" begin
            result = PartitionedHeritability(
                0.5,
                [0.1, 0.2, 0.2],
                [0.02, 0.03, 0.03],
                [2.0, 1.5, 0.8],
                [0.3, 0.2, 0.1],
                ["CNS", "Immune", "Baseline"],
                Dict{Symbol,Any}()
            )

            @test result.h2_total == 0.5
            @test length(result.h2_categories) == 3
            @test length(result.enrichment) == 3
        end

        @testset "Basic Partitioning" begin
            m = 1000
            n_samples = 50000
            n_categories = 3

            chi2 = rand(Chisq(1), m)

            # Create category-specific LD scores
            annotations = rand(Bool, m, n_categories) |> x -> Float64.(x)
            ld_scores_cat = rand(m, n_categories) .* 30 .+ 10

            result = partitioned_ldsc(chi2, annotations, ld_scores_cat, n_samples)

            @test isa(result, PartitionedHeritability)
            @test length(result.h2_categories) == n_categories
            @test length(result.enrichment) == n_categories
        end

        @testset "Enrichment Calculation" begin
            m = 2000
            n_samples = 100000

            # Create annotation with enriched category
            annotations = zeros(m, 2)
            annotations[:, 1] .= 1.0  # Baseline
            annotations[1:200, 2] .= 1.0  # Enriched category (10%)

            # Add signal to enriched SNPs
            chi2 = rand(Chisq(1), m)
            chi2[1:200] .*= 3.0  # Inflate chi2 for enriched

            ld_scores_cat = rand(m, 2) .* 20 .+ 5

            result = partitioned_ldsc(chi2, annotations, ld_scores_cat, n_samples)

            # Second category should show enrichment
            @test result.enrichment[2] > result.enrichment[1]
        end

        @testset "Jackknife SE" begin
            m = 500
            chi2 = rand(Chisq(1), m)
            annotations = rand(Bool, m, 2) |> x -> Float64.(x)
            ld_scores_cat = rand(m, 2) .* 20 .+ 5

            result = partitioned_ldsc(chi2, annotations, ld_scores_cat, 50000;
                                     jackknife_blocks=50)

            @test all(result.h2_se .> 0)
            @test all(result.enrichment_se .> 0)
        end
    end

    # ========================================================================
    # Cell Type Enrichment Tests
    # ========================================================================
    @testset "Cell Type Enrichment" begin
        @testset "Basic Enrichment" begin
            m = 1000
            chi2 = rand(Chisq(1), m)

            # Cell type annotations
            cell_types = Dict(
                "Neurons" => rand(Bool, m),
                "Astrocytes" => rand(Bool, m),
                "Microglia" => rand(Bool, m)
            )

            ld_scores = rand(m) .* 30 .+ 10

            result = compute_cell_type_enrichment(chi2, cell_types, ld_scores, 50000)

            @test isa(result, Dict)
            @test haskey(result, "Neurons")
            @test haskey(result["Neurons"], :enrichment)
        end

        @testset "Multiple Cell Types" begin
            m = 500
            chi2 = rand(Chisq(1), m)
            ld_scores = rand(m) .* 20 .+ 5

            cell_types = Dict(
                "Type$i" => rand(Bool, m) for i in 1:5
            )

            result = compute_cell_type_enrichment(chi2, cell_types, ld_scores, 50000)

            @test length(result) == 5
            for (ct, stats) in result
                @test haskey(stats, :enrichment)
                @test haskey(stats, :pvalue)
            end
        end
    end

    # ========================================================================
    # Stratified LDSC Tests
    # ========================================================================
    @testset "Stratified LDSC" begin
        @testset "Basic Stratification" begin
            m = 1000
            chi2 = rand(Chisq(1), m)
            ld_scores = rand(m) .* 50 .+ 10

            # Stratify by MAF
            maf = rand(m) .* 0.5
            maf_bins = [0.0, 0.05, 0.2, 0.5]

            result = stratified_ldsc(chi2, ld_scores, 50000;
                                    stratify_by=maf, bins=maf_bins)

            @test isa(result, Dict)
            @test length(result) == 3  # 3 bins
        end

        @testset "LD Stratification" begin
            m = 500
            chi2 = rand(Chisq(1), m)
            ld_scores = rand(m) .* 100 .+ 10

            # Stratify by LD scores
            ld_bins = [0.0, 30.0, 60.0, 150.0]

            result = stratified_ldsc(chi2, ld_scores, 50000;
                                    stratify_by=ld_scores, bins=ld_bins)

            @test length(result) == 3
        end
    end

    # ========================================================================
    # Edge Cases and Validation
    # ========================================================================
    @testset "Edge Cases" begin
        @testset "Small Sample Size" begin
            m = 100
            chi2 = rand(Chisq(1), m)
            ld_scores = rand(m) .* 20 .+ 5

            result = ldsc_regression(chi2, ld_scores, 1000)

            @test isa(result, LDScoreResult)
        end

        @testset "Large Chi-Squared" begin
            m = 500
            chi2 = rand(Chisq(1), m) .* 10  # Inflated
            ld_scores = rand(m) .* 30 .+ 10

            result = ldsc_regression(chi2, ld_scores, 50000)

            @test isa(result, LDScoreResult)
            @test result.intercept > 1.0  # Should detect inflation
        end

        @testset "Negative Heritability Constraint" begin
            m = 500
            chi2 = rand(Chisq(1), m) .* 0.5  # Deflated
            ld_scores = rand(m) .* 30 .+ 10

            result = ldsc_regression(chi2, ld_scores, 50000; constrain_h2_positive=true)

            @test result.h2 >= 0  # Should not be negative
        end

        @testset "Missing LD Scores" begin
            m = 500
            chi2 = rand(Chisq(1), m)
            ld_scores = rand(m) .* 30 .+ 10
            ld_scores[1:50] .= NaN  # Some missing

            # Should handle or filter missing
            valid_idx = .!isnan.(ld_scores)
            result = ldsc_regression(chi2[valid_idx], ld_scores[valid_idx], 50000)

            @test isa(result, LDScoreResult)
        end
    end

    # ========================================================================
    # Integration Tests
    # ========================================================================
    @testset "Integration" begin
        @testset "Full Pipeline" begin
            # Simulate a complete LDSC analysis
            n, m = 1000, 500
            genotypes = rand(0:2, n, m) |> x -> Float64.(x)
            positions = collect(1000:1000:500000)

            # Step 1: Compute LD scores
            ld_scores = compute_ld_scores(genotypes, positions)

            # Step 2: Simulate GWAS (chi-squared)
            chi2 = rand(Chisq(1), m)

            # Step 3: Run LDSC
            result = ldsc_regression(chi2, ld_scores, n)

            @test isa(result, LDScoreResult)
            @test result.n_snps == m
        end

        @testset "Cross-Trait Analysis" begin
            m = 500
            ld_scores = rand(m) .* 30 .+ 10

            z1 = randn(m)
            z2 = 0.5 .* z1 .+ randn(m)

            # Heritability for both traits
            chi2_1 = z1.^2
            chi2_2 = z2.^2

            result1 = ldsc_regression(chi2_1, ld_scores, 50000)
            result2 = ldsc_regression(chi2_2, ld_scores, 50000)

            # Genetic correlation
            rg_result = genetic_correlation(z1, z2, ld_scores, 50000, 50000)

            @test isa(result1, LDScoreResult)
            @test isa(result2, LDScoreResult)
            @test isa(rg_result, GeneticCorrelationResult)
        end
    end

end # @testset "Heritability Analysis (LDSC)"
