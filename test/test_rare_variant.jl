# ============================================================================
# Comprehensive Tests for Rare Variant Analysis Module
# ============================================================================
# Tests for burden tests, SKAT, SKAT-O, and gene-based analysis
# ============================================================================

@testset "Rare Variant Analysis" begin

    # ========================================================================
    # Test Data Generation Helpers
    # ========================================================================
    function generate_rare_variant_data(n_samples::Int, n_variants::Int;
                                        causal_fraction::Float64=0.3,
                                        effect_size::Float64=0.5)
        # Generate rare variant genotypes (MAF < 0.05)
        mafs = rand(n_variants) .* 0.04 .+ 0.001
        genotypes = zeros(n_samples, n_variants)
        for j in 1:n_variants
            p = mafs[j]
            for i in 1:n_samples
                r = rand()
                if r < (1-p)^2
                    genotypes[i, j] = 0
                elseif r < (1-p)^2 + 2*p*(1-p)
                    genotypes[i, j] = 1
                else
                    genotypes[i, j] = 2
                end
            end
        end

        # Generate phenotype with some causal variants
        n_causal = max(1, round(Int, n_variants * causal_fraction))
        causal_idx = 1:n_causal
        effects = zeros(n_variants)
        effects[causal_idx] .= effect_size

        y = genotypes * effects + randn(n_samples)

        return genotypes, y, causal_idx
    end

    # ========================================================================
    # RareVariantResult Structure Tests
    # ========================================================================
    @testset "RareVariantResult Structure" begin
        result = RareVariantResult(
            :burden,
            5.0,
            0.025,
            1,
            10,
            Dict(:weights => ones(10))
        )

        @test result.test_type == :burden
        @test result.statistic == 5.0
        @test result.pvalue == 0.025
        @test result.n_samples == 1
        @test result.n_variants == 10
    end

    # ========================================================================
    # Burden Test Tests
    # ========================================================================
    @testset "Burden Test" begin
        @testset "Basic Burden Test" begin
            genotypes, y, _ = generate_rare_variant_data(200, 15; effect_size=0.8)

            result = burden_test(genotypes, y)

            @test isa(result, RareVariantResult)
            @test result.test_type == :burden
            @test 0 <= result.pvalue <= 1
            @test result.n_samples == 200
            @test result.n_variants == 15
        end

        @testset "With Covariates" begin
            genotypes, y, _ = generate_rare_variant_data(200, 10)
            covariates = hcat(ones(200), randn(200, 2))

            result = burden_test(genotypes, y; covariates=covariates)

            @test isa(result, RareVariantResult)
            @test 0 <= result.pvalue <= 1
        end

        @testset "Custom Weights" begin
            genotypes, y, _ = generate_rare_variant_data(150, 8)

            # MAF-based weights (beta distribution)
            mafs = mean(genotypes, dims=1)[:] ./ 2
            weights = pdf.(Beta(1, 25), mafs)

            result = burden_test(genotypes, y; weights=weights)

            @test isa(result, RareVariantResult)
            @test haskey(result.details, :weights)
        end

        @testset "Binary Phenotype" begin
            n = 200
            genotypes, _, _ = generate_rare_variant_data(n, 12)
            y_binary = rand(n) .< 0.4

            result = burden_test(genotypes, Float64.(y_binary); family=:binomial)

            @test isa(result, RareVariantResult)
            @test 0 <= result.pvalue <= 1
        end

        @testset "Causal Detection" begin
            # Strong signal should be detected
            genotypes, y, _ = generate_rare_variant_data(500, 20;
                                                         causal_fraction=0.5,
                                                         effect_size=1.5)

            result = burden_test(genotypes, y)
            @test result.pvalue < 0.05
        end
    end

    # ========================================================================
    # SKAT Tests
    # ========================================================================
    @testset "SKAT" begin
        @testset "Basic SKAT" begin
            genotypes, y, _ = generate_rare_variant_data(200, 15)

            result = skat(genotypes, y)

            @test isa(result, RareVariantResult)
            @test result.test_type == :skat
            @test 0 <= result.pvalue <= 1
            @test result.statistic >= 0  # SKAT statistic is non-negative
        end

        @testset "With Covariates" begin
            genotypes, y, _ = generate_rare_variant_data(200, 10)
            covariates = randn(200, 3)

            result = skat(genotypes, y; covariates=covariates)

            @test isa(result, RareVariantResult)
            @test 0 <= result.pvalue <= 1
        end

        @testset "Kernel Types" begin
            genotypes, y, _ = generate_rare_variant_data(150, 12)

            # Linear kernel
            result_linear = skat(genotypes, y; kernel=:linear)
            @test result_linear.test_type == :skat

            # IBS kernel
            result_ibs = skat(genotypes, y; kernel=:ibs)
            @test result_ibs.test_type == :skat

            # Quadratic kernel
            result_quad = skat(genotypes, y; kernel=:quadratic)
            @test result_quad.test_type == :skat
        end

        @testset "Binary Phenotype" begin
            n = 250
            genotypes, _, _ = generate_rare_variant_data(n, 10)
            y_binary = rand(n) .< 0.3

            result = skat(genotypes, Float64.(y_binary); family=:binomial)

            @test isa(result, RareVariantResult)
            @test 0 <= result.pvalue <= 1
        end

        @testset "P-value Accuracy" begin
            # Test that p-values are properly calibrated under null
            n_null_tests = 50
            pvalues = Float64[]

            for _ in 1:n_null_tests
                genotypes = rand([0, 1, 2], 100, 8) .* (rand(100, 8) .< 0.05)
                y = randn(100)  # Pure noise
                result = skat(genotypes, y)
                push!(pvalues, result.pvalue)
            end

            # P-values should be approximately uniform under null
            @test mean(pvalues .< 0.5) > 0.3
            @test mean(pvalues .< 0.5) < 0.7
        end
    end

    # ========================================================================
    # SKAT-O Tests
    # ========================================================================
    @testset "SKAT-O" begin
        @testset "Basic SKAT-O" begin
            genotypes, y, _ = generate_rare_variant_data(200, 15)

            result = skat_o(genotypes, y)

            @test isa(result, RareVariantResult)
            @test result.test_type == :skat_o
            @test 0 <= result.pvalue <= 1
            @test haskey(result.details, :optimal_rho)
        end

        @testset "Optimal Rho Selection" begin
            genotypes, y, _ = generate_rare_variant_data(200, 12)

            result = skat_o(genotypes, y)

            rho = result.details[:optimal_rho]
            @test 0 <= rho <= 1
        end

        @testset "Burden-Like Signal" begin
            # When all variants have same direction, SKAT-O should favor burden
            n = 200
            genotypes, _, _ = generate_rare_variant_data(n, 10)
            # All positive effects
            y = sum(genotypes, dims=2)[:] .+ randn(n) * 0.5

            result = skat_o(genotypes, y)

            # Optimal rho should be close to 1 (burden-like)
            @test result.details[:optimal_rho] > 0.5
        end

        @testset "SKAT-Like Signal" begin
            # Mixed effect directions favor SKAT
            n = 200
            genotypes, _, _ = generate_rare_variant_data(n, 10)
            # Alternating effects
            effects = repeat([1.0, -1.0], 5)
            y = genotypes * effects .+ randn(n) * 0.5

            result = skat_o(genotypes, y)

            # Optimal rho should be close to 0 (SKAT-like)
            @test result.details[:optimal_rho] < 0.5
        end

        @testset "Grid Search" begin
            genotypes, y, _ = generate_rare_variant_data(150, 8)

            # Custom rho grid
            result = skat_o(genotypes, y; rho_grid=[0.0, 0.25, 0.5, 0.75, 1.0])

            @test isa(result, RareVariantResult)
            @test result.details[:optimal_rho] in [0.0, 0.25, 0.5, 0.75, 1.0]
        end
    end

    # ========================================================================
    # CMC Test Tests
    # ========================================================================
    @testset "CMC Test" begin
        @testset "Basic CMC" begin
            genotypes, y, _ = generate_rare_variant_data(200, 15)

            result = cmc_test(genotypes, y)

            @test isa(result, RareVariantResult)
            @test result.test_type == :cmc
            @test 0 <= result.pvalue <= 1
        end

        @testset "MAF Threshold" begin
            genotypes, y, _ = generate_rare_variant_data(200, 12)

            # Different MAF thresholds
            result1 = cmc_test(genotypes, y; maf_threshold=0.01)
            result2 = cmc_test(genotypes, y; maf_threshold=0.05)

            @test isa(result1, RareVariantResult)
            @test isa(result2, RareVariantResult)
        end

        @testset "Binary Phenotype" begin
            n = 200
            genotypes, _, _ = generate_rare_variant_data(n, 10)
            y_binary = rand(n) .< 0.35

            result = cmc_test(genotypes, Float64.(y_binary); family=:binomial)

            @test isa(result, RareVariantResult)
        end
    end

    # ========================================================================
    # VT Test Tests
    # ========================================================================
    @testset "VT Test (Variable Threshold)" begin
        @testset "Basic VT Test" begin
            genotypes, y, _ = generate_rare_variant_data(200, 15)

            result = vt_test(genotypes, y)

            @test isa(result, RareVariantResult)
            @test result.test_type == :vt
            @test 0 <= result.pvalue <= 1
            @test haskey(result.details, :optimal_threshold)
        end

        @testset "Threshold Selection" begin
            genotypes, y, _ = generate_rare_variant_data(200, 12)

            result = vt_test(genotypes, y)

            threshold = result.details[:optimal_threshold]
            @test 0 < threshold < 0.1  # Should be in rare variant range
        end
    end

    # ========================================================================
    # ACAT-V Test Tests
    # ========================================================================
    @testset "ACAT-V Test" begin
        @testset "Basic ACAT-V" begin
            genotypes, y, _ = generate_rare_variant_data(200, 15)

            result = acatv_test(genotypes, y)

            @test isa(result, RareVariantResult)
            @test result.test_type == :acatv
            @test 0 <= result.pvalue <= 1
        end

        @testset "Aggregating Single-Variant Tests" begin
            genotypes, y, _ = generate_rare_variant_data(200, 10)

            result = acatv_test(genotypes, y)

            @test haskey(result.details, :variant_pvalues)
            @test length(result.details[:variant_pvalues]) == 10
        end

        @testset "With Weights" begin
            genotypes, y, _ = generate_rare_variant_data(150, 8)
            weights = rand(8)

            result = acatv_test(genotypes, y; weights=weights)

            @test isa(result, RareVariantResult)
        end

        @testset "Robust to Correlation" begin
            # Create correlated variants
            n, m = 200, 10
            base = rand([0, 1, 2], n, 2)
            genotypes = hcat(base, base .+ rand([0, 1], n, 8))
            genotypes = clamp.(genotypes, 0, 2)
            y = randn(n)

            result = acatv_test(genotypes, y)

            @test isa(result, RareVariantResult)
            @test 0 <= result.pvalue <= 1
        end
    end

    # ========================================================================
    # Gene-Based Test Tests
    # ========================================================================
    @testset "Gene-Based Test" begin
        @testset "Single Gene" begin
            genotypes, y, _ = generate_rare_variant_data(200, 30)
            variant_positions = collect(1000:1000:30000)

            gene_regions = DataFrame(
                gene = ["GENE1"],
                chr = [1],
                start = [1],
                stop = [15000]
            )

            results = gene_based_test(genotypes, y, variant_positions, gene_regions)

            @test isa(results, DataFrame)
            @test nrow(results) == 1
            @test :gene in propertynames(results)
            @test :pvalue in propertynames(results)
        end

        @testset "Multiple Genes" begin
            genotypes, y, _ = generate_rare_variant_data(200, 50)
            variant_positions = collect(1000:1000:50000)

            gene_regions = DataFrame(
                gene = ["GENE1", "GENE2", "GENE3"],
                chr = [1, 1, 1],
                start = [1, 20000, 40000],
                stop = [18000, 38000, 50000]
            )

            results = gene_based_test(genotypes, y, variant_positions, gene_regions)

            @test nrow(results) == 3
            @test all(0 .<= results.pvalue .<= 1)
        end

        @testset "Different Test Methods" begin
            genotypes, y, _ = generate_rare_variant_data(150, 20)
            variant_positions = collect(1:20)

            gene_regions = DataFrame(
                gene = ["GENE1"],
                chr = [1],
                start = [1],
                stop = [20]
            )

            # Burden test
            results_burden = gene_based_test(genotypes, y, variant_positions,
                                            gene_regions; method=:burden)
            @test results_burden.method[1] == "burden"

            # SKAT
            results_skat = gene_based_test(genotypes, y, variant_positions,
                                          gene_regions; method=:skat)
            @test results_skat.method[1] == "skat"

            # SKAT-O
            results_skato = gene_based_test(genotypes, y, variant_positions,
                                           gene_regions; method=:skat_o)
            @test results_skato.method[1] == "skat_o"
        end

        @testset "Multiple Testing Correction" begin
            genotypes, y, _ = generate_rare_variant_data(200, 100)
            variant_positions = collect(1:100)

            gene_regions = DataFrame(
                gene = ["G$i" for i in 1:10],
                chr = ones(Int, 10),
                start = collect(1:10:91),
                stop = collect(10:10:100)
            )

            results = gene_based_test(genotypes, y, variant_positions,
                                     gene_regions; correction=:bonferroni)

            @test :pvalue_corrected in propertynames(results)
            @test all(results.pvalue_corrected .>= results.pvalue)
        end

        @testset "Empty Gene" begin
            genotypes, y, _ = generate_rare_variant_data(100, 10)
            variant_positions = collect(100:100:1000)

            gene_regions = DataFrame(
                gene = ["GENE1", "GENE2"],
                chr = [1, 1],
                start = [1, 2000],  # GENE2 has no variants
                stop = [500, 3000]
            )

            results = gene_based_test(genotypes, y, variant_positions, gene_regions)

            # Should handle genes with no variants
            @test nrow(results) >= 1
        end
    end

    # ========================================================================
    # Comparison Tests
    # ========================================================================
    @testset "Method Comparisons" begin
        @testset "Power Comparison" begin
            genotypes, y, _ = generate_rare_variant_data(300, 20;
                                                         causal_fraction=0.4,
                                                         effect_size=1.0)

            result_burden = burden_test(genotypes, y)
            result_skat = skat(genotypes, y)
            result_skato = skat_o(genotypes, y)
            result_cmc = cmc_test(genotypes, y)

            # All should return valid p-values
            @test 0 <= result_burden.pvalue <= 1
            @test 0 <= result_skat.pvalue <= 1
            @test 0 <= result_skato.pvalue <= 1
            @test 0 <= result_cmc.pvalue <= 1
        end

        @testset "Consistency Across Runs" begin
            Random.seed!(42)
            genotypes, y, _ = generate_rare_variant_data(200, 15)

            result1 = skat(genotypes, y)

            # Same input should give same result
            result2 = skat(genotypes, y)

            @test result1.pvalue ≈ result2.pvalue
            @test result1.statistic ≈ result2.statistic
        end
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================
    @testset "Edge Cases" begin
        @testset "Single Variant" begin
            genotypes = rand([0, 1, 2], 100, 1)
            y = randn(100)

            result = burden_test(genotypes, y)
            @test isa(result, RareVariantResult)

            result_skat = skat(genotypes, y)
            @test isa(result_skat, RareVariantResult)
        end

        @testset "Monomorphic Variants" begin
            genotypes = zeros(100, 10)
            genotypes[:, 1:5] = rand([0, 1, 2], 100, 5)  # Half polymorphic
            y = randn(100)

            result = burden_test(genotypes, y)
            @test isa(result, RareVariantResult)
        end

        @testset "All Rare" begin
            # Very rare variants (singleton)
            genotypes = zeros(200, 10)
            for j in 1:10
                genotypes[rand(1:200), j] = 1
            end
            y = randn(200)

            result = skat(genotypes, y)
            @test isa(result, RareVariantResult)
        end

        @testset "High Missing Rate" begin
            genotypes = rand([0, 1, 2, missing], 100, 10)
            genotypes = coalesce.(genotypes, 0)  # Replace missing with 0
            y = randn(100)

            result = burden_test(Float64.(genotypes), y)
            @test isa(result, RareVariantResult)
        end
    end

end # @testset "Rare Variant Analysis"
