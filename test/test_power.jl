# ============================================================================
# Comprehensive Tests for Power Calculations Module
# ============================================================================
# Tests for GWAS, rare variant, heritability, and PRS power calculations
# ============================================================================

@testset "Power Calculations" begin

    # ========================================================================
    # PowerResult Structure Tests
    # ========================================================================
    @testset "PowerResult Structure" begin
        result = PowerResult(
            0.85,
            10000,
            Dict(:alpha => 5e-8, :beta => 0.2)
        )

        @test result.power == 0.85
        @test result.sample_size == 10000
        @test result.parameters[:alpha] == 5e-8
    end

    # ========================================================================
    # GWAS Power Tests
    # ========================================================================
    @testset "GWAS Power" begin
        @testset "Basic Power Calculation" begin
            power = gwas_power(
                10000,    # n
                0.3,      # maf
                0.1,      # beta (effect size)
                1.0       # sigma (residual SD)
            )

            @test 0 <= power <= 1
        end

        @testset "Sample Size Effect" begin
            power_small = gwas_power(1000, 0.3, 0.1, 1.0)
            power_large = gwas_power(100000, 0.3, 0.1, 1.0)

            @test power_large > power_small
        end

        @testset "MAF Effect" begin
            power_common = gwas_power(10000, 0.4, 0.1, 1.0)
            power_rare = gwas_power(10000, 0.05, 0.1, 1.0)

            @test power_common > power_rare
        end

        @testset "Effect Size Effect" begin
            power_small = gwas_power(10000, 0.3, 0.05, 1.0)
            power_large = gwas_power(10000, 0.3, 0.2, 1.0)

            @test power_large > power_small
        end

        @testset "Alpha Level" begin
            power_5e8 = gwas_power(10000, 0.3, 0.1, 1.0; alpha=5e-8)
            power_0_05 = gwas_power(10000, 0.3, 0.1, 1.0; alpha=0.05)

            @test power_0_05 > power_5e8
        end

        @testset "Known Result" begin
            # With large sample and effect, power should approach 1
            power = gwas_power(500000, 0.3, 0.2, 1.0)

            @test power > 0.99
        end
    end

    # ========================================================================
    # GWAS Sample Size Tests
    # ========================================================================
    @testset "GWAS Sample Size" begin
        @testset "Basic Sample Size" begin
            n = gwas_sample_size(
                0.80,     # power
                0.3,      # maf
                0.1,      # beta
                1.0       # sigma
            )

            @test n > 0
            @test isa(n, Int)
        end

        @testset "Power Target" begin
            n_80 = gwas_sample_size(0.80, 0.3, 0.1, 1.0)
            n_90 = gwas_sample_size(0.90, 0.3, 0.1, 1.0)

            @test n_90 > n_80
        end

        @testset "Effect Size Impact" begin
            n_small_effect = gwas_sample_size(0.80, 0.3, 0.05, 1.0)
            n_large_effect = gwas_sample_size(0.80, 0.3, 0.2, 1.0)

            @test n_small_effect > n_large_effect
        end

        @testset "Consistency" begin
            # Sample size should give target power
            n = gwas_sample_size(0.80, 0.3, 0.1, 1.0)
            achieved_power = gwas_power(n, 0.3, 0.1, 1.0)

            @test abs(achieved_power - 0.80) < 0.05
        end
    end

    # ========================================================================
    # Case-Control Power Tests
    # ========================================================================
    @testset "Case-Control Power" begin
        @testset "Basic Calculation" begin
            power = case_control_power(
                5000,     # n_cases
                5000,     # n_controls
                0.3,      # maf
                1.3       # odds ratio
            )

            @test 0 <= power <= 1
        end

        @testset "Balanced vs Unbalanced" begin
            power_balanced = case_control_power(5000, 5000, 0.3, 1.3)
            power_unbalanced = case_control_power(2000, 8000, 0.3, 1.3)

            # Balanced design typically more powerful
            @test power_balanced >= power_unbalanced * 0.8
        end

        @testset "Odds Ratio Effect" begin
            power_or_1_2 = case_control_power(5000, 5000, 0.3, 1.2)
            power_or_1_5 = case_control_power(5000, 5000, 0.3, 1.5)

            @test power_or_1_5 > power_or_1_2
        end

        @testset "Disease Prevalence" begin
            power = case_control_power(
                5000, 5000, 0.3, 1.3;
                prevalence=0.01
            )

            @test 0 <= power <= 1
        end

        @testset "Protective Effect" begin
            power_risk = case_control_power(5000, 5000, 0.3, 1.3)
            power_protective = case_control_power(5000, 5000, 0.3, 0.77)

            @test abs(power_risk - power_protective) < 0.1  # Similar power
        end
    end

    # ========================================================================
    # Rare Variant Power Tests
    # ========================================================================
    @testset "Rare Variant Power" begin
        @testset "Basic Calculation" begin
            power = rare_variant_power(
                5000,     # n
                20,       # n_variants
                0.01,     # mean MAF
                0.3       # mean effect
            )

            @test 0 <= power <= 1
        end

        @testset "Number of Variants" begin
            power_few = rare_variant_power(5000, 5, 0.01, 0.3)
            power_many = rare_variant_power(5000, 50, 0.01, 0.3)

            # More variants can increase or decrease power depending on effects
            @test 0 <= power_few <= 1
            @test 0 <= power_many <= 1
        end

        @testset "Causal Proportion" begin
            power_all = rare_variant_power(5000, 20, 0.01, 0.3; causal_prop=1.0)
            power_half = rare_variant_power(5000, 20, 0.01, 0.3; causal_prop=0.5)

            @test power_all >= power_half
        end

        @testset "Test Type" begin
            power_burden = rare_variant_power(5000, 20, 0.01, 0.3; test=:burden)
            power_skat = rare_variant_power(5000, 20, 0.01, 0.3; test=:skat)

            @test 0 <= power_burden <= 1
            @test 0 <= power_skat <= 1
        end

        @testset "Effect Direction" begin
            # SKAT more powerful when effects have mixed directions
            power_skat_mixed = rare_variant_power(5000, 20, 0.01, 0.3;
                                                   test=:skat, mixed_effects=true)
            power_burden_mixed = rare_variant_power(5000, 20, 0.01, 0.3;
                                                    test=:burden, mixed_effects=true)

            @test 0 <= power_skat_mixed <= 1
        end
    end

    # ========================================================================
    # Heritability Power Tests
    # ========================================================================
    @testset "Heritability Power" begin
        @testset "Basic Calculation" begin
            power = heritability_power(
                10000,    # n
                0.5       # h2
            )

            @test 0 <= power <= 1
        end

        @testset "Sample Size Effect" begin
            power_small = heritability_power(1000, 0.5)
            power_large = heritability_power(50000, 0.5)

            @test power_large > power_small
        end

        @testset "Heritability Effect" begin
            power_low = heritability_power(10000, 0.1)
            power_high = heritability_power(10000, 0.8)

            @test power_high > power_low
        end

        @testset "Number of SNPs" begin
            power = heritability_power(10000, 0.5; n_snps=100000)

            @test 0 <= power <= 1
        end

        @testset "LDSC vs GREML" begin
            power_ldsc = heritability_power(50000, 0.5; method=:ldsc)
            power_greml = heritability_power(5000, 0.5; method=:greml)

            @test 0 <= power_ldsc <= 1
            @test 0 <= power_greml <= 1
        end
    end

    # ========================================================================
    # PRS Power Tests
    # ========================================================================
    @testset "PRS Power" begin
        @testset "Basic Calculation" begin
            power = prs_power(
                100000,   # n_gwas
                10000,    # n_target
                0.5,      # h2
                0.2       # expected r2
            )

            @test 0 <= power <= 1
        end

        @testset "GWAS Size Effect" begin
            power_small = prs_power(10000, 5000, 0.5, 0.1)
            power_large = prs_power(500000, 5000, 0.5, 0.3)

            @test power_large > power_small
        end

        @testset "Target Size Effect" begin
            power_small = prs_power(100000, 1000, 0.5, 0.2)
            power_large = prs_power(100000, 50000, 0.5, 0.2)

            @test power_large > power_small
        end

        @testset "Heritability Ceiling" begin
            # R² cannot exceed h² in expectation
            power = prs_power(100000, 10000, 0.3, 0.3)

            @test power <= 1
        end

        @testset "AUC Target" begin
            power = prs_power(100000, 10000, 0.5, 0.2; auc_target=0.65)

            @test 0 <= power <= 1
        end
    end

    # ========================================================================
    # Genetic Correlation Power Tests
    # ========================================================================
    @testset "Genetic Correlation Power" begin
        @testset "Basic Calculation" begin
            power = genetic_correlation_power(
                50000,    # n1
                50000,    # n2
                0.5,      # rg
                0.3,      # h2_1
                0.3       # h2_2
            )

            @test 0 <= power <= 1
        end

        @testset "Sample Size Effect" begin
            power_small = genetic_correlation_power(10000, 10000, 0.5, 0.3, 0.3)
            power_large = genetic_correlation_power(100000, 100000, 0.5, 0.3, 0.3)

            @test power_large > power_small
        end

        @testset "Correlation Strength" begin
            power_low = genetic_correlation_power(50000, 50000, 0.2, 0.3, 0.3)
            power_high = genetic_correlation_power(50000, 50000, 0.8, 0.3, 0.3)

            @test power_high > power_low
        end

        @testset "Heritability Effect" begin
            power_low_h2 = genetic_correlation_power(50000, 50000, 0.5, 0.1, 0.1)
            power_high_h2 = genetic_correlation_power(50000, 50000, 0.5, 0.5, 0.5)

            @test power_high_h2 > power_low_h2
        end
    end

    # ========================================================================
    # Fine-Mapping Power Tests
    # ========================================================================
    @testset "Fine-Mapping Power" begin
        @testset "Basic Calculation" begin
            power = finemapping_power(
                10000,    # n
                100,      # n_variants in region
                0.1,      # causal effect
                0.3       # causal MAF
            )

            @test 0 <= power <= 1
        end

        @testset "Sample Size Effect" begin
            power_small = finemapping_power(5000, 100, 0.1, 0.3)
            power_large = finemapping_power(100000, 100, 0.1, 0.3)

            @test power_large > power_small
        end

        @testset "LD Pattern" begin
            power_low_ld = finemapping_power(50000, 100, 0.1, 0.3; mean_r2=0.2)
            power_high_ld = finemapping_power(50000, 100, 0.1, 0.3; mean_r2=0.8)

            @test power_low_ld > power_high_ld  # Lower LD = better resolution
        end

        @testset "Coverage Target" begin
            power_95 = finemapping_power(50000, 100, 0.1, 0.3; coverage=0.95)
            power_99 = finemapping_power(50000, 100, 0.1, 0.3; coverage=0.99)

            @test power_95 >= power_99
        end
    end

    # ========================================================================
    # Visualization Data Tests
    # ========================================================================
    @testset "Visualization Data" begin
        @testset "Power Summary Plot" begin
            result = power_summary_plot_data(
                maf=0.3,
                effect_size=0.1,
                sigma=1.0,
                sample_sizes=[1000, 5000, 10000, 50000, 100000]
            )

            @test isa(result, DataFrame)
            @test :n in propertynames(result)
            @test :power in propertynames(result)
            @test nrow(result) == 5
        end

        @testset "Sample Size Table" begin
            result = sample_size_table(
                power_target=0.80,
                mafs=[0.1, 0.2, 0.3, 0.4, 0.5],
                effect_sizes=[0.05, 0.1, 0.15, 0.2]
            )

            @test isa(result, DataFrame)
            @test nrow(result) == 20  # 5 MAFs × 4 effect sizes
        end
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================
    @testset "Edge Cases" begin
        @testset "Very Small Effect" begin
            power = gwas_power(100000, 0.3, 0.001, 1.0)

            @test 0 <= power <= 1
            @test power < 0.1  # Should be low
        end

        @testset "Very Large Sample" begin
            power = gwas_power(10000000, 0.3, 0.05, 1.0)

            @test power > 0.99
        end

        @testset "Rare Variant" begin
            power = gwas_power(100000, 0.001, 0.5, 1.0)

            @test 0 <= power <= 1
        end

        @testset "Zero Heritability" begin
            power = heritability_power(100000, 0.0)

            @test power < 0.1  # Should be near 0
        end

        @testset "Perfect Heritability" begin
            power = heritability_power(10000, 1.0)

            @test power > 0.9
        end
    end

    # ========================================================================
    # Integration Tests
    # ========================================================================
    @testset "Integration" begin
        @testset "Study Design" begin
            # Design a GWAS study
            target_power = 0.80
            maf = 0.3
            effect = 0.1

            # Calculate required sample size
            n_required = gwas_sample_size(target_power, maf, effect, 1.0)

            # Verify power at that sample size
            achieved_power = gwas_power(n_required, maf, effect, 1.0)

            @test abs(achieved_power - target_power) < 0.05
        end

        @testset "Multi-Stage Design" begin
            # Stage 1: Discovery
            n_discovery = 50000
            power_discovery = gwas_power(n_discovery, 0.3, 0.1, 1.0)

            # Stage 2: Replication (smaller sample)
            n_replication = 10000
            power_replication = gwas_power(n_replication, 0.3, 0.1, 1.0; alpha=0.05)

            # Combined power
            power_combined = power_discovery * power_replication

            @test 0 <= power_combined <= 1
        end

        @testset "Power Across Parameters" begin
            # Generate power curve across MAFs
            mafs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            powers = [gwas_power(50000, m, 0.1, 1.0) for m in mafs]

            # Power should increase with MAF (for fixed effect)
            @test issorted(powers)
        end
    end

end # @testset "Power Calculations"
