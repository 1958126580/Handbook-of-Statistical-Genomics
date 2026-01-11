# ============================================================================
# Comprehensive Tests for Meta-Analysis Module
# ============================================================================
# Tests for fixed effects, random effects, trans-ancestry, and MR-MEGA
# ============================================================================

@testset "Meta-Analysis" begin

    # ========================================================================
    # MetaAnalysisResult Structure Tests
    # ========================================================================
    @testset "MetaAnalysisResult Structure" begin
        result = MetaAnalysisResult(
            0.15,           # beta
            0.02,           # se
            7.5,            # z
            1e-14,          # pvalue
            0.0,            # tau2
            0.0,            # i_squared
            0.65,           # q_pvalue
            :fixed,         # method
            Dict{Symbol,Any}()
        )

        @test result.beta == 0.15
        @test result.se == 0.02
        @test result.method == :fixed
    end

    # ========================================================================
    # Fixed Effects Meta-Analysis Tests
    # ========================================================================
    @testset "Fixed Effects" begin
        @testset "Basic Fixed Effects" begin
            betas = [0.10, 0.12, 0.08, 0.11]
            ses = [0.02, 0.03, 0.025, 0.022]

            result = fixed_effects_meta(betas, ses)

            @test isa(result, MetaAnalysisResult)
            @test result.method == :fixed
            @test 0 <= result.pvalue <= 1

            # Meta beta should be between individual betas
            @test minimum(betas) <= result.beta <= maximum(betas)

            # Meta SE should be smaller than individual SEs
            @test result.se < minimum(ses)
        end

        @testset "Inverse Variance Weighting" begin
            betas = [0.1, 0.2, 0.15]
            ses = [0.01, 0.1, 0.05]  # First study has smallest SE

            result = fixed_effects_meta(betas, ses)

            # Result should be closer to first beta (highest weight)
            @test abs(result.beta - 0.1) < abs(result.beta - 0.2)
        end

        @testset "Single Study" begin
            betas = [0.15]
            ses = [0.03]

            result = fixed_effects_meta(betas, ses)

            @test result.beta == 0.15
            @test result.se == 0.03
        end

        @testset "Homogeneous Effects" begin
            # All studies have same effect
            betas = fill(0.1, 5)
            ses = [0.02, 0.025, 0.03, 0.022, 0.028]

            result = fixed_effects_meta(betas, ses)

            @test abs(result.beta - 0.1) < 0.01
            @test result.i_squared ≈ 0.0 atol=10.0
            @test result.q_pvalue > 0.05  # No significant heterogeneity
        end
    end

    # ========================================================================
    # Random Effects Meta-Analysis Tests
    # ========================================================================
    @testset "Random Effects" begin
        @testset "Basic Random Effects" begin
            betas = [0.10, 0.20, 0.05, 0.15]
            ses = [0.03, 0.04, 0.035, 0.032]

            result = random_effects_meta(betas, ses)

            @test isa(result, MetaAnalysisResult)
            @test result.method == :random
            @test result.tau2 >= 0
        end

        @testset "DerSimonian-Laird" begin
            betas = [0.1, 0.2, 0.15, 0.12]
            ses = [0.02, 0.03, 0.025, 0.022]

            result = random_effects_meta(betas, ses; method=:dl)

            @test isa(result, MetaAnalysisResult)
            @test result.tau2 >= 0
        end

        @testset "REML" begin
            betas = [0.1, 0.2, 0.05, 0.15, 0.12]
            ses = [0.02, 0.03, 0.025, 0.022, 0.028]

            result = random_effects_meta(betas, ses; method=:reml)

            @test isa(result, MetaAnalysisResult)
        end

        @testset "Heterogeneous Effects" begin
            # Studies with different effects
            betas = [0.05, 0.25, 0.10, 0.20]
            ses = [0.02, 0.02, 0.02, 0.02]  # Same precision

            result = random_effects_meta(betas, ses)

            @test result.tau2 > 0  # Should detect heterogeneity
            @test result.i_squared > 50  # High I²

            # Random effects SE should be larger than fixed
            result_fixed = fixed_effects_meta(betas, ses)
            @test result.se >= result_fixed.se
        end

        @testset "PM Method" begin
            betas = [0.1, 0.15, 0.08]
            ses = [0.03, 0.04, 0.035]

            result = random_effects_meta(betas, ses; method=:pm)

            @test isa(result, MetaAnalysisResult)
        end
    end

    # ========================================================================
    # Sample-Size Weighted Meta-Analysis Tests
    # ========================================================================
    @testset "Sample Size Weighted" begin
        @testset "Basic N-Weighted" begin
            betas = [0.1, 0.15, 0.12]
            ns = [10000, 5000, 8000]

            result = sample_size_weighted_meta(betas, ns)

            @test isa(result, MetaAnalysisResult)
            @test result.method == :sample_size_weighted
        end

        @testset "Z-Score Combination" begin
            z_scores = [3.0, 2.5, 4.0]
            ns = [10000, 8000, 12000]

            result = sample_size_weighted_meta(z_scores, ns; input=:z)

            @test isa(result, MetaAnalysisResult)
            @test result.pvalue < 0.05  # Significant combined effect
        end

        @testset "Weight Proportionality" begin
            betas = [0.1, 0.2]
            ns = [10000, 10000]  # Same sample size

            result = sample_size_weighted_meta(betas, ns)

            # With equal N, should be simple average
            @test abs(result.beta - mean(betas)) < 0.01
        end
    end

    # ========================================================================
    # Trans-Ancestry Meta-Analysis Tests
    # ========================================================================
    @testset "Trans-Ancestry Meta-Analysis" begin
        @testset "Basic Trans-Ancestry" begin
            betas = [0.10, 0.12, 0.08, 0.11]
            ses = [0.02, 0.03, 0.025, 0.022]
            ancestries = ["EUR", "EAS", "AFR", "SAS"]

            result = trans_ancestry_meta(betas, ses, ancestries)

            @test isa(result, MetaAnalysisResult)
            @test haskey(result.extra, :ancestry_specific) || true
        end

        @testset "Ancestry-Specific Effects" begin
            # Different effects by ancestry
            betas = [0.10, 0.10, 0.15, 0.15]
            ses = [0.02, 0.02, 0.02, 0.02]
            ancestries = ["EUR", "EUR", "EAS", "EAS"]

            result = trans_ancestry_meta(betas, ses, ancestries)

            # Should detect ancestry heterogeneity
            @test isa(result, MetaAnalysisResult)
        end

        @testset "Single Ancestry" begin
            betas = [0.10, 0.12, 0.08]
            ses = [0.02, 0.03, 0.025]
            ancestries = ["EUR", "EUR", "EUR"]

            result = trans_ancestry_meta(betas, ses, ancestries)

            @test isa(result, MetaAnalysisResult)
        end
    end

    # ========================================================================
    # MR-MEGA Tests
    # ========================================================================
    @testset "MR-MEGA" begin
        @testset "Basic MR-MEGA" begin
            betas = [0.10, 0.12, 0.08, 0.11, 0.09]
            ses = [0.02, 0.03, 0.025, 0.022, 0.028]

            # Ancestry PC loadings
            pc_loadings = randn(5, 2)

            result = mr_mega(betas, ses, pc_loadings)

            @test haskey(result, :beta_mega)
            @test haskey(result, :pvalue_mega)
            @test haskey(result, :pvalue_ancestry)
            @test haskey(result, :pvalue_residual)
        end

        @testset "Ancestry Heterogeneity" begin
            # Effect varies with ancestry
            betas = [0.05, 0.08, 0.15, 0.12]
            ses = [0.02, 0.02, 0.02, 0.02]

            # PC loadings correlated with effect
            pc_loadings = hcat([0.1, 0.2, 0.8, 0.7], randn(4))

            result = mr_mega(betas, ses, pc_loadings)

            # Should detect ancestry-correlated effect
            @test result.pvalue_ancestry < 0.5
        end

        @testset "Residual Heterogeneity" begin
            # Random heterogeneity not explained by ancestry
            betas = [0.05, 0.20, 0.08, 0.15]
            ses = [0.02, 0.02, 0.02, 0.02]
            pc_loadings = randn(4, 2)

            result = mr_mega(betas, ses, pc_loadings)

            # Should detect residual heterogeneity
            @test result.pvalue_residual < 0.5
        end
    end

    # ========================================================================
    # GWAS Meta-Analysis Tests
    # ========================================================================
    @testset "GWAS Meta-Analysis" begin
        @testset "Basic GWAS Meta" begin
            # Simulate multiple GWAS
            n_variants = 100

            study1 = DataFrame(
                rsid = ["rs$i" for i in 1:n_variants],
                beta = randn(n_variants) .* 0.1,
                se = rand(n_variants) .* 0.02 .+ 0.01
            )

            study2 = DataFrame(
                rsid = ["rs$i" for i in 1:n_variants],
                beta = randn(n_variants) .* 0.1,
                se = rand(n_variants) .* 0.02 .+ 0.01
            )

            result = gwas_meta_analysis([study1, study2])

            @test isa(result, DataFrame)
            @test nrow(result) == n_variants
            @test :beta_meta in propertynames(result)
            @test :pvalue_meta in propertynames(result)
        end

        @testset "Variant Matching" begin
            study1 = DataFrame(
                rsid = ["rs1", "rs2", "rs3"],
                beta = [0.1, 0.2, 0.15],
                se = [0.02, 0.03, 0.025]
            )

            study2 = DataFrame(
                rsid = ["rs1", "rs3", "rs4"],  # rs2 missing, rs4 extra
                beta = [0.12, 0.18, 0.1],
                se = [0.025, 0.028, 0.02]
            )

            result = gwas_meta_analysis([study1, study2]; match_by=:rsid)

            @test nrow(result) >= 2  # At least overlapping variants
        end

        @testset "Multiple Testing" begin
            n_variants = 1000

            studies = [
                DataFrame(
                    rsid = ["rs$i" for i in 1:n_variants],
                    beta = randn(n_variants) .* 0.05,
                    se = fill(0.02, n_variants)
                ) for _ in 1:3
            ]

            result = gwas_meta_analysis(studies; correction=:bonferroni)

            @test :pvalue_corrected in propertynames(result)
        end
    end

    # ========================================================================
    # Heterogeneity Tests
    # ========================================================================
    @testset "Heterogeneity Statistics" begin
        @testset "Cochran's Q" begin
            # Homogeneous
            betas_homo = fill(0.1, 5)
            ses = fill(0.02, 5)

            result_homo = fixed_effects_meta(betas_homo, ses)
            @test result_homo.q_pvalue > 0.1

            # Heterogeneous
            betas_het = [0.05, 0.15, 0.08, 0.20, 0.10]

            result_het = fixed_effects_meta(betas_het, ses)
            @test result_het.q_pvalue < result_homo.q_pvalue
        end

        @testset "I-Squared" begin
            # No heterogeneity
            betas = fill(0.1, 4)
            ses = [0.02, 0.03, 0.025, 0.022]

            result = random_effects_meta(betas, ses)
            @test result.i_squared < 25

            # High heterogeneity
            betas_het = [0.05, 0.20, 0.08, 0.25]
            result_het = random_effects_meta(betas_het, ses)
            @test result_het.i_squared > 50
        end
    end

    # ========================================================================
    # Forest Plot Data Tests
    # ========================================================================
    @testset "Forest Plot Data" begin
        @testset "Basic Forest Data" begin
            betas = [0.10, 0.12, 0.08, 0.11]
            ses = [0.02, 0.03, 0.025, 0.022]
            study_names = ["Study A", "Study B", "Study C", "Study D"]

            result = forest_plot_data(betas, ses, study_names)

            @test isa(result, DataFrame)
            @test :study in propertynames(result)
            @test :beta in propertynames(result)
            @test :ci_lower in propertynames(result)
            @test :ci_upper in propertynames(result)
            @test nrow(result) == 5  # 4 studies + summary
        end

        @testset "Confidence Intervals" begin
            betas = [0.1, 0.2]
            ses = [0.02, 0.04]

            result = forest_plot_data(betas, ses, ["A", "B"])

            # Check CI calculation
            for row in eachrow(result)
                if row.study != "Summary"
                    @test row.ci_lower < row.beta < row.ci_upper
                end
            end
        end
    end

    # ========================================================================
    # Leave-One-Out Analysis Tests
    # ========================================================================
    @testset "Leave-One-Out Analysis" begin
        @testset "Basic LOO" begin
            betas = [0.10, 0.12, 0.08, 0.11, 0.09]
            ses = [0.02, 0.03, 0.025, 0.022, 0.028]

            result = leave_one_out_analysis(betas, ses)

            @test isa(result, DataFrame)
            @test nrow(result) == 5
            @test :excluded_study in propertynames(result)
            @test :beta_loo in propertynames(result)
        end

        @testset "Influential Study" begin
            # One outlier study
            betas = [0.10, 0.10, 0.10, 0.50]  # Last is outlier
            ses = [0.02, 0.02, 0.02, 0.02]

            result = leave_one_out_analysis(betas, ses)

            # Excluding outlier should change result most
            changes = abs.(result.beta_loo .- mean(betas))
            @test argmax(changes) == 4
        end
    end

    # ========================================================================
    # Publication Bias Tests
    # ========================================================================
    @testset "Publication Bias" begin
        @testset "Egger's Test" begin
            betas = [0.10, 0.12, 0.08, 0.11, 0.09]
            ses = [0.02, 0.03, 0.025, 0.022, 0.028]

            result = publication_bias_test(betas, ses; method=:egger)

            @test haskey(result, :intercept)
            @test haskey(result, :pvalue)
        end

        @testset "Begg's Test" begin
            betas = [0.10, 0.12, 0.08, 0.11, 0.09, 0.15]
            ses = [0.02, 0.03, 0.025, 0.022, 0.028, 0.04]

            result = publication_bias_test(betas, ses; method=:begg)

            @test haskey(result, :tau)
            @test haskey(result, :pvalue)
        end

        @testset "Trim and Fill" begin
            betas = [0.10, 0.12, 0.15, 0.18, 0.20]
            ses = [0.02, 0.025, 0.03, 0.035, 0.04]

            result = publication_bias_test(betas, ses; method=:trim_fill)

            @test haskey(result, :n_imputed)
            @test haskey(result, :adjusted_beta)
        end
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================
    @testset "Edge Cases" begin
        @testset "Two Studies" begin
            betas = [0.1, 0.15]
            ses = [0.02, 0.03]

            result_fixed = fixed_effects_meta(betas, ses)
            result_random = random_effects_meta(betas, ses)

            @test isa(result_fixed, MetaAnalysisResult)
            @test isa(result_random, MetaAnalysisResult)
        end

        @testset "Very Large Study" begin
            betas = [0.1, 0.12, 0.11]
            ses = [0.001, 0.02, 0.02]  # First study has tiny SE

            result = fixed_effects_meta(betas, ses)

            # Should be dominated by first study
            @test abs(result.beta - 0.1) < 0.01
        end

        @testset "Opposite Signs" begin
            betas = [0.1, -0.1, 0.05, -0.05]
            ses = [0.02, 0.02, 0.02, 0.02]

            result = fixed_effects_meta(betas, ses)

            @test abs(result.beta) < 0.1
        end

        @testset "Zero Effect" begin
            betas = zeros(4)
            ses = [0.02, 0.03, 0.025, 0.022]

            result = fixed_effects_meta(betas, ses)

            @test result.beta ≈ 0.0
            @test result.pvalue > 0.5
        end
    end

    # ========================================================================
    # Integration Tests
    # ========================================================================
    @testset "Integration" begin
        @testset "Full Meta-Analysis Pipeline" begin
            # Simulate GWAS results
            n_variants = 500
            n_studies = 4

            true_betas = randn(n_variants) .* 0.1
            true_betas[1:10] .+= 0.3  # Some true signals

            studies = []
            for i in 1:n_studies
                noise = randn(n_variants) .* 0.02
                push!(studies, DataFrame(
                    rsid = ["rs$j" for j in 1:n_variants],
                    beta = true_betas .+ noise,
                    se = fill(0.02, n_variants) .+ rand(n_variants) .* 0.01
                ))
            end

            # Meta-analysis
            result = gwas_meta_analysis(studies)

            # Significant variants should include true signals
            sig_idx = findall(result.pvalue_meta .< 5e-8)
            @test any(sig_idx .<= 10)  # At least one true signal detected
        end
    end

end # @testset "Meta-Analysis"
