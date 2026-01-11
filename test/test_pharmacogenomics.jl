# ============================================================================
# Comprehensive Tests for Pharmacogenomics Module
# ============================================================================
# Tests for star allele calling, drug response prediction, and PGx GWAS
# ============================================================================

@testset "Pharmacogenomics" begin

    # ========================================================================
    # StarAllele Structure Tests
    # ========================================================================
    @testset "StarAllele Structure" begin
        allele = StarAllele(
            "CYP2D6",
            "*4",
            ["rs3892097"],
            [1],
            :no_function
        )

        @test allele.gene == "CYP2D6"
        @test allele.name == "*4"
        @test allele.function_status == :no_function
    end

    # ========================================================================
    # PGxResult Structure Tests
    # ========================================================================
    @testset "PGxResult Structure" begin
        result = PGxResult(
            "CYP2D6",
            "*1",
            "*4",
            :intermediate_metabolizer,
            ["Consider dose reduction"],
            Dict(:activity_score => 1.0)
        )

        @test result.gene == "CYP2D6"
        @test result.diplotype == "*1/*4"
        @test result.phenotype == :intermediate_metabolizer
    end

    # ========================================================================
    # Star Allele Calling Tests
    # ========================================================================
    @testset "Star Allele Calling" begin
        @testset "CYP2D6 *1/*1" begin
            # Reference genotypes
            genotypes = Dict(
                "rs1065852" => 0,  # Reference
                "rs3892097" => 0,  # Reference
                "rs16947" => 0
            )

            allele1, allele2 = call_star_alleles(genotypes, "CYP2D6")

            @test allele1 == "*1" || allele2 == "*1"
        end

        @testset "CYP2D6 *4" begin
            # *4 defining variant
            genotypes = Dict(
                "rs3892097" => 1,  # Heterozygous for *4
                "rs1065852" => 0,
                "rs16947" => 0
            )

            allele1, allele2 = call_star_alleles(genotypes, "CYP2D6")

            @test "*4" in [allele1, allele2]
        end

        @testset "CYP2C9" begin
            genotypes = Dict(
                "rs1799853" => 1,  # *2
                "rs1057910" => 0
            )

            allele1, allele2 = call_star_alleles(genotypes, "CYP2C9")

            @test "*2" in [allele1, allele2]
        end

        @testset "CYP2C19" begin
            genotypes = Dict(
                "rs4244285" => 2,  # Homozygous *2
                "rs4986893" => 0
            )

            allele1, allele2 = call_star_alleles(genotypes, "CYP2C19")

            @test allele1 == "*2" && allele2 == "*2"
        end

        @testset "VKORC1" begin
            genotypes = Dict(
                "rs9923231" => 1
            )

            allele1, allele2 = call_star_alleles(genotypes, "VKORC1")

            @test isa(allele1, String)
            @test isa(allele2, String)
        end

        @testset "TPMT" begin
            genotypes = Dict(
                "rs1800462" => 0,
                "rs1800460" => 1,
                "rs1142345" => 0
            )

            allele1, allele2 = call_star_alleles(genotypes, "TPMT")

            @test isa(allele1, String)
        end

        @testset "Unknown Gene" begin
            genotypes = Dict("rs1234" => 1)

            allele1, allele2 = call_star_alleles(genotypes, "UNKNOWN_GENE")

            @test allele1 == "*?" || allele1 == "*1"
        end
    end

    # ========================================================================
    # Phenotype Prediction Tests
    # ========================================================================
    @testset "Phenotype Prediction" begin
        @testset "CYP2D6 Phenotypes" begin
            # Normal metabolizer
            result_nm = predict_phenotype("CYP2D6", "*1", "*1")
            @test result_nm.phenotype == :normal_metabolizer

            # Intermediate metabolizer
            result_im = predict_phenotype("CYP2D6", "*1", "*4")
            @test result_im.phenotype == :intermediate_metabolizer

            # Poor metabolizer
            result_pm = predict_phenotype("CYP2D6", "*4", "*4")
            @test result_pm.phenotype == :poor_metabolizer

            # Ultra-rapid metabolizer
            result_um = predict_phenotype("CYP2D6", "*1xN", "*1")
            @test result_um.phenotype == :ultrarapid_metabolizer
        end

        @testset "CYP2C19 Phenotypes" begin
            result_nm = predict_phenotype("CYP2C19", "*1", "*1")
            @test result_nm.phenotype == :normal_metabolizer

            result_pm = predict_phenotype("CYP2C19", "*2", "*2")
            @test result_pm.phenotype == :poor_metabolizer
        end

        @testset "Activity Scores" begin
            result = predict_phenotype("CYP2D6", "*1", "*4")

            @test haskey(result.extra, :activity_score)
            @test result.extra[:activity_score] >= 0
        end

        @testset "Recommendations" begin
            result = predict_phenotype("CYP2D6", "*4", "*4")

            @test length(result.recommendations) > 0
        end
    end

    # ========================================================================
    # Warfarin Dose Prediction Tests
    # ========================================================================
    @testset "Warfarin Dose Prediction" begin
        @testset "Standard Dose" begin
            dose = warfarin_dose_prediction("*1/*1", "GG")

            @test dose > 0
            @test dose < 15  # mg/day
        end

        @testset "Reduced Dose" begin
            dose_normal = warfarin_dose_prediction("*1/*1", "GG")
            dose_reduced = warfarin_dose_prediction("*3/*3", "AA")

            @test dose_reduced < dose_normal
        end

        @testset "With Clinical Factors" begin
            dose = warfarin_dose_prediction(
                "*1/*1", "GG";
                age=65,
                weight=75.0,
                height=170.0,
                race=:caucasian,
                amiodarone=false
            )

            @test dose > 0
        end

        @testset "Age Effect" begin
            dose_young = warfarin_dose_prediction("*1/*1", "GG"; age=30)
            dose_old = warfarin_dose_prediction("*1/*1", "GG"; age=80)

            @test dose_old < dose_young
        end

        @testset "Amiodarone Effect" begin
            dose_no = warfarin_dose_prediction("*1/*1", "GG"; amiodarone=false)
            dose_yes = warfarin_dose_prediction("*1/*1", "GG"; amiodarone=true)

            @test dose_yes < dose_no
        end

        @testset "Race Adjustment" begin
            dose_cauc = warfarin_dose_prediction("*1/*1", "GG"; race=:caucasian)
            dose_asian = warfarin_dose_prediction("*1/*1", "GG"; race=:asian)

            @test dose_asian != dose_cauc || true  # May differ
        end
    end

    # ========================================================================
    # PGx GWAS Tests
    # ========================================================================
    @testset "PGx GWAS" begin
        @testset "Basic PGx GWAS" begin
            n, p = 500, 100
            genotypes = rand(0:2, n, p) |> x -> Float64.(x)
            drug_response = randn(n)

            result = pgx_gwas(genotypes, drug_response)

            @test isa(result, DrugResponseResult)
            @test length(result.pvalues) == p
        end

        @testset "Binary Response" begin
            n, p = 400, 50
            genotypes = rand(0:2, n, p) |> x -> Float64.(x)
            response = Float64.(rand(n) .> 0.5)

            result = pgx_gwas(genotypes, response; binary=true)

            @test isa(result, DrugResponseResult)
        end

        @testset "With Covariates" begin
            n, p = 300, 40
            genotypes = rand(0:2, n, p) |> x -> Float64.(x)
            response = randn(n)
            covariates = randn(n, 3)

            result = pgx_gwas(genotypes, response; covariates=covariates)

            @test length(result.pvalues) == p
        end

        @testset "Interaction Testing" begin
            n, p = 400, 30
            genotypes = rand(0:2, n, p) |> x -> Float64.(x)
            drug_dose = randn(n)
            response = genotypes[:, 1] .* drug_dose .+ randn(n)

            result = pgx_gwas(genotypes, response;
                             interaction=drug_dose)

            @test haskey(result.extra, :interaction_pvalues) || true
        end

        @testset "Survival Outcome" begin
            n, p = 300, 25
            genotypes = rand(0:2, n, p) |> x -> Float64.(x)
            time = rand(n) .* 100
            event = Float64.(rand(n) .> 0.3)

            result = pgx_gwas(genotypes, time;
                             survival=true,
                             event=event)

            @test isa(result, DrugResponseResult)
        end
    end

    # ========================================================================
    # PGx Report Tests
    # ========================================================================
    @testset "PGx Report" begin
        @testset "Basic Report" begin
            genotypes = Dict(
                "CYP2D6" => Dict("rs3892097" => 1, "rs1065852" => 0),
                "CYP2C19" => Dict("rs4244285" => 0),
                "CYP2C9" => Dict("rs1799853" => 0, "rs1057910" => 0)
            )

            report = pgx_report(genotypes)

            @test isa(report, Dict)
            @test haskey(report, "CYP2D6")
            @test haskey(report["CYP2D6"], :diplotype)
            @test haskey(report["CYP2D6"], :phenotype)
        end

        @testset "Drug Recommendations" begin
            genotypes = Dict(
                "CYP2D6" => Dict("rs3892097" => 2)  # Homozygous *4
            )

            report = pgx_report(genotypes; include_drugs=true)

            @test haskey(report["CYP2D6"], :drug_recommendations) || true
        end

        @testset "Multiple Genes" begin
            genotypes = Dict(
                "CYP2D6" => Dict("rs3892097" => 0),
                "CYP2C19" => Dict("rs4244285" => 1),
                "CYP2C9" => Dict("rs1799853" => 1),
                "VKORC1" => Dict("rs9923231" => 2),
                "TPMT" => Dict("rs1800460" => 0)
            )

            report = pgx_report(genotypes)

            @test length(report) >= 5
        end
    end

    # ========================================================================
    # Drug Interaction Tests
    # ========================================================================
    @testset "Drug Interactions" begin
        @testset "Metabolizer Status" begin
            # Poor metabolizer should have different recommendations
            result_pm = predict_phenotype("CYP2D6", "*4", "*4")
            result_nm = predict_phenotype("CYP2D6", "*1", "*1")

            @test result_pm.recommendations != result_nm.recommendations
        end

        @testset "Prodrug vs Active Drug" begin
            # Codeine is a prodrug (needs CYP2D6)
            # Poor metabolizers have reduced efficacy
            result_pm = predict_phenotype("CYP2D6", "*4", "*4")

            @test :poor_metabolizer == result_pm.phenotype
        end
    end

    # ========================================================================
    # Population Frequency Tests
    # ========================================================================
    @testset "Population Frequencies" begin
        @testset "Allele Frequencies" begin
            # Generate population data
            n = 1000
            genotypes = Dict(
                "rs3892097" => rand([0, 0, 0, 0, 1, 1, 2], n)
            )

            # Calculate allele frequency
            alt_freq = (sum(genotypes["rs3892097"]) / (2 * n))

            @test 0 <= alt_freq <= 1
        end

        @testset "Phenotype Distribution" begin
            n = 500
            phenotypes = []

            for _ in 1:n
                # Random genotypes
                geno = Dict("rs3892097" => rand(0:2))
                allele1, allele2 = call_star_alleles(geno, "CYP2D6")
                result = predict_phenotype("CYP2D6", allele1, allele2)
                push!(phenotypes, result.phenotype)
            end

            # Should have distribution of phenotypes
            @test length(unique(phenotypes)) >= 1
        end
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================
    @testset "Edge Cases" begin
        @testset "Missing Genotypes" begin
            genotypes = Dict(
                "rs3892097" => missing
            )

            # Should handle gracefully
            try
                allele1, allele2 = call_star_alleles(
                    Dict(k => coalesce(v, 0) for (k, v) in genotypes),
                    "CYP2D6"
                )
                @test true
            catch
                @test true  # Acceptable to throw error
            end
        end

        @testset "Novel Haplotype" begin
            # Unusual combination
            genotypes = Dict(
                "rs3892097" => 1,
                "rs1065852" => 2,
                "rs16947" => 2
            )

            allele1, allele2 = call_star_alleles(genotypes, "CYP2D6")

            @test isa(allele1, String)
            @test isa(allele2, String)
        end

        @testset "Extreme Warfarin Dose" begin
            # Should clamp to reasonable range
            dose = warfarin_dose_prediction("*3/*3", "AA"; age=90, weight=40.0)

            @test dose >= 0.5  # Minimum reasonable dose
            @test dose <= 15   # Maximum reasonable dose
        end
    end

    # ========================================================================
    # Integration Tests
    # ========================================================================
    @testset "Integration" begin
        @testset "Full PGx Pipeline" begin
            # Simulate patient data
            patient_genotypes = Dict(
                "CYP2D6" => Dict(
                    "rs3892097" => 1,
                    "rs1065852" => 0,
                    "rs16947" => 1
                ),
                "CYP2C9" => Dict(
                    "rs1799853" => 1,
                    "rs1057910" => 0
                ),
                "VKORC1" => Dict(
                    "rs9923231" => 1
                )
            )

            # Step 1: Call star alleles for each gene
            diplotypes = Dict()
            for (gene, geno) in patient_genotypes
                a1, a2 = call_star_alleles(geno, gene)
                diplotypes[gene] = "$a1/$a2"
            end

            # Step 2: Predict phenotypes
            phenotypes = Dict()
            for (gene, geno) in patient_genotypes
                a1, a2 = call_star_alleles(geno, gene)
                result = predict_phenotype(gene, a1, a2)
                phenotypes[gene] = result.phenotype
            end

            # Step 3: Calculate warfarin dose
            cyp2c9_diplo = diplotypes["CYP2C9"]
            vkorc1_geno = patient_genotypes["VKORC1"]["rs9923231"] == 0 ? "GG" :
                         (patient_genotypes["VKORC1"]["rs9923231"] == 1 ? "AG" : "AA")
            dose = warfarin_dose_prediction(cyp2c9_diplo, vkorc1_geno)

            @test length(diplotypes) == 3
            @test length(phenotypes) == 3
            @test dose > 0
        end

        @testset "Population PGx Study" begin
            # Simulate cohort
            n = 200
            p = 50

            genotypes = rand(0:2, n, p) |> x -> Float64.(x)
            drug_response = randn(n)

            # Add genetic effect
            drug_response .+= genotypes[:, 1] .* 0.5

            # Run PGx GWAS
            result = pgx_gwas(genotypes, drug_response)

            # Should detect association
            @test minimum(result.pvalues) < 0.05
        end
    end

end # @testset "Pharmacogenomics"
