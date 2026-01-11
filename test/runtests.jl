# ============================================================================
# Test Suite for StatisticalGenomics.jl
# ============================================================================

using Test
using StatisticalGenomics
using Statistics
using Random
using LinearAlgebra

# Set random seed for reproducibility
Random.seed!(12345)

@testset "StatisticalGenomics.jl Tests" begin

    # ========================================================================
    # Core Types Tests
    # ========================================================================
    @testset "Core Types" begin
        @testset "GenotypeMatrix" begin
            # Create test data
            data = [0 1 2; 1 1 0; missing 2 1]
            gm = GenotypeMatrix(data, 
                               ["S1", "S2", "S3"], 
                               ["rs1", "rs2", "rs3"])
            
            @test n_samples(gm) == 3
            @test n_variants(gm) == 3
            
            # Test MAF calculation
            mafs = minor_allele_frequency(gm)
            @test length(mafs) == 3
            @test all(0 .<= mafs .<= 0.5)
        end
        
        @testset "Phenotypes" begin
            # Continuous phenotype
            values = [1.0, 2.5, 3.0, 4.5, 5.0]
            pheno = ContinuousPhenotype(values, "BMI")
            @test length(pheno.values) == 5
            
            # Standardization
            std_pheno = standardize(pheno)
            @test abs(mean(std_pheno.values)) < 1e-10
            @test abs(std(std_pheno.values) - 1.0) < 1e-10
            
            # Binary phenotype
            binary = BinaryPhenotype([true, false, true, true, false])
            @test case_count(binary) == 3
            @test control_count(binary) == 2
        end
    end

    # ========================================================================
    # Statistics Utilities Tests
    # ========================================================================
    @testset "Statistical Utilities" begin
        @testset "T-tests" begin
            x = randn(100) .+ 0.5
            y = randn(100)
            result = welch_t_test(x, y)
            @test haskey(result, :t_statistic)
            @test haskey(result, :pvalue)
            @test 0 <= result.pvalue <= 1
        end
        
        @testset "Linear Regression" begin
            X = hcat(ones(100), randn(100))
            β_true = [1.0, 2.0]
            y = X * β_true + randn(100) * 0.1
            
            result = linear_regression(X, y)
            @test length(result.coefficients) == 2
            @test abs(result.coefficients[1] - 1.0) < 0.5
            @test abs(result.coefficients[2] - 2.0) < 0.5
        end
        
        @testset "Chi-squared Test" begin
            observed = [25, 25, 50]
            expected = [20, 30, 50]
            result = chi_squared_test(observed, expected)
            @test haskey(result, :chi_squared)
            @test haskey(result, :pvalue)
        end
    end

    # ========================================================================
    # Hardy-Weinberg Tests
    # ========================================================================
    @testset "Hardy-Weinberg Equilibrium" begin
        @testset "Allele Frequencies" begin
            genotypes = [0, 0, 1, 1, 1, 2, 2]
            freqs = allele_frequencies(genotypes)
            @test haskey(freqs, :p)
            @test haskey(freqs, :q)
            @test abs(freqs.p + freqs.q - 1.0) < 1e-10
        end
        
        @testset "HWE Test" begin
            # Create HWE genotypes
            p = 0.3
            n = 1000
            aa = round(Int, (1-p)^2 * n)
            ab = round(Int, 2*p*(1-p) * n)
            bb = n - aa - ab
            genotypes = vcat(zeros(Int, aa), ones(Int, ab), fill(2, bb))
            
            result = hwe_test(genotypes)
            @test haskey(result, :pvalue)
            @test result.pvalue > 0.01  # Should not reject HWE
        end
    end

    # ========================================================================
    # Linkage Disequilibrium Tests
    # ========================================================================
    @testset "Linkage Disequilibrium" begin
        @testset "LD Calculation" begin
            geno1 = rand(0:2, 200)
            geno2 = rand(0:2, 200)
            
            result = calculate_ld(geno1, geno2)
            @test haskey(result, :r_squared)
            @test 0 <= result.r_squared <= 1
        end
        
        @testset "LD Matrix" begin
            data = rand(0:2, 100, 10)
            gm = GenotypeMatrix(data)
            
            ld_mat = ld_matrix(gm)
            @test size(ld_mat) == (10, 10)
            @test all(0 .<= ld_mat .<= 1)
            @test all(diag(ld_mat) .== 1.0)  # Self-LD should be 1
        end
    end

    # ========================================================================
    # Wright-Fisher Model Tests
    # ========================================================================
    @testset "Wright-Fisher Model" begin
        @testset "Basic Simulation" begin
            traj = wright_fisher_simulate(100, 0.5, 50)
            @test length(traj) == 51
            @test all(0 .<= traj .<= 1)
        end
        
        @testset "Fixation Probability" begin
            # For neutral allele, Pr(fixation) ≈ initial frequency
            p0 = 0.3
            prob = fixation_probability_theory(100, p0, 0.0)
            @test abs(prob - p0) < 0.01
        end
    end

    # ========================================================================
    # Substitution Model Tests
    # ========================================================================
    @testset "Substitution Models" begin
        @testset "JC69" begin
            model = JC69()
            Q = rate_matrix(model)
            @test size(Q) == (4, 4)
            @test all(sum(Q, dims=2) .≈ 0)  # Rows sum to 0
        end
        
        @testset "K80" begin
            model = K80(2.0)
            Q = rate_matrix(model)
            @test size(Q) == (4, 4)
            # Transition rate should be higher than transversion
            @test Q[1, 3] > Q[1, 2]  # A→G > A→C
        end
    end

    # ========================================================================
    # Coalescent Tests
    # ========================================================================
    @testset "Coalescent Theory" begin
        @testset "Basic Coalescent" begin
            tree = coalescent_simulate(10)
            @test tree.n_samples == 10
            @test length(tree.coalescence_times) == 9
            @test tree.tree_height > 0
        end
        
        @testset "TMRCA" begin
            result = time_to_mrca(10; n_simulations=100)
            @test haskey(result, :empirical_mean)
            @test haskey(result, :theoretical_mean)
            @test result.empirical_mean > 0
        end
    end

    # ========================================================================
    # Phylogenetics Tests
    # ========================================================================
    @testset "Phylogenetics" begin
        @testset "Distance Matrix" begin
            seqs = [rand(1:4, 100) for _ in 1:5]
            D = distance_matrix(seqs)
            @test size(D) == (5, 5)
            @test all(diag(D) .== 0)
            @test D == D'  # Symmetric
        end
        
        @testset "Neighbor Joining" begin
            D = [0.0 0.2 0.3 0.4;
                 0.2 0.0 0.25 0.35;
                 0.3 0.25 0.0 0.28;
                 0.4 0.35 0.28 0.0]
            tree = neighbor_joining(D)
            @test tree.n_tips == 4
        end
    end

    # ========================================================================
    # Population Structure Tests
    # ========================================================================
    @testset "Population Structure" begin
        @testset "PCA" begin
            data = rand(0:2, 50, 100)
            gm = GenotypeMatrix(data)
            
            result = genetic_pca(gm; n_components=5)
            @test size(result.scores, 1) == 50
            @test size(result.scores, 2) == 5
            @test sum(result.variance_explained) <= 1.0
        end
        
        @testset "Clustering" begin
            data = rand(0:2, 30, 50)
            gm = GenotypeMatrix(data)
            
            result = structure_clustering(gm, 2; maxiter=20)
            @test result.K == 2
            @test length(result.assignments) == 30
            @test size(result.proportions) == (30, 2)
        end
    end

    # ========================================================================
    # GWAS Tests
    # ========================================================================
    @testset "GWAS" begin
        @testset "Single Variant Test" begin
            # Create test data with one causal variant
            n = 200
            m = 20
            data = rand(0:2, n, m)
            gm = GenotypeMatrix(data)
            
            # Create phenotype correlated with first variant
            y = 0.5 .* Float64.(data[:, 1]) .+ randn(n) * 0.5
            pheno = ContinuousPhenotype(y)
            
            result = gwas_single_variant(gm, pheno)
            @test length(result.pvalues) == m
            @test result.pvalues[1] < 0.05  # Causal variant should be significant
        end
        
        @testset "Multiple Testing" begin
            pvals = rand(1000)
            
            bonf = bonferroni_correction(pvals)
            @test haskey(bonf, :threshold)
            @test bonf.threshold ≈ 0.05 / 1000
            
            fdr = fdr_correction(pvals)
            @test haskey(fdr, :qvalues)
            @test length(fdr.qvalues) == 1000
        end
        
        @testset "Genomic Control" begin
            # Create inflated p-values
            pvals = rand(1000) .^ 2  # Inflate
            result = genomic_control(pvals)
            @test haskey(result, :lambda_gc)
            @test result.lambda_gc > 1.0  # Should detect inflation
        end
    end

    # ========================================================================
    # Mixed Model Tests
    # ========================================================================
    @testset "Mixed Models" begin
        @testset "GRM Calculation" begin
            data = rand(0:2, 50, 100)
            gm = GenotypeMatrix(data)
            
            G = grm_matrix(gm)
            @test size(G) == (50, 50)
            @test G ≈ G'  # Symmetric
        end
    end

    # ========================================================================
    # Expression Tests
    # ========================================================================
    @testset "Expression Analysis" begin
        @testset "Differential Expression" begin
            # Create expression data
            expr = randn(100, 20)
            groups = vcat(ones(Int, 10), fill(2, 10))
            
            # Add differential expression for first genes
            expr[1:5, 1:10] .+= 2.0
            
            result = differential_expression(expr, groups)
            @test length(result.pvalues) == 100
            @test all(result.pvalues[1:5] .< 0.05)
        end
        
        @testset "Diversity Indices" begin
            abundance = [10.0, 20.0, 30.0, 40.0, 50.0]
            
            H = shannon_diversity(abundance)
            @test H > 0
            
            D = simpson_diversity(abundance)
            @test 0 <= D <= 1
        end
    end

    # ========================================================================
    # Mendelian Randomization Tests
    # ========================================================================
    @testset "Mendelian Randomization" begin
        @testset "IVW Method" begin
            # Create test data
            betas_x = [0.1, 0.2, 0.15, 0.12]
            ses_x = [0.02, 0.03, 0.025, 0.02]
            betas_y = [0.05, 0.1, 0.075, 0.06]  # True causal effect ≈ 0.5
            ses_y = [0.01, 0.015, 0.012, 0.01]
            
            result = ivw_method(betas_x, ses_x, betas_y, ses_y)
            @test haskey(result, :beta)
            @test haskey(result, :pvalue)
            @test abs(result.beta - 0.5) < 0.2
        end
    end

    # ========================================================================
    # Forensics Tests
    # ========================================================================
    @testset "Forensics" begin
        @testset "Kinship" begin
            # Create related individuals (siblings share ~50% alleles)
            data = rand(0:2, 4, 100)
            # Make individuals 1 and 2 more similar
            data[2, :] = data[1, :]
            data[2, 1:50] = rand(0:2, 50)  # Partially different
            
            gm = GenotypeMatrix(data)
            
            k_related = kinship_coefficient(gm, 1, 2)
            k_unrelated = kinship_coefficient(gm, 1, 3)
            
            @test k_related > k_unrelated
        end
    end

end # @testset

println("All tests completed!")
