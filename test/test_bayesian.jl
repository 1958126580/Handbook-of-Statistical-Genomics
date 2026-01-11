# ============================================================================
# Comprehensive Tests for Bayesian Methods Module
# ============================================================================
# Tests for MCMC sampling and Variational Bayes inference
# ============================================================================

@testset "Bayesian Methods" begin

    # ========================================================================
    # MCMC Chain Structure Tests
    # ========================================================================
    @testset "MCMCChain Structure" begin
        # Create a mock MCMC chain for structure testing
        samples = randn(100, 3)
        log_posterior = randn(100)
        chain = MCMCChain(
            samples,
            log_posterior,
            0.45,
            ["param1", "param2", "param3"],
            100,
            10,
            1
        )

        @test chain.n_iterations == 100
        @test chain.n_burnin == 10
        @test chain.thinning == 1
        @test chain.acceptance_rate == 0.45
        @test length(chain.parameter_names) == 3
        @test size(chain.samples) == (100, 3)
    end

    # ========================================================================
    # Metropolis-Hastings Algorithm Tests
    # ========================================================================
    @testset "Metropolis-Hastings" begin
        @testset "Normal Target Distribution" begin
            # Sample from standard normal
            log_target(x) = -0.5 * sum(x.^2)
            initial = [0.0]
            proposal_sd = [1.0]

            chain = metropolis_hastings(
                log_target,
                initial,
                proposal_sd;
                n_iterations=1000,
                n_burnin=200,
                thinning=1
            )

            @test chain.n_iterations == 1000
            @test chain.n_burnin == 200
            @test 0.0 < chain.acceptance_rate < 1.0

            # Check that samples are approximately from standard normal
            posterior_samples = chain.samples[201:end, 1]
            @test abs(mean(posterior_samples)) < 0.3
            @test abs(std(posterior_samples) - 1.0) < 0.3
        end

        @testset "Multivariate Target" begin
            # Sample from bivariate normal with correlation
            Σ = [1.0 0.5; 0.5 1.0]
            Σ_inv = inv(Σ)
            log_target(x) = -0.5 * dot(x, Σ_inv * x)

            initial = [0.0, 0.0]
            proposal_sd = [0.5, 0.5]

            chain = metropolis_hastings(
                log_target,
                initial,
                proposal_sd;
                n_iterations=2000,
                n_burnin=500
            )

            @test size(chain.samples, 2) == 2
            @test chain.acceptance_rate > 0.1
        end

        @testset "Adaptive Proposal" begin
            log_target(x) = -0.5 * sum(x.^2)
            initial = [0.0, 0.0]
            proposal_sd = [0.1, 0.1]  # Start with small proposal

            chain = metropolis_hastings(
                log_target,
                initial,
                proposal_sd;
                n_iterations=1000,
                adaptive=true,
                target_acceptance=0.44
            )

            # Adaptive should improve acceptance rate
            @test chain.acceptance_rate > 0.2
        end
    end

    # ========================================================================
    # Gibbs Sampler Tests
    # ========================================================================
    @testset "Gibbs Sampler" begin
        @testset "Simple Two-Component" begin
            # Gibbs sampling for bivariate normal
            ρ = 0.7

            # Conditional samplers
            sampler1(state) = [randn() * sqrt(1 - ρ^2) + ρ * state[2][1]]
            sampler2(state) = [randn() * sqrt(1 - ρ^2) + ρ * state[1][1]]

            initial = [[0.0], [0.0]]

            chain = gibbs_sampler(
                [sampler1, sampler2],
                initial;
                n_iterations=1000,
                n_burnin=200
            )

            @test size(chain.samples, 2) == 2

            # Check marginal variances
            samples_post = chain.samples[201:end, :]
            @test abs(var(samples_post[:, 1]) - 1.0) < 0.3
            @test abs(var(samples_post[:, 2]) - 1.0) < 0.3

            # Check correlation
            @test abs(cor(samples_post[:, 1], samples_post[:, 2]) - ρ) < 0.2
        end

        @testset "Block Gibbs" begin
            # Test with blocked parameters
            sampler1(state) = randn(2)
            sampler2(state) = randn(3)

            initial = [zeros(2), zeros(3)]

            chain = gibbs_sampler(
                [sampler1, sampler2],
                initial;
                n_iterations=500
            )

            @test size(chain.samples, 2) == 5  # 2 + 3 parameters
        end
    end

    # ========================================================================
    # Hamiltonian Monte Carlo Tests
    # ========================================================================
    @testset "Hamiltonian Monte Carlo" begin
        @testset "Standard Normal" begin
            log_target(x) = -0.5 * sum(x.^2)
            grad_log_target(x) = -x

            initial = [0.0, 0.0]

            chain = hamiltonian_monte_carlo(
                log_target,
                grad_log_target,
                initial;
                n_iterations=500,
                n_burnin=100,
                step_size=0.1,
                n_leapfrog=10
            )

            @test size(chain.samples, 2) == 2
            @test chain.acceptance_rate > 0.5  # HMC should have high acceptance

            # Check posterior samples
            samples_post = chain.samples[101:end, :]
            @test abs(mean(samples_post[:, 1])) < 0.3
            @test abs(mean(samples_post[:, 2])) < 0.3
        end

        @testset "Funnel Distribution" begin
            # Neal's funnel - challenging for MCMC
            function log_target(x)
                v = x[1]
                z = x[2:end]
                return -0.5 * v^2 / 9 - 0.5 * sum(z.^2) * exp(-v)
            end

            function grad_log_target(x)
                v = x[1]
                z = x[2:end]
                grad = zeros(length(x))
                grad[1] = -v/9 + 0.5 * sum(z.^2) * exp(-v)
                grad[2:end] = -z .* exp(-v)
                return grad
            end

            initial = zeros(3)

            chain = hamiltonian_monte_carlo(
                log_target,
                grad_log_target,
                initial;
                n_iterations=300,
                n_burnin=50,
                step_size=0.05,
                n_leapfrog=20
            )

            @test chain.acceptance_rate > 0.3
        end
    end

    # ========================================================================
    # Slice Sampler Tests
    # ========================================================================
    @testset "Slice Sampler" begin
        @testset "Univariate Normal" begin
            log_target(x) = -0.5 * x[1]^2
            initial = [0.0]

            chain = slice_sampler(
                log_target,
                initial;
                n_iterations=500,
                n_burnin=100,
                width=2.0
            )

            @test size(chain.samples, 2) == 1
            samples_post = chain.samples[101:end, 1]
            @test abs(mean(samples_post)) < 0.3
            @test abs(std(samples_post) - 1.0) < 0.3
        end

        @testset "Multimodal" begin
            # Mixture of two normals
            log_target(x) = log(0.5 * exp(-0.5 * (x[1] - 3)^2) +
                              0.5 * exp(-0.5 * (x[1] + 3)^2))
            initial = [0.0]

            chain = slice_sampler(
                log_target,
                initial;
                n_iterations=2000,
                width=5.0
            )

            # Should visit both modes
            samples = chain.samples[:, 1]
            @test minimum(samples) < -1.0
            @test maximum(samples) > 1.0
        end
    end

    # ========================================================================
    # Parallel Tempering Tests
    # ========================================================================
    @testset "Parallel Tempering" begin
        @testset "Multimodal Target" begin
            # Double-well potential
            log_target(x) = -((x[1]^2 - 1)^2)
            initial = [0.0]
            temperatures = [1.0, 2.0, 5.0, 10.0]

            chain = parallel_tempering(
                log_target,
                initial,
                temperatures;
                n_iterations=500,
                n_burnin=100
            )

            @test size(chain.samples, 1) == 500

            # Check that samples explore both modes
            samples_post = chain.samples[101:end, 1]
            @test minimum(samples_post) < 0.0
            @test maximum(samples_post) > 0.0
        end
    end

    # ========================================================================
    # MCMC Diagnostics Tests
    # ========================================================================
    @testset "MCMC Diagnostics" begin
        @testset "Convergence Diagnostics" begin
            # Create test chain
            log_target(x) = -0.5 * sum(x.^2)
            initial = [0.0, 0.0]

            chain = metropolis_hastings(
                log_target,
                initial,
                [1.0, 1.0];
                n_iterations=1000
            )

            diag = compute_diagnostics(chain)

            @test haskey(diag, :ess)  # Effective sample size
            @test haskey(diag, :rhat)  # Potential scale reduction
            @test haskey(diag, :geweke_z)  # Geweke diagnostic
            @test haskey(diag, :autocorrelation)  # Autocorrelation

            # ESS should be less than total samples
            @test all(diag.ess .<= 1000)
            @test all(diag.ess .> 0)
        end

        @testset "Chain Summary" begin
            samples = randn(500, 3)
            log_posterior = randn(500)
            chain = MCMCChain(
                samples,
                log_posterior,
                0.4,
                ["a", "b", "c"],
                500, 0, 1
            )

            summary = summarize_chain(chain)

            @test haskey(summary, :mean)
            @test haskey(summary, :std)
            @test haskey(summary, :quantiles)
            @test length(summary.mean) == 3
        end
    end

    # ========================================================================
    # Variational Inference Tests
    # ========================================================================
    @testset "Variational Bayes" begin
        @testset "ADVI Basic" begin
            # Simple normal target
            log_joint(z) = -0.5 * sum(z.^2)

            result = advi(log_joint, 2; n_iterations=500, n_samples=10)

            @test isa(result, VariationalResult)
            @test length(result.mean) == 2
            @test length(result.std) == 2
            @test all(result.std .> 0)
            @test length(result.elbo_history) > 0
        end

        @testset "Coordinate Ascent VI" begin
            # Mean-field VI for multivariate normal
            log_joint(z) = -0.5 * sum(z.^2)

            result = coordinate_ascent_vi(log_joint, 3; n_iterations=200)

            @test isa(result, VariationalResult)
            @test length(result.mean) == 3
            @test all(abs.(result.mean) .< 1.0)  # Should be close to 0
        end

        @testset "ELBO Computation" begin
            log_joint(z) = -0.5 * sum(z.^2)
            result = advi(log_joint, 2; n_iterations=500)

            # ELBO should increase (less negative)
            elbo = result.elbo_history
            @test elbo[end] >= elbo[1] - 1.0  # Allow for stochastic variation
        end
    end

    # ========================================================================
    # Variational Linear Regression Tests
    # ========================================================================
    @testset "Variational Linear Regression" begin
        @testset "Simple Regression" begin
            n, p = 100, 3
            X = hcat(ones(n), randn(n, p-1))
            β_true = [1.0, 2.0, -1.0]
            y = X * β_true + randn(n) * 0.5

            result = variational_linear_regression(y, X; n_iterations=500)

            @test isa(result, VariationalResult)
            @test length(result.mean) == p

            # Check coefficient recovery
            @test abs(result.mean[1] - 1.0) < 0.5
            @test abs(result.mean[2] - 2.0) < 0.5
            @test abs(result.mean[3] + 1.0) < 0.5
        end

        @testset "With Prior" begin
            n, p = 50, 5
            X = hcat(ones(n), randn(n, p-1))
            y = randn(n)

            # Strong prior should shrink coefficients
            result = variational_linear_regression(
                y, X;
                prior_precision=10.0,
                n_iterations=300
            )

            # Coefficients should be shrunk toward zero
            @test all(abs.(result.mean[2:end]) .< 1.0)
        end
    end

    # ========================================================================
    # Spike-and-Slab Prior Tests
    # ========================================================================
    @testset "Variational Spike-Slab" begin
        @testset "Sparse Signal Recovery" begin
            n, p = 200, 50
            X = randn(n, p)

            # Sparse true signal
            β_true = zeros(p)
            β_true[1:3] = [2.0, -1.5, 1.0]

            y = X * β_true + randn(n) * 0.5

            result = variational_spike_slab(y, X; n_iterations=500)

            @test isa(result, VariationalResult)
            @test length(result.mean) == p
            @test haskey(result.extra, :pip)  # Posterior inclusion probabilities

            pip = result.extra[:pip]
            @test length(pip) == p

            # True signals should have higher PIPs
            @test pip[1] > 0.5
            @test pip[2] > 0.5
            @test pip[3] > 0.5

            # Most null signals should have low PIPs
            @test mean(pip[6:end]) < 0.3
        end

        @testset "High-Dimensional" begin
            n, p = 100, 200  # p > n
            X = randn(n, p)
            β_true = zeros(p)
            β_true[1:2] = [1.5, -1.0]
            y = X * β_true + randn(n) * 0.5

            result = variational_spike_slab(y, X; n_iterations=300)

            @test isa(result, VariationalResult)
            pip = result.extra[:pip]

            # Should identify true signals even in p > n setting
            @test pip[1] > pip[end]
            @test pip[2] > pip[end]
        end
    end

    # ========================================================================
    # Stochastic VI Tests
    # ========================================================================
    @testset "Stochastic VI" begin
        @testset "Large Dataset" begin
            n, p = 1000, 5
            X = hcat(ones(n), randn(n, p-1))
            β_true = randn(p)
            y = X * β_true + randn(n) * 0.5

            result = stochastic_vi(
                y, X;
                batch_size=100,
                n_iterations=200,
                learning_rate=0.01
            )

            @test isa(result, VariationalResult)
            @test length(result.mean) == p
        end
    end

    # ========================================================================
    # Sampling from Variational Distribution Tests
    # ========================================================================
    @testset "Variational Sampling" begin
        result = VariationalResult(
            [1.0, 2.0],
            [0.5, 0.3],
            [-100.0],
            Dict{Symbol,Any}()
        )

        samples = sample_from_variational(result, 1000)

        @test size(samples) == (1000, 2)
        @test abs(mean(samples[:, 1]) - 1.0) < 0.1
        @test abs(mean(samples[:, 2]) - 2.0) < 0.1
        @test abs(std(samples[:, 1]) - 0.5) < 0.1
        @test abs(std(samples[:, 2]) - 0.3) < 0.1
    end

end # @testset "Bayesian Methods"
