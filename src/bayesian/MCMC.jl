# ============================================================================
# MCMC.jl - Markov Chain Monte Carlo Methods for Bayesian Genomics
# ============================================================================
# Comprehensive MCMC sampling algorithms for Bayesian statistical genetics
# Implements Metropolis-Hastings, Gibbs sampling, Hamiltonian MC, and more
# ============================================================================

"""
    MCMCChain

Structure to store MCMC chain results with samples, acceptance rates, and diagnostics.

# Fields
- `samples::Matrix{Float64}`: Parameter samples (iterations × parameters)
- `log_posterior::Vector{Float64}`: Log posterior at each iteration
- `acceptance_rate::Float64`: Overall acceptance rate
- `parameter_names::Vector{String}`: Names of parameters
- `n_iterations::Int`: Total number of iterations
- `n_burnin::Int`: Number of burn-in iterations discarded
- `thinning::Int`: Thinning interval used
"""
struct MCMCChain
    samples::Matrix{Float64}
    log_posterior::Vector{Float64}
    acceptance_rate::Float64
    parameter_names::Vector{String}
    n_iterations::Int
    n_burnin::Int
    thinning::Int
end

"""
    MCMCDiagnostics

Structure containing MCMC convergence diagnostics.

# Fields
- `rhat::Vector{Float64}`: Gelman-Rubin R-hat statistics
- `ess::Vector{Float64}`: Effective sample sizes
- `geweke_z::Vector{Float64}`: Geweke diagnostic z-scores
- `autocorrelation::Matrix{Float64}`: Autocorrelation functions
- `converged::Bool`: Whether chains appear converged
"""
struct MCMCDiagnostics
    rhat::Vector{Float64}
    ess::Vector{Float64}
    geweke_z::Vector{Float64}
    autocorrelation::Matrix{Float64}
    converged::Bool
end

"""
    metropolis_hastings(log_target::Function, initial::Vector{Float64},
                       proposal_sd::Vector{Float64}; kwargs...) -> MCMCChain

Run Metropolis-Hastings MCMC algorithm to sample from a target distribution.

# Arguments
- `log_target`: Function computing log of (unnormalized) target density
- `initial`: Initial parameter values
- `proposal_sd`: Standard deviations for Gaussian proposal distribution

# Keyword Arguments
- `n_iterations::Int=10000`: Number of MCMC iterations
- `n_burnin::Int=1000`: Number of burn-in iterations to discard
- `thinning::Int=1`: Keep every nth sample
- `adapt::Bool=true`: Whether to adapt proposal during burn-in
- `target_acceptance::Float64=0.234`: Target acceptance rate for adaptation
- `verbose::Bool=true`: Print progress information

# Returns
- `MCMCChain`: Chain of posterior samples with diagnostics

# Algorithm Details
The Metropolis-Hastings algorithm generates samples from a target distribution
π(θ) by constructing a Markov chain with π as its stationary distribution.

At each iteration:
1. Propose θ* ~ q(θ*|θ_t) where q is a symmetric Gaussian proposal
2. Compute acceptance probability: α = min(1, π(θ*)/π(θ_t))
3. Accept θ_{t+1} = θ* with probability α, else θ_{t+1} = θ_t

The proposal standard deviations are adapted during burn-in to achieve
the target acceptance rate (0.234 is optimal for high-dimensional targets).

# Example
```julia
# Define log posterior for Bayesian linear regression
function log_posterior(θ)
    β, σ² = θ[1:end-1], θ[end]
    σ² > 0 || return -Inf
    residuals = y - X * β
    log_lik = -n/2 * log(2π*σ²) - sum(residuals.^2) / (2σ²)
    log_prior = -sum(β.^2) / (2*prior_var) - log(σ²)  # Normal-InvGamma priors
    return log_lik + log_prior
end

chain = metropolis_hastings(log_posterior, initial, proposal_sd; n_iterations=50000)
```

# References
- Metropolis et al. (1953) J. Chem. Phys.
- Hastings (1970) Biometrika
- Roberts et al. (1997) Ann. Appl. Probab. (optimal scaling)
"""
function metropolis_hastings(
    log_target::Function,
    initial::Vector{Float64},
    proposal_sd::Vector{Float64};
    n_iterations::Int=10000,
    n_burnin::Int=1000,
    thinning::Int=1,
    adapt::Bool=true,
    target_acceptance::Float64=0.234,
    verbose::Bool=true,
    parameter_names::Union{Vector{String}, Nothing}=nothing
)
    n_params = length(initial)

    # Initialize storage
    n_samples = div(n_iterations - n_burnin, thinning)
    samples = zeros(n_samples, n_params)
    log_posteriors = zeros(n_samples)

    # Current state
    current = copy(initial)
    current_log_prob = log_target(current)

    # Proposal standard deviations (will be adapted)
    σ_prop = copy(proposal_sd)

    # Tracking acceptance
    n_accepted = 0
    n_total = 0
    adaptation_window = 100
    recent_accepts = zeros(Bool, adaptation_window)

    # Parameter names
    if parameter_names === nothing
        parameter_names = ["θ_$i" for i in 1:n_params]
    end

    sample_idx = 1

    # Progress meter
    if verbose
        prog = Progress(n_iterations; desc="MCMC Sampling: ")
    end

    for iter in 1:n_iterations
        # Propose new state
        proposed = current + σ_prop .* randn(n_params)
        proposed_log_prob = log_target(proposed)

        # Metropolis-Hastings acceptance probability
        log_α = proposed_log_prob - current_log_prob

        # Accept or reject
        if log(rand()) < log_α
            current = proposed
            current_log_prob = proposed_log_prob
            n_accepted += 1
            recent_accepts[mod1(iter, adaptation_window)] = true
        else
            recent_accepts[mod1(iter, adaptation_window)] = false
        end
        n_total += 1

        # Adapt proposal during burn-in
        if adapt && iter <= n_burnin && iter % adaptation_window == 0
            recent_rate = mean(recent_accepts)
            # Adjust proposal scale
            if recent_rate < target_acceptance - 0.05
                σ_prop .*= 0.9  # Decrease to increase acceptance
            elseif recent_rate > target_acceptance + 0.05
                σ_prop .*= 1.1  # Increase to decrease acceptance
            end
        end

        # Store sample after burn-in
        if iter > n_burnin && (iter - n_burnin) % thinning == 0
            samples[sample_idx, :] = current
            log_posteriors[sample_idx] = current_log_prob
            sample_idx += 1
        end

        if verbose
            next!(prog)
        end
    end

    acceptance_rate = n_accepted / n_total

    if verbose
        println("\nMCMC completed:")
        println("  Final acceptance rate: $(round(acceptance_rate, digits=3))")
        println("  Samples collected: $n_samples")
    end

    return MCMCChain(
        samples,
        log_posteriors,
        acceptance_rate,
        parameter_names,
        n_iterations,
        n_burnin,
        thinning
    )
end

"""
    gibbs_sampler(conditional_samplers::Vector{Function},
                  initial::Vector{Vector{Float64}}; kwargs...) -> MCMCChain

Run Gibbs sampling algorithm using full conditional distributions.

# Arguments
- `conditional_samplers`: Vector of functions, each sampling one parameter block
  given all others. Each function takes (current_params, data) and returns sampled value.
- `initial`: Initial values for each parameter block

# Keyword Arguments
- `n_iterations::Int=10000`: Number of Gibbs iterations
- `n_burnin::Int=1000`: Burn-in iterations
- `thinning::Int=1`: Thinning interval
- `data::Any=nothing`: Data passed to conditional samplers
- `verbose::Bool=true`: Print progress

# Algorithm Details
Gibbs sampling is a special case of Metropolis-Hastings where proposals
are drawn from the full conditional distributions, guaranteeing acceptance.

At each iteration, for each parameter block j:
θ_j^{t+1} ~ p(θ_j | θ_{-j}^{t+1}, y)

where θ_{-j} denotes all parameters except θ_j.

# Example
```julia
# Gibbs sampler for Normal-Normal model
# y ~ N(μ, σ²), μ ~ N(μ_0, τ²)
function sample_mu(params, data)
    y, σ², μ_0, τ² = data
    precision = length(y)/σ² + 1/τ²
    mean_post = (sum(y)/σ² + μ_0/τ²) / precision
    return randn() * sqrt(1/precision) + mean_post
end

function sample_sigma2(params, data)
    y, μ = data[1], params[1]
    n = length(y)
    ss = sum((y .- μ).^2)
    # InverseGamma(a + n/2, b + ss/2) via InverseGamma
    return rand(InverseGamma(0.01 + n/2, 0.01 + ss/2))
end

chain = gibbs_sampler([sample_mu, sample_sigma2], [[0.0], [1.0]]; data=(y, 1.0, 0.0, 100.0))
```

# References
- Geman & Geman (1984) IEEE Trans. PAMI
- Gelfand & Smith (1990) J. Am. Stat. Assoc.
"""
function gibbs_sampler(
    conditional_samplers::Vector{<:Function},
    initial::Vector{<:Vector{Float64}};
    n_iterations::Int=10000,
    n_burnin::Int=1000,
    thinning::Int=1,
    data::Any=nothing,
    verbose::Bool=true,
    parameter_names::Union{Vector{String}, Nothing}=nothing
)
    n_blocks = length(conditional_samplers)
    @assert length(initial) == n_blocks "Number of initial values must match samplers"

    # Flatten parameters for storage
    block_sizes = [length(init) for init in initial]
    n_params = sum(block_sizes)
    block_starts = cumsum([1; block_sizes[1:end-1]])

    # Initialize storage
    n_samples = div(n_iterations - n_burnin, thinning)
    samples = zeros(n_samples, n_params)

    # Current state
    current = deepcopy(initial)

    # Parameter names
    if parameter_names === nothing
        parameter_names = String[]
        for (b, size) in enumerate(block_sizes)
            if size == 1
                push!(parameter_names, "block_$b")
            else
                for j in 1:size
                    push!(parameter_names, "block_$(b)_$j")
                end
            end
        end
    end

    sample_idx = 1

    if verbose
        prog = Progress(n_iterations; desc="Gibbs Sampling: ")
    end

    for iter in 1:n_iterations
        # Update each block from its full conditional
        for b in 1:n_blocks
            current[b] = conditional_samplers[b](current, data)
        end

        # Store sample after burn-in
        if iter > n_burnin && (iter - n_burnin) % thinning == 0
            flat_idx = 1
            for b in 1:n_blocks
                for val in current[b]
                    samples[sample_idx, flat_idx] = val
                    flat_idx += 1
                end
            end
            sample_idx += 1
        end

        if verbose
            next!(prog)
        end
    end

    # No acceptance rate for Gibbs (always accept)
    return MCMCChain(
        samples,
        zeros(n_samples),  # Log posterior not tracked
        1.0,  # Acceptance rate is always 1 for Gibbs
        parameter_names,
        n_iterations,
        n_burnin,
        thinning
    )
end

"""
    hamiltonian_monte_carlo(log_target::Function, grad_log_target::Function,
                           initial::Vector{Float64}; kwargs...) -> MCMCChain

Run Hamiltonian Monte Carlo (HMC) for efficient sampling from continuous distributions.

# Arguments
- `log_target`: Function computing log of target density
- `grad_log_target`: Function computing gradient of log target
- `initial`: Initial parameter values

# Keyword Arguments
- `n_iterations::Int=5000`: Number of HMC iterations
- `n_burnin::Int=1000`: Burn-in iterations
- `step_size::Float64=0.1`: Leapfrog step size ε
- `n_leapfrog::Int=10`: Number of leapfrog steps L
- `mass_matrix::Union{Matrix{Float64}, Nothing}=nothing`: Mass matrix M
- `adapt_step_size::Bool=true`: Adapt step size during burn-in

# Algorithm Details
HMC augments the parameter space with momentum variables and simulates
Hamiltonian dynamics to propose distant moves while maintaining high acceptance.

The Hamiltonian is: H(θ, p) = -log π(θ) + p'M⁻¹p/2

Leapfrog integration:
1. p_{t+ε/2} = p_t + (ε/2) ∇ log π(θ_t)
2. θ_{t+ε} = θ_t + ε M⁻¹ p_{t+ε/2}
3. p_{t+ε} = p_{t+ε/2} + (ε/2) ∇ log π(θ_{t+ε})

# Example
```julia
function log_posterior(θ)
    -0.5 * θ' * Σ_inv * θ  # Multivariate normal
end

function grad_log_posterior(θ)
    -Σ_inv * θ
end

chain = hamiltonian_monte_carlo(log_posterior, grad_log_posterior, zeros(10))
```

# References
- Duane et al. (1987) Physics Letters B
- Neal (2011) Handbook of MCMC, Chapter 5
- Betancourt (2017) arXiv:1701.02434
"""
function hamiltonian_monte_carlo(
    log_target::Function,
    grad_log_target::Function,
    initial::Vector{Float64};
    n_iterations::Int=5000,
    n_burnin::Int=1000,
    thinning::Int=1,
    step_size::Float64=0.1,
    n_leapfrog::Int=10,
    mass_matrix::Union{Matrix{Float64}, Nothing}=nothing,
    adapt_step_size::Bool=true,
    target_acceptance::Float64=0.65,
    verbose::Bool=true,
    parameter_names::Union{Vector{String}, Nothing}=nothing
)
    n_params = length(initial)

    # Mass matrix (default to identity)
    if mass_matrix === nothing
        M = Matrix{Float64}(I, n_params, n_params)
        M_inv = M
    else
        M = mass_matrix
        M_inv = inv(M)
    end
    M_chol = cholesky(M).L

    # Initialize storage
    n_samples = div(n_iterations - n_burnin, thinning)
    samples = zeros(n_samples, n_params)
    log_posteriors = zeros(n_samples)

    # Current state
    current_q = copy(initial)
    current_log_prob = log_target(current_q)

    # Adaptive step size
    ε = step_size

    # Tracking
    n_accepted = 0
    n_total = 0

    # Parameter names
    if parameter_names === nothing
        parameter_names = ["θ_$i" for i in 1:n_params]
    end

    sample_idx = 1

    if verbose
        prog = Progress(n_iterations; desc="HMC Sampling: ")
    end

    for iter in 1:n_iterations
        # Sample momentum
        p = M_chol * randn(n_params)
        current_p = copy(p)

        # Current position
        q = copy(current_q)

        # Leapfrog integration
        # Half step for momentum
        p = p + (ε / 2) * grad_log_target(q)

        # Full steps
        for _ in 1:n_leapfrog-1
            q = q + ε * (M_inv * p)
            p = p + ε * grad_log_target(q)
        end

        # Final half step
        q = q + ε * (M_inv * p)
        p = p + (ε / 2) * grad_log_target(q)

        # Negate momentum for reversibility
        p = -p

        # Compute Hamiltonian
        proposed_log_prob = log_target(q)
        current_H = -current_log_prob + 0.5 * current_p' * M_inv * current_p
        proposed_H = -proposed_log_prob + 0.5 * p' * M_inv * p

        # Accept/reject
        log_α = current_H - proposed_H

        if log(rand()) < log_α
            current_q = q
            current_log_prob = proposed_log_prob
            n_accepted += 1
        end
        n_total += 1

        # Adapt step size during burn-in
        if adapt_step_size && iter <= n_burnin && iter % 50 == 0
            current_rate = n_accepted / n_total
            if current_rate < target_acceptance - 0.05
                ε *= 0.9
            elseif current_rate > target_acceptance + 0.05
                ε *= 1.1
            end
            # Keep step size reasonable
            ε = clamp(ε, 1e-5, 1.0)
        end

        # Store sample after burn-in
        if iter > n_burnin && (iter - n_burnin) % thinning == 0
            samples[sample_idx, :] = current_q
            log_posteriors[sample_idx] = current_log_prob
            sample_idx += 1
        end

        if verbose
            next!(prog)
        end
    end

    acceptance_rate = n_accepted / n_total

    if verbose
        println("\nHMC completed:")
        println("  Final acceptance rate: $(round(acceptance_rate, digits=3))")
        println("  Final step size: $(round(ε, digits=4))")
    end

    return MCMCChain(
        samples,
        log_posteriors,
        acceptance_rate,
        parameter_names,
        n_iterations,
        n_burnin,
        thinning
    )
end

"""
    slice_sampler(log_target::Function, initial::Float64; kwargs...) -> Vector{Float64}

Univariate slice sampling algorithm.

# Arguments
- `log_target`: Log of target density
- `initial`: Initial value

# Keyword Arguments
- `n_iterations::Int=10000`: Number of iterations
- `n_burnin::Int=1000`: Burn-in
- `width::Float64=1.0`: Initial slice width

# Algorithm Details
Slice sampling samples uniformly from the region under the density curve.
For target p(x), we sample (x, u) uniformly from {(x, u): 0 < u < p(x)}.

1. Given x_t, sample u ~ Uniform(0, p(x_t))
2. Find slice S = {x: p(x) > u} using stepping out
3. Sample x_{t+1} uniformly from S using shrinkage

# References
- Neal (2003) Ann. Statist.
"""
function slice_sampler(
    log_target::Function,
    initial::Float64;
    n_iterations::Int=10000,
    n_burnin::Int=1000,
    width::Float64=1.0,
    max_steps::Int=100
)
    samples = zeros(n_iterations - n_burnin)
    x = initial

    sample_idx = 1

    for iter in 1:n_iterations
        # Sample slice height
        log_y = log_target(x) - randexp()

        # Step out to find slice bounds
        L = x - width * rand()
        R = L + width

        # Expand left
        for _ in 1:max_steps
            if log_target(L) <= log_y
                break
            end
            L -= width
        end

        # Expand right
        for _ in 1:max_steps
            if log_target(R) <= log_y
                break
            end
            R += width
        end

        # Shrink and sample
        while true
            x_new = L + rand() * (R - L)
            if log_target(x_new) > log_y
                x = x_new
                break
            end
            if x_new < x
                L = x_new
            else
                R = x_new
            end
        end

        # Store sample
        if iter > n_burnin
            samples[sample_idx] = x
            sample_idx += 1
        end
    end

    return samples
end

"""
    compute_diagnostics(chains::Vector{MCMCChain}) -> MCMCDiagnostics

Compute convergence diagnostics for multiple MCMC chains.

# Arguments
- `chains`: Vector of MCMCChain objects from parallel runs

# Returns
- `MCMCDiagnostics`: Comprehensive diagnostics including R-hat, ESS, Geweke

# Diagnostics Computed
1. **R-hat (Gelman-Rubin)**: Compares within-chain and between-chain variance.
   Values close to 1.0 indicate convergence (< 1.1 recommended).

2. **Effective Sample Size (ESS)**: Accounts for autocorrelation.
   ESS = n / (1 + 2 Σ_k ρ_k) where ρ_k is autocorrelation at lag k.

3. **Geweke diagnostic**: Compares first 10% and last 50% of chain.
   Should be approximately N(0,1) if chain is stationary.

# Example
```julia
# Run multiple chains from different starting points
chains = [metropolis_hastings(log_target, rand(p), σ) for _ in 1:4]
diag = compute_diagnostics(chains)
diag.converged  # Check overall convergence
```

# References
- Gelman & Rubin (1992) Statistical Science
- Geweke (1992) Bayesian Statistics 4
"""
function compute_diagnostics(chains::Vector{MCMCChain})
    n_chains = length(chains)
    n_params = size(chains[1].samples, 2)
    n_samples = size(chains[1].samples, 1)

    rhat = zeros(n_params)
    ess = zeros(n_params)
    geweke_z = zeros(n_params)

    for p in 1:n_params
        # Extract parameter across chains
        param_samples = [chains[c].samples[:, p] for c in 1:n_chains]

        # Gelman-Rubin R-hat
        chain_means = [mean(s) for s in param_samples]
        chain_vars = [var(s) for s in param_samples]

        overall_mean = mean(chain_means)
        between_var = n_samples * var(chain_means)
        within_var = mean(chain_vars)

        var_estimate = ((n_samples - 1) / n_samples) * within_var +
                       (1 / n_samples) * between_var
        rhat[p] = sqrt(var_estimate / within_var)

        # Effective sample size (using first chain)
        x = param_samples[1]
        acf = autocor(x, 1:min(100, n_samples-1))
        # Sum autocorrelations until they become negligible
        sum_acf = 0.0
        for k in 1:length(acf)
            if acf[k] < 0.05
                break
            end
            sum_acf += acf[k]
        end
        ess[p] = n_samples * n_chains / (1 + 2 * sum_acf)

        # Geweke diagnostic (first chain)
        n_first = div(n_samples, 10)
        n_last = div(n_samples, 2)
        first_part = x[1:n_first]
        last_part = x[end-n_last+1:end]

        se_first = std(first_part) / sqrt(n_first)
        se_last = std(last_part) / sqrt(n_last)
        geweke_z[p] = (mean(first_part) - mean(last_part)) / sqrt(se_first^2 + se_last^2)
    end

    # Compute autocorrelation matrix for first chain
    max_lag = min(50, n_samples - 1)
    autocorrelation = zeros(max_lag, n_params)
    for p in 1:n_params
        autocorrelation[:, p] = autocor(chains[1].samples[:, p], 1:max_lag)
    end

    # Check convergence criteria
    converged = all(rhat .< 1.1) && all(ess .> 100) && all(abs.(geweke_z) .< 2.0)

    return MCMCDiagnostics(rhat, ess, geweke_z, autocorrelation, converged)
end

"""
    summarize_chain(chain::MCMCChain; credible_level::Float64=0.95) -> DataFrame

Compute posterior summary statistics from MCMC chain.

# Arguments
- `chain`: MCMCChain object
- `credible_level`: Level for credible intervals (default 0.95)

# Returns
DataFrame with columns: parameter, mean, std, median, lower_ci, upper_ci, ess

# Example
```julia
chain = metropolis_hastings(log_target, initial, σ)
summary = summarize_chain(chain)
```
"""
function summarize_chain(chain::MCMCChain; credible_level::Float64=0.95)
    n_params = length(chain.parameter_names)
    α = (1 - credible_level) / 2

    results = DataFrame(
        parameter = chain.parameter_names,
        mean = zeros(n_params),
        std = zeros(n_params),
        median = zeros(n_params),
        lower_ci = zeros(n_params),
        upper_ci = zeros(n_params)
    )

    for p in 1:n_params
        samples = chain.samples[:, p]
        results.mean[p] = mean(samples)
        results.std[p] = std(samples)
        results.median[p] = median(samples)
        results.lower_ci[p] = quantile(samples, α)
        results.upper_ci[p] = quantile(samples, 1 - α)
    end

    return results
end

"""
    autocor(x::Vector{Float64}, lags::AbstractVector{Int}) -> Vector{Float64}

Compute autocorrelation function at specified lags.
"""
function autocor(x::Vector{Float64}, lags::AbstractVector{Int})
    n = length(x)
    x_centered = x .- mean(x)
    var_x = var(x)

    acf = zeros(length(lags))
    for (i, lag) in enumerate(lags)
        if lag >= n
            acf[i] = 0.0
        else
            acf[i] = sum(x_centered[1:n-lag] .* x_centered[lag+1:n]) / ((n - lag) * var_x)
        end
    end

    return acf
end

"""
    parallel_tempering(log_target::Function, initial::Vector{Float64},
                      temperatures::Vector{Float64}; kwargs...) -> MCMCChain

Parallel tempering (replica exchange) MCMC for multimodal distributions.

# Arguments
- `log_target`: Log target density at temperature 1
- `initial`: Initial parameter values
- `temperatures`: Vector of temperatures (including 1.0)

# Algorithm Details
Runs multiple chains at different temperatures T_i where the tempered
target is π_i(θ) ∝ π(θ)^{1/T_i}. Higher temperatures flatten the
distribution, allowing escape from local modes.

Periodically proposes swaps between adjacent temperature chains:
α = min(1, [π(θ_i)/π(θ_j)]^{1/T_i - 1/T_j})

# References
- Geyer (1991) Computing Science and Statistics
- Earl & Deem (2005) Phys. Chem. Chem. Phys.
"""
function parallel_tempering(
    log_target::Function,
    initial::Vector{Float64},
    temperatures::Vector{Float64};
    proposal_sd::Vector{Float64}=ones(length(initial)),
    n_iterations::Int=10000,
    n_burnin::Int=2000,
    swap_interval::Int=10,
    verbose::Bool=true
)
    n_temps = length(temperatures)
    n_params = length(initial)

    # Sort temperatures (coldest = 1.0 should be first or last)
    temp_order = sortperm(temperatures)
    temperatures = temperatures[temp_order]

    # Ensure temperature 1.0 is included
    @assert 1.0 in temperatures "Temperatures must include 1.0"
    cold_idx = findfirst(==(1.0), temperatures)

    # Initialize chains at each temperature
    chains = [copy(initial) for _ in 1:n_temps]
    log_probs = [log_target(initial) for _ in 1:n_temps]

    # Storage for cold chain
    n_samples = div(n_iterations - n_burnin, 1)
    samples = zeros(n_samples, n_params)
    log_posteriors = zeros(n_samples)

    n_swaps_accepted = 0
    n_swaps_proposed = 0
    sample_idx = 1

    if verbose
        prog = Progress(n_iterations; desc="Parallel Tempering: ")
    end

    for iter in 1:n_iterations
        # Update each chain with MH at its temperature
        for t in 1:n_temps
            proposed = chains[t] + proposal_sd .* randn(n_params)
            proposed_log_prob = log_target(proposed)

            # Tempered acceptance
            log_α = (proposed_log_prob - log_probs[t]) / temperatures[t]

            if log(rand()) < log_α
                chains[t] = proposed
                log_probs[t] = proposed_log_prob
            end
        end

        # Propose temperature swaps
        if iter % swap_interval == 0
            for t in 1:n_temps-1
                n_swaps_proposed += 1

                # Swap acceptance probability
                log_α = (log_probs[t] - log_probs[t+1]) *
                        (1/temperatures[t+1] - 1/temperatures[t])

                if log(rand()) < log_α
                    # Swap chains
                    chains[t], chains[t+1] = chains[t+1], chains[t]
                    log_probs[t], log_probs[t+1] = log_probs[t+1], log_probs[t]
                    n_swaps_accepted += 1
                end
            end
        end

        # Store sample from cold chain
        if iter > n_burnin
            samples[sample_idx, :] = chains[cold_idx]
            log_posteriors[sample_idx] = log_probs[cold_idx]
            sample_idx += 1
        end

        if verbose
            next!(prog)
        end
    end

    if verbose
        println("\nParallel tempering completed:")
        println("  Swap acceptance rate: $(round(n_swaps_accepted/n_swaps_proposed, digits=3))")
    end

    return MCMCChain(
        samples,
        log_posteriors,
        n_swaps_accepted / n_swaps_proposed,
        ["θ_$i" for i in 1:n_params],
        n_iterations,
        n_burnin,
        1
    )
end
