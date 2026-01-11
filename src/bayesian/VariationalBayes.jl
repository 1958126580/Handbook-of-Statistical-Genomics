# ============================================================================
# VariationalBayes.jl - Variational Inference for Genomics
# ============================================================================
# Scalable approximate Bayesian inference using variational methods
# Implements CAVI, Stochastic VI, and automatic differentiation VI
# ============================================================================

"""
    VariationalResult

Structure containing results from variational inference.

# Fields
- `parameters::Dict{Symbol, Any}`: Variational parameters
- `elbo_history::Vector{Float64}`: ELBO values during optimization
- `converged::Bool`: Whether optimization converged
- `n_iterations::Int`: Number of iterations run
"""
struct VariationalResult
    parameters::Dict{Symbol, Any}
    elbo_history::Vector{Float64}
    converged::Bool
    n_iterations::Int
end

"""
    MeanFieldVariational

Mean-field variational family where q(θ) = Π_i q_i(θ_i).

# Fields
- `means::Vector{Float64}`: Variational means
- `log_stds::Vector{Float64}`: Log of variational standard deviations
"""
mutable struct MeanFieldVariational
    means::Vector{Float64}
    log_stds::Vector{Float64}
end

"""
    compute_elbo(log_joint::Function, q::MeanFieldVariational,
                n_samples::Int=100) -> Float64

Estimate Evidence Lower BOund (ELBO) using Monte Carlo sampling.

# Arguments
- `log_joint`: Function computing log p(y, θ)
- `q`: Variational distribution
- `n_samples`: Number of MC samples for estimation

# Returns
- `Float64`: Estimated ELBO value

# Mathematical Definition
ELBO = E_q[log p(y, θ)] - E_q[log q(θ)]
     = E_q[log p(y, θ)] + H[q]

where H[q] is the entropy of the variational distribution.

For mean-field Gaussian: H[q] = Σ_i (log σ_i + 0.5 * log(2πe))
"""
function compute_elbo(
    log_joint::Function,
    q::MeanFieldVariational,
    n_samples::Int=100
)
    n_params = length(q.means)
    stds = exp.(q.log_stds)

    # Monte Carlo estimate of E_q[log p(y, θ)]
    log_joint_sum = 0.0
    for _ in 1:n_samples
        # Reparameterization trick: θ = μ + σ * ε, ε ~ N(0, I)
        ε = randn(n_params)
        θ = q.means + stds .* ε
        log_joint_sum += log_joint(θ)
    end
    expected_log_joint = log_joint_sum / n_samples

    # Entropy of mean-field Gaussian
    entropy = sum(q.log_stds) + 0.5 * n_params * (1 + log(2π))

    return expected_log_joint + entropy
end

"""
    advi(log_joint::Function, n_params::Int; kwargs...) -> VariationalResult

Automatic Differentiation Variational Inference (ADVI).

# Arguments
- `log_joint`: Function computing log p(y, θ) for unconstrained θ
- `n_params`: Number of parameters

# Keyword Arguments
- `n_iterations::Int=10000`: Maximum iterations
- `learning_rate::Float64=0.01`: Initial learning rate
- `n_samples::Int=10`: MC samples per gradient estimate
- `tolerance::Float64=1e-6`: Convergence tolerance
- `verbose::Bool=true`: Print progress

# Algorithm Details
ADVI transforms constrained parameters to unconstrained space and
optimizes mean-field Gaussian variational parameters using stochastic
gradient ascent on the ELBO.

Uses the reparameterization gradient estimator:
∇_φ ELBO ≈ (1/S) Σ_s ∇_φ [log p(y, g(ε_s; φ)) + log |det J_g|]

where g(ε; φ) = μ + σ * ε is the reparameterization.

# Example
```julia
function log_joint(θ)
    β, log_σ = θ[1:end-1], θ[end]
    σ = exp(log_σ)
    log_lik = -n/2*log(2π) - n*log_σ - sum((y - X*β).^2)/(2σ^2)
    log_prior = -sum(β.^2)/2 - 2*log_σ  # Normal + log-normal priors
    return log_lik + log_prior
end

result = advi(log_joint, p+1)
posterior_mean = result.parameters[:means]
```

# References
- Kucukelbir et al. (2017) J. Mach. Learn. Res.
- Ranganath et al. (2014) AISTATS
"""
function advi(
    log_joint::Function,
    n_params::Int;
    n_iterations::Int=10000,
    learning_rate::Float64=0.01,
    n_samples::Int=10,
    tolerance::Float64=1e-6,
    momentum::Float64=0.9,
    verbose::Bool=true
)
    # Initialize variational parameters
    q = MeanFieldVariational(
        zeros(n_params),      # means
        zeros(n_params)       # log_stds (σ = 1)
    )

    # Adam optimizer state
    m_mean = zeros(n_params)
    v_mean = zeros(n_params)
    m_logstd = zeros(n_params)
    v_logstd = zeros(n_params)
    β1, β2 = 0.9, 0.999
    ε_adam = 1e-8

    elbo_history = Float64[]

    if verbose
        prog = Progress(n_iterations; desc="ADVI: ")
    end

    for iter in 1:n_iterations
        stds = exp.(q.log_stds)

        # Compute gradients using reparameterization trick
        grad_mean = zeros(n_params)
        grad_logstd = zeros(n_params)
        elbo_est = 0.0

        for _ in 1:n_samples
            ε = randn(n_params)
            θ = q.means + stds .* ε

            # Compute log joint and its gradient
            lj = log_joint(θ)
            elbo_est += lj

            # Numerical gradient approximation
            δ = 1e-5
            for i in 1:n_params
                θ_plus = copy(θ)
                θ_plus[i] += δ
                θ_minus = copy(θ)
                θ_minus[i] -= δ
                grad_θ_i = (log_joint(θ_plus) - log_joint(θ_minus)) / (2δ)

                # Chain rule for reparameterization
                grad_mean[i] += grad_θ_i
                grad_logstd[i] += grad_θ_i * stds[i] * ε[i]
            end
        end

        grad_mean ./= n_samples
        grad_logstd ./= n_samples

        # Add entropy gradient: ∂H/∂log_σ = 1
        grad_logstd .+= 1.0

        # Adam update for means
        m_mean = β1 * m_mean + (1 - β1) * grad_mean
        v_mean = β2 * v_mean + (1 - β2) * grad_mean.^2
        m_hat = m_mean / (1 - β1^iter)
        v_hat = v_mean / (1 - β2^iter)
        q.means .+= learning_rate * m_hat ./ (sqrt.(v_hat) .+ ε_adam)

        # Adam update for log_stds
        m_logstd = β1 * m_logstd + (1 - β1) * grad_logstd
        v_logstd = β2 * v_logstd + (1 - β2) * grad_logstd.^2
        m_hat = m_logstd / (1 - β1^iter)
        v_hat = v_logstd / (1 - β2^iter)
        q.log_stds .+= learning_rate * m_hat ./ (sqrt.(v_hat) .+ ε_adam)

        # Compute and store ELBO
        entropy = sum(q.log_stds) + 0.5 * n_params * (1 + log(2π))
        elbo = elbo_est / n_samples + entropy
        push!(elbo_history, elbo)

        # Check convergence
        if iter > 100
            recent_change = abs(elbo_history[end] - elbo_history[end-100]) /
                           (abs(elbo_history[end-100]) + 1e-10)
            if recent_change < tolerance
                if verbose
                    println("\nConverged at iteration $iter")
                end
                break
            end
        end

        if verbose
            next!(prog)
        end
    end

    converged = length(elbo_history) < n_iterations

    return VariationalResult(
        Dict(
            :means => q.means,
            :stds => exp.(q.log_stds),
            :log_stds => q.log_stds
        ),
        elbo_history,
        converged,
        length(elbo_history)
    )
end

"""
    coordinate_ascent_vi(E_step::Function, M_step::Function,
                        initial::Dict; kwargs...) -> VariationalResult

Coordinate Ascent Variational Inference (CAVI) for conjugate models.

# Arguments
- `E_step`: Function updating variational parameters for latent variables
- `M_step`: Function updating model parameters (if applicable)
- `initial`: Initial variational parameters

# Keyword Arguments
- `n_iterations::Int=1000`: Maximum iterations
- `tolerance::Float64=1e-6`: Convergence tolerance
- `compute_elbo::Function=nothing`: Optional ELBO computation function

# Algorithm Details
CAVI iteratively updates each variational factor while holding others fixed:
q_j^{t+1}(θ_j) ∝ exp(E_{q_{-j}}[log p(y, θ)])

For exponential family models with conjugate priors, these updates
have closed-form solutions.

# Example
```julia
# Variational EM for Gaussian mixture
function E_step(params, data)
    # Update responsibilities
    r = compute_responsibilities(params, data)
    return merge(params, Dict(:r => r))
end

function M_step(params, data)
    # Update cluster parameters
    μ, Σ = update_cluster_params(params[:r], data)
    return merge(params, Dict(:μ => μ, :Σ => Σ))
end

result = coordinate_ascent_vi(E_step, M_step, initial_params; data=y)
```

# References
- Bishop (2006) Pattern Recognition and Machine Learning, Ch. 10
- Blei et al. (2017) J. Am. Stat. Assoc.
"""
function coordinate_ascent_vi(
    E_step::Function,
    M_step::Function,
    initial::Dict;
    n_iterations::Int=1000,
    tolerance::Float64=1e-6,
    compute_elbo_fn::Union{Function, Nothing}=nothing,
    data::Any=nothing,
    verbose::Bool=true
)
    params = deepcopy(initial)
    elbo_history = Float64[]

    if verbose
        prog = Progress(n_iterations; desc="CAVI: ")
    end

    for iter in 1:n_iterations
        # E-step: update variational parameters
        params = E_step(params, data)

        # M-step: update model parameters (if applicable)
        params = M_step(params, data)

        # Compute ELBO if function provided
        if compute_elbo_fn !== nothing
            elbo = compute_elbo_fn(params, data)
            push!(elbo_history, elbo)

            # Check convergence
            if iter > 1
                rel_change = abs(elbo - elbo_history[end-1]) /
                            (abs(elbo_history[end-1]) + 1e-10)
                if rel_change < tolerance
                    if verbose
                        println("\nConverged at iteration $iter")
                    end
                    break
                end
            end
        end

        if verbose
            next!(prog)
        end
    end

    converged = length(elbo_history) > 0 && length(elbo_history) < n_iterations

    return VariationalResult(params, elbo_history, converged, length(elbo_history))
end

"""
    variational_linear_regression(y::Vector{Float64}, X::Matrix{Float64};
                                 prior_precision::Float64=1.0) -> VariationalResult

Variational Bayesian linear regression with conjugate priors.

# Model
y ~ N(Xβ, σ²I)
β ~ N(0, α⁻¹I)
α ~ Gamma(a_0, b_0)
τ = 1/σ² ~ Gamma(c_0, d_0)

# Arguments
- `y`: Response vector (n × 1)
- `X`: Design matrix (n × p)
- `prior_precision`: Prior precision for β (default 1.0)

# Returns
- `VariationalResult` with posterior parameters

# References
- Bishop (2006) PRML, Section 10.3
"""
function variational_linear_regression(
    y::Vector{Float64},
    X::Matrix{Float64};
    prior_precision::Float64=1.0,
    n_iterations::Int=100,
    tolerance::Float64=1e-8,
    verbose::Bool=true
)
    n, p = size(X)

    # Prior hyperparameters
    a_0, b_0 = 1e-3, 1e-3  # For α
    c_0, d_0 = 1e-3, 1e-3  # For τ

    # Initialize variational parameters
    E_α = prior_precision
    E_τ = 1.0

    # Precompute
    XtX = X' * X
    Xty = X' * y
    yty = y' * y

    elbo_history = Float64[]

    for iter in 1:n_iterations
        # Update q(β) = N(m_β, S_β)
        S_β_inv = E_τ * XtX + E_α * I
        S_β = inv(S_β_inv)
        m_β = E_τ * S_β * Xty

        # E[β'β]
        E_βtβ = m_β' * m_β + tr(S_β)

        # Update q(α) = Gamma(a_n, b_n)
        a_n = a_0 + p / 2
        b_n = b_0 + E_βtβ / 2
        E_α = a_n / b_n

        # Update q(τ) = Gamma(c_n, d_n)
        # E[(y - Xβ)'(y - Xβ)] = yty - 2y'X*m_β + tr(XtX * (S_β + m_β*m_β'))
        residual_ss = yty - 2 * Xty' * m_β + tr(XtX * (S_β + m_β * m_β'))
        c_n = c_0 + n / 2
        d_n = d_0 + residual_ss / 2
        E_τ = c_n / d_n

        # Compute ELBO
        elbo = compute_vb_linear_elbo(
            m_β, S_β, a_n, b_n, c_n, d_n,
            a_0, b_0, c_0, d_0, n, p, yty, Xty, XtX
        )
        push!(elbo_history, elbo)

        # Check convergence
        if iter > 1
            if abs(elbo - elbo_history[end-1]) < tolerance
                if verbose
                    println("Converged at iteration $iter")
                end
                break
            end
        end
    end

    return VariationalResult(
        Dict(
            :beta_mean => m_β,
            :beta_cov => S_β,
            :alpha_shape => a_n,
            :alpha_rate => b_n,
            :tau_shape => c_n,
            :tau_rate => d_n,
            :E_alpha => E_α,
            :E_tau => E_τ
        ),
        elbo_history,
        length(elbo_history) < n_iterations,
        length(elbo_history)
    )
end

"""
    compute_vb_linear_elbo(m_β, S_β, a_n, b_n, c_n, d_n,
                          a_0, b_0, c_0, d_0, n, p, yty, Xty, XtX) -> Float64

Compute ELBO for variational Bayesian linear regression.
"""
function compute_vb_linear_elbo(
    m_β, S_β, a_n, b_n, c_n, d_n,
    a_0, b_0, c_0, d_0, n, p, yty, Xty, XtX
)
    E_α = a_n / b_n
    E_log_α = digamma(a_n) - log(b_n)
    E_τ = c_n / d_n
    E_log_τ = digamma(c_n) - log(d_n)

    # E[log p(y|β, τ)]
    E_βtβ = m_β' * m_β + tr(S_β)
    residual_ss = yty - 2 * Xty' * m_β + tr(XtX * (S_β + m_β * m_β'))
    term1 = n/2 * (E_log_τ - log(2π)) - E_τ/2 * residual_ss

    # E[log p(β|α)]
    term2 = p/2 * (E_log_α - log(2π)) - E_α/2 * E_βtβ

    # E[log p(α)]
    term3 = a_0 * log(b_0) - loggamma(a_0) + (a_0 - 1) * E_log_α - b_0 * E_α

    # E[log p(τ)]
    term4 = c_0 * log(d_0) - loggamma(c_0) + (c_0 - 1) * E_log_τ - d_0 * E_τ

    # -E[log q(β)]
    term5 = 0.5 * logdet(S_β) + p/2 * (1 + log(2π))

    # -E[log q(α)]
    term6 = -a_n * log(b_n) + loggamma(a_n) - (a_n - 1) * digamma(a_n) + a_n

    # -E[log q(τ)]
    term7 = -c_n * log(d_n) + loggamma(c_n) - (c_n - 1) * digamma(c_n) + c_n

    return term1 + term2 + term3 + term4 + term5 + term6 + term7
end

"""
    variational_spike_slab(y::Vector{Float64}, X::Matrix{Float64};
                          π0::Float64=0.9) -> VariationalResult

Variational inference for spike-and-slab regression (variable selection).

# Model
y ~ N(Xβ, σ²I)
β_j | γ_j ~ γ_j * N(0, σ²_β) + (1 - γ_j) * δ_0
γ_j ~ Bernoulli(1 - π_0)

where π_0 is the prior probability of exclusion (spike).

# Arguments
- `y`: Response vector
- `X`: Design matrix
- `π0`: Prior probability of exclusion (default 0.9)

# Returns
Posterior inclusion probabilities (PIPs) and effect estimates

# References
- Carbonetto & Stephens (2012) Bayesian Anal.
- George & McCulloch (1993) J. Am. Stat. Assoc.
"""
function variational_spike_slab(
    y::Vector{Float64},
    X::Matrix{Float64};
    π0::Float64=0.9,
    σ2_β::Float64=1.0,
    n_iterations::Int=1000,
    tolerance::Float64=1e-6,
    verbose::Bool=true
)
    n, p = size(X)

    # Initialize
    α = fill(1 - π0, p)  # Posterior inclusion probabilities
    μ = zeros(p)         # Posterior means (conditional on inclusion)
    σ2_post = fill(σ2_β, p)  # Posterior variances
    σ2 = var(y)          # Residual variance estimate

    # Precompute
    Xty = X' * y
    X_norms = sum(X.^2, dims=1)[:]

    elbo_history = Float64[]

    if verbose
        prog = Progress(n_iterations; desc="Spike-Slab VI: ")
    end

    for iter in 1:n_iterations
        # Update each variable
        for j in 1:p
            # Residual excluding variable j
            r_j = y - X * (α .* μ) + X[:, j] * (α[j] * μ[j])

            # Posterior variance
            σ2_post[j] = 1 / (X_norms[j] / σ2 + 1 / σ2_β)

            # Posterior mean (conditional on inclusion)
            μ[j] = σ2_post[j] * (X[:, j]' * r_j) / σ2

            # Log Bayes factor for inclusion
            log_BF = 0.5 * log(σ2_post[j] / σ2_β) +
                     0.5 * μ[j]^2 / σ2_post[j]

            # Posterior inclusion probability
            log_odds = log_BF + log(1 - π0) - log(π0)
            α[j] = 1 / (1 + exp(-log_odds))
        end

        # Update residual variance
        expected_residuals = y - X * (α .* μ)
        expected_ss = expected_residuals' * expected_residuals
        for j in 1:p
            expected_ss += X_norms[j] * α[j] * (σ2_post[j] + μ[j]^2 * (1 - α[j]))
        end
        σ2 = expected_ss / n

        # Simple convergence check based on PIPs
        if iter > 1
            # Could compute full ELBO here
            push!(elbo_history, -expected_ss / (2 * σ2))
            if iter > 10
                recent_change = abs(elbo_history[end] - elbo_history[end-1])
                if recent_change < tolerance
                    if verbose
                        println("\nConverged at iteration $iter")
                    end
                    break
                end
            end
        else
            push!(elbo_history, -expected_ss / (2 * σ2))
        end

        if verbose
            next!(prog)
        end
    end

    return VariationalResult(
        Dict(
            :pip => α,
            :beta_mean => α .* μ,
            :beta_conditional_mean => μ,
            :beta_var => σ2_post,
            :sigma2 => σ2,
            :selected => findall(α .> 0.5)
        ),
        elbo_history,
        length(elbo_history) < n_iterations,
        length(elbo_history)
    )
end

"""
    stochastic_vi(log_joint::Function, n_params::Int, n_data::Int;
                 kwargs...) -> VariationalResult

Stochastic Variational Inference for large datasets.

# Arguments
- `log_joint`: Function computing log p(y_batch, θ) for a mini-batch
- `n_params`: Number of parameters
- `n_data`: Total number of data points

# Keyword Arguments
- `batch_size::Int=100`: Mini-batch size
- `n_iterations::Int=10000`: Number of iterations
- `learning_rate::Float64=0.01`: Initial learning rate
- `decay_rate::Float64=0.0001`: Learning rate decay

# Algorithm Details
SVI uses stochastic gradients estimated from mini-batches:
∇_φ ELBO ≈ (N/B) * ∇_φ Σ_{i∈batch} log p(y_i | θ) + ∇_φ [log p(θ) - log q(θ)]

Uses Robbins-Monro learning rate: ρ_t = ρ_0 * (1 + t * κ)^{-0.75}

# References
- Hoffman et al. (2013) J. Mach. Learn. Res.
"""
function stochastic_vi(
    log_joint_batch::Function,
    n_params::Int,
    n_data::Int;
    batch_size::Int=100,
    n_iterations::Int=10000,
    learning_rate::Float64=0.01,
    decay_rate::Float64=0.0001,
    n_samples::Int=5,
    verbose::Bool=true
)
    # Initialize mean-field parameters
    q = MeanFieldVariational(
        zeros(n_params),
        zeros(n_params)
    )

    # Natural gradient accumulators (for momentum)
    nat_grad_mean = zeros(n_params)
    nat_grad_var = zeros(n_params)

    elbo_history = Float64[]
    scale_factor = n_data / batch_size

    if verbose
        prog = Progress(n_iterations; desc="Stochastic VI: ")
    end

    for iter in 1:n_iterations
        # Learning rate schedule (Robbins-Monro)
        ρ = learning_rate * (1 + iter * decay_rate)^(-0.75)

        stds = exp.(q.log_stds)

        # Estimate stochastic gradient
        grad_mean = zeros(n_params)
        grad_logstd = zeros(n_params)

        for _ in 1:n_samples
            ε = randn(n_params)
            θ = q.means + stds .* ε

            # Numerical gradient (batch is sampled inside log_joint_batch)
            δ = 1e-5
            for i in 1:n_params
                θ_plus = copy(θ)
                θ_plus[i] += δ
                θ_minus = copy(θ)
                θ_minus[i] -= δ

                grad_θ_i = scale_factor *
                          (log_joint_batch(θ_plus) - log_joint_batch(θ_minus)) / (2δ)

                grad_mean[i] += grad_θ_i
                grad_logstd[i] += grad_θ_i * stds[i] * ε[i]
            end
        end

        grad_mean ./= n_samples
        grad_logstd ./= n_samples
        grad_logstd .+= 1.0  # Entropy gradient

        # Stochastic natural gradient update
        nat_grad_mean = (1 - ρ) * nat_grad_mean + ρ * grad_mean
        nat_grad_var = (1 - ρ) * nat_grad_var + ρ * grad_logstd

        q.means .+= ρ * nat_grad_mean
        q.log_stds .+= ρ * nat_grad_var

        # Periodically estimate ELBO
        if iter % 100 == 0
            elbo_est = 0.0
            for _ in 1:10
                θ = q.means + stds .* randn(n_params)
                elbo_est += scale_factor * log_joint_batch(θ)
            end
            entropy = sum(q.log_stds) + 0.5 * n_params * (1 + log(2π))
            push!(elbo_history, elbo_est / 10 + entropy)
        end

        if verbose
            next!(prog)
        end
    end

    return VariationalResult(
        Dict(
            :means => q.means,
            :stds => exp.(q.log_stds)
        ),
        elbo_history,
        true,  # No convergence check for SVI
        n_iterations
    )
end

"""
    sample_from_variational(result::VariationalResult, n_samples::Int) -> Matrix{Float64}

Draw samples from the fitted variational distribution.
"""
function sample_from_variational(result::VariationalResult, n_samples::Int)
    means = result.parameters[:means]
    stds = result.parameters[:stds]
    n_params = length(means)

    samples = zeros(n_samples, n_params)
    for i in 1:n_samples
        samples[i, :] = means + stds .* randn(n_params)
    end

    return samples
end
