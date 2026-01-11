# ============================================================================
# Clustering.jl - Population Clustering Methods
# ============================================================================

"""
    ClusteringResult

Result of population clustering analysis.
"""
struct ClusteringResult
    assignments::Vector{Int}
    proportions::Matrix{Float64}  # Admixture-style proportions (n_samples × K)
    K::Int
    log_likelihood::Float64
    bic::Float64
end

"""
    structure_clustering(gm::GenotypeMatrix, K::Int; maxiter::Int=100)

STRUCTURE-like model-based clustering.
Uses simplified EM algorithm for computational efficiency.
"""
function structure_clustering(gm::GenotypeMatrix, K::Int; maxiter::Int=100)
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    # Initialize proportions randomly
    Q = rand(n_samp, K)
    Q ./= sum(Q, dims=2)
    
    # Initialize allele frequencies per population
    P = rand(K, n_var) * 0.8 .+ 0.1
    
    log_lik = -Inf
    
    for iter in 1:maxiter
        prev_ll = log_lik
        
        # E-step: Update Q given P
        for i in 1:n_samp
            for k in 1:K
                prob = 1.0
                for j in 1:n_var
                    g = gm.data[i, j]
                    if !ismissing(g)
                        p_k = P[k, j]
                        # Binomial probability
                        prob *= binomial_prob(g, p_k)
                    end
                end
                Q[i, k] = prob * Q[i, k]
            end
        end
        Q ./= sum(Q, dims=2)
        Q .= max.(Q, 1e-6)
        Q ./= sum(Q, dims=2)
        
        # M-step: Update P given Q
        for k in 1:K
            for j in 1:n_var
                num = 0.0
                denom = 0.0
                for i in 1:n_samp
                    g = gm.data[i, j]
                    if !ismissing(g)
                        num += Q[i, k] * g
                        denom += Q[i, k] * 2
                    end
                end
                P[k, j] = denom > 0 ? clamp(num / denom, 0.01, 0.99) : 0.5
            end
        end
        
        # Calculate log-likelihood
        log_lik = 0.0
        for i in 1:n_samp
            for j in 1:n_var
                g = gm.data[i, j]
                if !ismissing(g)
                    prob = sum(Q[i, k] * binomial_prob(g, P[k, j]) for k in 1:K)
                    log_lik += log(max(prob, 1e-10))
                end
            end
        end
        
        if abs(log_lik - prev_ll) < 1e-4
            break
        end
    end
    
    # Hard assignments
    assignments = [argmax(Q[i, :]) for i in 1:n_samp]
    
    # BIC
    n_params = K * n_var + n_samp * (K - 1)
    n_obs = sum(!ismissing(gm.data[i, j]) for i in 1:n_samp for j in 1:n_var)
    bic = -2 * log_lik + n_params * log(n_obs)
    
    ClusteringResult(assignments, Q, K, log_lik, bic)
end

"""Binomial probability for genotype given allele frequency."""
function binomial_prob(g::Integer, p::Float64)
    if g == 0
        return (1-p)^2
    elseif g == 1
        return 2*p*(1-p)
    else
        return p^2
    end
end

"""
    optimal_k(gm::GenotypeMatrix; K_range::UnitRange{Int}=2:10)

Determine optimal K using Evanno's ΔK method.
"""
function optimal_k(gm::GenotypeMatrix; K_range::UnitRange{Int}=2:10, n_reps::Int=3)
    log_liks = Dict{Int, Vector{Float64}}()
    
    for K in K_range
        log_liks[K] = Float64[]
        for rep in 1:n_reps
            result = structure_clustering(gm, K; maxiter=50)
            push!(log_liks[K], result.log_likelihood)
        end
    end
    
    # Calculate ΔK (Evanno et al. 2005)
    K_vals = collect(K_range)
    L_mean = [mean(log_liks[K]) for K in K_vals]
    L_sd = [std(log_liks[K]) for K in K_vals]
    
    # First derivative: L'(K)
    L_prime = diff(L_mean)
    
    # Second derivative: L''(K)
    L_double_prime = diff(L_prime)
    
    # ΔK = |L''(K)| / sd(L(K))
    delta_K = Float64[]
    for i in 1:(length(K_vals)-2)
        K = K_vals[i+1]
        if L_sd[i+1] > 0
            push!(delta_K, abs(L_double_prime[i]) / L_sd[i+1])
        else
            push!(delta_K, 0.0)
        end
    end
    
    if isempty(delta_K)
        best_K = K_vals[1]
    else
        best_K = K_vals[argmax(delta_K) + 1]
    end
    
    return (best_K=best_K, delta_K=delta_K, log_likelihoods=L_mean,
            K_values=K_vals[2:end-1])
end
