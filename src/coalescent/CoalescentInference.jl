# ============================================================================
# CoalescentInference.jl - Parameter Inference from Coalescent
# ============================================================================

"""
    demographic_inference(sfs::Vector{Float64}; n_samples::Int)

Infer demographic history from site frequency spectrum.
Returns estimate of population size changes.
"""
function demographic_inference(sfs::Vector{Float64}; n_samples::Int)
    n = n_samples
    k = length(sfs)
    
    @assert k == n - 1 "SFS should have n-1 entries"
    
    # Under neutrality: ξ_i = θ/i
    # Deviations suggest demographic changes
    
    # Estimate θ from each frequency class
    theta_estimates = [sfs[i] * i for i in 1:k]
    theta_mean = mean(theta_estimates)
    
    # Calculate deviation from expected
    expected_sfs = [theta_mean / i for i in 1:k]
    deviations = sfs .- expected_sfs
    
    # Infer expansion/contraction
    # Excess of singletons: population expansion
    # Deficit of singletons: population contraction
    
    singleton_ratio = sfs[1] / expected_sfs[1]
    
    demographic_signal = if singleton_ratio > 1.2
        :expansion
    elseif singleton_ratio < 0.8
        :contraction
    else
        :stable
    end
    
    return (theta=theta_mean,
            observed_sfs=sfs,
            expected_sfs=expected_sfs,
            deviations=deviations,
            demographic_signal=demographic_signal,
            singleton_excess=singleton_ratio)
end

"""
    skyline_plot_data(coalescent_times::Vector{Vector{Float64}}, Ne0::Float64)

Prepare data for skyline plot from multiple gene trees.
Estimates effective population size through time.
"""
function skyline_plot_data(coalescent_times::Vector{Vector{Float64}}, Ne0::Float64=1.0)
    # Combine all coalescence times
    all_times = Float64[]
    for ct in coalescent_times
        append!(all_times, ct)
    end
    
    if isempty(all_times)
        return (times=Float64[], Ne_estimates=Float64[])
    end
    
    sort!(all_times)
    
    # Binned skyline
    n_bins = min(20, length(all_times))
    time_bins = range(0, maximum(all_times), length=n_bins+1)
    
    Ne_estimates = Float64[]
    time_points = Float64[]
    
    for i in 1:n_bins
        t_start, t_end = time_bins[i], time_bins[i+1]
        
        # Count coalescence events in this interval
        n_events = count(t -> t_start <= t < t_end, all_times)
        
        # Estimate Ne from coalescence rate
        # Rate = k(k-1)/(4Ne), so Ne ≈ k(k-1) * Δt / (4 * n_events)
        interval_length = t_end - t_start
        
        if n_events > 0
            # Rough estimate assuming k decreases linearly
            k_avg = length(coalescent_times[1]) / 2  # Simplified
            Ne_est = k_avg * (k_avg - 1) * interval_length / (4 * n_events)
            push!(Ne_estimates, Ne_est * Ne0)
        else
            push!(Ne_estimates, Ne0)
        end
        
        push!(time_points, (t_start + t_end) / 2)
    end
    
    return (times=time_points, Ne_estimates=Ne_estimates)
end

"""
    coalescent_model_selection(sfs::Vector{Float64}, n_samples::Int)

Compare coalescent models (constant size vs growth) using AIC.
"""
function coalescent_model_selection(sfs::Vector{Float64}, n_samples::Int)
    # Model 1: Constant population size
    constant_ll = constant_size_likelihood(sfs, n_samples)
    aic_constant = 2 * 1 - 2 * constant_ll  # 1 parameter (θ)
    
    # Model 2: Exponential growth
    growth_ll, growth_rate = growth_model_likelihood(sfs, n_samples)
    aic_growth = 2 * 2 - 2 * growth_ll  # 2 parameters (θ, r)
    
    best_model = aic_constant <= aic_growth ? :constant : :growth
    
    return (constant_aic=aic_constant,
            growth_aic=aic_growth,
            growth_rate=growth_rate,
            best_model=best_model,
            delta_aic=abs(aic_constant - aic_growth))
end

"""Calculate likelihood under constant size model."""
function constant_size_likelihood(sfs::Vector{Float64}, n_samples::Int)
    k = length(sfs)
    
    # MLE for θ
    theta_mle = sum(sfs) / sum(1.0/i for i in 1:k)
    
    # Log likelihood (Poisson for each class)
    ll = 0.0
    for i in 1:k
        expected = theta_mle / i
        if expected > 0 && sfs[i] >= 0
            ll += sfs[i] * log(expected) - expected  # Poisson log-likelihood
        end
    end
    
    return ll
end

"""Calculate likelihood under exponential growth model."""
function growth_model_likelihood(sfs::Vector{Float64}, n_samples::Int)
    k = length(sfs)
    
    # Grid search for growth rate
    best_ll = -Inf
    best_r = 0.0
    
    for r in range(-0.1, 0.1, length=21)
        # Expected SFS under growth (simplified)
        # More low-frequency variants under expansion
        theta_trial = sum(sfs) / sum(1.0/i for i in 1:k)
        
        ll = 0.0
        for i in 1:k
            # Growth modifies expected counts
            modifier = exp(-r * i / k)
            expected = theta_trial / i * modifier
            if expected > 0 && sfs[i] >= 0
                ll += sfs[i] * log(expected) - expected
            end
        end
        
        if ll > best_ll
            best_ll = ll
            best_r = r
        end
    end
    
    return (best_ll, best_r)
end

"""
    estimate_theta_watterson(n_segregating::Int, n_samples::Int, seq_length::Int)

Watterson's estimator of θ = 4Neμ.
"""
function estimate_theta_watterson(n_segregating::Int, n_samples::Int, seq_length::Int)
    a_n = sum(1.0/i for i in 1:(n_samples-1))
    theta_w = n_segregating / (a_n * seq_length)
    return theta_w
end

"""
    estimate_theta_pi(pairwise_diffs::Float64, seq_length::Int)

Nucleotide diversity (π) estimator of θ.
"""
function estimate_theta_pi(pairwise_diffs::Float64, seq_length::Int)
    return pairwise_diffs / seq_length
end
