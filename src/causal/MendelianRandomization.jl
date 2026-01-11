# ============================================================================
# MendelianRandomization.jl - Causal Inference Methods
# ============================================================================

"""
    mendelian_randomization(betas_x::Vector{Float64}, ses_x::Vector{Float64},
                           betas_y::Vector{Float64}, ses_y::Vector{Float64};
                           method::Symbol=:ivw)

Perform Mendelian randomization analysis.
"""
function mendelian_randomization(betas_x::Vector{Float64}, ses_x::Vector{Float64},
                                betas_y::Vector{Float64}, ses_y::Vector{Float64};
                                method::Symbol=:ivw)
    if method == :ivw
        return ivw_method(betas_x, ses_x, betas_y, ses_y)
    elseif method == :egger
        return mr_egger(betas_x, ses_x, betas_y, ses_y)
    elseif method == :weighted_median
        return weighted_median(betas_x, ses_x, betas_y, ses_y)
    else
        throw(ArgumentError("Unknown method: $method"))
    end
end

"""
    ivw_method(betas_x, ses_x, betas_y, ses_y)

Inverse-variance weighted MR estimator.
"""
function ivw_method(betas_x::Vector{Float64}, ses_x::Vector{Float64},
                   betas_y::Vector{Float64}, ses_y::Vector{Float64})
    # Ratio estimates
    ratios = betas_y ./ betas_x
    
    # Weights (inverse variance of ratio)
    weights = (betas_x.^2) ./ (ses_y.^2)
    
    # IVW estimate
    beta_ivw = sum(weights .* ratios) / sum(weights)
    se_ivw = sqrt(1 / sum(weights))
    
    z = beta_ivw / se_ivw
    pvalue = 2 * ccdf(Normal(), abs(z))
    
    return (beta=beta_ivw, se=se_ivw, z_statistic=z, pvalue=pvalue, method="IVW")
end

"""
    mr_egger(betas_x, ses_x, betas_y, ses_y)

MR-Egger regression for pleiotropy-robust estimation.
"""
function mr_egger(betas_x::Vector{Float64}, ses_x::Vector{Float64},
                 betas_y::Vector{Float64}, ses_y::Vector{Float64})
    n = length(betas_x)
    
    # Weighted regression of betas_y on betas_x with intercept
    weights = 1.0 ./ (ses_y.^2)
    
    X = hcat(ones(n), betas_x)
    y = betas_y
    W = Diagonal(weights)
    
    beta = (X' * W * X) \ (X' * W * y)
    intercept = beta[1]
    slope = beta[2]
    
    # Standard errors
    residuals = y - X * beta
    sigma2 = sum(weights .* residuals.^2) / (n - 2)
    var_beta = sigma2 * inv(X' * W * X)
    
    se_intercept = sqrt(var_beta[1, 1])
    se_slope = sqrt(var_beta[2, 2])
    
    z = slope / se_slope
    pvalue = 2 * ccdf(Normal(), abs(z))
    
    # Intercept test for pleiotropy
    z_intercept = intercept / se_intercept
    pleiotropy_pvalue = 2 * ccdf(Normal(), abs(z_intercept))
    
    return (beta=slope, se=se_slope, z_statistic=z, pvalue=pvalue,
            intercept=intercept, intercept_pvalue=pleiotropy_pvalue, method="MR-Egger")
end

"""
    weighted_median(betas_x, ses_x, betas_y, ses_y)

Weighted median MR estimator (robust to up to 50% invalid instruments).
"""
function weighted_median(betas_x::Vector{Float64}, ses_x::Vector{Float64},
                        betas_y::Vector{Float64}, ses_y::Vector{Float64})
    ratios = betas_y ./ betas_x
    weights = (betas_x.^2) ./ (ses_y.^2)
    weights ./= sum(weights)
    
    # Sort by ratio
    order = sortperm(ratios)
    sorted_ratios = ratios[order]
    sorted_weights = weights[order]
    
    # Find weighted median
    cumulative_weight = cumsum(sorted_weights)
    median_idx = findfirst(x -> x >= 0.5, cumulative_weight)
    beta_wm = sorted_ratios[median_idx]
    
    # Bootstrap SE
    n_boot = 1000
    boot_estimates = Float64[]
    for _ in 1:n_boot
        boot_idx = sample(1:length(ratios), length(ratios))
        boot_ratios = ratios[boot_idx]
        boot_weights = weights[boot_idx]
        boot_weights ./= sum(boot_weights)
        
        order_boot = sortperm(boot_ratios)
        cum_w = cumsum(boot_weights[order_boot])
        mid = findfirst(x -> x >= 0.5, cum_w)
        push!(boot_estimates, boot_ratios[order_boot][mid])
    end
    
    se_wm = std(boot_estimates)
    z = beta_wm / se_wm
    pvalue = 2 * ccdf(Normal(), abs(z))
    
    return (beta=beta_wm, se=se_wm, z_statistic=z, pvalue=pvalue, method="Weighted median")
end
