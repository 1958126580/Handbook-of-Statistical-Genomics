# ============================================================================
# Statistics.jl - Statistical Utility Functions
# ============================================================================

"""
    welch_t_test(x::AbstractVector, y::AbstractVector)

Perform Welch's t-test for unequal variances.

# Returns
- StatisticalTestResult with t-statistic, p-value, and degrees of freedom
"""
function welch_t_test(x::AbstractVector, y::AbstractVector)
    x_clean = collect(skipmissing(x))
    y_clean = collect(skipmissing(y))
    
    n1, n2 = length(x_clean), length(y_clean)
    m1, m2 = mean(x_clean), mean(y_clean)
    v1, v2 = var(x_clean), var(y_clean)
    
    # Welch's t-statistic
    se = sqrt(v1/n1 + v2/n2)
    t_stat = (m1 - m2) / se
    
    # Welch-Satterthwaite degrees of freedom
    df = (v1/n1 + v2/n2)^2 / ((v1/n1)^2/(n1-1) + (v2/n2)^2/(n2-1))
    
    # Two-tailed p-value
    pval = 2 * ccdf(TDist(df), abs(t_stat))
    
    StatisticalTestResult(t_stat, pval, df, "Welch's t-test")
end

"""
    chi_squared_test(observed::AbstractVector, expected::AbstractVector)

Perform chi-squared goodness of fit test.
"""
function chi_squared_test(observed::AbstractVector, expected::AbstractVector)
    @assert length(observed) == length(expected)
    @assert all(expected .> 0) "Expected values must be positive"
    
    chi2 = sum((observed .- expected).^2 ./ expected)
    df = length(observed) - 1
    pval = ccdf(Chisq(df), chi2)
    
    StatisticalTestResult(chi2, pval, df, "Chi-squared test")
end

"""
    fisher_exact_2x2(a::Int, b::Int, c::Int, d::Int)

Fisher's exact test for a 2x2 contingency table.
"""
function fisher_exact_2x2(a::Int, b::Int, c::Int, d::Int)
    # Calculate hypergeometric probability
    n = a + b + c + d
    row1 = a + b
    col1 = a + c
    
    # Use log factorials for numerical stability
    function log_factorial(n)
        n <= 1 ? 0.0 : sum(log(i) for i in 2:n)
    end
    
    log_p = (log_factorial(row1) + log_factorial(n - row1) + 
             log_factorial(col1) + log_factorial(n - col1) -
             log_factorial(n) - log_factorial(a) - log_factorial(b) - 
             log_factorial(c) - log_factorial(d))
    
    # For a proper implementation, sum probabilities for more extreme tables
    # This is a simplified version
    pval = min(1.0, 2 * exp(log_p))
    
    odds_ratio = (a * d) / (b * c + 1e-10)
    
    StatisticalTestResult(odds_ratio, pval, nothing, "Fisher's exact test")
end

"""
    linear_regression(X::AbstractMatrix, y::AbstractVector)

Ordinary least squares linear regression.

# Returns
- NamedTuple with coefficients, standard errors, t-statistics, and p-values
"""
function linear_regression(X::AbstractMatrix, y::AbstractVector)
    n, p = size(X)
    
    # OLS: β = (X'X)^(-1) X'y
    XtX = X' * X
    Xty = X' * y
    β = XtX \ Xty
    
    # Residuals and variance
    residuals = y - X * β
    σ² = sum(residuals.^2) / (n - p)
    
    # Standard errors
    var_β = σ² * inv(XtX)
    se = sqrt.(diag(var_β))
    
    # t-statistics and p-values
    t_stats = β ./ se
    df = n - p
    pvals = 2 .* ccdf.(TDist(df), abs.(t_stats))
    
    # R-squared
    ss_res = sum(residuals.^2)
    ss_tot = sum((y .- mean(y)).^2)
    r_squared = 1 - ss_res / ss_tot
    
    (coefficients=β, se=se, t_statistics=t_stats, pvalues=pvals, 
     r_squared=r_squared, residuals=residuals, df=df)
end

"""
    logistic_regression(X::AbstractMatrix, y::AbstractVector; maxiter::Int=100, tol::Float64=1e-8)

Logistic regression using iteratively reweighted least squares (IRLS).
"""
function logistic_regression(X::AbstractMatrix, y::AbstractVector; 
                            maxiter::Int=100, tol::Float64=1e-8)
    n, p = size(X)
    β = zeros(p)
    
    for iter in 1:maxiter
        # Linear predictor
        η = X * β
        
        # Predicted probabilities (with numerical safeguards)
        μ = 1.0 ./ (1.0 .+ exp.(-clamp.(η, -500, 500)))
        
        # Weights
        w = μ .* (1.0 .- μ)
        w = max.(w, 1e-10)  # Avoid zero weights
        
        # Working response
        z = η + (y .- μ) ./ w
        
        # Weighted least squares step
        W = Diagonal(w)
        XtWX = X' * W * X
        XtWz = X' * W * z
        
        β_new = XtWX \ XtWz
        
        # Check convergence
        if maximum(abs.(β_new - β)) < tol
            β = β_new
            break
        end
        β = β_new
    end
    
    # Final predictions and variance
    η = X * β
    μ = 1.0 ./ (1.0 .+ exp.(-clamp.(η, -500, 500)))
    w = μ .* (1.0 .- μ)
    
    # Variance-covariance matrix
    W = Diagonal(max.(w, 1e-10))
    var_β = inv(X' * W * X)
    se = sqrt.(diag(var_β))
    
    # Wald statistics and p-values
    z_stats = β ./ se
    pvals = 2 .* ccdf.(Normal(), abs.(z_stats))
    
    (coefficients=β, se=se, z_statistics=z_stats, pvalues=pvals,
     fitted=μ, deviance=-2*sum(y.*log.(μ.+1e-10) + (1 .-y).*log.(1 .-μ.+1e-10)))
end

"""
    permutation_test(x::AbstractVector, y::AbstractVector, stat_func::Function; 
                    n_perm::Int=10000)

Permutation test for comparing two groups.
"""
function permutation_test(x::AbstractVector, y::AbstractVector, 
                         stat_func::Function; n_perm::Int=10000)
    combined = vcat(collect(skipmissing(x)), collect(skipmissing(y)))
    n_x = count(!ismissing, x)
    n_total = length(combined)
    
    observed_stat = stat_func(collect(skipmissing(x)), collect(skipmissing(y)))
    
    count_extreme = 0
    for _ in 1:n_perm
        perm = shuffle(combined)
        perm_x = perm[1:n_x]
        perm_y = perm[(n_x+1):end]
        perm_stat = stat_func(perm_x, perm_y)
        if abs(perm_stat) >= abs(observed_stat)
            count_extreme += 1
        end
    end
    
    pval = (count_extreme + 1) / (n_perm + 1)
    StatisticalTestResult(observed_stat, pval, nothing, "Permutation test")
end

"""
    correlation_test(x::AbstractVector, y::AbstractVector)

Test for Pearson correlation coefficient.
"""
function correlation_test(x::AbstractVector, y::AbstractVector)
    x_clean = collect(skipmissing(x))
    y_clean = collect(skipmissing(y))
    
    @assert length(x_clean) == length(y_clean)
    n = length(x_clean)
    
    r = cor(x_clean, y_clean)
    
    # Fisher's z transformation for confidence interval
    z = 0.5 * log((1 + r) / (1 - r))
    se_z = 1 / sqrt(n - 3)
    
    # t-statistic for significance test
    t_stat = r * sqrt(n - 2) / sqrt(1 - r^2)
    df = n - 2
    pval = 2 * ccdf(TDist(df), abs(t_stat))
    
    StatisticalTestResult(r, pval, df, "Pearson correlation test")
end
