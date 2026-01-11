# ============================================================================
# MachineLearning.jl - Machine Learning Methods for Genomics
# ============================================================================
# Penalized regression, random forests, gradient boosting, and feature selection
# for genomic prediction and variant prioritization
# ============================================================================

"""
    PenalizedRegressionResult

Structure containing results from penalized regression.

# Fields
- `coefficients::Vector{Float64}`: Estimated coefficients
- `intercept::Float64`: Intercept term
- `lambda::Float64`: Regularization parameter used
- `n_nonzero::Int`: Number of non-zero coefficients
- `cv_error::Float64`: Cross-validation error
- `method::String`: Method used (LASSO, Ridge, Elastic Net)
"""
struct PenalizedRegressionResult
    coefficients::Vector{Float64}
    intercept::Float64
    lambda::Float64
    n_nonzero::Int
    cv_error::Float64
    method::String
end

"""
    lasso(X::Matrix{Float64}, y::Vector{Float64}; kwargs...) -> PenalizedRegressionResult

LASSO (L1-penalized) regression using coordinate descent.

# Arguments
- `X`: Design matrix (n × p)
- `y`: Response vector

# Keyword Arguments
- `lambda::Union{Float64, Nothing}=nothing`: Regularization parameter (auto if nothing)
- `standardize::Bool=true`: Standardize predictors
- `max_iter::Int=1000`: Maximum iterations
- `tol::Float64=1e-6`: Convergence tolerance
- `n_folds::Int=5`: CV folds for lambda selection

# Mathematical Details
LASSO minimizes:
L(β) = (1/2n) ||y - Xβ||² + λ ||β||₁

Solution via coordinate descent:
β_j = S(z_j, λ) / (1 + 2λ)

where S(z, λ) = sign(z)(|z| - λ)₊ is the soft-thresholding operator.

# Example
```julia
# Fit LASSO for variant selection
result = lasso(genotypes, phenotype; lambda=0.01)

# Get selected variants
selected = findall(result.coefficients .!= 0)
```

# References
- Tibshirani (1996) J. R. Stat. Soc. B
- Friedman et al. (2010) J. Stat. Softw. (coordinate descent)
"""
function lasso(
    X::Matrix{Float64},
    y::Vector{Float64};
    lambda::Union{Float64, Nothing}=nothing,
    standardize::Bool=true,
    max_iter::Int=1000,
    tol::Float64=1e-6,
    n_folds::Int=5,
    verbose::Bool=false
)
    n, p = size(X)

    # Standardize if requested
    if standardize
        X_mean = mean(X, dims=1)
        X_std = std(X, dims=1)
        X_std[X_std .== 0] .= 1
        X_scaled = (X .- X_mean) ./ X_std
    else
        X_scaled = X
        X_mean = zeros(1, p)
        X_std = ones(1, p)
    end

    y_mean = mean(y)
    y_centered = y .- y_mean

    # Select lambda via CV if not provided
    if lambda === nothing
        lambda = select_lambda_cv(X_scaled, y_centered; n_folds=n_folds)
    end

    # Coordinate descent
    β = zeros(p)
    residuals = copy(y_centered)

    # Precompute X'X diagonal
    X_sq_sum = vec(sum(X_scaled.^2, dims=1))

    for iter in 1:max_iter
        β_old = copy(β)

        for j in 1:p
            # Partial residual
            r_j = residuals + X_scaled[:, j] * β[j]

            # Compute z_j = X_j' r_j / n
            z_j = dot(X_scaled[:, j], r_j) / n

            # Soft thresholding
            β[j] = soft_threshold(z_j, lambda) * n / X_sq_sum[j]

            # Update residuals
            residuals = r_j - X_scaled[:, j] * β[j]
        end

        # Check convergence
        if maximum(abs.(β - β_old)) < tol
            if verbose
                println("Converged at iteration $iter")
            end
            break
        end
    end

    # Transform back to original scale
    β_original = β ./ vec(X_std)
    intercept = y_mean - dot(vec(X_mean), β_original)

    # CV error estimate
    cv_error = mean(residuals.^2)

    n_nonzero = sum(β .!= 0)

    return PenalizedRegressionResult(
        vec(β_original),
        intercept,
        lambda,
        n_nonzero,
        cv_error,
        "LASSO"
    )
end

"""
    soft_threshold(z::Float64, lambda::Float64) -> Float64

Soft thresholding operator for LASSO.
"""
function soft_threshold(z::Float64, lambda::Float64)
    if z > lambda
        return z - lambda
    elseif z < -lambda
        return z + lambda
    else
        return 0.0
    end
end

"""
    ridge(X::Matrix{Float64}, y::Vector{Float64}; kwargs...) -> PenalizedRegressionResult

Ridge (L2-penalized) regression.

# Arguments
- `X`: Design matrix
- `y`: Response vector
- `lambda`: Regularization parameter

# Mathematical Details
Ridge minimizes:
L(β) = (1/2n) ||y - Xβ||² + λ ||β||²

Closed-form solution:
β = (X'X + λI)⁻¹ X'y

# Example
```julia
result = ridge(X, y; lambda=1.0)
```
"""
function ridge(
    X::Matrix{Float64},
    y::Vector{Float64};
    lambda::Union{Float64, Nothing}=nothing,
    standardize::Bool=true,
    n_folds::Int=5
)
    n, p = size(X)

    # Standardize
    if standardize
        X_mean = mean(X, dims=1)
        X_std = std(X, dims=1)
        X_std[X_std .== 0] .= 1
        X_scaled = (X .- X_mean) ./ X_std
    else
        X_scaled = X
        X_mean = zeros(1, p)
        X_std = ones(1, p)
    end

    y_mean = mean(y)
    y_centered = y .- y_mean

    # Select lambda via CV if not provided
    if lambda === nothing
        lambda = select_lambda_cv_ridge(X_scaled, y_centered; n_folds=n_folds)
    end

    # Closed-form solution
    XtX = X_scaled' * X_scaled
    Xty = X_scaled' * y_centered

    β = (XtX + lambda * I) \ Xty

    # Transform back
    β_original = β ./ vec(X_std)
    intercept = y_mean - dot(vec(X_mean), β_original)

    # CV error
    residuals = y_centered - X_scaled * β
    cv_error = mean(residuals.^2)

    return PenalizedRegressionResult(
        vec(β_original),
        intercept,
        lambda,
        sum(abs.(β) .> 1e-10),
        cv_error,
        "Ridge"
    )
end

"""
    elastic_net(X::Matrix{Float64}, y::Vector{Float64}; kwargs...) -> PenalizedRegressionResult

Elastic net regression (combined L1 and L2 penalty).

# Arguments
- `X`: Design matrix
- `y`: Response vector
- `alpha`: Mixing parameter (0=Ridge, 1=LASSO)
- `lambda`: Overall regularization strength

# Mathematical Details
Elastic net minimizes:
L(β) = (1/2n) ||y - Xβ||² + λ [α||β||₁ + (1-α)/2 ||β||²]

# Example
```julia
result = elastic_net(X, y; alpha=0.5, lambda=0.01)
```

# References
- Zou & Hastie (2005) J. R. Stat. Soc. B
"""
function elastic_net(
    X::Matrix{Float64},
    y::Vector{Float64};
    alpha::Float64=0.5,
    lambda::Union{Float64, Nothing}=nothing,
    standardize::Bool=true,
    max_iter::Int=1000,
    tol::Float64=1e-6,
    n_folds::Int=5
)
    n, p = size(X)

    # Standardize
    if standardize
        X_mean = mean(X, dims=1)
        X_std = std(X, dims=1)
        X_std[X_std .== 0] .= 1
        X_scaled = (X .- X_mean) ./ X_std
    else
        X_scaled = X
        X_mean = zeros(1, p)
        X_std = ones(1, p)
    end

    y_mean = mean(y)
    y_centered = y .- y_mean

    # Select lambda
    if lambda === nothing
        lambda = select_lambda_cv(X_scaled, y_centered; n_folds=n_folds) / alpha
    end

    # Coordinate descent with elastic net penalty
    β = zeros(p)
    residuals = copy(y_centered)
    X_sq_sum = vec(sum(X_scaled.^2, dims=1))

    for iter in 1:max_iter
        β_old = copy(β)

        for j in 1:p
            r_j = residuals + X_scaled[:, j] * β[j]
            z_j = dot(X_scaled[:, j], r_j) / n

            # Elastic net update
            λ1 = lambda * alpha
            λ2 = lambda * (1 - alpha)

            β[j] = soft_threshold(z_j, λ1) * n / (X_sq_sum[j] + n * λ2)

            residuals = r_j - X_scaled[:, j] * β[j]
        end

        if maximum(abs.(β - β_old)) < tol
            break
        end
    end

    # Transform back
    β_original = β ./ vec(X_std)
    intercept = y_mean - dot(vec(X_mean), β_original)

    cv_error = mean(residuals.^2)

    return PenalizedRegressionResult(
        vec(β_original),
        intercept,
        lambda,
        sum(β .!= 0),
        cv_error,
        "Elastic Net (α=$alpha)"
    )
end

"""
    select_lambda_cv(X, y; n_folds=5) -> Float64

Select optimal lambda using cross-validation.
"""
function select_lambda_cv(X, y; n_folds::Int=5, n_lambda::Int=50)
    n, p = size(X)

    # Lambda sequence
    lambda_max = maximum(abs.(X' * y)) / n
    lambda_min = lambda_max * 0.001
    lambdas = exp.(range(log(lambda_max), log(lambda_min), length=n_lambda))

    # CV folds
    fold_idx = [mod(i-1, n_folds) + 1 for i in randperm(n)]

    cv_errors = zeros(n_lambda)

    for (l_idx, λ) in enumerate(lambdas)
        fold_errors = zeros(n_folds)

        for fold in 1:n_folds
            train_idx = findall(fold_idx .!= fold)
            test_idx = findall(fold_idx .== fold)

            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]

            # Fit LASSO
            β = lasso_fit(X_train, y_train, λ)

            # Test error
            pred = X_test * β
            fold_errors[fold] = mean((y_test .- pred).^2)
        end

        cv_errors[l_idx] = mean(fold_errors)
    end

    # Select lambda with minimum CV error
    best_idx = argmin(cv_errors)

    return lambdas[best_idx]
end

"""
    lasso_fit(X, y, lambda; max_iter=100) -> Vector{Float64}

Fast LASSO fit for CV inner loop.
"""
function lasso_fit(X, y, lambda; max_iter::Int=100, tol::Float64=1e-4)
    n, p = size(X)
    β = zeros(p)
    X_sq_sum = vec(sum(X.^2, dims=1))

    for _ in 1:max_iter
        β_old = copy(β)

        for j in 1:p
            r_j = y - X * β + X[:, j] * β[j]
            z_j = dot(X[:, j], r_j) / n
            β[j] = soft_threshold(z_j, lambda) * n / X_sq_sum[j]
        end

        if maximum(abs.(β - β_old)) < tol
            break
        end
    end

    return β
end

"""
    select_lambda_cv_ridge(X, y; n_folds=5) -> Float64

Select optimal lambda for ridge regression.
"""
function select_lambda_cv_ridge(X, y; n_folds::Int=5, n_lambda::Int=50)
    n, p = size(X)

    lambdas = 10.0 .^ range(-4, 4, length=n_lambda)
    fold_idx = [mod(i-1, n_folds) + 1 for i in randperm(n)]

    cv_errors = zeros(n_lambda)

    for (l_idx, λ) in enumerate(lambdas)
        fold_errors = zeros(n_folds)

        for fold in 1:n_folds
            train_idx = findall(fold_idx .!= fold)
            test_idx = findall(fold_idx .== fold)

            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]

            β = (X_train' * X_train + λ * I) \ (X_train' * y_train)
            pred = X_test * β
            fold_errors[fold] = mean((y_test .- pred).^2)
        end

        cv_errors[l_idx] = mean(fold_errors)
    end

    return lambdas[argmin(cv_errors)]
end

"""
    RandomForestResult

Structure for random forest results.

# Fields
- `predictions::Vector{Float64}`: Predictions
- `importance::Vector{Float64}`: Variable importance scores
- `oob_error::Float64`: Out-of-bag error
- `n_trees::Int`: Number of trees
"""
struct RandomForestResult
    predictions::Vector{Float64}
    importance::Vector{Float64}
    oob_error::Float64
    n_trees::Int
end

"""
    random_forest(X::Matrix{Float64}, y::Vector{Float64}; kwargs...) -> RandomForestResult

Random forest for genomic prediction.

# Arguments
- `X`: Feature matrix
- `y`: Response vector

# Keyword Arguments
- `n_trees::Int=100`: Number of trees
- `max_depth::Int=10`: Maximum tree depth
- `min_samples_leaf::Int=5`: Minimum samples in leaf
- `mtry::Union{Int, Nothing}=nothing`: Features to consider at each split

# Example
```julia
result = random_forest(genotypes, phenotype; n_trees=500)

# Get variable importance
top_features = sortperm(result.importance, rev=true)[1:20]
```

# References
- Breiman (2001) Machine Learning
"""
function random_forest(
    X::Matrix{Float64},
    y::Vector{Float64};
    n_trees::Int=100,
    max_depth::Int=10,
    min_samples_leaf::Int=5,
    mtry::Union{Int, Nothing}=nothing,
    verbose::Bool=false
)
    n, p = size(X)

    if mtry === nothing
        mtry = max(1, round(Int, sqrt(p)))
    end

    # Storage for predictions and importance
    oob_predictions = zeros(n)
    oob_counts = zeros(Int, n)
    importance = zeros(p)

    trees = Vector{Dict}(undef, n_trees)

    if verbose
        prog = Progress(n_trees; desc="Building trees: ")
    end

    for t in 1:n_trees
        # Bootstrap sample
        boot_idx = rand(1:n, n)
        oob_idx = setdiff(1:n, unique(boot_idx))

        X_boot = X[boot_idx, :]
        y_boot = y[boot_idx]

        # Build tree
        tree = build_tree(X_boot, y_boot, 1, max_depth, min_samples_leaf, mtry)
        trees[t] = tree

        # OOB predictions
        for i in oob_idx
            pred = predict_tree(tree, X[i, :])
            oob_predictions[i] += pred
            oob_counts[i] += 1
        end

        # Variable importance (permutation-based)
        # Simplified: use split counts
        importance .+= get_split_counts(tree, p)

        if verbose
            next!(prog)
        end
    end

    # Aggregate OOB predictions
    oob_mask = oob_counts .> 0
    oob_predictions[oob_mask] ./= oob_counts[oob_mask]
    oob_error = mean((y[oob_mask] .- oob_predictions[oob_mask]).^2)

    # Normalize importance
    importance ./= sum(importance)

    # Full predictions (average across all trees)
    predictions = zeros(n)
    for t in 1:n_trees
        for i in 1:n
            predictions[i] += predict_tree(trees[t], X[i, :])
        end
    end
    predictions ./= n_trees

    return RandomForestResult(predictions, importance, oob_error, n_trees)
end

"""
    build_tree(X, y, depth, max_depth, min_samples, mtry) -> Dict

Build a single decision tree.
"""
function build_tree(X, y, depth, max_depth, min_samples_leaf, mtry)
    n, p = size(X)

    # Check stopping conditions
    if depth >= max_depth || n <= min_samples_leaf || var(y) < 1e-10
        return Dict(:is_leaf => true, :value => mean(y))
    end

    # Random feature subset
    feature_idx = randperm(p)[1:mtry]

    best_split = nothing
    best_gain = -Inf

    for j in feature_idx
        # Find best split for this feature
        split_val, gain = find_best_split(X[:, j], y)

        if gain > best_gain
            best_gain = gain
            best_split = (feature=j, threshold=split_val)
        end
    end

    if best_split === nothing || best_gain <= 0
        return Dict(:is_leaf => true, :value => mean(y))
    end

    # Split data
    left_mask = X[:, best_split.feature] .<= best_split.threshold
    right_mask = .!left_mask

    if sum(left_mask) < min_samples_leaf || sum(right_mask) < min_samples_leaf
        return Dict(:is_leaf => true, :value => mean(y))
    end

    # Recursively build children
    left_child = build_tree(X[left_mask, :], y[left_mask], depth+1, max_depth, min_samples_leaf, mtry)
    right_child = build_tree(X[right_mask, :], y[right_mask], depth+1, max_depth, min_samples_leaf, mtry)

    return Dict(
        :is_leaf => false,
        :feature => best_split.feature,
        :threshold => best_split.threshold,
        :left => left_child,
        :right => right_child
    )
end

"""
    find_best_split(x, y) -> (split_value, gain)

Find best split point for a single feature.
"""
function find_best_split(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    sorted_idx = sortperm(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

    best_split = x_sorted[1]
    best_gain = -Inf

    # Try splits between consecutive unique values
    total_var = var(y)
    left_sum = 0.0
    left_sq_sum = 0.0

    for i in 1:n-1
        left_sum += y_sorted[i]
        left_sq_sum += y_sorted[i]^2

        if x_sorted[i] == x_sorted[i+1]
            continue  # Skip if same value
        end

        n_left = i
        n_right = n - i

        left_mean = left_sum / n_left
        right_mean = (sum(y) - left_sum) / n_right

        # Variance reduction
        left_var = left_sq_sum / n_left - left_mean^2
        right_sq_sum = sum(y.^2) - left_sq_sum
        right_var = right_sq_sum / n_right - right_mean^2

        gain = total_var - (n_left * max(0, left_var) + n_right * max(0, right_var)) / n

        if gain > best_gain
            best_gain = gain
            best_split = (x_sorted[i] + x_sorted[i+1]) / 2
        end
    end

    return best_split, best_gain
end

"""
    predict_tree(tree, x) -> Float64

Predict using a single tree.
"""
function predict_tree(tree::Dict, x::Vector{Float64})
    if tree[:is_leaf]
        return tree[:value]
    end

    if x[tree[:feature]] <= tree[:threshold]
        return predict_tree(tree[:left], x)
    else
        return predict_tree(tree[:right], x)
    end
end

"""
    get_split_counts(tree, n_features) -> Vector{Float64}

Count feature usage in tree splits.
"""
function get_split_counts(tree::Dict, n_features::Int)
    counts = zeros(n_features)
    _count_splits!(tree, counts)
    return counts
end

function _count_splits!(tree::Dict, counts::Vector{Float64})
    if tree[:is_leaf]
        return
    end
    counts[tree[:feature]] += 1
    _count_splits!(tree[:left], counts)
    _count_splits!(tree[:right], counts)
end

"""
    gradient_boosting(X::Matrix{Float64}, y::Vector{Float64}; kwargs...) -> NamedTuple

Gradient boosting for genomic prediction.

# Arguments
- `X`: Feature matrix
- `y`: Response vector

# Keyword Arguments
- `n_estimators::Int=100`: Number of boosting iterations
- `learning_rate::Float64=0.1`: Shrinkage parameter
- `max_depth::Int=3`: Maximum tree depth
- `subsample::Float64=0.8`: Fraction of samples for each tree

# Example
```julia
result = gradient_boosting(genotypes, phenotype; n_estimators=200)
```

# References
- Friedman (2001) Ann. Stat.
"""
function gradient_boosting(
    X::Matrix{Float64},
    y::Vector{Float64};
    n_estimators::Int=100,
    learning_rate::Float64=0.1,
    max_depth::Int=3,
    subsample::Float64=0.8,
    min_samples_leaf::Int=5,
    verbose::Bool=false
)
    n, p = size(X)
    n_subsample = round(Int, n * subsample)

    # Initialize with mean
    predictions = fill(mean(y), n)
    residuals = y .- predictions

    trees = Vector{Dict}(undef, n_estimators)
    train_losses = zeros(n_estimators)

    if verbose
        prog = Progress(n_estimators; desc="Boosting: ")
    end

    for t in 1:n_estimators
        # Subsample
        sample_idx = randperm(n)[1:n_subsample]

        # Fit tree to residuals
        tree = build_tree(X[sample_idx, :], residuals[sample_idx], 1, max_depth, min_samples_leaf, p)
        trees[t] = tree

        # Update predictions
        for i in 1:n
            pred = predict_tree(tree, X[i, :])
            predictions[i] += learning_rate * pred
        end

        # Update residuals
        residuals = y .- predictions
        train_losses[t] = mean(residuals.^2)

        if verbose
            next!(prog)
        end
    end

    # Variable importance from all trees
    importance = zeros(p)
    for tree in trees
        importance .+= get_split_counts(tree, p)
    end
    importance ./= sum(importance)

    return (
        predictions = predictions,
        importance = importance,
        train_losses = train_losses,
        n_estimators = n_estimators,
        learning_rate = learning_rate
    )
end

"""
    feature_selection_stability(X::Matrix{Float64}, y::Vector{Float64};
                               method=:lasso, n_bootstrap=100) -> Vector{Float64}

Stability selection for robust feature selection.

# Arguments
- `X`: Feature matrix
- `y`: Response vector
- `method`: Selection method - :lasso, :elastic_net
- `n_bootstrap`: Number of bootstrap resamples

# Returns
Stability scores (selection frequency) for each feature

# Algorithm
1. Bootstrap resample data
2. Apply feature selection method
3. Record selected features
4. Repeat and compute selection frequencies

# Example
```julia
stability = feature_selection_stability(genotypes, phenotype; n_bootstrap=100)
stable_features = findall(stability .> 0.6)
```

# References
- Meinshausen & Bühlmann (2010) J. R. Stat. Soc. B
"""
function feature_selection_stability(
    X::Matrix{Float64},
    y::Vector{Float64};
    method::Symbol=:lasso,
    n_bootstrap::Int=100,
    verbose::Bool=false
)
    n, p = size(X)
    selection_counts = zeros(p)

    if verbose
        prog = Progress(n_bootstrap; desc="Stability selection: ")
    end

    for b in 1:n_bootstrap
        # Bootstrap sample (half of data for stability)
        sample_idx = randperm(n)[1:div(n, 2)]
        X_boot = X[sample_idx, :]
        y_boot = y[sample_idx]

        # Apply selection method
        if method == :lasso
            result = lasso(X_boot, y_boot; standardize=true)
            selected = findall(result.coefficients .!= 0)
        elseif method == :elastic_net
            result = elastic_net(X_boot, y_boot; alpha=0.5)
            selected = findall(result.coefficients .!= 0)
        end

        selection_counts[selected] .+= 1

        if verbose
            next!(prog)
        end
    end

    stability_scores = selection_counts ./ n_bootstrap

    return stability_scores
end
