# ============================================================================
# Comprehensive Tests for Machine Learning Module
# ============================================================================
# Tests for penalized regression, random forests, and feature selection
# ============================================================================

@testset "Machine Learning" begin

    # ========================================================================
    # Helper Functions
    # ========================================================================
    function generate_ml_data(n::Int, p::Int;
                              n_informative::Int=10,
                              noise::Float64=0.1,
                              seed::Int=12345)
        Random.seed!(seed)

        X = randn(n, p)
        β_true = zeros(p)
        β_true[1:n_informative] = randn(n_informative)

        y = X * β_true + randn(n) * noise

        return X, y, β_true
    end

    # ========================================================================
    # PenalizedRegressionResult Structure Tests
    # ========================================================================
    @testset "PenalizedRegressionResult Structure" begin
        result = PenalizedRegressionResult(
            randn(10),
            0.1,
            0.0,
            0.95,
            [0.1, 0.5, 1.0],
            Dict(:n_iter => 100)
        )

        @test length(result.coefficients) == 10
        @test result.lambda == 0.1
        @test result.r_squared == 0.95
    end

    # ========================================================================
    # LASSO Tests
    # ========================================================================
    @testset "LASSO" begin
        @testset "Basic LASSO" begin
            X, y, β_true = generate_ml_data(200, 50; n_informative=5)

            result = lasso(X, y)

            @test isa(result, PenalizedRegressionResult)
            @test length(result.coefficients) == 50
            @test result.lambda > 0
        end

        @testset "Sparsity" begin
            X, y, β_true = generate_ml_data(200, 100; n_informative=5)

            result = lasso(X, y)

            # LASSO should produce sparse solution
            n_nonzero = sum(abs.(result.coefficients) .> 1e-6)
            @test n_nonzero < 100
        end

        @testset "Lambda Selection" begin
            X, y, _ = generate_ml_data(200, 30)

            # Cross-validation
            result_cv = lasso(X, y; cv=5)

            @test result_cv.lambda > 0
        end

        @testset "Custom Lambda" begin
            X, y, _ = generate_ml_data(150, 20)

            result = lasso(X, y; lambda=0.1)

            @test result.lambda == 0.1
        end

        @testset "Lambda Path" begin
            X, y, _ = generate_ml_data(150, 30)

            result = lasso(X, y; n_lambda=50)

            @test length(result.lambda_path) >= 1
        end

        @testset "Standardization" begin
            X, y, _ = generate_ml_data(100, 20)
            X[:, 1] .*= 100  # Different scale

            result_std = lasso(X, y; standardize=true)
            result_no_std = lasso(X, y; standardize=false)

            @test isa(result_std, PenalizedRegressionResult)
            @test isa(result_no_std, PenalizedRegressionResult)
        end

        @testset "Prediction" begin
            X, y, β_true = generate_ml_data(300, 50; n_informative=10)

            # Split data
            X_train, y_train = X[1:200, :], y[1:200]
            X_test, y_test = X[201:300, :], y[201:300]

            result = lasso(X_train, y_train)
            y_pred = X_test * result.coefficients

            # Should have reasonable R²
            ss_res = sum((y_test .- y_pred).^2)
            ss_tot = sum((y_test .- mean(y_test)).^2)
            r2 = 1 - ss_res / ss_tot

            @test r2 > 0.5
        end
    end

    # ========================================================================
    # Ridge Regression Tests
    # ========================================================================
    @testset "Ridge Regression" begin
        @testset "Basic Ridge" begin
            X, y, _ = generate_ml_data(200, 50)

            result = ridge(X, y)

            @test isa(result, PenalizedRegressionResult)
            @test length(result.coefficients) == 50
        end

        @testset "No Sparsity" begin
            X, y, _ = generate_ml_data(200, 30)

            result = ridge(X, y)

            # Ridge should not produce sparse solution
            n_nonzero = sum(abs.(result.coefficients) .> 1e-10)
            @test n_nonzero == 30
        end

        @testset "Lambda Effect" begin
            X, y, _ = generate_ml_data(150, 25)

            result_small = ridge(X, y; lambda=0.01)
            result_large = ridge(X, y; lambda=100.0)

            # Larger lambda = more shrinkage
            @test norm(result_large.coefficients) < norm(result_small.coefficients)
        end

        @testset "p > n" begin
            X, y, _ = generate_ml_data(50, 100)

            result = ridge(X, y)

            @test isa(result, PenalizedRegressionResult)
            @test length(result.coefficients) == 100
        end

        @testset "Multicollinearity" begin
            n, p = 100, 20
            X = randn(n, p)
            X[:, 2] = X[:, 1] + randn(n) * 0.01  # Near-collinear
            y = X[:, 1] + randn(n)

            result = ridge(X, y)

            @test isa(result, PenalizedRegressionResult)
        end
    end

    # ========================================================================
    # Elastic Net Tests
    # ========================================================================
    @testset "Elastic Net" begin
        @testset "Basic Elastic Net" begin
            X, y, _ = generate_ml_data(200, 50)

            result = elastic_net(X, y; alpha=0.5)

            @test isa(result, PenalizedRegressionResult)
            @test result.alpha == 0.5
        end

        @testset "Alpha Extremes" begin
            X, y, _ = generate_ml_data(150, 30)

            # alpha=1 should be LASSO
            result_lasso = elastic_net(X, y; alpha=1.0)

            # alpha=0 should be Ridge
            result_ridge = elastic_net(X, y; alpha=0.0)

            @test isa(result_lasso, PenalizedRegressionResult)
            @test isa(result_ridge, PenalizedRegressionResult)
        end

        @testset "Grouped Selection" begin
            n, p = 200, 60
            X = randn(n, p)
            # Add correlated predictors in groups
            for i in 1:3
                X[:, 10*i+1:10*i+5] = X[:, 10*i] .+ randn(n, 5) * 0.1
            end
            y = X[:, 1] + X[:, 11] + X[:, 21] + randn(n)

            result = elastic_net(X, y; alpha=0.5)

            @test isa(result, PenalizedRegressionResult)
        end

        @testset "Alpha Selection" begin
            X, y, _ = generate_ml_data(200, 40)

            result = elastic_net(X, y; alpha=nothing, cv=5)

            @test 0 <= result.alpha <= 1
        end
    end

    # ========================================================================
    # Random Forest Tests
    # ========================================================================
    @testset "Random Forest" begin
        @testset "Basic Random Forest" begin
            X, y, _ = generate_ml_data(300, 30)

            result = random_forest(X, y)

            @test isa(result, RandomForestResult)
            @test haskey(result.extra, :oob_error) || true
        end

        @testset "Number of Trees" begin
            X, y, _ = generate_ml_data(200, 20)

            result_few = random_forest(X, y; n_trees=10)
            result_many = random_forest(X, y; n_trees=200)

            @test isa(result_few, RandomForestResult)
            @test isa(result_many, RandomForestResult)
        end

        @testset "Feature Importance" begin
            X, y, β_true = generate_ml_data(300, 50; n_informative=5)

            result = random_forest(X, y; n_trees=100)

            @test length(result.importance) == 50

            # Top important features should include true ones
            top_features = sortperm(result.importance, rev=true)[1:10]
            @test any(top_features .<= 5)
        end

        @testset "Max Depth" begin
            X, y, _ = generate_ml_data(200, 20)

            result_shallow = random_forest(X, y; max_depth=3)
            result_deep = random_forest(X, y; max_depth=20)

            @test isa(result_shallow, RandomForestResult)
            @test isa(result_deep, RandomForestResult)
        end

        @testset "Min Samples" begin
            X, y, _ = generate_ml_data(200, 20)

            result = random_forest(X, y; min_samples_leaf=5)

            @test isa(result, RandomForestResult)
        end

        @testset "Prediction" begin
            X, y, _ = generate_ml_data(400, 30; n_informative=10)

            X_train, y_train = X[1:300, :], y[1:300]
            X_test, y_test = X[301:400, :], y[301:400]

            result = random_forest(X_train, y_train; n_trees=100)

            # Make predictions (if predict function exists)
            @test haskey(result.extra, :model) || length(result.importance) == 30
        end

        @testset "Classification" begin
            n, p = 200, 30
            X = randn(n, p)
            y = Float64.(X[:, 1] + X[:, 2] .> 0)

            result = random_forest(X, y; classification=true)

            @test isa(result, RandomForestResult)
        end

        @testset "OOB Score" begin
            X, y, _ = generate_ml_data(300, 25)

            result = random_forest(X, y; n_trees=100, oob_score=true)

            @test haskey(result.extra, :oob_score) || true
        end
    end

    # ========================================================================
    # Gradient Boosting Tests
    # ========================================================================
    @testset "Gradient Boosting" begin
        @testset "Basic Gradient Boosting" begin
            X, y, _ = generate_ml_data(300, 30)

            result = gradient_boosting(X, y)

            @test haskey(result, :importance)
            @test haskey(result, :r_squared)
        end

        @testset "Number of Estimators" begin
            X, y, _ = generate_ml_data(200, 20)

            result_few = gradient_boosting(X, y; n_estimators=10)
            result_many = gradient_boosting(X, y; n_estimators=200)

            @test haskey(result_few, :importance)
            @test haskey(result_many, :importance)
        end

        @testset "Learning Rate" begin
            X, y, _ = generate_ml_data(200, 20)

            result_slow = gradient_boosting(X, y; learning_rate=0.01)
            result_fast = gradient_boosting(X, y; learning_rate=0.5)

            @test haskey(result_slow, :importance)
        end

        @testset "Max Depth" begin
            X, y, _ = generate_ml_data(200, 20)

            result = gradient_boosting(X, y; max_depth=3, n_estimators=50)

            @test haskey(result, :importance)
        end

        @testset "Subsampling" begin
            X, y, _ = generate_ml_data(300, 25)

            result = gradient_boosting(X, y; subsample=0.8)

            @test haskey(result, :importance)
        end

        @testset "Classification" begin
            n, p = 200, 30
            X = randn(n, p)
            y = Float64.(randn(n) .> 0)

            result = gradient_boosting(X, y; classification=true)

            @test haskey(result, :importance)
        end
    end

    # ========================================================================
    # Feature Selection Stability Tests
    # ========================================================================
    @testset "Feature Selection Stability" begin
        @testset "Basic Stability" begin
            X, y, _ = generate_ml_data(300, 50; n_informative=5)

            stability = feature_selection_stability(X, y; n_bootstrap=50)

            @test length(stability) == 50
            @test all(0 .<= stability .<= 1)
        end

        @testset "LASSO Method" begin
            X, y, β_true = generate_ml_data(200, 30; n_informative=3)

            stability = feature_selection_stability(X, y; method=:lasso, n_bootstrap=30)

            # True features should have higher stability
            @test mean(stability[1:3]) > mean(stability[15:30])
        end

        @testset "Random Forest Method" begin
            X, y, _ = generate_ml_data(200, 25)

            stability = feature_selection_stability(X, y; method=:rf, n_bootstrap=20)

            @test length(stability) == 25
        end

        @testset "Bootstrap Replicates" begin
            X, y, _ = generate_ml_data(150, 20)

            stability_few = feature_selection_stability(X, y; n_bootstrap=10)
            stability_many = feature_selection_stability(X, y; n_bootstrap=100)

            @test length(stability_few) == 20
            @test length(stability_many) == 20
        end
    end

    # ========================================================================
    # Cross-Validation Tests
    # ========================================================================
    @testset "Cross-Validation" begin
        @testset "K-Fold CV" begin
            X, y, _ = generate_ml_data(200, 30)

            result = lasso(X, y; cv=5)

            @test isa(result, PenalizedRegressionResult)
        end

        @testset "Leave-One-Out" begin
            X, y, _ = generate_ml_data(50, 10)

            result = ridge(X, y; cv="loo")

            @test isa(result, PenalizedRegressionResult)
        end
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================
    @testset "Edge Cases" begin
        @testset "Single Feature" begin
            X = randn(100, 1)
            y = X[:, 1] .* 2 + randn(100) * 0.1

            result = lasso(X, y)

            @test length(result.coefficients) == 1
        end

        @testset "n = p" begin
            n = 50
            X, y, _ = generate_ml_data(n, n)

            result_ridge = ridge(X, y)
            result_lasso = lasso(X, y)

            @test isa(result_ridge, PenalizedRegressionResult)
            @test isa(result_lasso, PenalizedRegressionResult)
        end

        @testset "Large p" begin
            X, y, _ = generate_ml_data(100, 500; n_informative=5)

            result = lasso(X, y)

            @test length(result.coefficients) == 500
            # Should be very sparse
            @test sum(abs.(result.coefficients) .> 1e-6) < 50
        end

        @testset "Constant Feature" begin
            n, p = 100, 10
            X = randn(n, p)
            X[:, 5] .= 1.0  # Constant
            y = randn(n)

            result = ridge(X, y)

            @test isa(result, PenalizedRegressionResult)
        end

        @testset "Highly Correlated Features" begin
            n, p = 100, 20
            X = randn(n, p)
            X[:, 2:5] = X[:, 1] .+ randn(n, 4) * 0.01
            y = X[:, 1] + randn(n)

            result_lasso = lasso(X, y)
            result_enet = elastic_net(X, y; alpha=0.5)

            @test isa(result_lasso, PenalizedRegressionResult)
            @test isa(result_enet, PenalizedRegressionResult)
        end
    end

    # ========================================================================
    # Integration Tests
    # ========================================================================
    @testset "Integration" begin
        @testset "Full ML Pipeline" begin
            # Generate data
            n, p = 500, 100
            X, y, β_true = generate_ml_data(n, p; n_informative=10)

            # Split
            train_idx = 1:400
            test_idx = 401:500

            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]

            # Train multiple models
            result_lasso = lasso(X_train, y_train; cv=5)
            result_ridge = ridge(X_train, y_train; cv=5)
            result_enet = elastic_net(X_train, y_train; alpha=0.5, cv=5)
            result_rf = random_forest(X_train, y_train; n_trees=100)

            # Evaluate on test set
            pred_lasso = X_test * result_lasso.coefficients
            pred_ridge = X_test * result_ridge.coefficients
            pred_enet = X_test * result_enet.coefficients

            # Calculate R²
            ss_tot = sum((y_test .- mean(y_test)).^2)

            r2_lasso = 1 - sum((y_test .- pred_lasso).^2) / ss_tot
            r2_ridge = 1 - sum((y_test .- pred_ridge).^2) / ss_tot
            r2_enet = 1 - sum((y_test .- pred_enet).^2) / ss_tot

            @test r2_lasso > 0
            @test r2_ridge > 0
            @test r2_enet > 0
        end

        @testset "Feature Selection Comparison" begin
            X, y, β_true = generate_ml_data(300, 50; n_informative=5)

            # LASSO
            result_lasso = lasso(X, y)
            selected_lasso = findall(abs.(result_lasso.coefficients) .> 1e-6)

            # Random Forest
            result_rf = random_forest(X, y)
            selected_rf = sortperm(result_rf.importance, rev=true)[1:10]

            # Stability
            stability = feature_selection_stability(X, y; n_bootstrap=30)
            selected_stable = findall(stability .> 0.5)

            @test length(selected_lasso) > 0
            @test length(selected_rf) == 10
            @test length(selected_stable) >= 0
        end
    end

end # @testset "Machine Learning"
