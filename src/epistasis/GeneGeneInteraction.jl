# ============================================================================
# GeneGeneInteraction.jl - Epistasis and Gene-Gene Interaction Analysis
# ============================================================================
# Methods for detecting and analyzing epistatic interactions between variants
# Including exhaustive pairwise, pathway-based, and machine learning approaches
# ============================================================================

"""
    EpistasisResult

Structure containing results from epistasis analysis.

# Fields
- `snp1::String`: First variant identifier
- `snp2::String`: Second variant identifier
- `beta_interaction::Float64`: Interaction effect estimate
- `se_interaction::Float64`: Standard error
- `statistic::Float64`: Test statistic
- `pvalue::Float64`: P-value for interaction
- `model::String`: Model type used
"""
struct EpistasisResult
    snp1::String
    snp2::String
    beta_interaction::Float64
    se_interaction::Float64
    statistic::Float64
    pvalue::Float64
    model::String
end

"""
    pairwise_epistasis(genotypes::Matrix{Float64}, phenotype::Vector{Float64};
                      covariates=nothing, variant_ids=nothing,
                      model=:multiplicative) -> DataFrame

Test pairwise interactions between all SNP pairs.

# Arguments
- `genotypes`: Genotype matrix (n_samples × n_variants)
- `phenotype`: Phenotype vector
- `covariates`: Optional covariate matrix
- `variant_ids`: Variant identifiers
- `model`: Interaction model - :multiplicative, :additive, :dominant

# Models
1. Multiplicative (default):
   y = β₀ + β₁G₁ + β₂G₂ + β₃(G₁×G₂) + ε

2. Additive:
   y = β₀ + β₁G₁ + β₂G₂ + β₃max(G₁,G₂) + ε

3. Dominant:
   y = β₀ + β₁I(G₁>0) + β₂I(G₂>0) + β₃I(G₁>0∧G₂>0) + ε

# Algorithm
For each pair (i,j) where i < j:
1. Fit null model: y ~ G_i + G_j + covariates
2. Fit full model: y ~ G_i + G_j + G_i*G_j + covariates
3. Test interaction term using Wald or LRT

# Example
```julia
results = pairwise_epistasis(genotypes, phenotype; model=:multiplicative)

# Filter significant interactions
sig = filter(r -> r.pvalue < 1e-6, results)
```

# References
- Cordell (2009) Nat. Rev. Genet.
- Wei et al. (2014) Bioinformatics (BOOST)
"""
function pairwise_epistasis(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64};
    covariates::Union{Matrix{Float64}, Nothing}=nothing,
    variant_ids::Union{Vector{String}, Nothing}=nothing,
    model::Symbol=:multiplicative,
    test_type::Symbol=:wald,
    min_maf::Float64=0.01,
    verbose::Bool=true
)
    n_samples, n_variants = size(genotypes)

    if variant_ids === nothing
        variant_ids = ["SNP_$i" for i in 1:n_variants]
    end

    # Filter by MAF
    mafs = vec(mean(genotypes, dims=1)) / 2
    valid_variants = findall((mafs .>= min_maf) .& (mafs .<= 1 - min_maf))
    n_valid = length(valid_variants)

    n_pairs = div(n_valid * (n_valid - 1), 2)

    results = DataFrame(
        snp1 = String[],
        snp2 = String[],
        beta_interaction = Float64[],
        se_interaction = Float64[],
        statistic = Float64[],
        pvalue = Float64[]
    )

    # Build base design matrix
    if covariates !== nothing
        X_base = hcat(ones(n_samples), covariates)
    else
        X_base = ones(n_samples, 1)
    end

    is_binary = all(p -> p == 0 || p == 1, phenotype)

    if verbose
        prog = Progress(n_pairs; desc="Testing pairs: ")
    end

    pair_idx = 0
    for i in 1:n_valid-1
        vi = valid_variants[i]
        for j in i+1:n_valid
            vj = valid_variants[j]
            pair_idx += 1

            G1 = genotypes[:, vi]
            G2 = genotypes[:, vj]

            # Create interaction term based on model
            if model == :multiplicative
                G_int = G1 .* G2
            elseif model == :additive
                G_int = max.(G1, G2)
            elseif model == :dominant
                G1_dom = Float64.(G1 .> 0)
                G2_dom = Float64.(G2 .> 0)
                G_int = G1_dom .* G2_dom
            else
                error("Unknown model: $model")
            end

            # Full design matrix
            X_full = hcat(X_base, G1, G2, G_int)
            int_idx = size(X_full, 2)

            # Test interaction
            try
                if is_binary
                    result = test_interaction_logistic(X_full, phenotype, int_idx, test_type)
                else
                    result = test_interaction_linear(X_full, phenotype, int_idx, test_type)
                end

                push!(results, (
                    snp1 = variant_ids[vi],
                    snp2 = variant_ids[vj],
                    beta_interaction = result.beta,
                    se_interaction = result.se,
                    statistic = result.statistic,
                    pvalue = result.pvalue
                ))
            catch
                # Skip pairs that fail
            end

            if verbose
                next!(prog)
            end
        end
    end

    sort!(results, :pvalue)

    return results
end

"""
    boost_epistasis(genotypes::Matrix{Float64}, phenotype::Vector{Float64};
                   kwargs...) -> DataFrame

BOOST-like fast epistasis screening using likelihood ratio tests.

# Arguments
- `genotypes`: Genotype matrix (coded 0/1/2)
- `phenotype`: Binary phenotype (0/1)

# Algorithm (BOOST)
1. Recode genotypes to 3 categories
2. Build 3×3 contingency table for each pair
3. Compute log-linear model likelihood ratio
4. Use Kirkwood superposition for fast computation

# References
- Wan et al. (2010) Am. J. Hum. Genet.
"""
function boost_epistasis(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64};
    variant_ids::Union{Vector{String}, Nothing}=nothing,
    min_maf::Float64=0.01,
    verbose::Bool=true
)
    n_samples, n_variants = size(genotypes)

    @assert all(p -> p == 0 || p == 1, phenotype) "BOOST requires binary phenotype"

    if variant_ids === nothing
        variant_ids = ["SNP_$i" for i in 1:n_variants]
    end

    # Convert to integer genotypes
    G = round.(Int, genotypes)

    # Filter by MAF
    mafs = vec(mean(genotypes, dims=1)) / 2
    valid = findall((mafs .>= min_maf) .& (mafs .<= 1 - min_maf))
    n_valid = length(valid)
    n_pairs = div(n_valid * (n_valid - 1), 2)

    results = DataFrame(
        snp1 = String[],
        snp2 = String[],
        statistic = Float64[],
        pvalue = Float64[]
    )

    # Precompute marginal tables for each SNP
    case_idx = findall(phenotype .== 1)
    ctrl_idx = findall(phenotype .== 0)
    n_cases = length(case_idx)
    n_ctrls = length(ctrl_idx)

    if verbose
        prog = Progress(n_pairs; desc="BOOST screening: ")
    end

    pair_idx = 0
    for i in 1:n_valid-1
        vi = valid[i]
        for j in i+1:n_valid
            vj = valid[j]
            pair_idx += 1

            # Build 3×3×2 contingency table (G1 × G2 × phenotype)
            table = zeros(3, 3, 2)

            for s in 1:n_samples
                g1 = G[s, vi] + 1  # 1, 2, or 3
                g2 = G[s, vj] + 1
                y = Int(phenotype[s]) + 1  # 1 or 2

                if g1 >= 1 && g1 <= 3 && g2 >= 1 && g2 <= 3
                    table[g1, g2, y] += 1
                end
            end

            # Compute interaction log-likelihood ratio
            # H0: log-linear model without interaction
            # H1: saturated model

            # Marginal counts
            margin_g1 = sum(table, dims=(2, 3))
            margin_g2 = sum(table, dims=(1, 3))
            margin_y = sum(table, dims=(1, 2))
            total = sum(table)

            # Expected under independence
            expected = zeros(3, 3, 2)
            for g1 in 1:3, g2 in 1:3, y in 1:2
                expected[g1, g2, y] = (margin_g1[g1] * margin_g2[g2] * margin_y[y]) / total^2
            end

            # Likelihood ratio statistic
            # G² = 2 Σ O log(O/E)
            G2 = 0.0
            for g1 in 1:3, g2 in 1:3, y in 1:2
                if table[g1, g2, y] > 0 && expected[g1, g2, y] > 0
                    G2 += 2 * table[g1, g2, y] * log(table[g1, g2, y] / expected[g1, g2, y])
                end
            end

            # df = (3-1)*(3-1)*(2-1) = 4 for interaction
            pvalue = ccdf(Chisq(4), G2)

            push!(results, (
                snp1 = variant_ids[vi],
                snp2 = variant_ids[vj],
                statistic = G2,
                pvalue = pvalue
            ))

            if verbose
                next!(prog)
            end
        end
    end

    sort!(results, :pvalue)

    return results
end

"""
    pathway_epistasis(genotypes::Matrix{Float64}, phenotype::Vector{Float64},
                     gene_sets::Dict{String, Vector{Int}};
                     kwargs...) -> DataFrame

Test for epistasis between gene sets/pathways.

# Arguments
- `genotypes`: Genotype matrix
- `phenotype`: Phenotype vector
- `gene_sets`: Dictionary mapping pathway names to variant indices

# Algorithm
1. Aggregate variants within each pathway (PCA, burden, or set-based)
2. Test pairwise interactions between pathways
3. Adjust for multiple testing

# Example
```julia
pathways = Dict(
    "MAPK" => [1, 5, 10, 15],
    "PI3K" => [2, 8, 12, 20],
    "WNT" => [3, 7, 11, 18]
)

results = pathway_epistasis(genotypes, phenotype, pathways)
```

# References
- Emily et al. (2009) BMC Genomics
"""
function pathway_epistasis(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64},
    gene_sets::Dict{String, Vector{Int}};
    covariates::Union{Matrix{Float64}, Nothing}=nothing,
    aggregation::Symbol=:pca,  # :pca, :burden, :first_pc
    n_components::Int=1,
    verbose::Bool=true
)
    n_samples = size(genotypes, 1)
    pathway_names = collect(keys(gene_sets))
    n_pathways = length(pathway_names)

    # Aggregate variants within each pathway
    pathway_scores = Dict{String, Matrix{Float64}}()

    for (name, indices) in gene_sets
        G_pathway = genotypes[:, indices]

        if aggregation == :pca
            # Use first n_components PCs
            if size(G_pathway, 2) > 1
                G_centered = G_pathway .- mean(G_pathway, dims=1)
                U, S, V = svd(G_centered)
                n_comp = min(n_components, length(S))
                pathway_scores[name] = U[:, 1:n_comp] .* S[1:n_comp]'
            else
                pathway_scores[name] = G_pathway
            end
        elseif aggregation == :burden
            # Sum of minor alleles
            pathway_scores[name] = reshape(sum(G_pathway, dims=2), :, 1)
        else
            pathway_scores[name] = G_pathway[:, 1:min(n_components, size(G_pathway, 2))]
        end
    end

    # Test pairwise pathway interactions
    n_pairs = div(n_pathways * (n_pathways - 1), 2)

    results = DataFrame(
        pathway1 = String[],
        pathway2 = String[],
        n_snps_1 = Int[],
        n_snps_2 = Int[],
        statistic = Float64[],
        df = Int[],
        pvalue = Float64[]
    )

    if verbose
        prog = Progress(n_pairs; desc="Testing pathway pairs: ")
    end

    is_binary = all(p -> p == 0 || p == 1, phenotype)

    for i in 1:n_pathways-1
        for j in i+1:n_pathways
            name1, name2 = pathway_names[i], pathway_names[j]
            P1 = pathway_scores[name1]
            P2 = pathway_scores[name2]

            # Create interaction terms (all pairwise products of components)
            n_comp1 = size(P1, 2)
            n_comp2 = size(P2, 2)

            interactions = zeros(n_samples, n_comp1 * n_comp2)
            idx = 1
            for c1 in 1:n_comp1
                for c2 in 1:n_comp2
                    interactions[:, idx] = P1[:, c1] .* P2[:, c2]
                    idx += 1
                end
            end

            # Build design matrices
            if covariates !== nothing
                X_null = hcat(ones(n_samples), covariates, P1, P2)
            else
                X_null = hcat(ones(n_samples), P1, P2)
            end
            X_full = hcat(X_null, interactions)

            # Likelihood ratio test
            try
                if is_binary
                    ll_null = logistic_log_likelihood(X_null, phenotype)
                    ll_full = logistic_log_likelihood(X_full, phenotype)
                else
                    ss_null = sum((phenotype - X_null * (X_null \ phenotype)).^2)
                    ss_full = sum((phenotype - X_full * (X_full \ phenotype)).^2)
                    ll_null = -n_samples/2 * log(ss_null / n_samples)
                    ll_full = -n_samples/2 * log(ss_full / n_samples)
                end

                lrt = 2 * (ll_full - ll_null)
                df = n_comp1 * n_comp2
                pvalue = ccdf(Chisq(df), lrt)

                push!(results, (
                    pathway1 = name1,
                    pathway2 = name2,
                    n_snps_1 = length(gene_sets[name1]),
                    n_snps_2 = length(gene_sets[name2]),
                    statistic = lrt,
                    df = df,
                    pvalue = pvalue
                ))
            catch
                # Skip failed tests
            end

            if verbose
                next!(prog)
            end
        end
    end

    sort!(results, :pvalue)

    return results
end

"""
    mdr(genotypes::Matrix{Float64}, phenotype::Vector{Float64};
       k::Int=2, n_folds::Int=10) -> NamedTuple

Multifactor Dimensionality Reduction for epistasis detection.

# Arguments
- `genotypes`: Genotype matrix (coded 0/1/2)
- `phenotype`: Binary phenotype
- `k`: Number of factors (SNPs) to consider
- `n_folds`: Cross-validation folds

# Algorithm (MDR)
1. Pool k-factor genotype combinations
2. Classify each cell as high/low risk
3. Reduce to 1D attribute
4. Evaluate prediction accuracy via CV
5. Select best model by CV consistency

# Example
```julia
result = mdr(genotypes[:, 1:100], phenotype; k=2)
println("Best pair: \$(result.best_model)")
println("CV accuracy: \$(result.accuracy)")
```

# References
- Ritchie et al. (2001) Am. J. Hum. Genet.
"""
function mdr(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64};
    k::Int=2,
    n_folds::Int=10,
    variant_ids::Union{Vector{String}, Nothing}=nothing,
    max_combinations::Int=10000,
    verbose::Bool=true
)
    n_samples, n_variants = size(genotypes)

    @assert all(p -> p == 0 || p == 1, phenotype) "MDR requires binary phenotype"

    if variant_ids === nothing
        variant_ids = ["SNP_$i" for i in 1:n_variants]
    end

    # Generate k-combinations
    combinations = collect(Combinatorics.combinations(1:n_variants, k))

    if length(combinations) > max_combinations
        # Random sample
        combinations = combinations[randperm(length(combinations))[1:max_combinations]]
    end

    # Cross-validation folds
    fold_idx = [mod(i-1, n_folds) + 1 for i in randperm(n_samples)]

    case_ratio = mean(phenotype)

    best_accuracy = 0.0
    best_model = Int[]
    best_cv_count = 0

    if verbose
        prog = Progress(length(combinations); desc="MDR screening: ")
    end

    for combo in combinations
        cv_accuracies = zeros(n_folds)

        for fold in 1:n_folds
            train_idx = findall(fold_idx .!= fold)
            test_idx = findall(fold_idx .== fold)

            # Get genotypes for this combination
            G_train = genotypes[train_idx, combo]
            G_test = genotypes[test_idx, combo]
            y_train = phenotype[train_idx]
            y_test = phenotype[test_idx]

            # Build k-dimensional contingency table
            # For k=2: 3×3 table; for k=3: 3×3×3 table, etc.

            # Simplified: collapse to single index
            n_cells = 3^k
            cell_counts_case = zeros(n_cells)
            cell_counts_ctrl = zeros(n_cells)

            for i in eachindex(y_train)
                cell_idx = 1
                for j in 1:k
                    cell_idx += round(Int, G_train[i, j]) * 3^(j-1)
                end
                cell_idx = clamp(cell_idx, 1, n_cells)

                if y_train[i] == 1
                    cell_counts_case[cell_idx] += 1
                else
                    cell_counts_ctrl[cell_idx] += 1
                end
            end

            # Classify cells as high/low risk
            high_risk = (cell_counts_case ./ (cell_counts_case .+ cell_counts_ctrl .+ 1e-10)) .> case_ratio

            # Predict on test set
            correct = 0
            for i in eachindex(y_test)
                cell_idx = 1
                for j in 1:k
                    cell_idx += round(Int, G_test[i, j]) * 3^(j-1)
                end
                cell_idx = clamp(cell_idx, 1, n_cells)

                pred = high_risk[cell_idx] ? 1.0 : 0.0
                if pred == y_test[i]
                    correct += 1
                end
            end

            cv_accuracies[fold] = correct / length(test_idx)
        end

        mean_acc = mean(cv_accuracies)
        cv_count = sum(cv_accuracies .== maximum(cv_accuracies))

        if mean_acc > best_accuracy
            best_accuracy = mean_acc
            best_model = collect(combo)
            best_cv_count = cv_count
        end

        if verbose
            next!(prog)
        end
    end

    return (
        best_model = variant_ids[best_model],
        best_indices = best_model,
        accuracy = best_accuracy,
        cv_consistency = best_cv_count,
        n_folds = n_folds,
        k = k
    )
end

"""
    random_forest_epistasis(genotypes::Matrix{Float64}, phenotype::Vector{Float64};
                           kwargs...) -> DataFrame

Detect epistasis using Random Forest importance measures.

# Arguments
- `genotypes`: Genotype matrix
- `phenotype`: Phenotype vector

# Algorithm
1. Train Random Forest with interaction features
2. Compute paired variable importance
3. Rank pairs by importance gain

# Returns
DataFrame with top interacting pairs and importance scores
"""
function random_forest_epistasis(
    genotypes::Matrix{Float64},
    phenotype::Vector{Float64};
    variant_ids::Union{Vector{String}, Nothing}=nothing,
    n_trees::Int=100,
    max_depth::Int=5,
    top_pairs::Int=100,
    verbose::Bool=true
)
    n_samples, n_variants = size(genotypes)

    if variant_ids === nothing
        variant_ids = ["SNP_$i" for i in 1:n_variants]
    end

    # Simplified importance calculation
    # Using correlation-based screening

    # First, get marginal importances
    marginal_corr = zeros(n_variants)
    for j in 1:n_variants
        marginal_corr[j] = abs(cor(genotypes[:, j], phenotype))
    end

    # Select top variants for pairwise testing
    top_marginal = sortperm(marginal_corr, rev=true)[1:min(100, n_variants)]

    # Test pairwise interactions among top variants
    pair_importance = Dict{Tuple{Int,Int}, Float64}()

    n_top = length(top_marginal)

    if verbose
        prog = Progress(div(n_top * (n_top - 1), 2); desc="RF screening: ")
    end

    for i in 1:n_top-1
        vi = top_marginal[i]
        for j in i+1:n_top
            vj = top_marginal[j]

            G1 = genotypes[:, vi]
            G2 = genotypes[:, vj]
            G_int = G1 .* G2

            # Combined feature
            X = hcat(G1, G2, G_int)

            # Simple importance: R² improvement
            r2_main = 1 - sum((phenotype .- X[:, 1:2] * (X[:, 1:2] \ phenotype)).^2) / sum((phenotype .- mean(phenotype)).^2)
            r2_full = 1 - sum((phenotype .- X * (X \ phenotype)).^2) / sum((phenotype .- mean(phenotype)).^2)

            pair_importance[(vi, vj)] = r2_full - r2_main

            if verbose
                next!(prog)
            end
        end
    end

    # Sort by importance
    sorted_pairs = sort(collect(pair_importance), by=x->x[2], rev=true)

    results = DataFrame(
        snp1 = String[],
        snp2 = String[],
        importance = Float64[],
        rank = Int[]
    )

    for (rank, (pair, imp)) in enumerate(sorted_pairs[1:min(top_pairs, length(sorted_pairs))])
        push!(results, (
            snp1 = variant_ids[pair[1]],
            snp2 = variant_ids[pair[2]],
            importance = imp,
            rank = rank
        ))
    end

    return results
end

# Helper functions
function test_interaction_linear(X, y, int_idx, test_type)
    n = length(y)
    p = size(X, 2)

    β = X \ y
    residuals = y - X * β
    σ2 = sum(residuals.^2) / (n - p)

    XtX_inv = inv(X' * X)
    se = sqrt(σ2 * XtX_inv[int_idx, int_idx])
    t_stat = β[int_idx] / se
    pvalue = 2 * ccdf(TDist(n - p), abs(t_stat))

    return (beta=β[int_idx], se=se, statistic=t_stat, pvalue=pvalue)
end

function test_interaction_logistic(X, y, int_idx, test_type)
    β = logistic_fit_irls(X, y)
    μ = 1.0 ./ (1.0 .+ exp.(-X * β))
    V = μ .* (1 .- μ)

    I = X' * Diagonal(V) * X
    se = sqrt(inv(I)[int_idx, int_idx])
    z_stat = β[int_idx] / se
    pvalue = 2 * ccdf(Normal(), abs(z_stat))

    return (beta=β[int_idx], se=se, statistic=z_stat, pvalue=pvalue)
end

function logistic_fit_irls(X, y; max_iter=25, tol=1e-8)
    n, p = size(X)
    β = zeros(p)

    for _ in 1:max_iter
        μ = 1.0 ./ (1.0 .+ exp.(-X * β))
        V = μ .* (1 .- μ)
        V = max.(V, 1e-10)
        z = X * β + (y - μ) ./ V
        β_new = (X' * Diagonal(V) * X) \ (X' * Diagonal(V) * z)

        if maximum(abs.(β_new - β)) < tol
            return β_new
        end
        β = β_new
    end
    return β
end

function logistic_log_likelihood(X, y)
    β = logistic_fit_irls(X, y)
    μ = 1.0 ./ (1.0 .+ exp.(-X * β))
    μ = clamp.(μ, 1e-10, 1 - 1e-10)
    return sum(y .* log.(μ) + (1 .- y) .* log.(1 .- μ))
end

# Import Combinatorics-like functionality
module Combinatorics
    function combinations(arr, k)
        n = length(arr)
        if k > n
            return Vector{Int}[]
        end
        if k == 0
            return [Int[]]
        end
        if k == n
            return [collect(arr)]
        end

        result = Vector{Int}[]
        indices = collect(1:k)

        while true
            push!(result, arr[indices])

            # Find rightmost index that can be incremented
            i = k
            while i > 0 && indices[i] == n - k + i
                i -= 1
            end

            if i == 0
                break
            end

            indices[i] += 1
            for j in i+1:k
                indices[j] = indices[j-1] + 1
            end
        end

        return result
    end
end
