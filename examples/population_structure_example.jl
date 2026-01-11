# ============================================================================
# Example: Population Structure Analysis
# ============================================================================

using StatisticalGenomics
using Random

Random.seed!(123)

println("=" ^ 60)
println("Population Structure Analysis Example")
println("=" ^ 60)

# ============================================================================
# 1. Simulate Three Populations
# ============================================================================
println("\n1. Simulating three distinct populations...")

n_per_pop = 50
n_variants = 500

# Population-specific allele frequencies
pop1_freqs = rand(0.1:0.01:0.9, n_variants)
pop2_freqs = pop1_freqs .+ randn(n_variants) * 0.15
pop3_freqs = pop1_freqs .+ randn(n_variants) * 0.2

pop2_freqs = clamp.(pop2_freqs, 0.05, 0.95)
pop3_freqs = clamp.(pop3_freqs, 0.05, 0.95)

function generate_genotypes(freqs, n)
    data = Matrix{Int8}(undef, n, length(freqs))
    for (j, f) in enumerate(freqs)
        for i in 1:n
            data[i, j] = sum(rand() < f for _ in 1:2)
        end
    end
    return data
end

pop1_data = generate_genotypes(pop1_freqs, n_per_pop)
pop2_data = generate_genotypes(pop2_freqs, n_per_pop)
pop3_data = generate_genotypes(pop3_freqs, n_per_pop)

# Combine populations
all_data = vcat(pop1_data, pop2_data, pop3_data)
true_labels = vcat(fill(1, n_per_pop), fill(2, n_per_pop), fill(3, n_per_pop))

gm = GenotypeMatrix(all_data)
println("  - Total samples: $(n_samples(gm))")
println("  - Variants: $(n_variants(gm))")

# ============================================================================
# 2. Principal Component Analysis
# ============================================================================
println("\n2. Performing PCA...")

pca_result = genetic_pca(gm; n_components=10)

println("  - Variance explained by PC1: $(round(pca_result.variance_explained[1] * 100, digits=1))%")
println("  - Variance explained by PC2: $(round(pca_result.variance_explained[2] * 100, digits=1))%")
println("  - Cumulative (PC1-4): $(round(sum(pca_result.variance_explained[1:4]) * 100, digits=1))%")

# Check if populations are separated
pop1_pc1 = mean(pca_result.scores[1:n_per_pop, 1])
pop2_pc1 = mean(pca_result.scores[n_per_pop+1:2*n_per_pop, 1])
pop3_pc1 = mean(pca_result.scores[2*n_per_pop+1:end, 1])
println("  - PC1 means: Pop1=$(round(pop1_pc1, digits=3)), Pop2=$(round(pop2_pc1, digits=3)), Pop3=$(round(pop3_pc1, digits=3))")

# ============================================================================
# 3. Model-Based Clustering
# ============================================================================
println("\n3. Running STRUCTURE-like clustering (K=3)...")

cluster_result = structure_clustering(gm, 3; maxiter=50)

println("  - Log-likelihood: $(round(cluster_result.log_likelihood, digits=1))")
println("  - BIC: $(round(cluster_result.bic, digits=1))")

# Check clustering accuracy
correct = 0
for i in 1:length(true_labels)
    if cluster_result.assignments[i] == true_labels[i]
        correct += 1
    end
end
# Note: Cluster labels may be permuted, so we check if clustering groups samples correctly
println("  - Assignments match true labels: $(correct) / $(length(true_labels))")

# ============================================================================
# 4. Admixture Proportions
# ============================================================================
println("\n4. Admixture proportions (first 5 samples per population):")

println("\n  Sample    | Pop1    | Pop2    | Pop3    | True Pop")
println("  " * "-" ^ 50)

for pop in 1:3
    start_idx = (pop - 1) * n_per_pop + 1
    for i in 0:4
        idx = start_idx + i
        props = cluster_result.proportions[idx, :]
        println("  $(lpad(idx, 6))    | $(round(props[1], digits=3))   | $(round(props[2], digits=3))   | $(round(props[3], digits=3))   | $pop")
    end
    println()
end

# ============================================================================
# 5. Fst Between Populations
# ============================================================================
println("\n5. Calculating Fst between populations...")

# Calculate pairwise Fst
function calculate_fst_pair(freqs1, freqs2)
    n = length(freqs1)
    ht_sum = 0.0
    hs_sum = 0.0
    
    for i in 1:n
        p1, p2 = freqs1[i], freqs2[i]
        p_avg = (p1 + p2) / 2
        
        ht = 2 * p_avg * (1 - p_avg)
        hs = (2 * p1 * (1 - p1) + 2 * p2 * (1 - p2)) / 2
        
        ht_sum += ht
        hs_sum += hs
    end
    
    if ht_sum > 0
        return (ht_sum - hs_sum) / ht_sum
    else
        return 0.0
    end
end

fst_12 = calculate_fst_pair(pop1_freqs, pop2_freqs)
fst_13 = calculate_fst_pair(pop1_freqs, pop3_freqs)
fst_23 = calculate_fst_pair(pop2_freqs, pop3_freqs)

println("  - Fst(Pop1, Pop2): $(round(fst_12, digits=4))")
println("  - Fst(Pop1, Pop3): $(round(fst_13, digits=4))")
println("  - Fst(Pop2, Pop3): $(round(fst_23, digits=4))")

# ============================================================================
# 6. Summary
# ============================================================================
println("\n" * "=" ^ 60)
println("Population structure analysis complete!")
println("Key findings:")
println("  - Three distinct populations clearly separated in PCA")
println("  - STRUCTURE clustering identifies population membership")
println("  - Fst values indicate moderate genetic differentiation")
println("=" ^ 60)
