# ============================================================================
# Example: Coalescent Simulation and Analysis
# ============================================================================

using StatisticalGenomics
using Random
using Statistics

Random.seed!(456)

println("=" ^ 60)
println("Coalescent Theory Example")
println("=" ^ 60)

# ============================================================================
# 1. Basic Coalescent Simulation
# ============================================================================
println("\n1. Simulating coalescent genealogies...")

n_samples = 20
Ne = 10000.0

# Single simulation
tree = coalescent_simulate(n_samples; Ne=Ne)

println("  - Sample size: $(tree.n_samples)")
println("  - Number of coalescence events: $(length(tree.coalescence_times))")
println("  - TMRCA: $(round(tree.tree_height, digits=1)) generations")
println("  - Total branch length: $(round(tree.total_branch_length, digits=1))")

# ============================================================================
# 2. Expected vs Observed TMRCA
# ============================================================================
println("\n2. Comparing observed TMRCA to theoretical expectation...")

result = time_to_mrca(n_samples; Ne=Ne, n_simulations=1000)

println("  - Theoretical E[TMRCA]: $(round(result.theoretical_mean, digits=1)) generations")
println("  - Simulated mean TMRCA: $(round(result.empirical_mean, digits=1)) generations")
println("  - Simulated SD: $(round(result.empirical_sd, digits=1))")
println("  - Difference: $(round(abs(result.empirical_mean - result.theoretical_mean) / result.theoretical_mean * 100, digits=1))%")

# ============================================================================
# 3. Expected Branch Length
# ============================================================================
println("\n3. Expected branch length...")

expected_L = expected_branch_lengths(n_samples; Ne=Ne)
println("  - Theoretical E[L]: $(round(expected_L, digits=1))")
println("  - Observed L: $(round(tree.total_branch_length, digits=1))")

# ============================================================================
# 4. Site Frequency Spectrum
# ============================================================================
println("\n4. Simulating site frequency spectrum...")

θ = 0.01  # Mutation rate
sfs_result = site_frequency_spectrum(n_samples, θ; n_simulations=500)

println("\n  Frequency class | Observed | Expected")
println("  " * "-" ^ 40)
for i in 1:min(10, length(sfs_result.observed))
    println("  $(lpad(i, 14)) | $(round(sfs_result.observed[i], digits=3))    | $(round(sfs_result.expected[i], digits=3))")
end

# ============================================================================
# 5. Theta Estimation
# ============================================================================
println("\n5. Estimating theta (4Neμ)...")

# Simulate many trees and count segregating sites
S_values = Int[]
for _ in 1:100
    t = coalescent_simulate(n_samples; Ne=Ne)
    muts = simulate_mutations_on_tree(t, θ)
    push!(S_values, muts.n_mutations)
end

S_mean = mean(S_values)
theta_est = estimate_theta_watterson(round(Int, S_mean), n_samples, 1000)

println("  - Mean segregating sites: $(round(S_mean, digits=1))")
println("  - Watterson's θ estimate: $(round(theta_est.theta_w * 1000, digits=4))")
println("  - True θ: $(θ)")

# ============================================================================
# 6. Coalescent with Recombination
# ============================================================================
println("\n6. Simulating ancestral recombination graph...")

ρ = 10.0  # Recombination rate
L = 1000.0  # Sequence length

arg = arg_simulate(10; ρ=ρ, L=L, Ne=Ne)

println("  - Sample size: $(arg.n_samples)")
println("  - Recombination events: $(arg.n_recombinations)")
println("  - Sequence length: $(arg.sequence_length)")

# ============================================================================
# 7. Tajima's D
# ============================================================================
println("\n7. Calculating Tajima's D...")

# Simulate under neutral model
n_test = 30
S_test = 25
π_test = 0.005 * 1000  # Pairwise differences

tajima = tajima_D(n_test, S_test, π_test)

println("  - Tajima's D: $(round(tajima.D, digits=3))")
println("  - P-value: $(round(tajima.pvalue, digits=4))")

interpretation = if tajima.D > 2
    "Balancing selection or population contraction"
elseif tajima.D < -2
    "Positive selection or population expansion"
else
    "Consistent with neutral evolution"
end
println("  - Interpretation: $interpretation")

# ============================================================================
# 8. Multi-species Coalescent
# ============================================================================
println("\n8. Multi-species coalescent simulation...")

species_tree = SpeciesTree(
    3,
    ["Human", "Chimpanzee", "Gorilla"],
    [6e6, 9e6],  # Divergence times (generations)
    [1e4, 1e4, 1e4],  # Population sizes
    Dict{Int, Vector{Int}}()
)

msc_result = gene_tree_species_tree(species_tree, 100; samples_per_species=2)

println("  - Gene trees simulated: 100")
println("  - Mean gene tree TMRCA: $(round(msc_result.mean_tmrca, digits=0)) generations")
println("  - SD of TMRCA: $(round(msc_result.sd_tmrca, digits=0))")

# ILS probability for Human-Chimp split
τ = 6e6
ils_prob = incomplete_lineage_sorting_probability(τ, 1e4)
println("  - P(ILS) at Human-Chimp divergence: $(round(ils_prob, digits=4))")

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 60)
println("Coalescent analysis complete!")
println("Key concepts demonstrated:")
println("  - Coalescent tree simulation and TMRCA")
println("  - Site frequency spectrum under neutrality")
println("  - Theta estimation methods")
println("  - Ancestral recombination graphs")
println("  - Tajima's D for selection/demography")
println("  - Multi-species coalescent and ILS")
println("=" ^ 60)
