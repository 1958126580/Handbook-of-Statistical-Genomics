# ============================================================================
# Example: Basic GWAS Analysis
# ============================================================================
# This example demonstrates a typical genome-wide association study workflow

using StatisticalGenomics
using Random
using Plots

Random.seed!(42)

println("=" ^ 60)
println("StatisticalGenomics.jl - GWAS Example")
println("=" ^ 60)

# ============================================================================
# 1. Generate Simulated Data
# ============================================================================
println("\n1. Generating simulated genotype data...")

n_samples = 500
n_variants = 1000
n_causal = 5

# Generate random genotypes
genotype_data = rand(0:2, n_samples, n_variants)

# Create GenotypeMatrix
chromosomes = vcat([fill(c, div(n_variants, 22)) for c in 1:22]...)[1:n_variants]
positions = collect(1:n_variants) .* 10000
variant_ids = ["rs$(100000 + i)" for i in 1:n_variants]

gm = GenotypeMatrix(
    genotype_data,
    ["Sample_$i" for i in 1:n_samples],
    variant_ids,
    chromosomes,
    positions
)

println("  - Samples: $(n_samples(gm))")
println("  - Variants: $(n_variants(gm))")

# ============================================================================
# 2. Generate Phenotype with Causal Variants
# ============================================================================
println("\n2. Generating phenotype with causal variants...")

causal_indices = [10, 50, 200, 500, 800]
effect_sizes = [0.3, 0.25, 0.2, 0.35, 0.28]

phenotype_values = zeros(n_samples)
for (idx, effect) in zip(causal_indices, effect_sizes)
    phenotype_values .+= effect .* Float64.(genotype_data[:, idx])
end
phenotype_values .+= randn(n_samples) * 0.5  # Add noise

phenotype = ContinuousPhenotype(phenotype_values, "Trait1")
println("  - Causal variants: $(causal_indices)")
println("  - Phenotype variance: $(round(var(phenotype_values), digits=3))")

# ============================================================================
# 3. Quality Control
# ============================================================================
println("\n3. Performing quality control...")

# Calculate minor allele frequencies
mafs = minor_allele_frequency(gm)
println("  - MAF range: $(round(minimum(mafs), digits=3)) - $(round(maximum(mafs), digits=3))")

# Calculate missing rates
miss_rates = missing_rate(gm)
println("  - Max missing rate: $(round(maximum(miss_rates), digits=3))")

# Hardy-Weinberg equilibrium test (for first 100 variants as example)
println("  - Testing HWE for first 100 variants...")
hwe_pvals = Float64[]
for j in 1:min(100, n_variants)
    genos = collect(skipmissing(gm.data[:, j]))
    result = hwe_test(genos)
    push!(hwe_pvals, result.pvalue)
end
n_hwe_fail = count(p -> p < 0.001, hwe_pvals)
println("  - HWE failures (p < 0.001): $n_hwe_fail / 100")

# ============================================================================
# 4. Run GWAS
# ============================================================================
println("\n4. Running genome-wide association study...")

gwas_result = gwas_single_variant(gm, phenotype)

n_tested = count(!isnan, gwas_result.pvalues)
println("  - Variants tested: $n_tested")
println("  - Test type: $(gwas_result.test_type)")

# ============================================================================
# 5. Multiple Testing Correction
# ============================================================================
println("\n5. Applying multiple testing correction...")

# Genomic control
gc_result = genomic_control(gwas_result.pvalues)
println("  - Genomic inflation factor (λ): $(round(gc_result.lambda_gc, digits=3))")

# Bonferroni correction
bonf = bonferroni_correction(gwas_result.pvalues)
println("  - Bonferroni threshold: $(bonf.threshold)")
println("  - Bonferroni significant: $(bonf.n_significant)")

# FDR correction
fdr = fdr_correction(gwas_result.pvalues)
println("  - FDR significant (q < 0.05): $(fdr.n_significant)")

# ============================================================================
# 6. Identify Top Hits
# ============================================================================
println("\n6. Top association signals:")

# Sort by p-value
order = sortperm(gwas_result.pvalues)
println("\n  Rank | Variant    | Chr | Position   | Beta   | SE     | P-value")
println("  " * "-" ^ 70)

for i in 1:min(10, length(order))
    j = order[i]
    if !isnan(gwas_result.pvalues[j])
        is_causal = j in causal_indices ? " *" : ""
        println("  $(lpad(i, 4)) | $(rpad(gwas_result.variant_ids[j], 10)) | " *
                "$(lpad(gwas_result.chromosomes[j], 3)) | " *
                "$(lpad(gwas_result.positions[j], 10)) | " *
                "$(lpad(round(gwas_result.betas[j], digits=3), 6)) | " *
                "$(lpad(round(gwas_result.standard_errors[j], digits=3), 6)) | " *
                "$(gwas_result.pvalues[j])$is_causal")
    end
end
println("\n  * = Known causal variant")

# ============================================================================
# 7. Summary Statistics
# ============================================================================
println("\n7. Summary Statistics:")
println("  - Genome-wide significance (p < 5e-8): $(count(p -> !isnan(p) && p < 5e-8, gwas_result.pvalues))")
println("  - Suggestive significance (p < 1e-5): $(count(p -> !isnan(p) && p < 1e-5, gwas_result.pvalues))")

# Check how many causal variants were detected
detected = 0
for idx in causal_indices
    if gwas_result.pvalues[idx] < 0.05 / n_variants
        detected += 1
    end
end
println("  - Causal variants detected: $detected / $(length(causal_indices))")

println("\n" * "=" ^ 60)
println("GWAS analysis complete!")
println("=" ^ 60)
