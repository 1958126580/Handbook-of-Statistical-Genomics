# Getting Started

This guide will help you get started with StatisticalGenomics.jl.

## Installation

### Prerequisites

- Julia 1.9 or later
- At least 4GB RAM recommended for large-scale analyses

### Installing the Package

```julia
using Pkg
Pkg.add(url="https://github.com/1958126580/Handbook-of-Statistical-Genomics")
```

For development:
```julia
Pkg.develop(path="/path/to/Handbook-of-Statistical-Genomics")
```

## Loading the Package

```julia
using StatisticalGenomics
```

## Basic Workflow

### 1. Loading Data

#### From PLINK files
```julia
gm = read_plink("path/to/data")  # Reads .bed, .bim, .fam
```

#### From VCF files
```julia
gm = read_vcf("path/to/data.vcf")
```

### 2. Quality Control

```julia
# Calculate allele frequencies
mafs = minor_allele_frequency(gm)

# Test Hardy-Weinberg equilibrium
hwe_results = [hwe_test(gm.data[:, j]) for j in 1:n_variants(gm)]

# Calculate linkage disequilibrium
ld_mat = ld_matrix(gm)
```

### 3. Association Analysis

```julia
# Create phenotype
phenotype = ContinuousPhenotype(values, "trait_name")

# Run GWAS
result = gwas_single_variant(gm, phenotype)

# Apply FDR correction
fdr = fdr_correction(result.pvalues)
```

### 4. Population Structure

```julia
# PCA
pca_result = genetic_pca(gm; n_components=10)

# Clustering
cluster_result = structure_clustering(gm, 3)
```

## Next Steps

- See the [GWAS Tutorial](tutorials/gwas.md) for a complete example
- Explore the [API Reference](api/types.md) for all available functions
