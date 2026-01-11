# StatisticalGenomics.jl Documentation

Welcome to the documentation for **StatisticalGenomics.jl**, a comprehensive Julia package for statistical genomics analysis.

## Overview

StatisticalGenomics.jl provides a complete toolkit for genetic and genomic data analysis, implementing methods from the *Handbook of Statistical Genomics, 4th Edition*.

## Key Features

- **Population Genetics**: Hardy-Weinberg testing, linkage disequilibrium, haplotype estimation
- **Evolutionary Genetics**: Wright-Fisher models, substitution models, selection detection
- **Coalescent Theory**: Tree simulation, demographic inference, site frequency spectra
- **GWAS**: Association testing, mixed models, multiple testing correction
- **Population Structure**: PCA, clustering, admixture analysis
- **Expression Analysis**: eQTL mapping, differential expression, pathway enrichment
- **And more**: Epigenetics, microbiome, forensics, causal inference

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/1958126580/Handbook-of-Statistical-Genomics")
```

## Quick Start

```julia
using StatisticalGenomics

# Load data
gm = read_plink("data/study")

# Run GWAS
result = gwas_single_variant(gm, phenotype)

# Correct for multiple testing
fdr = fdr_correction(result.pvalues)
```

## Documentation Sections

- [Getting Started](getting_started.md) - Installation and first steps
- [Tutorials](tutorials/gwas.md) - Step-by-step guides
- [API Reference](api/types.md) - Complete function documentation

## Citation

```bibtex
@book{balding2019handbook,
  title={Handbook of Statistical Genomics},
  edition={4th},
  year={2019},
  publisher={Wiley}
}
```
