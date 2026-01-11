# StatisticalGenomics.jl

[![Build Status](https://github.com/1958126580/Handbook-of-Statistical-Genomics/workflows/CI/badge.svg)](https://github.com/1958126580/Handbook-of-Statistical-Genomics/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Julia package for statistical genomics analysis based on the **Handbook of Statistical Genomics, 4th Edition**.

## Features

### Population Genetics
- Hardy-Weinberg equilibrium testing
- Allele and genotype frequency estimation
- Inbreeding coefficient calculation
- Linkage disequilibrium analysis (r², D, D')
- Haplotype estimation (EM algorithm)
- Genotype imputation (mean, mode, k-NN, EM)

### Evolutionary Genetics
- Wright-Fisher population simulation
- Nucleotide substitution models (JC69, K80, HKY85, GTR)
- Natural selection models
- Genetic drift analysis
- Effective population size estimation

### Coalescent Theory
- Kingman's coalescent simulation
- Ancestral recombination graphs (ARG)
- Multi-species coalescent
- Demographic inference from genealogies
- Site frequency spectrum analysis

### Phylogenetics
- Tree estimation (Neighbor-Joining, UPGMA)
- Molecular evolution models
- dN/dS ratio calculation
- McDonald-Kreitman test
- Positive selection detection

### GWAS
- Single-variant association testing (linear, logistic)
- Mixed model association (EMMA algorithm)
- Multiple testing correction (Bonferroni, FDR, genomic control)
- Gene-environment interaction testing
- Polygenic risk score calculation

### Population Structure
- Principal Component Analysis (PCA)
- Model-based clustering (STRUCTURE-like)
- Admixture analysis (f3, f4, D statistics)
- Ancient DNA analysis

### Gene Expression
- eQTL mapping (cis and trans)
- Differential expression analysis
- Expression normalization (TMM, DESeq)
- Co-expression network analysis
- GSEA pathway enrichment

### Epigenetics
- Methylation data processing
- Differential methylation analysis
- DMR detection
- EWAS association testing

### Microbiome
- Alpha diversity (Shannon, Simpson, Chao1)
- Beta diversity (Bray-Curtis, Jaccard)
- Compositional data analysis
- Metagenome-phenotype association

### Forensics & Conservation
- DNA profile matching
- Kinship estimation (KING-robust)
- IBD sharing analysis
- Population viability analysis

### Causal Inference
- Mendelian randomization (IVW, MR-Egger, weighted median)
- Heritability estimation
- Variant annotation
- Effect size estimation

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/1958126580/Handbook-of-Statistical-Genomics")
```

Or in development mode:
```julia
Pkg.develop(path="path/to/Handbook-of-Statistical-Genomics")
```

## Quick Start

```julia
using StatisticalGenomics

# Load PLINK data
gm = read_plink("data/study")

# Calculate allele frequencies
freqs = allele_frequencies(gm)

# Test Hardy-Weinberg equilibrium
hwe_results = hwe_test(gm)

# Run GWAS
phenotype = ContinuousPhenotype(pheno_values)
gwas_result = gwas_single_variant(gm, phenotype)

# Multiple testing correction
fdr = fdr_correction(gwas_result.pvalues)

# Create Manhattan plot
manhattan_plot(gwas_result)
```

## Examples

### Population Structure Analysis
```julia
# PCA for population stratification
pca_result = genetic_pca(gm; n_components=10)

# STRUCTURE-like clustering
cluster_result = structure_clustering(gm, 3)

# Admixture analysis
f3_result = f3_statistic(gm, target_pop, source1, source2)
```

### Mixed Model GWAS
```julia
# Calculate genetic relationship matrix
K = grm_matrix(gm)

# Run mixed model GWAS
result = mixed_model_gwas(gm, phenotype, K)
```

### Mendelian Randomization
```julia
# Two-sample MR
mr_result = mendelian_randomization(
    betas_exposure, ses_exposure,
    betas_outcome, ses_outcome;
    method=:ivw
)
```

### Coalescent Simulation
```julia
# Simulate coalescent tree
tree = coalescent_simulate(100; Ne=10000.0)

# Place mutations
mutations = simulate_mutations_on_tree(tree, 0.01)
```

## Project Structure

```
StatisticalGenomics.jl/
├── src/
│   ├── StatisticalGenomics.jl    # Main module
│   ├── types/                     # Core data types
│   ├── utils/                     # Utilities (stats, I/O, viz)
│   ├── population/                # Population genetics
│   ├── evolution/                 # Evolutionary models
│   ├── coalescent/                # Coalescent theory
│   ├── phylogenetics/             # Phylogenetic analysis
│   ├── structure/                 # Population structure
│   ├── gwas/                      # Association studies
│   ├── expression/                # Gene expression
│   ├── epigenetics/               # Methylation analysis
│   ├── microbiome/                # Microbiome analysis
│   ├── forensics/                 # Forensics & conservation
│   └── causal/                    # Causal inference
├── test/
│   └── runtests.jl
├── docs/
├── examples/
├── Project.toml
└── README.md
```

## Dependencies

- Julia 1.6+
- Distributions.jl
- DataFrames.jl
- StatsBase.jl
- HypothesisTests.jl
- Optim.jl
- ProgressMeter.jl
- JLD2.jl

## Documentation

Full documentation is available in the `docs/` directory.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Citation

If you use this package, please cite:

```bibtex
@book{balding2019handbook,
  title={Handbook of Statistical Genomics},
  author={Balding, David J and Moltke, Ida and Marioni, John},
  edition={4th},
  year={2019},
  publisher={John Wiley \& Sons}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

For questions or issues, please open a GitHub issue.
