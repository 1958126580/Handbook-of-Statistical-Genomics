using Documenter
using StatisticalGenomics

makedocs(
    sitename = "StatisticalGenomics.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [StatisticalGenomics],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorials" => [
            "GWAS Analysis" => "tutorials/gwas.md",
            "Population Structure" => "tutorials/structure.md",
            "Coalescent Theory" => "tutorials/coalescent.md",
        ],
        "API Reference" => [
            "Core Types" => "api/types.md",
            "Population Genetics" => "api/population.md",
            "GWAS" => "api/gwas.md",
            "Evolution" => "api/evolution.md",
            "Phylogenetics" => "api/phylogenetics.md",
        ],
    ]
)

deploydocs(
    repo = "github.com/1958126580/Handbook-of-Statistical-Genomics.git",
    devbranch = "main"
)
